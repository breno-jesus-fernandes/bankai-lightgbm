//! Gradient quantization foundations.
//!
//! upstream: src/treelearner/gradient_discretizer.{cpp,hpp} (v4.7.0),
//! `DiscretizeGradients` and `SetNumBitsInHistogramBin`.
//! These primitives are not yet connected to the serial tree learner (T2.7).

pub mod int_histogram;
pub mod mt19937;

use crate::error::{Error, Result};
use crate::utils::common::madd;

/// Reusable nearest-rounding discretizer with upstream's packed per-row layout.
///
/// The low byte is the Hessian and the signed high byte is the gradient. Scales
/// remain `f64`, while input derivatives are upstream's `score_t` (`f32`).
/// Stochastic rounding and the quantized tree learner are separate T2.7 work.
pub struct GradientDiscretizer {
    packed: Vec<i16>,
    bins: u8,
    constant_hessian: bool,
    gradient_scale: f64,
    hessian_scale: f64,
}

impl GradientDiscretizer {
    /// Allocate the per-row buffer once, for a nonempty dataset and 2..=127 bins.
    ///
    /// Larger Hessian bin values cannot be represented as upstream's signed
    /// `int8_t` derivatives. Refuse that case rather than reproduce undefined
    /// floating-to-integer conversions (SPEC-NUM-10).
    pub fn new(num_data: usize, bins: u8, constant_hessian: bool) -> Result<Self> {
        if num_data == 0 {
            return Err(Error::InvalidParameter(
                "gradient discretization requires at least one row".into(),
            ));
        }
        if !(2..=127).contains(&bins) {
            return Err(Error::Unsupported(
                "gradient discretization requires num_grad_quant_bins in 2..=127".into(),
            ));
        }
        Ok(Self {
            packed: vec![0; num_data],
            bins,
            constant_hessian,
            gradient_scale: 0.0,
            hessian_scale: 0.0,
        })
    }

    /// Discretize using signed nearest rounding (halfway values round away from zero).
    ///
    /// upstream: GradientDiscretizer::DiscretizeGradients, stochastic_rounding=false.
    /// Scales are recalculated for every call, without allocating a new row buffer.
    /// Nonfinite derivatives and negative Hessians are rejected before changing it.
    /// A zero maximum gradient or variable Hessian is refused because upstream's
    /// zero scale leads to an undefined NaN-to-int8 conversion (SPEC-NUM-10).
    /// Constant Hessians always have integer value one and need no such cast.
    pub fn discretize_nearest(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<()> {
        if gradients.len() != self.packed.len() || hessians.len() != self.packed.len() {
            return Err(Error::InvalidParameter(
                "gradient and Hessian lengths must match the discretizer row count".into(),
            ));
        }
        if gradients.iter().any(|x| !x.is_finite())
            || hessians.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(Error::InvalidParameter(
                "gradient discretization requires finite gradients and nonnegative finite Hessians"
                    .into(),
            ));
        }
        let max_gradient = gradients.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        let max_hessian = hessians.iter().copied().fold(0.0_f32, f32::max);
        if max_gradient == 0.0 || (!self.constant_hessian && max_hessian == 0.0) {
            return Err(Error::Unsupported(
                "gradient discretization with zero maximum gradient or Hessian (upstream undefined conversion)"
                    .into(),
            ));
        }
        // Integer division is deliberate: upstream divides num_grad_quant_bins by 2
        // before converting it to double, including for odd quantization bin counts.
        self.gradient_scale = f64::from(max_gradient) / f64::from(self.bins / 2);
        self.hessian_scale = if self.constant_hessian {
            f64::from(max_hessian)
        } else {
            f64::from(max_hessian) / f64::from(self.bins)
        };
        let inverse_gradient = 1.0 / self.gradient_scale;
        let inverse_hessian = 1.0 / self.hessian_scale;
        // Round one gradient to its signed int8, away from zero at the halfway point.
        let quantize_gradient = |gradient: f32| -> i16 {
            let gradient = f64::from(gradient);
            let offset = if gradient >= 0.0 { 0.5 } else { -0.5 };
            i16::from(madd(gradient, inverse_gradient, offset) as i8)
        };
        // The constant-Hessian branch is lifted out of the per-row loop: each row's Hessian is
        // the integer one, so the loop body is a pure gradient quantization. This keeps the two
        // cases in separate loops rather than a per-row test, which also stabilizes the hot
        // loop's codegen against unrelated changes elsewhere in the crate.
        if self.constant_hessian {
            for (packed, &gradient) in self.packed.iter_mut().zip(gradients) {
                *packed = (quantize_gradient(gradient) << 8) | 1;
            }
        } else {
            for ((packed, &gradient), &hessian) in
                self.packed.iter_mut().zip(gradients).zip(hessians)
            {
                let hessian = madd(f64::from(hessian), inverse_hessian, 0.5) as i8;
                *packed = (quantize_gradient(gradient) << 8) | i16::from(hessian as u8);
            }
        }
        Ok(())
    }

    /// Packed per-row signed gradient/high-byte and Hessian/low-byte statistics.
    pub fn packed_gradients_hessians(&self) -> &[i16] {
        &self.packed
    }

    /// Multiply an integer gradient sum by this scale to recover its score units.
    pub fn gradient_scale(&self) -> f64 {
        self.gradient_scale
    }

    /// Multiply an integer Hessian sum by this scale to recover its score units.
    pub fn hessian_scale(&self) -> f64 {
        self.hessian_scale
    }
}

/// Upstream's per-leaf statistic width, using strict 256 and 65536 boundaries.
///
/// upstream: GradientDiscretizer::SetNumBitsInHistogramBin (v4.7.0).
/// The packed histogram needs two components of this width per bin.
pub fn histogram_bits(num_data: usize, num_grad_quant_bins: u8) -> u8 {
    let max_stat = num_data.saturating_mul(usize::from(num_grad_quant_bins));
    if max_stat < 256 {
        8
    } else if max_stat < 65536 {
        16
    } else {
        32
    }
}
