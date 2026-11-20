//! Gradient quantization foundations.
//!
//! upstream: src/treelearner/gradient_discretizer.{cpp,hpp} (v4.7.0),
//! `DiscretizeGradients` and `SetNumBitsInHistogramBin`.
//! These primitives are not yet connected to the serial tree learner (T2.7).

pub mod int_histogram;
pub mod mt19937;

use self::mt19937::Mt19937;
use crate::error::{Error, Result};
use crate::utils::common::madd;

/// Bytes of alignment for upstream's `SIZE_ALIGNED` (`Threading::BlockInfo`).
const K_ALIGNED_SIZE: usize = 32;

/// upstream `Threading::BlockInfo(num_threads, cnt, min_cnt_per_block, ...)`: the block count and
/// aligned block size used to seed the stochastic rounding noise per block.
fn block_info(num_threads: usize, cnt: usize, min_cnt_per_block: usize) -> (usize, usize) {
    let n_block = num_threads
        .max(1)
        .min(cnt.div_ceil(min_cnt_per_block.max(1)));
    if n_block > 1 {
        let size = cnt.div_ceil(n_block);
        (n_block, size.div_ceil(K_ALIGNED_SIZE) * K_ALIGNED_SIZE)
    } else {
        (n_block, cnt)
    }
}

/// Reusable discretizer with upstream's packed per-row layout, for nearest or stochastic rounding.
///
/// The low byte is the Hessian and the signed high byte is the gradient. Scales
/// remain `f64`, while input derivatives are upstream's `score_t` (`f32`).
pub struct GradientDiscretizer {
    packed: Vec<i16>,
    bins: u8,
    constant_hessian: bool,
    gradient_scale: f64,
    hessian_scale: f64,
    /// Stochastic rounding state (`stochastic_rounding=true`); empty for nearest rounding.
    stochastic: bool,
    /// Per-row uniform `[0, 1)` rounding noise, precomputed once (upstream `Init`).
    gradient_random_values: Vec<f64>,
    hessian_random_values: Vec<f64>,
    /// Draws the per-tree rotation offset into the noise arrays (upstream
    /// `random_values_use_start_eng_`); advances once per `discretize_stochastic`.
    use_start_eng: Mt19937,
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
            stochastic: false,
            gradient_random_values: Vec::new(),
            hessian_random_values: Vec::new(),
            use_start_eng: Mt19937::new(0),
        })
    }

    /// Enable stochastic rounding: precompute the per-row `[0, 1)` noise and seed the rotation
    /// engine (upstream `GradientDiscretizer::Init`). The noise is generated per block exactly as
    /// upstream does — block `b` seeds `mt19937(seed + b)` for gradients and `mt19937(seed + b +
    /// num_threads)` for Hessians — so the values match for the same `num_threads`, which also
    /// makes the result reproducible for a fixed thread count (upstream is not thread-invariant
    /// here, since the block layout depends on the thread count).
    pub fn init_stochastic(&mut self, seed: u32, num_threads: usize) {
        let num_data = self.packed.len();
        self.use_start_eng = Mt19937::new(seed);
        self.gradient_random_values = vec![0.0; num_data];
        self.hessian_random_values = vec![0.0; num_data];
        let threads = num_threads.max(1);
        let (n_block, block_size) = block_info(threads, num_data, 512);
        for b in 0..n_block {
            let start = b * block_size;
            let end = (start + block_size).min(num_data);
            let mut grad_eng = Mt19937::new(seed.wrapping_add(b as u32));
            let mut hess_eng =
                Mt19937::new(seed.wrapping_add(b as u32).wrapping_add(threads as u32));
            for i in start..end {
                self.gradient_random_values[i] = grad_eng.uniform_real_01();
                self.hessian_random_values[i] = hess_eng.uniform_real_01();
            }
        }
        self.stochastic = true;
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
        let (inverse_gradient, inverse_hessian) = self.update_scales(gradients, hessians)?;
        for ((packed, &gradient), &hessian) in self.packed.iter_mut().zip(gradients).zip(hessians) {
            let gradient = f64::from(gradient);
            let offset = if gradient >= 0.0 { 0.5 } else { -0.5 };
            let gradient = madd(gradient, inverse_gradient, offset) as i8;
            let hessian = if self.constant_hessian {
                1
            } else {
                madd(f64::from(hessian), inverse_hessian, 0.5) as i8
            };
            *packed = (i16::from(gradient) << 8) | i16::from(hessian as u8);
        }
        Ok(())
    }

    /// Validate the derivatives and recompute `(gradient_scale, Hessian_scale)` and their
    /// inverses (upstream `DiscretizeGradients`'s scale update). Shared by both rounding modes.
    fn update_scales(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<(f64, f64)> {
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
        Ok((1.0 / self.gradient_scale, 1.0 / self.hessian_scale))
    }

    /// Discretize with stochastic rounding (upstream `DiscretizeGradients`, stochastic path).
    ///
    /// Each row's gradient (and non-constant Hessian) is rounded by adding the precomputed
    /// `[0, 1)` noise before truncation, the noise indexed through a per-tree rotation offset
    /// drawn here (upstream `random_values_use_start`). `init_stochastic` must have run first.
    pub fn discretize_stochastic(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<()> {
        if !self.stochastic {
            return Err(Error::InvalidParameter(
                "discretize_stochastic called before init_stochastic".into(),
            ));
        }
        let (inverse_gradient, inverse_hessian) = self.update_scales(gradients, hessians)?;
        let num_data = self.packed.len();
        // upstream draws the rotation offset once per call: uniform_int_distribution(0, num_data).
        let start = self.use_start_eng.uniform_int_inclusive(num_data as u32) as usize;
        for (i, packed) in self.packed.iter_mut().enumerate() {
            let pos = (i + start) % num_data;
            let gradient = f64::from(gradients[i]);
            // Signed noise: + for a non-negative gradient, - otherwise (upstream).
            let noise = self.gradient_random_values[pos];
            let signed_noise = if gradient >= 0.0 { noise } else { -noise };
            let gradient = madd(gradient, inverse_gradient, signed_noise) as i8;
            let hessian = if self.constant_hessian {
                1
            } else {
                madd(
                    f64::from(hessians[i]),
                    inverse_hessian,
                    self.hessian_random_values[pos],
                ) as i8
            };
            *packed = (i16::from(gradient) << 8) | i16::from(hessian as u8);
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
