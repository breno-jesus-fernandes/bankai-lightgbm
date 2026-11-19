//! Packed integer gradient/Hessian histogram for quantized training.
//!
//! upstream: src/io/dense_bin.hpp (v4.7.0) `ConstructHistogramIntInner`.
//!
//! The discretized per-row derivative is a packed `int16`: the signed `int8`
//! gradient in the high byte and the `uint8` Hessian in the low byte (produced by
//! [`super::GradientDiscretizer`]). Upstream accumulates both parts into a single
//! integer per bin: the Hessian occupies the low `HIST_BITS` bits and the gradient
//! the next `HIST_BITS` bits, so `out[bin] += packed` sums both halves in one add.
//! The accumulator width (`HIST_BITS` = 8, 16 or 32, held in `int16`/`int32`/`int64`)
//! is chosen per leaf so the Hessian sum never carries into the gradient part
//! (`GradientDiscretizer::SetNumBitsInHistogramBin`, mirrored by
//! [`super::histogram_bits`]).

/// Accumulator width for one packed histogram, matching upstream's three kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistWidth {
    /// `int16` accumulator, gradient and Hessian in 8-bit halves.
    Bits8,
    /// `int32` accumulator, gradient and Hessian in 16-bit halves.
    Bits16,
    /// `int64` accumulator, gradient and Hessian in 32-bit halves.
    Bits32,
}

/// Build the packed integer histogram and return `(gradient_sum, Hessian_sum)`
/// per bin, both widened to `i64`.
///
/// `packed[i]` is the discretized derivative of the row whose feature bin is
/// `bins[i]`. With `constant_hessian`, each row contributes a Hessian of one (the
/// low byte is ignored), matching upstream's `| 1` packing. The caller selects
/// `width` from [`super::histogram_bits`]; a narrower width than the data needs
/// overflows exactly as upstream's fixed-width accumulator would.
pub fn construct_int_histogram(
    packed: &[i16],
    bins: &[u32],
    num_bins: usize,
    width: HistWidth,
    constant_hessian: bool,
) -> Vec<(i64, i64)> {
    match width {
        HistWidth::Bits8 => construct::<i16>(packed, bins, num_bins, 8, constant_hessian),
        HistWidth::Bits16 => construct::<i32>(packed, bins, num_bins, 16, constant_hessian),
        HistWidth::Bits32 => construct::<i64>(packed, bins, num_bins, 32, constant_hessian),
    }
}

/// A packed accumulator whose low/high halves hold the Hessian/gradient sums.
trait Packed: Copy {
    const ZERO: Self;
    /// Pack a signed gradient (high half) and a Hessian low byte (low half).
    fn pack(gradient: i8, hessian_low: u8, hist_bits: u32, constant_hessian: bool) -> Self;
    fn add(self, other: Self) -> Self;
    /// Unpack to `(gradient_sum, Hessian_sum)` using the low `hist_bits` for the Hessian.
    fn unpack(self, hist_bits: u32) -> (i64, i64);
}

macro_rules! impl_packed {
    ($t:ty, $wide:ty) => {
        impl Packed for $t {
            const ZERO: Self = 0;

            #[inline]
            fn pack(gradient: i8, hessian_low: u8, hist_bits: u32, constant_hessian: bool) -> Self {
                let hessian: $t = if constant_hessian { 1 } else { hessian_low as $t };
                ((gradient as $t) << hist_bits) | hessian
            }

            #[inline]
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }

            #[inline]
            fn unpack(self, hist_bits: u32) -> (i64, i64) {
                // The Hessian is non-negative and fits the low `hist_bits`, so an
                // arithmetic shift recovers the signed gradient sum exactly.
                let mask: $t = ((1 as $wide << hist_bits) - 1) as $t;
                let hessian = (self & mask) as i64;
                let gradient = (self >> hist_bits) as i64;
                (gradient, hessian)
            }
        }
    };
}

impl_packed!(i16, i32);
impl_packed!(i32, i64);
impl_packed!(i64, i128);

fn construct<P: Packed>(
    packed: &[i16],
    bins: &[u32],
    num_bins: usize,
    hist_bits: u32,
    constant_hessian: bool,
) -> Vec<(i64, i64)> {
    let mut acc = vec![P::ZERO; num_bins];
    for (&p, &b) in packed.iter().zip(bins) {
        let gradient = (p >> 8) as i8;
        let hessian_low = (p & 0xff) as u8;
        let value = P::pack(gradient, hessian_low, hist_bits, constant_hessian);
        let slot = &mut acc[b as usize];
        *slot = slot.add(value);
    }
    acc.into_iter().map(|a| a.unpack(hist_bits)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_gradient_unpacks_correctly_at_each_width() {
        let packed = [(i16::from(-3i8) << 8) | 5];
        let bins = [0u32];
        for width in [HistWidth::Bits8, HistWidth::Bits16, HistWidth::Bits32] {
            assert_eq!(
                construct_int_histogram(&packed, &bins, 1, width, false),
                vec![(-3, 5)]
            );
        }
    }
}
