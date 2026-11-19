//! Invariant tests for the packed integer gradient/Hessian histogram.
//!
//! upstream: src/io/dense_bin.hpp (v4.7.0) `ConstructHistogramIntInner`. The
//! discretized per-row derivative is a packed int16 (signed int8 gradient in the
//! high byte, uint8 Hessian in the low byte). The histogram accumulates both into
//! one integer per bin: the gradient in the high half, the Hessian in the low
//! half, at 8/16/32-bit width. The reference here is the brute-force per-bin sum
//! of the discretized derivatives, which the packing must reproduce exactly.
#![allow(clippy::unwrap_used)]

use lgbm_core::treelearner::gradient_discretizer::int_histogram::{
    HistWidth, construct_int_histogram,
};

/// Brute-force per-bin (gradient_sum, Hessian_sum) over packed derivatives.
fn reference(
    packed: &[i16],
    bins: &[u32],
    num_bins: usize,
    constant_hessian: bool,
) -> Vec<(i64, i64)> {
    let mut out = vec![(0i64, 0i64); num_bins];
    for (&p, &b) in packed.iter().zip(bins) {
        let grad = i64::from((p >> 8) as i8);
        let hess = if constant_hessian {
            1
        } else {
            i64::from((p & 0xff) as u8)
        };
        out[b as usize].0 += grad;
        out[b as usize].1 += hess;
    }
    out
}

fn pack(grad: i8, hess: u8) -> i16 {
    (i16::from(grad) << 8) | i16::from(hess)
}

#[test]
fn int_histogram_matches_bruteforce_all_widths() {
    // Small signed gradients and Hessians, kept within the 8-bit per-bin range
    // so all three widths accumulate the same sums without overflow.
    let packed: Vec<i16> = [
        (3i8, 2u8),
        (-5, 4),
        (1, 1),
        (-2, 7),
        (6, 3),
        (-1, 2),
        (0, 5),
        (4, 1),
    ]
    .iter()
    .map(|&(g, h)| pack(g, h))
    .collect();
    let bins = [0u32, 1, 0, 2, 1, 2, 0, 1];
    let num_bins = 3;

    for constant in [false, true] {
        let expected = reference(&packed, &bins, num_bins, constant);
        for width in [HistWidth::Bits8, HistWidth::Bits16, HistWidth::Bits32] {
            let got = construct_int_histogram(&packed, &bins, num_bins, width, constant);
            assert_eq!(got, expected, "width {width:?}, constant {constant}");
        }
    }
}

#[test]
fn int_histogram_holds_sums_overflowing_the_narrower_width() {
    // The gradient occupies the high half and the Hessian the low half of the
    // accumulator, so a width holds a leaf only while |gradient_sum| < 2^(bits-1)
    // and Hessian_sum < 2^bits (upstream picks the width from rows * bins).

    // Gradient/Hessian sums that overflow 8 bits but fit a 16-bit half.
    let packed16: Vec<i16> = (0..500).map(|_| pack(50, 50)).collect();
    let bins16 = vec![0u32; 500];
    let expected16 = reference(&packed16, &bins16, 1, false);
    assert_eq!(expected16, vec![(25_000, 25_000)]);
    for width in [HistWidth::Bits16, HistWidth::Bits32] {
        assert_eq!(
            construct_int_histogram(&packed16, &bins16, 1, width, false),
            expected16
        );
    }

    // Sums that overflow a 16-bit half and need the 32-bit width.
    let packed32: Vec<i16> = (0..1000).map(|_| pack(100, 100)).collect();
    let bins32 = vec![0u32; 1000];
    let expected32 = reference(&packed32, &bins32, 1, false);
    assert_eq!(expected32, vec![(100_000, 100_000)]);
    assert_eq!(
        construct_int_histogram(&packed32, &bins32, 1, HistWidth::Bits32, false),
        expected32
    );
}

#[test]
fn int_histogram_empty_bins_are_zero() {
    let packed = [pack(7, 3)];
    let bins = [2u32];
    let got = construct_int_histogram(&packed, &bins, 4, HistWidth::Bits16, false);
    assert_eq!(got, vec![(0, 0), (0, 0), (7, 3), (0, 0)]);
}
