//! Property tests for bin mappers (AGENTS.md section 4, ROADMAP P2 binning).
//!
//! These check invariants that must hold for every input, not only the golden fixtures:
//! bins stay in range, binning is monotone in the value, and the binary form round-trips.

#![allow(clippy::expect_used)]

use lgbm_core::io::bin::{BinMapper, BinType, FindBinParams};
use proptest::prelude::*;

/// Builds a numerical mapper from a sample, with upstream's default binning parameters.
fn numerical_mapper(mut sample: Vec<f64>, max_bin: i32, use_missing: bool) -> BinMapper {
    let total = sample.len();
    let params = FindBinParams {
        total_sample_cnt: total,
        max_bin,
        min_data_in_bin: 1,
        min_split_data: 1,
        pre_filter: false,
        bin_type: BinType::Numerical,
        use_missing,
        zero_as_missing: false,
        forced_upper_bounds: &[],
    };
    BinMapper::find_bin(&mut sample, &params)
}

fn finite_sample() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1.0e6f64..1.0e6, 2..200)
}

proptest! {
    /// Every value, including out-of-range ones, maps to a bin below `num_bin`.
    #[test]
    fn value_to_bin_stays_in_range(
        sample in finite_sample(),
        max_bin in 2i32..64,
        probe in -2.0e6f64..2.0e6,
    ) {
        let mapper = numerical_mapper(sample, max_bin, true);
        prop_assert!(mapper.value_to_bin(probe) < mapper.num_bin as u32);
    }

    /// Binning is monotone: a larger value never lands in a smaller bin.
    #[test]
    fn value_to_bin_is_monotone(
        sample in finite_sample(),
        max_bin in 2i32..64,
        a in -2.0e6f64..2.0e6,
        b in -2.0e6f64..2.0e6,
    ) {
        let mapper = numerical_mapper(sample, max_bin, true);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(mapper.value_to_bin(lo) <= mapper.value_to_bin(hi));
    }

    /// The binary form decodes to an identical mapper, and the decoder reports the bytes it read.
    #[test]
    fn binary_form_round_trips(
        sample in finite_sample(),
        max_bin in 2i32..64,
        use_missing in any::<bool>(),
    ) {
        let mapper = numerical_mapper(sample, max_bin, use_missing);
        let mut bytes = Vec::new();
        mapper.write(&mut bytes);
        let (decoded, used) = BinMapper::from_bytes(&bytes).expect("own output must decode");
        prop_assert_eq!(used, bytes.len());
        prop_assert!(decoded.bitwise_eq(&mapper));
        prop_assert_eq!(decoded.num_bin, mapper.num_bin);
    }
}
