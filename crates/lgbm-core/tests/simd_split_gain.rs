//! Independent lanes must preserve upstream's scalar split-gain arithmetic.
use lgbm_core::treelearner::simd::{split_gain, split_gain_scalar};

#[test]
fn simd_split_gain_is_bit_exact_to_scalar_formula() {
    let gradients: [f64; 8] = [-1e100, -127.125, -0.0, 0.0, 1e-100, 0.75, 42.0, 1e100];
    for gl in gradients {
        for gr in gradients {
            for hl in [1e-200, 0.125, 800.0, 1e100] {
                for hr in [1e-200, 0.25, 400.0, 1e100] {
                    for l2 in [0.0, 0.125, 10.0] {
                        let oracle = gl * gl / (hl + l2) + gr * gr / (hr + l2);
                        assert_eq!(
                            split_gain_scalar(gl, hl, gr, hr, l2).to_bits(),
                            oracle.to_bits()
                        );
                        assert_eq!(split_gain(gl, hl, gr, hr, l2).to_bits(), oracle.to_bits());
                    }
                }
            }
        }
    }
}
