//! Training output must not depend on `num_threads`. The parallel split search and the
//! multi-value histograms split their work across threads but keep every sum in the same
//! order, so the model text is identical for any thread count.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

/// Deterministic data from splitmix64, so the test does not depend on any RNG crate.
fn synthetic(nrow: usize, ncol: usize) -> (Vec<f64>, Vec<f64>) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut x = vec![0.0; nrow * ncol];
    let mut y = vec![0.0; nrow];
    for r in 0..nrow {
        for c in 0..ncol {
            let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
            // Every third column is mostly zero, so bundling creates multi-value groups.
            x[r * ncol + c] = if c % 3 == 0 && u < 0.8 {
                0.0
            } else {
                u * (c as f64 + 1.0)
            };
        }
        y[r] = x[r * ncol] + 2.0 * x[r * ncol + 1] - x[r * ncol + 2] * x[r * ncol + 3];
    }
    (x, y)
}

fn train_text(num_threads: u32, x: &[f64], y: &[f64], nrow: usize, ncol: usize) -> String {
    let params: HashMap<String, String> = [
        ("objective", "regression"),
        ("num_leaves", "31"),
        ("num_iterations", "15"),
        ("min_data_in_leaf", "20"),
        ("deterministic", "true"),
        ("force_col_wise", "true"),
        ("verbose", "-1"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .chain([("num_threads".to_string(), num_threads.to_string())])
    .collect();
    let cfg = Config::from_params(&params).unwrap();
    let ds = Dataset::from_dense(x, nrow, ncol, y, &cfg, &[]).unwrap();
    let model = Gbdt::train(&cfg, &ds).unwrap();
    let text = model.save_model_to_string(0, -1, 0).unwrap();
    // The trailing parameters section echoes `num_threads`; only the trees must match.
    text.split("\nparameters:").next().unwrap().to_string()
}

#[test]
fn model_text_is_independent_of_num_threads() {
    // 120 features with up to 255 bins each: well above the parallel-search threshold.
    let (nrow, ncol) = (4000, 120);
    let (x, y) = synthetic(nrow, ncol);
    let serial = train_text(1, &x, &y, nrow, ncol);
    assert!(serial.contains("Tree=14"), "expected 15 trees in the model");
    for threads in [2, 3] {
        let parallel = train_text(threads, &x, &y, nrow, ncol);
        assert!(
            parallel == serial,
            "num_threads={threads} changed the model text"
        );
    }
}
