//! `Gbdt::merge` appends another model's trees (upstream `GBDT::MergeFrom` / `LGBM_BoosterMerge`):
//! the merged model holds both models' trees and its raw score is the sum of the two.
#![allow(clippy::unwrap_used)]

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect()
}

fn reg_labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| data[r * ncol] * 2.0 - data[r * ncol + ncol - 1] + (r % 5) as f64 * 0.3)
        .collect()
}

fn train(params: &str, data: &[f64], nrow: usize, ncol: usize, label: &[f64]) -> Gbdt {
    let cfg = Config::from_str_params(params).unwrap();
    let ds = Dataset::from_dense(data, nrow, ncol, label, &cfg, &[]).unwrap();
    Gbdt::train(&cfg, &ds).unwrap()
}

fn raw_scores(model: &Gbdt, data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    let cfg = PredictConfig {
        kind: PredictKind::RawScore,
        num_threads: 1,
        ..PredictConfig::default()
    };
    Predictor::new(model, &cfg)
        .unwrap()
        .predict_dense(data, nrow, ncol)
        .unwrap()
}

const BASE: &str = "max_bin=31 num_leaves=7 learning_rate=0.1 min_data_in_leaf=1 \
                    deterministic=true force_col_wise=true num_threads=1 verbosity=-1";

#[test]
fn merge_appends_trees_and_sums_raw_scores() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = reg_labels(&data, nrow, ncol);

    let a = train(
        &format!("objective=regression num_iterations=8 seed=42 {BASE}"),
        &data,
        nrow,
        ncol,
        &label,
    );
    let b = train(
        &format!("objective=regression num_iterations=6 seed=7 {BASE}"),
        &data,
        nrow,
        ncol,
        &label,
    );

    let a_raw = raw_scores(&a, &data, nrow, ncol);
    let b_raw = raw_scores(&b, &data, nrow, ncol);

    let mut merged = a.clone();
    merged.merge(&b).unwrap();

    assert_eq!(merged.num_trees(), a.num_trees() + b.num_trees());
    assert_eq!(merged.num_trees(), 14);

    let merged_raw = raw_scores(&merged, &data, nrow, ncol);
    assert_eq!(merged_raw.len(), nrow);
    for (i, &m) in merged_raw.iter().enumerate() {
        let want = a_raw[i] + b_raw[i];
        assert!(
            (m - want).abs() <= 1e-9 * want.abs().max(1.0),
            "row {i}: merged raw {m} != a+b {want}"
        );
    }
}

#[test]
fn merge_rejects_mismatched_classes() {
    let (nrow, ncol) = (60usize, 4usize);
    let data = matrix(nrow, ncol);
    let reg = reg_labels(&data, nrow, ncol);
    let multi: Vec<f64> = (0..nrow).map(|r| (r % 3) as f64).collect();

    let a = train(
        &format!("objective=regression num_iterations=5 seed=42 {BASE}"),
        &data,
        nrow,
        ncol,
        &reg,
    );
    let b = train(
        &format!("objective=multiclass num_class=3 num_iterations=5 seed=42 {BASE}"),
        &data,
        nrow,
        ncol,
        &multi,
    );

    // Regression has 1 tree per iteration; 3-class multiclass has 3 — merging is rejected.
    let mut a2 = a.clone();
    assert!(a2.merge(&b).is_err());
}
