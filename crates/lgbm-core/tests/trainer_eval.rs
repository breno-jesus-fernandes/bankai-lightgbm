//! `GbdtTrainer::eval_values` computes each configured metric over the training score updater.
//! Anchored independently: for a regression booster, `l2` is the mean squared error and `l1` the
//! mean absolute error between the training score and the labels.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use lgbm_core::boosting::GbdtTrainer;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const PARAMS: &str = "objective=regression metric=l2,l1 max_bin=31 num_iterations=6 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect()
}

fn labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| {
            let row = &data[r * ncol..(r + 1) * ncol];
            row[0] * 2.0 - row[ncol - 1] + (r % 5) as f64 * 0.3
        })
        .collect()
}

#[test]
fn eval_values_are_the_training_mse_and_mae() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let cfg = Config::from_str_params(PARAMS).unwrap();
    let ds = Dataset::from_dense(&data, nrow, ncol, &label, &cfg, &[]).unwrap();

    let mut trainer = GbdtTrainer::new(Arc::new(cfg), Arc::new(ds)).unwrap();
    for _ in 0..6 {
        trainer.update_one_iter();
    }

    // Names fix the order of the values.
    assert_eq!(trainer.eval_names().unwrap(), vec!["l2", "l1"]);

    // Independent oracle over the training score (num_class == 1, so one score per row). The
    // metric reads labels from the metadata, where they are stored as f32, so round-trip through
    // f32 to match.
    let scores = trainer.training_score().to_vec();
    assert_eq!(scores.len(), nrow);
    let y: Vec<f64> = label.iter().map(|&v| f64::from(v as f32)).collect();
    let mse: f64 = scores
        .iter()
        .zip(&y)
        .map(|(&s, &y)| (s - y) * (s - y))
        .sum::<f64>()
        / nrow as f64;
    let mae: f64 = scores
        .iter()
        .zip(&y)
        .map(|(&s, &y)| (s - y).abs())
        .sum::<f64>()
        / nrow as f64;

    let values = trainer.eval_values().unwrap();
    assert_eq!(values.len(), 2);
    assert!(
        (values[0] - mse).abs() <= 1e-9 * mse.abs().max(1.0),
        "l2 {} vs mse {mse}",
        values[0]
    );
    assert!(
        (values[1] - mae).abs() <= 1e-9 * mae.abs().max(1.0),
        "l1 {} vs mae {mae}",
        values[1]
    );
}
