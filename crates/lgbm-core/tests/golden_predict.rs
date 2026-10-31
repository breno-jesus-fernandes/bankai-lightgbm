//! P1 gate: models produced by upstream LightGBM predict bit-exactly.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};
use support::*;

fn run(kind: PredictKind, npy: &str) {
    run_range(kind, npy, 0, -1);
}

fn run_range(kind: PredictKind, npy: &str, start_iteration: i32, num_iteration: i32) {
    for case in cases() {
        if !case.join(npy).exists() {
            continue;
        }
        let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
        let (x, ncol) = load_test_features(&case);
        let cfg = PredictConfig {
            kind,
            start_iteration,
            num_iteration,
            ..PredictConfig::default()
        };
        let pred = Predictor::new(&model, &cfg).unwrap();
        let got = pred.predict_dense(&x, x.len() / ncol, ncol).unwrap();
        let (want, _) = load_npy(&case.join(npy));
        assert_bits_eq(&case, npy, &got, &want);
    }
}

#[test]
fn normal_prediction_matches_upstream() {
    run(PredictKind::Normal, "pred.npy");
}

#[test]
fn raw_score_matches_upstream() {
    run(PredictKind::RawScore, "pred_raw.npy");
}

#[test]
fn leaf_index_matches_upstream() {
    run(PredictKind::LeafIndex, "pred_leaf.npy");
}

#[test]
fn shap_contrib_matches_upstream() {
    run(PredictKind::Contrib, "pred_contrib.npy");
}

#[test]
fn start_and_num_iteration_match_upstream() {
    run_range(PredictKind::Normal, "pred_iter_3_5.npy", 3, 5);
}

#[test]
fn linear_tree_contrib_is_rejected_like_upstream() {
    let model = Gbdt::from_model_file(fixtures_dir().join("linear_tree/model.txt")).unwrap();
    let cfg = PredictConfig {
        kind: PredictKind::Contrib,
        ..PredictConfig::default()
    };
    assert!(Predictor::new(&model, &cfg).is_err());
}

#[test]
fn iteration_range_sums_to_full_raw_score() {
    for case in cases() {
        let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
        let (x, ncol) = load_test_features(&case);
        let nrow = x.len() / ncol;
        let raw = |start, num| {
            let cfg = PredictConfig {
                kind: PredictKind::RawScore,
                start_iteration: start,
                num_iteration: num,
                ..PredictConfig::default()
            };
            Predictor::new(&model, &cfg)
                .unwrap()
                .predict_dense(&x, nrow, ncol)
                .unwrap()
        };
        let full = raw(0, -1);
        let head = raw(0, 5);
        let tail = raw(5, -1);
        for i in 0..full.len() {
            let sum = head[i] + tail[i];
            assert!(
                (full[i] - sum).abs() <= 1e-12 * full[i].abs().max(1.0),
                "{}",
                case.display()
            );
        }
    }
}
