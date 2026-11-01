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

/// Dense rows as CSR, dropping exact zeros so absent entries are exercised.
fn to_csr(x: &[f64], ncol: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let mut indptr = vec![0i64];
    let (mut idx, mut val) = (Vec::new(), Vec::new());
    for row in x.chunks(ncol) {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                idx.push(c as i32);
                val.push(v);
            }
        }
        indptr.push(idx.len() as i64);
    }
    (indptr, idx, val)
}

/// The same matrix as CSC.
fn to_csc(x: &[f64], ncol: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let nrow = x.len() / ncol;
    let mut col_ptr = vec![0i64];
    let (mut idx, mut val) = (Vec::new(), Vec::new());
    for c in 0..ncol {
        for r in 0..nrow {
            let v = x[r * ncol + c];
            if v != 0.0 {
                idx.push(r as i32);
                val.push(v);
            }
        }
        col_ptr.push(idx.len() as i64);
    }
    (col_ptr, idx, val)
}

/// CSR and CSC input reproduce upstream's (dense-input) fixtures bit for bit: an absent
/// entry and a stored `0.0` both read as `0.0` (upstream: CopyToPredictBuffer).
#[test]
fn sparse_input_matches_upstream() {
    for (kind, npy) in [
        (PredictKind::Normal, "pred.npy"),
        (PredictKind::RawScore, "pred_raw.npy"),
        (PredictKind::LeafIndex, "pred_leaf.npy"),
        (PredictKind::Contrib, "pred_contrib.npy"),
    ] {
        for case in cases() {
            if !case.join(npy).exists() {
                continue;
            }
            let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
            let (x, ncol) = load_test_features(&case);
            let nrow = x.len() / ncol;
            let cfg = PredictConfig {
                kind,
                ..PredictConfig::default()
            };
            let p = Predictor::new(&model, &cfg).unwrap();
            let (want, _) = load_npy(&case.join(npy));
            let (ip, ix, v) = to_csr(&x, ncol);
            assert_bits_eq(&case, npy, &p.predict_csr(&ip, &ix, &v).unwrap(), &want);
            let (cp, rx, v) = to_csc(&x, ncol);
            assert_bits_eq(
                &case,
                npy,
                &p.predict_csc(&cp, &rx, &v, nrow).unwrap(),
                &want,
            );
        }
    }
}

/// Sparse SHAP output stores every split feature of the in-range trees plus the bias for
/// each row, holding the dense contributions (upstream: TreeSHAPByMap).
#[test]
fn sparse_contrib_output_matches_dense_contrib() {
    for case in cases() {
        if !case.join("pred_contrib.npy").exists() {
            continue;
        }
        let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
        let (x, ncol) = load_test_features(&case);
        let nrow = x.len() / ncol;
        let cfg = PredictConfig {
            kind: PredictKind::Contrib,
            ..PredictConfig::default()
        };
        let p = Predictor::new(&model, &cfg).unwrap();
        let dense = p.predict_dense(&x, nrow, ncol).unwrap();
        let width = model.num_features() + 1;
        let k = p.num_pred_one_row();
        let cols = p.contrib_columns();
        let (ip, ix, v) = to_csr(&x, ncol);
        let csr = p.predict_contrib_csr(&ip, &ix, &v).unwrap();
        let (cp, rx, cv) = to_csc(&x, ncol);
        let csc = p.predict_contrib_csc(&cp, &rx, &cv, nrow).unwrap();
        assert_eq!(csr.len(), model.num_tree_per_iteration());
        for (class, (m, c)) in csr.iter().zip(&csc).enumerate() {
            assert_eq!(
                cols[class].last(),
                Some(&(width as i32 - 1)),
                "bias is stored"
            );
            let mut from_csr = vec![0.0; nrow * width];
            for r in 0..nrow {
                for j in m.indptr[r] as usize..m.indptr[r + 1] as usize {
                    from_csr[r * width + m.indices[j] as usize] = m.data[j];
                }
                assert_eq!(
                    &m.indices[m.indptr[r] as usize..m.indptr[r + 1] as usize],
                    &cols[class][..]
                );
            }
            let mut from_csc = vec![0.0; nrow * width];
            for col in 0..width {
                for j in c.indptr[col] as usize..c.indptr[col + 1] as usize {
                    from_csc[c.indices[j] as usize * width + col] = c.data[j];
                }
            }
            let want: Vec<f64> = (0..nrow)
                .flat_map(|r| dense[r * k + class * width..r * k + (class + 1) * width].to_vec())
                .collect();
            assert_bits_eq(&case, "sparse contrib (csr)", &from_csr, &want);
            assert_bits_eq(&case, "sparse contrib (csc)", &from_csc, &want);
        }
    }
}
