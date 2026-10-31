//! P3 gate: models trained by bankai are byte-identical to upstream's
//! (`deterministic=true`, `num_threads=1`, `force_col_wise=true`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::path::PathBuf;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use support::*;

fn train_cases() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures_dir().join("train"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("model.txt").exists())
        .collect();
    out.sort();
    assert!(
        out.len() >= 75,
        "expected training fixtures; run scripts/gen_train_fixtures.py"
    );
    out
}

#[test]
fn trained_models_match_upstream() {
    let mut failures = Vec::new();
    for case in train_cases() {
        let (mut params, cats) = dataset_params(&case);
        // The Python Dataset forwards `categorical_feature` into the training parameters.
        if !cats.is_empty() {
            let list: Vec<String> = cats.iter().map(ToString::to_string).collect();
            params.insert("categorical_feature".into(), list.join(","));
        }
        let rounds: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(case.join("params.json")).unwrap())
                .unwrap();
        params.insert(
            "num_iterations".into(),
            rounds["num_boost_round"].to_string(),
        );
        let cfg = Config::from_params(&params).unwrap();
        let (x, shape) = load_npy(&case.join("x.npy"));
        let (y, _) = load_npy(&case.join("y.npy"));
        let mut ds = Dataset::from_dense(&x, shape[0], shape[1], &y, &cfg, &cats).unwrap();
        if case.join("w.npy").exists() {
            let (w, _) = load_npy(&case.join("w.npy"));
            let w: Vec<f32> = w.iter().map(|&v| v as f32).collect();
            ds.set_weights(&w).unwrap();
        }
        if case.join("group.npy").exists() {
            let (g, _) = load_npy(&case.join("group.npy"));
            let g: Vec<i32> = g.iter().map(|&v| v as i32).collect();
            ds.set_group(&g).unwrap();
        }
        let model = match Gbdt::train(&cfg, &ds) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{}: {e}", case.display()));
                continue;
            }
        };
        let ours = format!(
            "{}\npandas_categorical:null\n",
            model.save_model_to_string(0, -1, 0).unwrap()
        );
        let want = std::fs::read_to_string(case.join("model.txt")).unwrap();
        if ours != want {
            let (i, (a, b)) = ours
                .lines()
                .zip(want.lines())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .unwrap_or((0, ("<len>", "<len>")));
            failures.push(format!(
                "{}: line {i}\n  ours {a}\n  want {b}",
                case.display()
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
