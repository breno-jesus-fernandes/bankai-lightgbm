//! `lightgbm` with `task=refit` reloads `input_model`, refits its leaves to a new labelled data
//! file, and writes the result to `output_model`. Oracle: `lgbm-core` refits the same way
//! (`GbdtTrainer::refit` over the model's leaf-index predictions on the new data).
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use lgbm_core::boosting::{Gbdt, GbdtTrainer};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const PARAMS: &str = "objective=regression max_bin=31 num_iterations=8 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";

fn temp_dir(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("bankai_cli_{tag}_{pid}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A CSV with the label in column 0 and `ncol` feature columns; `salt` shifts the data so refit
/// sees a different distribution than the model was trained on.
fn csv(nrow: usize, ncol: usize, salt: usize) -> String {
    let mut s = String::new();
    for r in 0..nrow {
        let feats: Vec<f64> = (0..ncol)
            .map(|c| ((((r * ncol + c + salt) * 37) % 101) as f64) * 0.1 - 5.0)
            .collect();
        let label = feats[0] * 1.5 - feats[ncol - 1] + ((r + salt) % 5) as f64 * 0.4;
        let mut row = vec![label.to_string()];
        row.extend(feats.iter().map(f64::to_string));
        s.push_str(&row.join(","));
        s.push('\n');
    }
    s
}

fn trees_only(model: &str) -> &str {
    model
        .split_once("\nparameters:\n")
        .map_or(model, |(trees, _)| trees)
}

fn load_dataset(csv: &str, cfg: &Config) -> (lgbm_core::io::text_loader::LoadedText, Dataset) {
    let opts = TextLoadOptions {
        header: false,
        label_column: "",
        weight_column: "",
        group_column: "",
        ignore_column: "",
        precise_float_parser: false,
        num_features: 0,
    };
    let loaded = load_text(csv, &opts).unwrap();
    let ds = Dataset::from_dense(
        &loaded.features,
        loaded.num_data,
        loaded.num_features,
        &loaded.label,
        cfg,
        &[],
    )
    .unwrap();
    (loaded, ds)
}

/// Train a model on one dataset and write it to `path`.
fn write_model(path: &std::path::Path, ncol: usize) {
    let cfg = Config::from_str_params(PARAMS).unwrap();
    let (_loaded, ds) = load_dataset(&csv(60, ncol, 0), &cfg);
    let model = Gbdt::train(&cfg, &ds).unwrap();
    std::fs::write(path, model.save_model_to_string(0, 0, 0).unwrap()).unwrap();
}

/// Core oracle: refit `model_path` to `new_csv` and return the refitted model text.
fn core_refit(model_path: &std::path::Path, new_csv: &str) -> String {
    let cfg = Config::from_str_params(PARAMS).unwrap();
    let init = Gbdt::from_model_file(model_path).unwrap();
    let (loaded, ds) = load_dataset(new_csv, &cfg);
    let leaf_cfg = PredictConfig {
        kind: PredictKind::LeafIndex,
        num_threads: 1,
        ..PredictConfig::default()
    };
    let predictor = Predictor::new(&init, &leaf_cfg).unwrap();
    let leaf_f = predictor
        .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
        .unwrap();
    let ncol = predictor.num_pred_one_row();
    let leaf_pred: Vec<i32> = leaf_f.iter().map(|&v| v as i32).collect();
    let trainer =
        GbdtTrainer::refit(Arc::new(cfg), Arc::new(ds), &init, &leaf_pred, ncol).unwrap();
    trainer.snapshot().save_model_to_string(0, 0, 0).unwrap()
}

#[test]
fn refit_task_matches_core() {
    let ncol = 4usize;
    let dir = temp_dir("refit");
    let model_path = dir.join("model.txt");
    let data_path = dir.join("refit.csv");
    let out_path = dir.join("refit_model.txt");
    let conf_path = dir.join("refit.conf");

    write_model(&model_path, ncol);
    let new_csv = csv(45, ncol, 7);
    std::fs::write(&data_path, &new_csv).unwrap();

    let conf = format!(
        "task=refit\ninput_model={}\ndata={}\noutput_model={}\n{}\n",
        model_path.display(),
        data_path.display(),
        out_path.display(),
        PARAMS.replace(' ', "\n"),
    );
    std::fs::write(&conf_path, conf).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(status.success(), "CLI refit failed: {status:?}");

    let got = std::fs::read_to_string(&out_path).unwrap();
    let want = core_refit(&model_path, &new_csv);
    assert_eq!(trees_only(&got), trees_only(&want));

    // Sanity: refit changed the leaves (the new model differs from the original).
    let original = std::fs::read_to_string(&model_path).unwrap();
    assert_ne!(trees_only(&got), trees_only(&original), "refit should change the leaves");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_input_model_fails() {
    let dir = temp_dir("refit_missing");
    let data_path = dir.join("refit.csv");
    std::fs::write(&data_path, csv(10, 4, 1)).unwrap();
    let conf_path = dir.join("refit.conf");
    std::fs::write(
        &conf_path,
        format!(
            "task=refit\ninput_model={}\ndata={}\noutput_model={}\nobjective=regression\nverbosity=-1\n",
            dir.join("nope.txt").display(),
            data_path.display(),
            dir.join("out.txt").display(),
        ),
    )
    .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(!status.success(), "a missing input_model must fail");
    assert!(!dir.join("out.txt").exists());
    std::fs::remove_dir_all(&dir).ok();
}
