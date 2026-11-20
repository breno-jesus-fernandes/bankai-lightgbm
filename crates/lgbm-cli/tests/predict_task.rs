//! `lightgbm` with `task=predict` loads a model and a data file and writes per-row predictions to
//! `output_result`. Oracle: `lgbm-core`'s `Predictor` over the features `load_text` reads from the
//! same file, compared byte-for-byte against the result file.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const TRAIN_PARAMS: &str = "objective=regression max_bin=31 num_iterations=8 num_leaves=7 \
                            learning_rate=0.1 min_data_in_leaf=1 deterministic=true \
                            force_col_wise=true num_threads=1 seed=42 verbosity=-1";

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

/// A row-major feature matrix (no label column) as CSV text.
fn feature_csv(nrow: usize, ncol: usize) -> (Vec<f64>, String) {
    let mut data = Vec::with_capacity(nrow * ncol);
    let mut s = String::new();
    for r in 0..nrow {
        let row: Vec<f64> = (0..ncol)
            .map(|c| ((((r * ncol + c) * 37) % 101) as f64) * 0.1 - 5.0)
            .collect();
        s.push_str(&row.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
        s.push('\n');
        data.extend(row);
    }
    (data, s)
}

/// Train a small model and write it to `path`.
fn write_model(path: &std::path::Path) {
    let (nrow, ncol) = (60usize, 4usize);
    let data: Vec<f64> = (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect();
    let label: Vec<f64> = (0..nrow)
        .map(|r| data[r * ncol] * 2.0 - data[r * ncol + ncol - 1] + (r % 5) as f64 * 0.3)
        .collect();
    let cfg = Config::from_str_params(TRAIN_PARAMS).unwrap();
    let ds = Dataset::from_dense(&data, nrow, ncol, &label, &cfg, &[]).unwrap();
    let model = Gbdt::train(&cfg, &ds).unwrap();
    std::fs::write(path, model.save_model_to_string(0, 0, 0).unwrap()).unwrap();
}

fn parse_result(text: &str) -> Vec<f64> {
    text.lines()
        .flat_map(|line| line.split('\t').map(|v| v.parse::<f64>().unwrap()))
        .collect()
}

fn oracle(model_path: &std::path::Path, csv: &str, ncol: usize, kind: PredictKind) -> Vec<f64> {
    let opts = TextLoadOptions {
        header: false,
        label_column: "",
        weight_column: "",
        group_column: "",
        ignore_column: "",
        precise_float_parser: false,
        num_features: ncol,
    };
    let loaded = load_text(csv, &opts).unwrap();
    let model = Gbdt::from_model_file(model_path).unwrap();
    let cfg = PredictConfig {
        kind,
        num_threads: 1,
        ..PredictConfig::default()
    };
    let predictor = Predictor::new(&model, &cfg).unwrap();
    predictor
        .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
        .unwrap()
}

fn run_predict(dir: &std::path::Path, extra: &[&str]) -> (PathBuf, std::process::ExitStatus) {
    let model_path = dir.join("model.txt");
    let data_path = dir.join("predict.csv");
    let result_path = dir.join("preds.txt");
    let conf_path = dir.join("predict.conf");
    let mut conf = format!(
        "task=predict\ninput_model={}\ndata={}\noutput_result={}\nnum_threads=1\nverbosity=-1\n",
        model_path.display(),
        data_path.display(),
        result_path.display(),
    );
    for line in extra {
        conf.push_str(line);
        conf.push('\n');
    }
    std::fs::write(&conf_path, conf).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(dir)
        .status()
        .unwrap();
    (result_path, status)
}

#[test]
fn predict_task_matches_core() {
    let ncol = 4usize;
    let dir = temp_dir("predict");
    write_model(&dir.join("model.txt"));
    let (_data, csv) = feature_csv(23, ncol);
    std::fs::write(dir.join("predict.csv"), &csv).unwrap();

    let (result_path, status) = run_predict(&dir, &[]);
    assert!(status.success(), "CLI predict failed: {status:?}");

    let got = parse_result(&std::fs::read_to_string(&result_path).unwrap());
    let want = oracle(&dir.join("model.txt"), &csv, ncol, PredictKind::Normal);
    assert_eq!(got.len(), 23);
    assert_eq!(got, want);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn predict_leaf_index_flag_selects_leaf_output() {
    let ncol = 4usize;
    let dir = temp_dir("predict_leaf");
    write_model(&dir.join("model.txt"));
    let (_data, csv) = feature_csv(15, ncol);
    std::fs::write(dir.join("predict.csv"), &csv).unwrap();

    let (result_path, status) = run_predict(&dir, &["predict_leaf_index=true"]);
    assert!(status.success(), "CLI predict (leaf) failed: {status:?}");

    let got = parse_result(&std::fs::read_to_string(&result_path).unwrap());
    let want = oracle(&dir.join("model.txt"), &csv, ncol, PredictKind::LeafIndex);
    // 8 trees → 8 leaf indices per row.
    assert_eq!(got.len(), 15 * 8);
    assert_eq!(got, want);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_input_model_fails() {
    let dir = temp_dir("predict_missing");
    let (_data, csv) = feature_csv(5, 4);
    std::fs::write(dir.join("predict.csv"), &csv).unwrap();
    // No model written.
    let (result_path, status) = run_predict(&dir, &[]);
    assert!(!status.success(), "a missing input_model must fail");
    assert!(!result_path.exists(), "no result file on error");
    std::fs::remove_dir_all(&dir).ok();
}
