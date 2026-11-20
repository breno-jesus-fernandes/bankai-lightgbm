//! `lightgbm config=<file>` with `task=train` reads a config file + data file, trains, and writes
//! the model to `output_model`. Oracle: `lgbm-core` loads the same data and trains with the same
//! parameters, and the saved model's tree section matches byte-for-byte.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};

/// Parameters shared by the CLI config and the core oracle (no `data`/`output_model`, which are
/// CLI/file concerns).
const MODEL_PARAMS: &str = "objective=regression max_bin=31 num_iterations=8 num_leaves=7 \
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

/// A CSV with the label in column 0 and `ncol` feature columns, no header.
fn csv(nrow: usize, ncol: usize) -> String {
    let mut s = String::new();
    for r in 0..nrow {
        let feats: Vec<f64> = (0..ncol)
            .map(|c| ((((r * ncol + c) * 37) % 101) as f64) * 0.1 - 5.0)
            .collect();
        let label = feats[0] * 2.0 - feats[ncol - 1] + (r % 5) as f64 * 0.3;
        let mut row = vec![label.to_string()];
        row.extend(feats.iter().map(f64::to_string));
        s.push_str(&row.join(","));
        s.push('\n');
    }
    s
}

/// Everything up to the trailing `parameters:` block, which echoes CLI-only keys (`data`, …).
fn trees_only(model: &str) -> &str {
    model
        .split_once("\nparameters:\n")
        .map_or(model, |(trees, _)| trees)
}

#[test]
fn train_task_writes_a_model_matching_core() {
    let (nrow, ncol) = (60usize, 5usize);
    let dir = temp_dir("train");
    let data_path = dir.join("train.csv");
    let model_path = dir.join("model.txt");
    let conf_path = dir.join("train.conf");

    let data_csv = csv(nrow, ncol);
    std::fs::write(&data_path, &data_csv).unwrap();

    // Config file with comments and `key = value` spacing to exercise the parser.
    let conf = format!(
        "task = train\n# the training data\ndata = {}\noutput_model = {}\n{}\n",
        data_path.display(),
        model_path.display(),
        MODEL_PARAMS.replace(' ', "\n")
    );
    std::fs::write(&conf_path, conf).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(status.success(), "CLI exited with failure: {status:?}");

    let got = std::fs::read_to_string(&model_path).unwrap();

    // Oracle: core trains on the same loaded data with the same parameters.
    let cfg = Config::from_str_params(MODEL_PARAMS).unwrap();
    let opts = TextLoadOptions {
        header: false,
        label_column: "",
        weight_column: "",
        group_column: "",
        ignore_column: "",
        precise_float_parser: false,
        num_features: 0,
    };
    let loaded = load_text(&data_csv, &opts).unwrap();
    assert_eq!(loaded.num_features, ncol);
    let ds = Dataset::from_dense(
        &loaded.features,
        loaded.num_data,
        loaded.num_features,
        &loaded.label,
        &cfg,
        &[],
    )
    .unwrap();
    let model = Gbdt::train(&cfg, &ds).unwrap();
    let want = model.save_model_to_string(0, 0, 0).unwrap();

    assert_eq!(trees_only(&got), trees_only(&want));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn cli_args_override_the_config_file() {
    let (nrow, ncol) = (40usize, 4usize);
    let dir = temp_dir("override");
    let data_path = dir.join("train.csv");
    let model_path = dir.join("model.txt");
    let conf_path = dir.join("train.conf");

    let data_csv = csv(nrow, ncol);
    std::fs::write(&data_path, &data_csv).unwrap();

    // The config file asks for 3 iterations; the command line overrides it to 8.
    let conf = format!(
        "task=train\ndata={}\noutput_model={}\nobjective=regression\nnum_iterations=3\n\
         max_bin=31\nnum_leaves=7\nlearning_rate=0.1\nmin_data_in_leaf=1\ndeterministic=true\n\
         force_col_wise=true\nnum_threads=1\nseed=42\nverbosity=-1\n",
        data_path.display(),
        model_path.display(),
    );
    std::fs::write(&conf_path, conf).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg("num_iterations=8")
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(status.success(), "CLI exited with failure: {status:?}");

    let got = std::fs::read_to_string(&model_path).unwrap();
    // 8 iterations (CLI) must win: the model has 8 trees, not 3.
    let tree_count = got.matches("\nTree=").count();
    assert_eq!(tree_count, 8, "CLI num_iterations=8 should override the file's 3");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_data_file_fails() {
    let dir = temp_dir("missing");
    let model_path = dir.join("model.txt");
    let conf_path = dir.join("train.conf");
    std::fs::write(
        &conf_path,
        format!(
            "task=train\ndata={}\noutput_model={}\nobjective=regression\nverbosity=-1\n",
            dir.join("does_not_exist.csv").display(),
            model_path.display(),
        ),
    )
    .unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(!status.success(), "a missing data file must fail");
    assert!(!model_path.exists(), "no model should be written on error");

    std::fs::remove_dir_all(&dir).ok();
}
