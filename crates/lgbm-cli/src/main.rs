//! `lightgbm` command-line tool, config-file compatible with upstream.
//!
//! Parameters come from a config file (`config=<file>`) and/or `key=value` command-line arguments,
//! merged into one parameter string with the command line taking precedence, then parsed by
//! `lgbm_core::config::Config` (which resolves aliases and reports unknown keys). `task` selects the
//! action; `train`, `predict`, `save_binary` and `refit` are implemented.

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use lgbm_core::boosting::{Gbdt, GbdtTrainer};
use lgbm_core::config::{Config, TaskType};
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{LoadedText, TextLoadOptions, load_text};
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lightgbm: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let params = collect_params(&args)?;
    let cfg = Config::from_str_params(&params).map_err(|e| e.to_string())?;
    match cfg.task {
        TaskType::Train => train(&cfg),
        TaskType::Predict => predict(&cfg),
        TaskType::SaveBinary => save_binary(&cfg),
        TaskType::RefitTree => refit(&cfg),
        other => Err(format!(
            "task {other:?} is not implemented yet (supported: train, predict, save_binary, refit)"
        )),
    }
}

/// Merge command-line `key=value` arguments and an optional `config=<file>` into one whitespace-
/// separated parameter string. Command-line tokens come first so they win the "first occurrence
/// wins" resolution in `Config::from_str_params`; `config=` is expanded in place from the file.
fn collect_params(args: &[String]) -> Result<String, String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut config_file: Option<String> = None;
    for arg in args {
        match arg.split_once('=') {
            Some(("config", path)) => config_file = Some(path.to_string()),
            Some(_) => tokens.push(arg.clone()),
            None => return Err(format!("argument {arg:?} is not in key=value form")),
        }
    }
    if let Some(path) = config_file {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading config file {path}: {e}"))?;
        tokens.extend(config_tokens(&text));
    }
    Ok(tokens.join(" "))
}

/// Turn config-file lines into `key=value` tokens: drop `#` comments and blank lines, and trim the
/// spacing around the first `=` (upstream allows `key = value`).
fn config_tokens(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            line.split_once('=')
                .map(|(key, value)| format!("{}={}", key.trim(), value.trim()))
        })
        .collect()
}

/// Load the `data` text file and bin it into a training `Dataset` (column 0 is the label unless
/// `label_column` says otherwise), returning the loaded dense rows alongside it. Shared by
/// `task=train`, `task=save_binary` and `task=refit`.
fn build_training_dataset(cfg: &Config) -> Result<(LoadedText, Dataset), String> {
    if cfg.data.is_empty() {
        return Err("no training data: set data=<file>".into());
    }
    let content = std::fs::read_to_string(&cfg.data)
        .map_err(|e| format!("reading data file {}: {e}", cfg.data))?;
    let opts = TextLoadOptions {
        header: cfg.header,
        label_column: &cfg.label_column,
        weight_column: &cfg.weight_column,
        group_column: &cfg.group_column,
        ignore_column: &cfg.ignore_column,
        precise_float_parser: cfg.precise_float_parser,
        num_features: 0,
    };
    let loaded = load_text(&content, &opts).map_err(|e| e.to_string())?;
    let dataset = Dataset::from_dense(
        &loaded.features,
        loaded.num_data,
        loaded.num_features,
        &loaded.label,
        cfg,
        &[],
    )
    .map_err(|e| e.to_string())?;
    Ok((loaded, dataset))
}

fn train(cfg: &Config) -> Result<(), String> {
    let (_loaded, dataset) = build_training_dataset(cfg)?;
    let model = Gbdt::train(cfg, &dataset).map_err(|e| e.to_string())?;
    let text = model
        .save_model_to_string(0, 0, cfg.saved_feature_importance_type)
        .map_err(|e| e.to_string())?;
    let output = if cfg.output_model.is_empty() {
        "LightGBM_model.txt"
    } else {
        &cfg.output_model
    };
    std::fs::write(Path::new(output), text)
        .map_err(|e| format!("writing model to {output}: {e}"))?;
    Ok(())
}

fn save_binary(cfg: &Config) -> Result<(), String> {
    let (_loaded, dataset) = build_training_dataset(cfg)?;
    // Upstream writes the binary dataset next to the data file as `<data>.bin`.
    let output = format!("{}.bin", cfg.data);
    // The core writer refuses to overwrite (it warns and skips); fail loudly here so a silent
    // no-op is not mistaken for a successful save.
    if Path::new(&output).exists() {
        return Err(format!(
            "output file {output} already exists; refusing to overwrite"
        ));
    }
    dataset.save_binary(&output).map_err(|e| e.to_string())?;
    Ok(())
}

fn refit(cfg: &Config) -> Result<(), String> {
    if cfg.input_model.is_empty() {
        return Err("no model to refit: set input_model=<file>".into());
    }
    let init = Gbdt::from_model_file(Path::new(&cfg.input_model)).map_err(|e| e.to_string())?;
    let (loaded, dataset) = build_training_dataset(cfg)?;
    // Refit maps each new row to the leaf it falls into in every existing tree, then recomputes the
    // leaf outputs from the new labels (upstream `Booster::Refit`). The leaf indices come from the
    // existing model's leaf-index prediction over the new data.
    let leaf_cfg = PredictConfig {
        kind: PredictKind::LeafIndex,
        num_threads: cfg.num_threads.max(0) as usize,
        ..PredictConfig::default()
    };
    let predictor = Predictor::new(&init, &leaf_cfg).map_err(|e| e.to_string())?;
    let leaves = predictor
        .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
        .map_err(|e| e.to_string())?;
    let trees_per_row = predictor.num_pred_one_row();
    let leaf_pred: Vec<i32> = leaves.iter().map(|&v| v as i32).collect();
    let trainer = GbdtTrainer::refit(
        Arc::new(cfg.clone()),
        Arc::new(dataset),
        &init,
        &leaf_pred,
        trees_per_row,
    )
    .map_err(|e| e.to_string())?;
    let text = trainer
        .snapshot()
        .save_model_to_string(0, 0, cfg.saved_feature_importance_type)
        .map_err(|e| e.to_string())?;
    let output = if cfg.output_model.is_empty() {
        "LightGBM_model.txt"
    } else {
        &cfg.output_model
    };
    std::fs::write(Path::new(output), text)
        .map_err(|e| format!("writing model to {output}: {e}"))?;
    Ok(())
}

/// The predict type upstream derives from the boolean flags (leaf index, then contributions, then
/// raw score, else the normal transformed output).
fn predict_kind(cfg: &Config) -> PredictKind {
    if cfg.predict_leaf_index {
        PredictKind::LeafIndex
    } else if cfg.predict_contrib {
        PredictKind::Contrib
    } else if cfg.predict_raw_score {
        PredictKind::RawScore
    } else {
        PredictKind::Normal
    }
}

/// Format predictions as one row per line, values tab-separated in shortest round-trippable form.
fn format_predictions(preds: &[f64], per_row: usize, num_rows: usize) -> String {
    if per_row == 0 {
        return "\n".repeat(num_rows);
    }
    let mut out = String::with_capacity(preds.len() * 8);
    for chunk in preds.chunks(per_row) {
        out.push_str(
            &chunk
                .iter()
                .map(f64::to_string)
                .collect::<Vec<_>>()
                .join("\t"),
        );
        out.push('\n');
    }
    out
}

fn predict(cfg: &Config) -> Result<(), String> {
    if cfg.input_model.is_empty() {
        return Err("no model to predict with: set input_model=<file>".into());
    }
    if cfg.data.is_empty() {
        return Err("no data to predict: set data=<file>".into());
    }
    let model = Gbdt::from_model_file(Path::new(&cfg.input_model)).map_err(|e| e.to_string())?;
    let content = std::fs::read_to_string(&cfg.data)
        .map_err(|e| format!("reading data file {}: {e}", cfg.data))?;
    let opts = TextLoadOptions {
        header: cfg.header,
        label_column: &cfg.label_column,
        weight_column: &cfg.weight_column,
        group_column: &cfg.group_column,
        ignore_column: &cfg.ignore_column,
        precise_float_parser: cfg.precise_float_parser,
        // Prediction: the model's feature count tells the loader whether a leading label column is
        // present (detected from the column count).
        num_features: model.num_features(),
    };
    let loaded = load_text(&content, &opts).map_err(|e| e.to_string())?;
    let pred_cfg = PredictConfig {
        kind: predict_kind(cfg),
        start_iteration: cfg.start_iteration_predict,
        num_iteration: cfg.num_iteration_predict,
        pred_early_stop: cfg.pred_early_stop,
        num_threads: cfg.num_threads.max(0) as usize,
        ..PredictConfig::default()
    };
    let predictor = Predictor::new(&model, &pred_cfg).map_err(|e| e.to_string())?;
    let preds = predictor
        .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
        .map_err(|e| e.to_string())?;
    let body = format_predictions(&preds, predictor.num_pred_one_row(), loaded.num_data);
    let output = if cfg.output_result.is_empty() {
        "LightGBM_predict_result.txt"
    } else {
        &cfg.output_result
    };
    std::fs::write(Path::new(output), body)
        .map_err(|e| format!("writing predictions to {output}: {e}"))?;
    Ok(())
}
