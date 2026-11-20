//! `lightgbm` command-line tool, config-file compatible with upstream.
//!
//! Parameters come from a config file (`config=<file>`) and/or `key=value` command-line arguments,
//! merged into one parameter string with the command line taking precedence, then parsed by
//! `lgbm_core::config::Config` (which resolves aliases and reports unknown keys). `task` selects the
//! action; `task=train` is implemented.

use std::path::Path;
use std::process::ExitCode;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::{Config, TaskType};
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};

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
        other => Err(format!(
            "task {other:?} is not implemented yet (only task=train is supported)"
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

fn train(cfg: &Config) -> Result<(), String> {
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
        // Training: column 0 is the label unless `label_column` says otherwise.
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
