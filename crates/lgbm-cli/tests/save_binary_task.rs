//! `lightgbm` with `task=save_binary` bins the data file and writes it to `<data>.bin`. Oracle:
//! `lgbm-core` builds the same dataset and writes the same bytes, and the file reloads to a dataset
//! of the expected shape.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

use lgbm_core::config::Config;
use lgbm_core::io::binary::BinaryDataset;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};

const PARAMS: &str = "objective=regression max_bin=31 min_data_in_bin=1 num_threads=1 verbosity=-1";

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

/// Core oracle: build the dataset the same way and save it to `out`.
fn core_save(csv: &str, out: &std::path::Path) -> (usize, usize) {
    let cfg = Config::from_str_params(PARAMS).unwrap();
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
        &cfg,
        &[],
    )
    .unwrap();
    ds.save_binary(out).unwrap();
    (loaded.num_data, loaded.num_features)
}

#[test]
fn save_binary_writes_the_same_bytes_as_core() {
    let (nrow, ncol) = (60usize, 5usize);
    let dir = temp_dir("savebin");
    let data_path = dir.join("train.csv");
    let conf_path = dir.join("save.conf");
    let data_csv = csv(nrow, ncol);
    std::fs::write(&data_path, &data_csv).unwrap();

    let conf = format!(
        "task=save_binary\ndata={}\n{}\n",
        data_path.display(),
        PARAMS.replace(' ', "\n")
    );
    std::fs::write(&conf_path, conf).unwrap();

    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(status.success(), "CLI save_binary failed: {status:?}");

    // Upstream writes to `<data>.bin`.
    let bin_path = dir.join("train.csv.bin");
    assert!(bin_path.exists(), "expected binary at {}", bin_path.display());

    // Oracle bytes.
    let oracle_path = dir.join("oracle.bin");
    let (want_rows, want_feats) = core_save(&data_csv, &oracle_path);
    assert_eq!(
        std::fs::read(&bin_path).unwrap(),
        std::fs::read(&oracle_path).unwrap(),
        "CLI binary differs from the core-written binary"
    );

    // And it reloads to the expected shape.
    let bin = BinaryDataset::read_file(&bin_path).unwrap();
    let reloaded = Dataset::from_binary(&bin).unwrap();
    assert_eq!(reloaded.num_data(), want_rows);
    assert_eq!(reloaded.layout.num_total_features, want_feats);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_data_file_fails() {
    let dir = temp_dir("savebin_missing");
    let conf_path = dir.join("save.conf");
    std::fs::write(
        &conf_path,
        format!(
            "task=save_binary\ndata={}\nverbosity=-1\n",
            dir.join("nope.csv").display()
        ),
    )
    .unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_lightgbm"))
        .arg(format!("config={}", conf_path.display()))
        .current_dir(&dir)
        .status()
        .unwrap();
    assert!(!status.success(), "a missing data file must fail");
    assert!(!dir.join("nope.csv.bin").exists());
    std::fs::remove_dir_all(&dir).ok();
}
