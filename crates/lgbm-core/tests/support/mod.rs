//! Helpers for golden tests: locate fixtures, read `.npy` and `test.csv` files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::path::{Path, PathBuf};

/// Reads a model text fixture with LF line endings. A Windows checkout turns the fixtures into
/// CRLF, while upstream writes LF, so the endings are normalized before any byte comparison.
pub fn read_model_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).unwrap().replace("\r\n", "\n")
}

pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// Every fixture case directory that contains a `model.txt`.
pub fn cases() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("model.txt").exists())
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no fixtures; run scripts/gen_fixtures.py");
    out
}

/// Every dataset fixture directory (`tests/fixtures/datasets/<case>`).
pub fn dataset_cases() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures_dir().join("datasets"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("dataset.bin").exists())
        .collect();
    out.sort();
    assert!(
        !out.is_empty(),
        "no dataset fixtures; run scripts/gen_dataset_fixtures.py"
    );
    out
}

/// Dataset params (`params.json`) as strings, the way the Python package passes them,
/// plus the categorical feature indices.
pub fn dataset_params(case: &Path) -> (std::collections::HashMap<String, String>, Vec<i32>) {
    let text = std::fs::read_to_string(case.join("params.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut params = std::collections::HashMap::new();
    for (k, val) in v["params"].as_object().unwrap() {
        let s = match val {
            serde_json::Value::Array(a) => a.iter().map(json_scalar).collect::<Vec<_>>().join(","),
            other => json_scalar(other),
        };
        params.insert(k.clone(), s);
    }
    let cats = v["categorical"]
        .as_array()
        .map(|a| a.iter().map(|c| c.as_i64().unwrap() as i32).collect())
        .unwrap_or_default();
    (params, cats)
}

fn json_scalar(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Bool(b) => {
            if *b {
                "true".into()
            } else {
                "false".into()
            }
        }
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Feature matrix (row-major) and column count from `test.npy`; column 0 is the label.
pub fn load_test_features(case: &Path) -> (Vec<f64>, usize) {
    let (all, shape) = load_npy(&case.join("test.npy"));
    let (nrow, ncol) = (shape[0], shape[1]);
    let mut data = Vec::with_capacity(nrow * (ncol - 1));
    for r in 0..nrow {
        data.extend_from_slice(&all[r * ncol + 1..(r + 1) * ncol]);
    }
    (data, ncol - 1)
}

/// Minimal `.npy` reader for little-endian `<f8`, `<f4`, `<i4`, `<i8`; returns values as f64 and shape.
pub fn load_npy(path: &Path) -> (Vec<f64>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let major = bytes[6];
    let (hlen, hstart) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[hstart..hstart + hlen]).unwrap();
    assert!(header.contains("'fortran_order': False"));
    let descr = header
        .split("'descr': '")
        .nth(1)
        .unwrap()
        .split('\'')
        .next()
        .unwrap();
    let shape_str = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let body = &bytes[hstart + hlen..];
    let vals = match descr {
        "<f8" => body
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        "<f4" => body
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        "<i4" => body
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        "<i8" => body
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect(),
        other => panic!("unsupported npy dtype {other}"),
    };
    (vals, shape)
}

/// Whether this build runs on the platform that generated the fixtures
/// (`tests/fixtures/platform.json`). Only then is bit-exact parity required.
pub fn on_fixture_platform() -> bool {
    let text = std::fs::read_to_string(fixtures_dir().join("platform.json")).unwrap_or_default();
    let (os, arch) = platform();
    text.contains(&format!("\"system\": \"{os}\""))
        && text.contains(&format!("\"machine\": \"{arch}\""))
}

/// This build's platform as Python's `platform.system().lower()` and
/// `platform.machine().lower()` spell it (the names `platform.json` records).
fn platform() -> (&'static str, &'static str) {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "aarch64" if os == "darwin" => "arm64",
        "x86_64" if os == "windows" => "amd64",
        other => other,
    };
    (os, arch)
}

/// `<system>-<machine>`, e.g. `linux-x86_64` or `darwin-arm64`.
pub fn platform_tag() -> String {
    let (os, arch) = platform();
    format!("{os}-{arch}")
}

/// The expected-output file for this platform: `<stem>.<platform_tag>.<ext>` when upstream
/// produced different bits here than on the fixture platform (written by
/// `scripts/gen_train_fixtures.py --platform-overrides`), otherwise `<stem>.<ext>`.
///
/// Upstream's own output is platform-specific (FMA contraction, vectorized reductions), so an
/// override keeps the comparison bit-exact against upstream on every platform that has one.
pub fn platform_fixture(case: &Path, file: &str) -> PathBuf {
    let (stem, ext) = file.rsplit_once('.').unwrap_or((file, ""));
    let specific = case.join(format!("{stem}.{}.{ext}", platform_tag()));
    if specific.exists() {
        specific
    } else {
        case.join(file)
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let i = x.to_bits() as i64;
        if i < 0 { i64::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// Bit-exact comparison on the fixture platform; elsewhere allow a few ULPs, since
/// upstream's own rounding differs between compilers/architectures (see AGENTS.md).
pub fn assert_bits_eq(case: &Path, what: &str, got: &[f64], want: &[f64]) {
    let max_ulps = if on_fixture_platform() { 0 } else { 64 };
    assert_eq!(got.len(), want.len(), "{}: {what} length", case.display());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let ok = (g.is_nan() && w.is_nan())
            || g.to_bits() == w.to_bits()
            || (max_ulps > 0 && ulp_distance(*g, *w) <= max_ulps)
            || (max_ulps > 0 && (g - w).abs() <= 1e-15);
        assert!(ok, "{}: {what}[{i}] got {g:e} want {w:e}", case.display());
    }
}
