//! Helpers for golden tests: locate fixtures, read `.npy` and `test.csv` files.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::path::{Path, PathBuf};

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

/// Feature matrix (row-major) and column count from `test.csv`; column 0 is the label.
pub fn load_test_features(case: &Path) -> (Vec<f64>, usize) {
    let text = std::fs::read_to_string(case.join("test.csv")).unwrap();
    let mut data = Vec::new();
    let mut ncol = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        let vals: Vec<f64> = line
            .split(',')
            .skip(1)
            .map(|s| s.trim().parse().unwrap())
            .collect();
        ncol = vals.len();
        data.extend(vals);
    }
    (data, ncol)
}

/// Minimal `.npy` reader for little-endian `<f8`, `<i4`, `<i8`; returns values as f64 and shape.
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

/// Bit-exact comparison with a readable failure message.
pub fn assert_bits_eq(case: &Path, what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{}: {what} length", case.display());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{}: {what}[{i}] got {g:e} want {w:e}",
            case.display()
        );
    }
}
