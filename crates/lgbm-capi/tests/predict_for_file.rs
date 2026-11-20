//! `LGBM_BoosterPredictForFile` reads a data file, predicts every row, and writes the predictions
//! to a result file (one row per line, tab-separated). Oracle: core's `Predictor::predict_dense`
//! over the features `lgbm-core`'s own file loader produces from the same file.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterFree, LGBM_BoosterPredictForFile,
    LGBM_BoosterSaveModelToString, LGBM_BoosterUpdateOneIter, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_PREDICT_NORMAL: c_int = 0;
const C_API_FEATURE_IMPORTANCE_SPLIT: c_int = 0;

const PARAMS: &str = "objective=regression max_bin=31 num_iterations=8 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

fn temp_path(tag: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("bankai_capi_{tag}_{pid}_{nanos}"))
}

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect()
}

fn labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| {
            let row = &data[r * ncol..(r + 1) * ncol];
            row[0] * 2.0 - row[ncol - 1] + (r % 5) as f64 * 0.3
        })
        .collect()
}

/// A row-major matrix as CSV text, each value with its shortest round-trippable form.
fn to_csv(data: &[f64], nrow: usize, ncol: usize, header: bool) -> String {
    let mut s = String::new();
    if header {
        let cols: Vec<String> = (0..ncol).map(|c| format!("f{c}")).collect();
        s.push_str(&cols.join(","));
        s.push('\n');
    }
    for r in 0..nrow {
        let row: Vec<String> = (0..ncol).map(|c| data[r * ncol + c].to_string()).collect();
        s.push_str(&row.join(","));
        s.push('\n');
    }
    s
}

/// Train a booster through the C API; returns the booster and dataset handles to free.
fn trained(data: &[f64], label: &[f64], nrow: usize, ncol: usize) -> (*mut c_void, *mut c_void) {
    let params = CString::new(PARAMS).unwrap();
    let mut ds: *mut c_void = ptr::null_mut();
    // SAFETY: data has nrow*ncol f64; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1,
            params.as_ptr(),
            ptr::null(),
            &mut ds,
        )
    };
    assert_eq!(rc, 0, "dataset create failed: {}", last_error());

    let label32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
    let field = CString::new("label").unwrap();
    // SAFETY: label32 has nrow f32; field valid.
    assert_eq!(
        unsafe {
            LGBM_DatasetSetField(
                ds,
                field.as_ptr(),
                label32.as_ptr().cast::<c_void>(),
                nrow as c_int,
                C_API_DTYPE_FLOAT32,
            )
        },
        0
    );

    let mut booster: *mut c_void = ptr::null_mut();
    // SAFETY: live dataset; params valid; out writable.
    assert_eq!(
        unsafe { LGBM_BoosterCreate(ds, params.as_ptr(), &mut booster) },
        0,
        "booster create failed: {}",
        last_error()
    );
    let cfg = Config::from_str_params(PARAMS).unwrap();
    for _ in 0..cfg.num_iterations {
        let mut empty: c_int = -1;
        // SAFETY: live booster; out writable.
        assert_eq!(unsafe { LGBM_BoosterUpdateOneIter(booster, &mut empty) }, 0);
        if empty == 1 {
            break;
        }
    }
    (booster, ds)
}

/// The model text the booster serializes to (two-call protocol), as the oracle's model source.
fn model_text(handle: *mut c_void) -> String {
    let mut needed: i64 = -1;
    // SAFETY: live handle; out_len writable; out_str null allowed when buffer_len is 0.
    assert_eq!(
        unsafe {
            LGBM_BoosterSaveModelToString(
                handle,
                0,
                0,
                C_API_FEATURE_IMPORTANCE_SPLIT,
                0,
                &mut needed,
                ptr::null_mut(),
            )
        },
        0
    );
    let mut buf = vec![0u8; needed as usize];
    let mut written: i64 = -1;
    // SAFETY: buf holds `needed` bytes; out_len writable.
    assert_eq!(
        unsafe {
            LGBM_BoosterSaveModelToString(
                handle,
                0,
                0,
                C_API_FEATURE_IMPORTANCE_SPLIT,
                needed,
                &mut written,
                buf.as_mut_ptr().cast::<c_char>(),
            )
        },
        0
    );
    String::from_utf8(buf[..buf.len() - 1].to_vec()).unwrap()
}

/// Parse a result file into one `f64` per tab-separated field, row-major.
fn parse_result(text: &str) -> Vec<f64> {
    text.lines()
        .flat_map(|line| line.split('\t').map(|v| v.parse::<f64>().unwrap()))
        .collect()
}

/// Oracle: the predictions core produces for the features `load_text` reads from `csv`.
fn oracle(csv: &str, model_txt: &str, ncol: usize, header: bool) -> Vec<f64> {
    let opts = TextLoadOptions {
        header,
        label_column: "",
        weight_column: "",
        group_column: "",
        ignore_column: "",
        precise_float_parser: false,
        num_features: ncol,
    };
    let loaded = load_text(csv, &opts).unwrap();
    let model = Gbdt::from_model_string(model_txt).unwrap();
    let cfg = PredictConfig {
        kind: PredictKind::Normal,
        ..PredictConfig::default()
    };
    let predictor = Predictor::new(&model, &cfg).unwrap();
    predictor
        .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
        .unwrap()
}

#[test]
fn predict_for_file_matches_core() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    // Predict a fresh matrix (no label column) written as CSV.
    let test = matrix(23, ncol);
    let csv = to_csv(&test, 23, ncol, false);
    let data_file = temp_path("file_data").with_extension("csv");
    let result_file = temp_path("file_out").with_extension("txt");
    std::fs::write(&data_file, &csv).unwrap();

    let c_data = CString::new(data_file.to_str().unwrap()).unwrap();
    let c_result = CString::new(result_file.to_str().unwrap()).unwrap();
    // SAFETY: live handle; valid C strings; null parameter.
    let rc = unsafe {
        LGBM_BoosterPredictForFile(
            booster,
            c_data.as_ptr(),
            0,
            C_API_PREDICT_NORMAL,
            0,
            0,
            ptr::null(),
            c_result.as_ptr(),
        )
    };
    assert_eq!(rc, 0, "predict for file failed: {}", last_error());

    let got = parse_result(&std::fs::read_to_string(&result_file).unwrap());
    let want = oracle(&csv, &model_text(booster), ncol, false);
    assert_eq!(got.len(), 23, "one prediction per row for regression");
    assert_eq!(got, want);

    std::fs::remove_file(&data_file).ok();
    std::fs::remove_file(&result_file).ok();
    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn predict_for_file_skips_the_header_row() {
    let (nrow, ncol) = (40usize, 4usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let test = matrix(15, ncol);
    let csv = to_csv(&test, 15, ncol, true);
    let data_file = temp_path("hdr_data").with_extension("csv");
    let result_file = temp_path("hdr_out").with_extension("txt");
    std::fs::write(&data_file, &csv).unwrap();

    let c_data = CString::new(data_file.to_str().unwrap()).unwrap();
    let c_result = CString::new(result_file.to_str().unwrap()).unwrap();
    // SAFETY: live handle; valid C strings; header flag set; null parameter.
    assert_eq!(
        unsafe {
            LGBM_BoosterPredictForFile(
                booster,
                c_data.as_ptr(),
                1,
                C_API_PREDICT_NORMAL,
                0,
                0,
                ptr::null(),
                c_result.as_ptr(),
            )
        },
        0,
        "predict for file failed: {}",
        last_error()
    );

    let got = parse_result(&std::fs::read_to_string(&result_file).unwrap());
    let want = oracle(&csv, &model_text(booster), ncol, true);
    assert_eq!(got.len(), 15);
    assert_eq!(got, want);

    std::fs::remove_file(&data_file).ok();
    std::fs::remove_file(&result_file).ok();
    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn rejects_null_pointers_and_unsupported_parameter() {
    let (nrow, ncol) = (20usize, 3usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let data_file = temp_path("err_data").with_extension("csv");
    std::fs::write(&data_file, to_csv(&matrix(5, ncol), 5, ncol, false)).unwrap();
    let c_data = CString::new(data_file.to_str().unwrap()).unwrap();
    let result_file = temp_path("err_out").with_extension("txt");
    let c_result = CString::new(result_file.to_str().unwrap()).unwrap();

    // Null handle.
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe {
            LGBM_BoosterPredictForFile(
                ptr::null_mut(),
                c_data.as_ptr(),
                0,
                C_API_PREDICT_NORMAL,
                0,
                0,
                ptr::null(),
                c_result.as_ptr(),
            )
        },
        -1
    );
    assert!(last_error().contains("null"));

    // Null result filename.
    // SAFETY: null result_filename is an explicitly handled error path.
    assert_eq!(
        unsafe {
            LGBM_BoosterPredictForFile(
                booster,
                c_data.as_ptr(),
                0,
                C_API_PREDICT_NORMAL,
                0,
                0,
                ptr::null(),
                ptr::null(),
            )
        },
        -1
    );

    // Non-empty prediction parameter string: not supported yet.
    let param = CString::new("num_threads=2").unwrap();
    // SAFETY: live handle; valid C strings; non-empty parameter is rejected.
    assert_eq!(
        unsafe {
            LGBM_BoosterPredictForFile(
                booster,
                c_data.as_ptr(),
                0,
                C_API_PREDICT_NORMAL,
                0,
                0,
                param.as_ptr(),
                c_result.as_ptr(),
            )
        },
        -1
    );
    assert!(last_error().contains("not supported"));

    std::fs::remove_file(&data_file).ok();
    std::fs::remove_file(&result_file).ok();
    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}
