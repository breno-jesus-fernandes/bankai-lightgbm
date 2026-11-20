//! `LGBM_BoosterCreate` + `LGBM_BoosterUpdateOneIter` train a booster from a dataset handle and
//! match `lgbm-core`'s `Gbdt::train` on the same dataset (SPEC-ERR-5). The oracle predicts with
//! the core model trained from the same data, params and labels; the C API path must agree
//! bit-for-bit, which pins that the trained trees are identical.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::Path;
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterPredictForMat, LGBM_BoosterUpdateOneIter, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_PREDICT_NORMAL: c_int = 0;

const PARAMS: &str = "objective=regression max_bin=31 num_iterations=12 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect()
}

/// A label that actually depends on the features, so the trees split.
fn labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| {
            let row = &data[r * ncol..(r + 1) * ncol];
            row[0] * 2.0 - row[ncol - 1] + (r % 5) as f64 * 0.3
        })
        .collect()
}

/// Build a dataset handle from `data` with `PARAMS` and set its `label`.
fn dataset_with_label(data: &[f64], label: &[f64], nrow: usize, ncol: usize) -> *mut c_void {
    let params = CString::new(PARAMS).unwrap();
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: data has nrow*ncol f64; params valid; reference null; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "dataset create failed: {}", last_error());

    let label32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
    let field = CString::new("label").unwrap();
    // SAFETY: label32 has nrow f32; field is a valid C string.
    let rc = unsafe {
        LGBM_DatasetSetField(
            handle,
            field.as_ptr(),
            label32.as_ptr().cast::<c_void>(),
            nrow as c_int,
            C_API_DTYPE_FLOAT32,
        )
    };
    assert_eq!(rc, 0, "set label failed: {}", last_error());
    handle
}

fn predict_mat(handle: *mut c_void, data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; nrow];
    let mut out_len: i64 = -1;
    let param = CString::new("").unwrap();
    // SAFETY: data has nrow*ncol f64; out holds nrow doubles (one output per row, regression).
    let rc = unsafe {
        LGBM_BoosterPredictForMat(
            handle,
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as c_int,
            ncol as c_int,
            1,
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "predict failed: {}", last_error());
    assert_eq!(out_len as usize, nrow);
    out
}

#[test]
fn train_matches_core_gbdt_train() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);

    let ds = dataset_with_label(&data, &label, nrow, ncol);

    // Train through the C API: num_iterations update calls, breaking on an empty tree, exactly as
    // core's `train_shared` does.
    let mut booster: *mut c_void = ptr::null_mut();
    let params = CString::new(PARAMS).unwrap();
    // SAFETY: live dataset handle; params valid; out writable.
    let rc = unsafe { LGBM_BoosterCreate(ds, params.as_ptr(), &mut booster) };
    assert_eq!(rc, 0, "booster create failed: {}", last_error());
    assert!(!booster.is_null());

    let cfg = Config::from_str_params(PARAMS).unwrap();
    for _ in 0..cfg.num_iterations {
        let mut empty: c_int = -1;
        // SAFETY: live booster handle; out writable.
        let rc = unsafe { LGBM_BoosterUpdateOneIter(booster, &mut empty) };
        assert_eq!(rc, 0, "update failed: {}", last_error());
        if empty == 1 {
            break;
        }
    }

    let got = predict_mat(booster, &data, nrow, ncol);

    // Oracle: core trains from the same data/params/labels.
    let reference_ds = Dataset::from_dense(&data, nrow, ncol, &label, &cfg, &[]).unwrap();
    let model = Gbdt::train(&cfg, &reference_ds).unwrap();
    let pcfg = PredictConfig {
        kind: PredictKind::Normal,
        ..PredictConfig::default()
    };
    let want = Predictor::new(&model, &pcfg)
        .unwrap()
        .predict_dense(&data, nrow, ncol)
        .unwrap();

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "row {i}: {g} != {w}");
    }

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn update_one_iter_rejects_a_loaded_model_booster() {
    // A booster loaded from a model file has no training state.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/binary/model.txt");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc =
        unsafe { LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut num_iterations, &mut handle) };
    assert_eq!(rc, 0, "load failed: {}", last_error());

    let mut empty: c_int = -1;
    // SAFETY: live (loaded) booster handle; out writable.
    let rc = unsafe { LGBM_BoosterUpdateOneIter(handle, &mut empty) };
    assert_eq!(rc, -1);
    assert!(last_error().contains("train"), "got: {}", last_error());

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn booster_create_rejects_nulls() {
    let params = CString::new(PARAMS).unwrap();
    let mut out: *mut c_void = ptr::null_mut();

    // Null dataset handle.
    // SAFETY: null dataset is an explicitly handled error path.
    let rc = unsafe { LGBM_BoosterCreate(ptr::null(), params.as_ptr(), &mut out) };
    assert_eq!(rc, -1);
    assert!(out.is_null());

    // Null out pointer.
    let data = matrix(8, 2);
    let label = vec![0.0f64; 8];
    let ds = dataset_with_label(&data, &label, 8, 2);
    // SAFETY: live dataset; null out is an explicitly handled error path.
    let rc = unsafe { LGBM_BoosterCreate(ds, params.as_ptr(), ptr::null_mut()) };
    assert_eq!(rc, -1);
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}
