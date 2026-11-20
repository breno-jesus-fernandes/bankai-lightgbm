//! `LGBM_BoosterGetEval` reports the training metric values of a training booster. Oracle: an
//! identically-built core `GbdtTrainer` evaluated with `eval_values` (the training-score source).
//! Only the training data (`data_idx == 0`) is supported; a loaded model reports no values.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree, LGBM_BoosterGetEval,
    LGBM_BoosterGetEvalCounts, LGBM_BoosterUpdateOneIter, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::boosting::GbdtTrainer;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;

const PARAMS: &str = "objective=regression metric=l2,l1 max_bin=31 num_iterations=6 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";
const NITER: usize = 6;

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

fn labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| {
            let row = &data[r * ncol..(r + 1) * ncol];
            row[0] * 2.0 - row[ncol - 1] + (r % 5) as f64 * 0.3
        })
        .collect()
}

/// Build a dataset handle and train a booster `NITER` iterations through the C API.
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
    for _ in 0..NITER {
        let mut empty: c_int = -1;
        // SAFETY: live booster; out writable.
        assert_eq!(unsafe { LGBM_BoosterUpdateOneIter(booster, &mut empty) }, 0);
    }
    (booster, ds)
}

/// The oracle: an identically built core trainer, evaluated the same way.
fn core_eval(data: &[f64], label: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    let cfg = Config::from_str_params(PARAMS).unwrap();
    let ds = Dataset::from_dense(data, nrow, ncol, label, &cfg, &[]).unwrap();
    let mut trainer = GbdtTrainer::new(Arc::new(cfg), Arc::new(ds)).unwrap();
    for _ in 0..NITER {
        trainer.update_one_iter();
    }
    trainer.eval_values().unwrap()
}

#[test]
fn get_eval_matches_core_over_the_training_data() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let mut count: c_int = -1;
    // SAFETY: live handle; out writable.
    assert_eq!(unsafe { LGBM_BoosterGetEvalCounts(booster, &mut count) }, 0);
    assert_eq!(count, 2);

    let mut out = vec![-1.0f64; count as usize];
    let mut out_len: c_int = -1;
    // SAFETY: live handle; out sized by the count; out_len writable.
    let rc = unsafe { LGBM_BoosterGetEval(booster, 0, &mut out_len, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "get eval failed: {}", last_error());
    assert_eq!(out_len, 2);

    let want = core_eval(&data, &label, nrow, ncol);
    assert_eq!(out, want);

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn validation_index_is_not_supported_yet() {
    let (nrow, ncol) = (30usize, 4usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let mut out = [0.0f64; 2];
    let mut out_len: c_int = -1;
    // SAFETY: live handle; out holds 2 f64; out_len writable.
    let rc = unsafe { LGBM_BoosterGetEval(booster, 1, &mut out_len, out.as_mut_ptr()) };
    assert_eq!(rc, -1, "validation data is not available yet");
    assert!(last_error().contains("validation"));

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn a_loaded_model_reports_no_values_and_nulls_are_rejected() {
    // Null handle.
    let mut out_len: c_int = -1;
    let mut out = [0.0f64; 1];
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetEval(ptr::null_mut(), 0, &mut out_len, out.as_mut_ptr()) },
        -1
    );
    assert!(last_error().contains("null"));

    // A loaded model has no training data: data_idx 0 reports zero values.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/binary/model.txt");
    let c_src = CString::new(src.to_str().unwrap()).unwrap();
    let mut ni: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    assert_eq!(
        unsafe { LGBM_BoosterCreateFromModelfile(c_src.as_ptr(), &mut ni, &mut handle) },
        0,
        "load failed: {}",
        last_error()
    );
    let mut out_len: c_int = -1;
    // SAFETY: live handle; out_len writable; out may be unused when the length is 0.
    let rc = unsafe { LGBM_BoosterGetEval(handle, 0, &mut out_len, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "get eval failed: {}", last_error());
    assert_eq!(out_len, 0);

    // Null out_len.
    // SAFETY: null out_len is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetEval(handle, 0, ptr::null_mut(), out.as_mut_ptr()) },
        -1
    );
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}
