//! `LGBM_BoosterRollbackOneIter` drops the last boosting iteration from a training booster.
//! Oracle: a booster trained to `n` iterations and rolled back once serializes exactly like a
//! booster trained to `n - 1` iterations (core `Gbdt::train`).
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterRollbackOneIter, LGBM_BoosterSaveModelToString, LGBM_BoosterUpdateOneIter,
    LGBM_DatasetCreateFromMat, LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
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

/// Build a dataset handle and a fresh training booster over it. Returns (booster, dataset).
fn fresh_booster(
    data: &[f64],
    label: &[f64],
    nrow: usize,
    ncol: usize,
) -> (*mut c_void, *mut c_void) {
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
    (booster, ds)
}

/// Run `n` boosting iterations on a training booster.
fn train_iters(booster: *mut c_void, n: usize) {
    for _ in 0..n {
        let mut empty: c_int = -1;
        // SAFETY: live booster; out writable.
        assert_eq!(unsafe { LGBM_BoosterUpdateOneIter(booster, &mut empty) }, 0);
        assert_eq!(empty, 0, "iteration produced an empty tree unexpectedly");
    }
}

/// The in-memory serializer (two-call protocol) used to compare boosters.
fn save_to_string(handle: *mut c_void) -> String {
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
        0,
        "length query failed: {}",
        last_error()
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
        0,
        "save failed: {}",
        last_error()
    );
    String::from_utf8(buf[..buf.len() - 1].to_vec()).unwrap()
}

/// Core oracle: the model text after training `n` iterations from scratch.
fn core_model_text(data: &[f64], label: &[f64], nrow: usize, ncol: usize, n: i32) -> String {
    let mut cfg = Config::from_str_params(PARAMS).unwrap();
    cfg.num_iterations = n;
    let ds = Dataset::from_dense(data, nrow, ncol, label, &cfg, &[]).unwrap();
    let model = Gbdt::train(&cfg, &ds).unwrap();
    model
        .save_model_to_string(0, 0, C_API_FEATURE_IMPORTANCE_SPLIT)
        .unwrap()
}

#[test]
fn rollback_matches_training_one_fewer_iteration() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);

    let (booster, ds) = fresh_booster(&data, &label, nrow, ncol);
    train_iters(booster, 6);
    // SAFETY: live training booster.
    assert_eq!(
        unsafe { LGBM_BoosterRollbackOneIter(booster) },
        0,
        "rollback failed: {}",
        last_error()
    );

    let got = save_to_string(booster);
    let want = core_model_text(&data, &label, nrow, ncol, 5);
    assert_eq!(got, want, "rollback to 5 should equal training 5 iterations");

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn rollback_then_train_reaches_the_same_model() {
    let (nrow, ncol) = (50usize, 4usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);

    let (booster, ds) = fresh_booster(&data, &label, nrow, ncol);
    train_iters(booster, 5);
    // SAFETY: live training booster.
    assert_eq!(unsafe { LGBM_BoosterRollbackOneIter(booster) }, 0);
    // Re-train the dropped iteration: deterministic, so we land on the 5-iteration model again.
    train_iters(booster, 1);

    let got = save_to_string(booster);
    let want = core_model_text(&data, &label, nrow, ncol, 5);
    assert_eq!(got, want);

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn rejects_null_handle_and_loaded_model() {
    // Null handle.
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(unsafe { LGBM_BoosterRollbackOneIter(ptr::null_mut()) }, -1);
    assert!(last_error().contains("null"));

    // A booster loaded from a model file has no training state to roll back.
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
    // SAFETY: live loaded booster; rollback is rejected for it.
    assert_eq!(unsafe { LGBM_BoosterRollbackOneIter(handle) }, -1);
    assert!(last_error().contains("loaded from a model"));
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}
