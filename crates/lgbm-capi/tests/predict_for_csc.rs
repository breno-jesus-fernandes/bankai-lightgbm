//! `LGBM_BoosterPredictForCSC` matches `lgbm-core`'s CSC predictor and upstream's C API
//! contract (SPEC-ERR-5). The oracle is `lgbm_core::predictor::Predictor::predict_csc`.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree, LGBM_BoosterPredictForCSC, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_DTYPE_INT32: c_int = 2;
const C_API_DTYPE_INT64: c_int = 3;
const C_API_PREDICT_NORMAL: c_int = 0;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

fn load(path: &Path) -> (*mut c_void, Gbdt) {
    let reference = Gbdt::from_model_file(path).unwrap();
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, 0, "load failed: {}", last_error());
    (handle, reference)
}

/// A CSC matrix with some structural zeros: `(col_ptr, row_indices, values)`.
fn csc(nrow: usize, ncol: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let mut col_ptr = vec![0i64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for c in 0..ncol {
        for r in 0..nrow {
            if (r + c) % 3 != 0 {
                indices.push(r as i32);
                values.push(((r * 7 + c * 3) % 50) as f64 * 0.2 - 4.0);
            }
        }
        col_ptr.push(indices.len() as i64);
    }
    (col_ptr, indices, values)
}

#[allow(clippy::too_many_arguments)]
fn predict_csc_capi(
    handle: *mut c_void,
    col_ptr: *const c_void,
    col_ptr_type: c_int,
    indices: &[i32],
    data: *const c_void,
    data_type: c_int,
    ncol_ptr: i64,
    nelem: i64,
    num_row: i64,
    out_capacity: usize,
) -> (Vec<f64>, i64) {
    let mut out = vec![0.0f64; out_capacity];
    let mut out_len: i64 = -1;
    let param = CString::new("").unwrap();
    // SAFETY: all pointers valid for the given lengths; param is a valid C string.
    let rc = unsafe {
        LGBM_BoosterPredictForCSC(
            handle,
            col_ptr,
            col_ptr_type,
            indices.as_ptr(),
            data,
            data_type,
            ncol_ptr,
            nelem,
            num_row,
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "csc predict failed: {}", last_error());
    (out, out_len)
}

#[test]
fn predict_for_csc_matches_core() {
    let path = fixtures_dir().join("binary/model.txt");
    let (handle, reference) = load(&path);
    let (nrow, ncol) = (6usize, reference.num_features());
    let (col_ptr, indices, values) = csc(nrow, ncol);

    let cfg = PredictConfig {
        kind: PredictKind::Normal,
        ..PredictConfig::default()
    };
    let want = Predictor::new(&reference, &cfg)
        .unwrap()
        .predict_csc(&col_ptr, &indices, &values, nrow)
        .unwrap();

    // int64 col_ptr + float64 values.
    let (got, out_len) = predict_csc_capi(
        handle,
        col_ptr.as_ptr().cast::<c_void>(),
        C_API_DTYPE_INT64,
        &indices,
        values.as_ptr().cast::<c_void>(),
        C_API_DTYPE_FLOAT64,
        col_ptr.len() as i64,
        values.len() as i64,
        nrow as i64,
        want.len(),
    );
    assert_eq!(out_len as usize, want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.to_bits(), w.to_bits(), "int64/f64: {g} != {w}");
    }

    // int32 col_ptr + float32 values.
    let col_ptr32: Vec<i32> = col_ptr.iter().map(|&x| x as i32).collect();
    let values32: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    let values32_as64: Vec<f64> = values32.iter().map(|&x| f64::from(x)).collect();
    let want32 = Predictor::new(&reference, &cfg)
        .unwrap()
        .predict_csc(&col_ptr, &indices, &values32_as64, nrow)
        .unwrap();
    let (got32, _) = predict_csc_capi(
        handle,
        col_ptr32.as_ptr().cast::<c_void>(),
        C_API_DTYPE_INT32,
        &indices,
        values32.as_ptr().cast::<c_void>(),
        C_API_DTYPE_FLOAT32,
        col_ptr32.len() as i64,
        values32.len() as i64,
        nrow as i64,
        want32.len(),
    );
    for (g, w) in got32.iter().zip(&want32) {
        assert_eq!(g.to_bits(), w.to_bits(), "int32/f32: {g} != {w}");
    }

    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn predict_for_csc_rejects_null_handle() {
    let col_ptr = [0i64, 1];
    let indices = [0i32];
    let values = [1.0f64];
    let mut out = [0.0f64; 1];
    let mut out_len = 0i64;
    let param = CString::new("").unwrap();
    // SAFETY: null handle is an explicitly handled error path; other pointers are valid.
    let rc = unsafe {
        LGBM_BoosterPredictForCSC(
            ptr::null_mut(),
            col_ptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            2,
            1,
            1,
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, -1);
    assert!(last_error().contains("null"), "got: {}", last_error());
}
