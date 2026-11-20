//! `LGBM_BoosterPredictForCSR` matches `lgbm-core`'s sparse predictor and upstream's C API
//! contract (SPEC-ERR-5). The oracle is `lgbm_core::predictor::Predictor::predict_csr`.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree, LGBM_BoosterPredictForCSR, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

// upstream c_api.h.
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

/// A small CSR matrix with some structural zeros, `(indptr, indices, values)`.
fn csr(nrow: usize, ncol: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..nrow {
        for c in 0..ncol {
            // Keep ~two thirds of the entries; skip the rest (structural zeros).
            if (r + c) % 3 != 0 {
                indices.push(c as i32);
                values.push(((r * 7 + c * 3) % 50) as f64 * 0.2 - 4.0);
            }
        }
        indptr.push(indices.len() as i64);
    }
    (indptr, indices, values)
}

#[allow(clippy::too_many_arguments)]
fn predict_csr_capi(
    handle: *mut c_void,
    indptr: *const c_void,
    indptr_type: c_int,
    indices: &[i32],
    data: *const c_void,
    data_type: c_int,
    nindptr: i64,
    nelem: i64,
    num_col: i64,
    out_capacity: usize,
) -> (Vec<f64>, i64) {
    let mut out = vec![0.0f64; out_capacity];
    let mut out_len: i64 = -1;
    let param = CString::new("").unwrap();
    // SAFETY: all pointers are valid for the given lengths; param is a valid C string.
    let rc = unsafe {
        LGBM_BoosterPredictForCSR(
            handle,
            indptr,
            indptr_type,
            indices.as_ptr(),
            data,
            data_type,
            nindptr,
            nelem,
            num_col,
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "csr predict failed: {}", last_error());
    (out, out_len)
}

#[test]
fn predict_for_csr_matches_core() {
    let path = fixtures_dir().join("binary/model.txt");
    let (handle, reference) = load(&path);
    let (nrow, ncol) = (6usize, reference.num_features());
    let (indptr, indices, values) = csr(nrow, ncol);

    let cfg = PredictConfig {
        kind: PredictKind::Normal,
        ..PredictConfig::default()
    };
    let want = Predictor::new(&reference, &cfg)
        .unwrap()
        .predict_csr(&indptr, &indices, &values)
        .unwrap();

    // int64 indptr + float64 values.
    let (got, out_len) = predict_csr_capi(
        handle,
        indptr.as_ptr().cast::<c_void>(),
        C_API_DTYPE_INT64,
        &indices,
        values.as_ptr().cast::<c_void>(),
        C_API_DTYPE_FLOAT64,
        indptr.len() as i64,
        values.len() as i64,
        ncol as i64,
        want.len(),
    );
    assert_eq!(out_len as usize, want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.to_bits(), w.to_bits(), "int64/f64: {g} != {w}");
    }

    // int32 indptr + float32 values: same values, compared to the core on the f32-rounded input.
    let indptr32: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
    let values32: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    let values32_as64: Vec<f64> = values32.iter().map(|&x| f64::from(x)).collect();
    let want32 = Predictor::new(&reference, &cfg)
        .unwrap()
        .predict_csr(&indptr, &indices, &values32_as64)
        .unwrap();
    let (got32, _) = predict_csr_capi(
        handle,
        indptr32.as_ptr().cast::<c_void>(),
        C_API_DTYPE_INT32,
        &indices,
        values32.as_ptr().cast::<c_void>(),
        C_API_DTYPE_FLOAT32,
        indptr32.len() as i64,
        values32.len() as i64,
        ncol as i64,
        want32.len(),
    );
    for (g, w) in got32.iter().zip(&want32) {
        assert_eq!(g.to_bits(), w.to_bits(), "int32/f32: {g} != {w}");
    }

    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn predict_for_csr_rejects_null_handle() {
    let indptr = [0i64, 1];
    let indices = [0i32];
    let values = [1.0f64];
    let mut out = [0.0f64; 1];
    let mut out_len = 0i64;
    let param = CString::new("").unwrap();
    // SAFETY: null handle is an explicitly handled error path; other pointers are valid.
    let rc = unsafe {
        LGBM_BoosterPredictForCSR(
            ptr::null_mut(),
            indptr.as_ptr().cast::<c_void>(),
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
