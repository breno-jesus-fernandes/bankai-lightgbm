//! `LGBM_BoosterPredictForMat` and `LGBM_BoosterCalcNumPredict` match `lgbm-core`'s predictor
//! and upstream's C API contract: `0`/`-1` return codes, a thread-local last error, and
//! panic-free boundaries (SPEC-ERR-5). The oracle is `lgbm_core::predictor::Predictor`.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCalcNumPredict, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterPredictForMat, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

// upstream c_api.h.
const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_PREDICT_NORMAL: c_int = 0;
const C_API_PREDICT_RAW_SCORE: c_int = 1;
const C_API_PREDICT_LEAF_INDEX: c_int = 2;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    let c = unsafe { CStr::from_ptr(LGBM_GetLastError()) };
    c.to_string_lossy().into_owned()
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

fn sample_matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    // Deterministic, spread across a plausible feature range.
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 100) as f64) * 0.1 - 5.0)
        .collect()
}

fn predict_via_capi(
    handle: *mut c_void,
    data: &[f64],
    nrow: usize,
    ncol: usize,
    predict_type: c_int,
) -> Vec<f64> {
    let mut num_pred: i64 = -1;
    // SAFETY: live handle, writable out-pointer.
    let rc = unsafe {
        LGBM_BoosterCalcNumPredict(handle, nrow as c_int, predict_type, 0, -1, &mut num_pred)
    };
    assert_eq!(rc, 0, "calc num predict failed: {}", last_error());
    assert!(num_pred >= 0);
    let mut out = vec![0.0f64; num_pred as usize];
    let mut out_len: i64 = -1;
    let param = CString::new("").unwrap();
    // SAFETY: data and out buffers are valid for the given shapes; param is a valid C string.
    let rc = unsafe {
        LGBM_BoosterPredictForMat(
            handle,
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1, // row-major
            predict_type,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "predict failed: {}", last_error());
    assert_eq!(out_len, num_pred, "out_len must equal the calc'd length");
    out
}

fn reference_predict(model: &Gbdt, data: &[f64], nrow: usize, ncol: usize, kind: PredictKind) -> Vec<f64> {
    let cfg = PredictConfig {
        kind,
        ..PredictConfig::default()
    };
    let p = Predictor::new(model, &cfg).unwrap();
    p.predict_dense(data, nrow, ncol).unwrap()
}

#[test]
fn predict_for_mat_matches_core_for_each_predict_type() {
    let path = fixtures_dir().join("binary/model.txt");
    let (handle, reference) = load(&path);
    let (nrow, ncol) = (5usize, reference.num_features());
    let data = sample_matrix(nrow, ncol);

    for (ptype, kind) in [
        (C_API_PREDICT_NORMAL, PredictKind::Normal),
        (C_API_PREDICT_RAW_SCORE, PredictKind::RawScore),
        (C_API_PREDICT_LEAF_INDEX, PredictKind::LeafIndex),
    ] {
        let got = predict_via_capi(handle, &data, nrow, ncol, ptype);
        let want = reference_predict(&reference, &data, nrow, ncol, kind);
        assert_eq!(got.len(), want.len(), "length for predict_type {ptype}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g.to_bits(), w.to_bits(), "predict_type {ptype}: {g} != {w}");
        }
    }
    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn predict_for_mat_accepts_float32_and_column_major() {
    let path = fixtures_dir().join("binary/model.txt");
    let (handle, reference) = load(&path);
    let (nrow, ncol) = (4usize, reference.num_features());
    let row_major = sample_matrix(nrow, ncol);
    let want = reference_predict(&reference, &row_major, nrow, ncol, PredictKind::Normal);

    // float32 row-major: same values, within the f32 round-trip (compare as f32-exact input).
    let f32_data: Vec<f32> = row_major.iter().map(|&x| x as f32).collect();
    let f64_from_f32: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
    let want32 = reference_predict(&reference, &f64_from_f32, nrow, ncol, PredictKind::Normal);
    let mut num_pred: i64 = 0;
    // SAFETY: live handle, writable out-pointer.
    unsafe { LGBM_BoosterCalcNumPredict(handle, nrow as c_int, C_API_PREDICT_NORMAL, 0, -1, &mut num_pred) };
    let mut out = vec![0.0f64; num_pred as usize];
    let mut out_len = 0i64;
    let param = CString::new("").unwrap();
    // SAFETY: f32 data valid for nrow*ncol; out buffer sized by the calc above.
    let rc = unsafe {
        LGBM_BoosterPredictForMat(
            handle,
            f32_data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT32,
            nrow as i32,
            ncol as i32,
            1,
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "float32 predict failed: {}", last_error());
    for (g, w) in out.iter().zip(&want32) {
        assert_eq!(g.to_bits(), w.to_bits(), "float32: {g} != {w}");
    }

    // column-major f64: transpose the same matrix; result must match the row-major prediction.
    let mut col_major = vec![0.0f64; nrow * ncol];
    for r in 0..nrow {
        for c in 0..ncol {
            col_major[c * nrow + r] = row_major[r * ncol + c];
        }
    }
    let mut out2 = vec![0.0f64; num_pred as usize];
    // SAFETY: col-major data valid for nrow*ncol; out buffer sized above.
    let rc = unsafe {
        LGBM_BoosterPredictForMat(
            handle,
            col_major.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            0, // column-major
            C_API_PREDICT_NORMAL,
            0,
            -1,
            param.as_ptr(),
            &mut out_len,
            out2.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "column-major predict failed: {}", last_error());
    for (g, w) in out2.iter().zip(&want) {
        assert_eq!(g.to_bits(), w.to_bits(), "column-major: {g} != {w}");
    }

    // SAFETY: handle freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn predict_for_mat_rejects_bad_inputs() {
    let mut out_len = 0i64;
    let mut out = [0.0f64; 4];
    let param = CString::new("").unwrap();
    // Null handle.
    // SAFETY: null handle is an explicitly handled error path.
    let rc = unsafe {
        LGBM_BoosterPredictForMat(
            ptr::null_mut(),
            [1.0f64].as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            1,
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
