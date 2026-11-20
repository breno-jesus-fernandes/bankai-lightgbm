//! `LGBM_BoosterPredictForMatSingleRow` matches a one-row `LGBM_BoosterPredictForMat` and
//! `lgbm-core`'s predictor (SPEC-ERR-5).
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCalcNumPredict, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterPredictForMatSingleRow, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_PREDICT_NORMAL: c_int = 0;
const C_API_PREDICT_RAW_SCORE: c_int = 1;

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

#[test]
fn predict_single_row_matches_core() {
    let path = fixtures_dir().join("binary/model.txt");
    let (handle, reference) = load(&path);
    let ncol = reference.num_features();
    let row: Vec<f64> = (0..ncol).map(|c| (c as f64) * 0.3 - 2.0).collect();

    for (ptype, kind) in [
        (C_API_PREDICT_NORMAL, PredictKind::Normal),
        (C_API_PREDICT_RAW_SCORE, PredictKind::RawScore),
    ] {
        let cfg = PredictConfig {
            kind,
            ..PredictConfig::default()
        };
        let want = Predictor::new(&reference, &cfg)
            .unwrap()
            .predict_dense(&row, 1, ncol)
            .unwrap();

        let mut num_pred: i64 = 0;
        // SAFETY: live handle, writable out-pointer.
        unsafe { LGBM_BoosterCalcNumPredict(handle, 1, ptype, 0, -1, &mut num_pred) };
        let mut out = vec![0.0f64; num_pred as usize];
        let mut out_len: i64 = -1;
        let param = CString::new("").unwrap();
        // SAFETY: row has ncol f64 values; out buffer sized by the calc above.
        let rc = unsafe {
            LGBM_BoosterPredictForMatSingleRow(
                handle,
                row.as_ptr().cast::<c_void>(),
                C_API_DTYPE_FLOAT64,
                ncol as c_int,
                1, // is_row_major (moot for one row)
                ptype,
                0,
                -1,
                param.as_ptr(),
                &mut out_len,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "single-row predict failed: {}", last_error());
        assert_eq!(out_len as usize, want.len());
        for (g, w) in out.iter().zip(&want) {
            assert_eq!(g.to_bits(), w.to_bits(), "predict_type {ptype}: {g} != {w}");
        }
    }
    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn predict_single_row_rejects_null_handle() {
    let row = [1.0f64, 2.0];
    let mut out = [0.0f64; 1];
    let mut out_len = 0i64;
    let param = CString::new("").unwrap();
    // SAFETY: null handle is an explicitly handled error path; other pointers are valid.
    let rc = unsafe {
        LGBM_BoosterPredictForMatSingleRow(
            ptr::null_mut(),
            row.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            2,
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
