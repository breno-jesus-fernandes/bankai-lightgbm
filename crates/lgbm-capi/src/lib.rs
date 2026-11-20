//! `c_api.h`-compatible FFI (`LGBM_*` symbols).
//!
//! upstream: include/LightGBM/c_api.h, src/c_api.cpp (v4.7.0). Each function returns `0` on
//! success and `-1` on error (upstream's `API_BEGIN`/`API_END`); the message is retrieved with
//! [`LGBM_GetLastError`]. Panics never cross the boundary: every entry point runs its body
//! inside [`std::panic::catch_unwind`] and reports a caught panic as an error (SPEC-ERR-4,
//! SPEC-ERR-5). This module is on the `unsafe` allowlist (AGENTS.md section 6).
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use lgbm_core::boosting::Gbdt;
use lgbm_core::predictor::{PredictConfig, PredictKind, Predictor};

/// upstream: `API_END()` returns `0` on success.
const C_API_SUCCESS: c_int = 0;
/// upstream: `API_END()` returns `-1` after storing the error.
const C_API_ERROR: c_int = -1;

// upstream: include/LightGBM/c_api.h `C_API_DTYPE_*` and `C_API_PREDICT_*`.
const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_DTYPE_INT32: c_int = 2;
const C_API_DTYPE_INT64: c_int = 3;
const C_API_PREDICT_NORMAL: c_int = 0;
const C_API_PREDICT_RAW_SCORE: c_int = 1;
const C_API_PREDICT_LEAF_INDEX: c_int = 2;
const C_API_PREDICT_CONTRIB: c_int = 3;

/// Map a C API `predict_type` to the core [`PredictKind`].
fn predict_kind(predict_type: c_int) -> Result<PredictKind, String> {
    match predict_type {
        C_API_PREDICT_NORMAL => Ok(PredictKind::Normal),
        C_API_PREDICT_RAW_SCORE => Ok(PredictKind::RawScore),
        C_API_PREDICT_LEAF_INDEX => Ok(PredictKind::LeafIndex),
        C_API_PREDICT_CONTRIB => Ok(PredictKind::Contrib),
        other => Err(format!("unknown predict_type {other}")),
    }
}

/// Build a predictor config for the given predict type and iteration window.
fn predict_config(
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
) -> Result<PredictConfig, String> {
    Ok(PredictConfig {
        kind: predict_kind(predict_type)?,
        start_iteration,
        num_iteration,
        ..PredictConfig::default()
    })
}

thread_local! {
    /// upstream keeps the last error in a thread-local string (`LGBM_GetLastError`).
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Store `msg` as the thread's last error, stripping any interior NUL so it stays a valid C
/// string.
fn set_last_error(msg: &str) {
    let cleaned = msg.replace('\0', " ");
    let stored = CString::new(cleaned).unwrap_or_default();
    LAST_ERROR.with(|cell| *cell.borrow_mut() = stored);
}

/// Run an entry point's body, turning a returned error or a caught panic into `-1` with the
/// message stored for [`LGBM_GetLastError`]. upstream: the `API_BEGIN`/`API_END` wrapper.
fn guard(body: impl FnOnce() -> Result<(), String>) -> c_int {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => C_API_SUCCESS,
        Ok(Err(message)) => {
            set_last_error(&message);
            C_API_ERROR
        }
        Err(_) => {
            set_last_error("unexpected panic in the bankai_lightgbm C API");
            C_API_ERROR
        }
    }
}

/// A booster handle's contents. The opaque `BoosterHandle` upstream exposes is a pointer to
/// this, owned by the caller until [`LGBM_BoosterFree`].
struct CBooster {
    model: Gbdt,
}

/// Borrow a C string as `&str`, erroring like upstream when it is null or not UTF-8.
///
/// # Safety
/// `ptr` must be null or point to a NUL-terminated string valid for the call's duration.
unsafe fn cstr_to_str<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err(format!("{what} must not be null"));
    }
    // SAFETY: `ptr` is non-null and the caller guarantees it is a valid NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(ptr) };
    bytes
        .to_str()
        .map_err(|_| format!("{what} is not valid UTF-8"))
}

/// Box `model` into a handle and publish it plus its iteration count through the out-pointers.
///
/// # Safety
/// `out_num_iterations` and `out_handle` must be valid, writable pointers (or null, which errors).
unsafe fn publish_booster(
    model: Gbdt,
    out_num_iterations: *mut c_int,
    out_handle: *mut *mut c_void,
) -> Result<(), String> {
    if out_handle.is_null() {
        return Err("out handle pointer must not be null".into());
    }
    let num_iterations = c_int::try_from(model.current_iteration())
        .map_err(|_| "model has too many iterations for the C API".to_string())?;
    if !out_num_iterations.is_null() {
        // SAFETY: checked non-null just above; the caller guarantees it is writable.
        unsafe { *out_num_iterations = num_iterations };
    }
    let boxed = Box::new(CBooster { model });
    // SAFETY: `out_handle` is non-null (checked) and writable per the contract.
    unsafe { *out_handle = Box::into_raw(boxed).cast::<c_void>() };
    Ok(())
}

/// Borrow the booster a handle points to.
///
/// # Safety
/// `handle` must be a pointer returned by a `LGBM_BoosterCreate*` call and not yet freed.
unsafe fn booster_ref<'a>(handle: *mut c_void) -> Result<&'a CBooster, String> {
    if handle.is_null() {
        return Err("booster handle must not be null".into());
    }
    // SAFETY: non-null and, per the contract, a live pointer from `publish_booster`.
    Ok(unsafe { &*handle.cast::<CBooster>() })
}

/// upstream: `LGBM_GetLastError`. The returned pointer is valid until the next C API call on
/// the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn LGBM_GetLastError() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// upstream: `LGBM_BoosterCreateFromModelfile`. Loads a model file into a new handle and
/// reports its iteration count.
///
/// # Safety
/// `filename` must be a valid C string; `out_num_iterations` and `out` valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterCreateFromModelfile(
    filename: *const c_char,
    out_num_iterations: *mut c_int,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's own safety contract.
        let path = unsafe { cstr_to_str(filename, "filename") }?;
        let model = Gbdt::from_model_file(Path::new(path)).map_err(|e| e.to_string())?;
        // SAFETY: out-pointers forwarded from this function's safety contract.
        unsafe { publish_booster(model, out_num_iterations, out) }
    })
}

/// upstream: `LGBM_BoosterLoadModelFromString`. Loads a model string into a new handle and
/// reports its iteration count.
///
/// # Safety
/// `model_str` must be a valid C string; `out_num_iterations` and `out` valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterLoadModelFromString(
    model_str: *const c_char,
    out_num_iterations: *mut c_int,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's own safety contract.
        let text = unsafe { cstr_to_str(model_str, "model_str") }?;
        let model = Gbdt::from_model_string(text).map_err(|e| e.to_string())?;
        // SAFETY: out-pointers forwarded from this function's safety contract.
        unsafe { publish_booster(model, out_num_iterations, out) }
    })
}

/// upstream: `LGBM_BoosterFree`. Releases a handle from a `LGBM_BoosterCreate*` call.
///
/// # Safety
/// `handle` must be a live pointer from a `LGBM_BoosterCreate*` call, freed at most once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterFree(handle: *mut c_void) -> c_int {
    guard(|| {
        if handle.is_null() {
            return Err("booster handle must not be null".into());
        }
        // SAFETY: non-null and, per the contract, a live pointer from `publish_booster` that is
        // freed at most once. Reconstructing the `Box` drops it.
        drop(unsafe { Box::from_raw(handle.cast::<CBooster>()) });
        Ok(())
    })
}

/// upstream: `LGBM_BoosterGetNumClasses`.
///
/// # Safety
/// `handle` must be a live booster handle; `out_len` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterGetNumClasses(
    handle: *mut c_void,
    out_len: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() {
            return Err("out_len must not be null".into());
        }
        let value = c_int::try_from(booster.model.num_class())
            .map_err(|_| "num_class does not fit in an int".to_string())?;
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out_len = value };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterGetCurrentIteration`.
///
/// # Safety
/// `handle` must be a live booster handle; `out_iteration` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterGetCurrentIteration(
    handle: *mut c_void,
    out_iteration: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_iteration.is_null() {
            return Err("out_iteration must not be null".into());
        }
        let value = c_int::try_from(booster.model.current_iteration())
            .map_err(|_| "current_iteration does not fit in an int".to_string())?;
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out_iteration = value };
        Ok(())
    })
}

/// Gather an `nrow x ncol` matrix (row- or column-major, element type `T`) into a row-major
/// `f64` buffer, converting each element with `conv`.
fn to_row_major<T: Copy>(
    src: &[T],
    nrow: usize,
    ncol: usize,
    row_major: bool,
    conv: impl Fn(T) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; nrow * ncol];
    for r in 0..nrow {
        for c in 0..ncol {
            let si = if row_major {
                r * ncol + c
            } else {
                c * nrow + r
            };
            out[r * ncol + c] = conv(src[si]);
        }
    }
    out
}

/// upstream: `LGBM_BoosterCalcNumPredict`. The number of `double`s a `num_row`-row prediction of
/// `predict_type` produces (`num_row * num_pred_one_row`).
///
/// # Safety
/// `handle` must be a live booster handle; `out_len` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterCalcNumPredict(
    handle: *mut c_void,
    num_row: c_int,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    out_len: *mut i64,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() {
            return Err("out_len must not be null".into());
        }
        if num_row < 0 {
            return Err("num_row must not be negative".into());
        }
        let cfg = predict_config(predict_type, start_iteration, num_iteration)?;
        let predictor = Predictor::new(&booster.model, &cfg).map_err(|e| e.to_string())?;
        let total = predictor.num_pred_one_row() as i64 * i64::from(num_row);
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out_len = total };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterPredictForMat`. Predict a dense `nrow x ncol` matrix (`float32` or
/// `float64`, row- or column-major) and write `nrow * num_pred_one_row` doubles to `out_result`.
///
/// The prediction `parameter` string is parsed in a later slice; only an empty string (or null)
/// is accepted now.
///
/// # Safety
/// `handle` must be a live booster handle. `data` must point to `nrow * ncol` elements of the
/// type `data_type` names. `out_result` must have room for the length `LGBM_BoosterCalcNumPredict`
/// reports, and `out_len` / `parameter` must be valid pointers (`parameter` null or a C string).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterPredictForMat(
    handle: *mut c_void,
    data: *const c_void,
    data_type: c_int,
    nrow: i32,
    ncol: i32,
    is_row_major: c_int,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    parameter: *const c_char,
    out_len: *mut i64,
    out_result: *mut f64,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() || out_result.is_null() {
            return Err("out_len and out_result must not be null".into());
        }
        if nrow < 0 || ncol < 0 {
            return Err("nrow and ncol must not be negative".into());
        }
        if !parameter.is_null() {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            let param = unsafe { cstr_to_str(parameter, "parameter") }?;
            if !param.trim().is_empty() {
                return Err("non-empty prediction parameter string is not supported yet".into());
            }
        }
        let (nrow, ncol) = (nrow as usize, ncol as usize);
        let n = nrow * ncol;
        if n != 0 && data.is_null() {
            return Err("data must not be null".into());
        }
        let row_major = is_row_major != 0;
        let gathered = match data_type {
            C_API_DTYPE_FLOAT64 => {
                // SAFETY: `data` points to `n` f64 values (checked non-null when `n != 0`).
                let src = unsafe { std::slice::from_raw_parts(data.cast::<f64>(), n) };
                to_row_major(src, nrow, ncol, row_major, |x| x)
            }
            C_API_DTYPE_FLOAT32 => {
                // SAFETY: `data` points to `n` f32 values (checked non-null when `n != 0`).
                let src = unsafe { std::slice::from_raw_parts(data.cast::<f32>(), n) };
                to_row_major(src, nrow, ncol, row_major, f64::from)
            }
            other => {
                return Err(format!(
                    "unsupported data_type {other} (only float32=0 and float64=1)"
                ));
            }
        };
        let cfg = predict_config(predict_type, start_iteration, num_iteration)?;
        let predictor = Predictor::new(&booster.model, &cfg).map_err(|e| e.to_string())?;
        let out = predictor
            .predict_dense(&gathered, nrow, ncol)
            .map_err(|e| e.to_string())?;
        // SAFETY: the caller sized `out_result` by `LGBM_BoosterCalcNumPredict`, which returns
        // this same length for the same shape and predict type.
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_result, out.len()) };
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = out.len() as i64 };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterPredictForCSR`. Predict a CSR matrix (`nindptr` row pointers of
/// `indptr_type`, `nelem` `int32` column indices and `data_type` values) and write
/// `(nindptr - 1) * num_pred_one_row` doubles to `out_result`.
///
/// The prediction `parameter` string is parsed in a later slice; only empty/null is accepted.
///
/// # Safety
/// `handle` must be a live booster handle. `indptr` must point to `nindptr` elements of
/// `indptr_type`; `indices` and `data` to `nelem` elements each (`int32` and `data_type`).
/// `out_result` must hold the length `LGBM_BoosterCalcNumPredict` reports for `nindptr - 1` rows,
/// and `out_len` / `parameter` must be valid (`parameter` null or a C string).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterPredictForCSR(
    handle: *mut c_void,
    indptr: *const c_void,
    indptr_type: c_int,
    indices: *const i32,
    data: *const c_void,
    data_type: c_int,
    nindptr: i64,
    nelem: i64,
    num_col: i64,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    parameter: *const c_char,
    out_len: *mut i64,
    out_result: *mut f64,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() || out_result.is_null() {
            return Err("out_len and out_result must not be null".into());
        }
        if nindptr < 1 {
            return Err("nindptr must be at least 1".into());
        }
        if nelem < 0 || num_col < 0 {
            return Err("nelem and num_col must not be negative".into());
        }
        if !parameter.is_null() {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            let param = unsafe { cstr_to_str(parameter, "parameter") }?;
            if !param.trim().is_empty() {
                return Err("non-empty prediction parameter string is not supported yet".into());
            }
        }
        let (nindptr, nelem) = (nindptr as usize, nelem as usize);
        if indptr.is_null() {
            return Err("indptr must not be null".into());
        }
        if nelem != 0 && (indices.is_null() || data.is_null()) {
            return Err("indices and data must not be null when nelem != 0".into());
        }
        let indptr_vec: Vec<i64> = match indptr_type {
            C_API_DTYPE_INT32 => {
                // SAFETY: `indptr` points to `nindptr` i32 values (checked non-null).
                unsafe { std::slice::from_raw_parts(indptr.cast::<i32>(), nindptr) }
                    .iter()
                    .map(|&x| i64::from(x))
                    .collect()
            }
            C_API_DTYPE_INT64 => {
                // SAFETY: `indptr` points to `nindptr` i64 values (checked non-null).
                unsafe { std::slice::from_raw_parts(indptr.cast::<i64>(), nindptr) }.to_vec()
            }
            other => {
                return Err(format!(
                    "unsupported indptr_type {other} (only int32=2 and int64=3)"
                ));
            }
        };
        // SAFETY: `indices` points to `nelem` i32 values (checked non-null when `nelem != 0`).
        let idx: &[i32] = if nelem == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(indices, nelem) }
        };
        let values: Vec<f64> = match data_type {
            C_API_DTYPE_FLOAT64 => {
                // SAFETY: `data` points to `nelem` f64 values (checked non-null when `nelem != 0`).
                unsafe { std::slice::from_raw_parts(data.cast::<f64>(), nelem) }.to_vec()
            }
            C_API_DTYPE_FLOAT32 => {
                // SAFETY: `data` points to `nelem` f32 values (checked non-null when `nelem != 0`).
                unsafe { std::slice::from_raw_parts(data.cast::<f32>(), nelem) }
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect()
            }
            other => {
                return Err(format!(
                    "unsupported data_type {other} (only float32=0 and float64=1)"
                ));
            }
        };
        let cfg = predict_config(predict_type, start_iteration, num_iteration)?;
        let predictor = Predictor::new(&booster.model, &cfg).map_err(|e| e.to_string())?;
        let out = predictor
            .predict_csr(&indptr_vec, idx, &values)
            .map_err(|e| e.to_string())?;
        // SAFETY: the caller sized `out_result` by `LGBM_BoosterCalcNumPredict` for these rows.
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_result, out.len()) };
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = out.len() as i64 };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterPredictForCSC`. Predict a CSC matrix (`ncol_ptr` column pointers of
/// `col_ptr_type`, `nelem` `int32` row indices and `data_type` values, over `num_row` rows) and
/// write `num_row * num_pred_one_row` doubles to `out_result`.
///
/// The prediction `parameter` string is parsed in a later slice; only empty/null is accepted.
///
/// # Safety
/// `handle` must be a live booster handle. `col_ptr` must point to `ncol_ptr` elements of
/// `col_ptr_type`; `indices` and `data` to `nelem` elements each (`int32` and `data_type`).
/// `out_result` must hold the length `LGBM_BoosterCalcNumPredict` reports for `num_row` rows, and
/// `out_len` / `parameter` must be valid (`parameter` null or a C string).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterPredictForCSC(
    handle: *mut c_void,
    col_ptr: *const c_void,
    col_ptr_type: c_int,
    indices: *const i32,
    data: *const c_void,
    data_type: c_int,
    ncol_ptr: i64,
    nelem: i64,
    num_row: i64,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    parameter: *const c_char,
    out_len: *mut i64,
    out_result: *mut f64,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() || out_result.is_null() {
            return Err("out_len and out_result must not be null".into());
        }
        if ncol_ptr < 1 {
            return Err("ncol_ptr must be at least 1".into());
        }
        if nelem < 0 || num_row < 0 {
            return Err("nelem and num_row must not be negative".into());
        }
        if !parameter.is_null() {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            let param = unsafe { cstr_to_str(parameter, "parameter") }?;
            if !param.trim().is_empty() {
                return Err("non-empty prediction parameter string is not supported yet".into());
            }
        }
        let (ncol_ptr, nelem, nrow) = (ncol_ptr as usize, nelem as usize, num_row as usize);
        if col_ptr.is_null() {
            return Err("col_ptr must not be null".into());
        }
        if nelem != 0 && (indices.is_null() || data.is_null()) {
            return Err("indices and data must not be null when nelem != 0".into());
        }
        let col_ptr_vec: Vec<i64> = match col_ptr_type {
            C_API_DTYPE_INT32 => {
                // SAFETY: `col_ptr` points to `ncol_ptr` i32 values (checked non-null).
                unsafe { std::slice::from_raw_parts(col_ptr.cast::<i32>(), ncol_ptr) }
                    .iter()
                    .map(|&x| i64::from(x))
                    .collect()
            }
            C_API_DTYPE_INT64 => {
                // SAFETY: `col_ptr` points to `ncol_ptr` i64 values (checked non-null).
                unsafe { std::slice::from_raw_parts(col_ptr.cast::<i64>(), ncol_ptr) }.to_vec()
            }
            other => {
                return Err(format!(
                    "unsupported col_ptr_type {other} (only int32=2 and int64=3)"
                ));
            }
        };
        // SAFETY: `indices` points to `nelem` i32 values (checked non-null when `nelem != 0`).
        let idx: &[i32] = if nelem == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(indices, nelem) }
        };
        let values: Vec<f64> = match data_type {
            C_API_DTYPE_FLOAT64 => {
                // SAFETY: `data` points to `nelem` f64 values (checked non-null when `nelem != 0`).
                unsafe { std::slice::from_raw_parts(data.cast::<f64>(), nelem) }.to_vec()
            }
            C_API_DTYPE_FLOAT32 => {
                // SAFETY: `data` points to `nelem` f32 values (checked non-null when `nelem != 0`).
                unsafe { std::slice::from_raw_parts(data.cast::<f32>(), nelem) }
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect()
            }
            other => {
                return Err(format!(
                    "unsupported data_type {other} (only float32=0 and float64=1)"
                ));
            }
        };
        let cfg = predict_config(predict_type, start_iteration, num_iteration)?;
        let predictor = Predictor::new(&booster.model, &cfg).map_err(|e| e.to_string())?;
        let out = predictor
            .predict_csc(&col_ptr_vec, idx, &values, nrow)
            .map_err(|e| e.to_string())?;
        // SAFETY: the caller sized `out_result` by `LGBM_BoosterCalcNumPredict` for these rows.
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_result, out.len()) };
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = out.len() as i64 };
        Ok(())
    })
}
