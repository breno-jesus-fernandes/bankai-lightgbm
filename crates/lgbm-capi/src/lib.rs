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
use std::sync::Arc;

use lgbm_core::boosting::{Gbdt, GbdtTrainer};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::sparse::{CsrMatrix, csc_to_csr};
use lgbm_core::io::text_loader::{TextLoadOptions, load_text};
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
/// this, owned by the caller until [`LGBM_BoosterFree`]. A handle is either a model loaded for
/// prediction or a trainer built from a dataset; both predict (a trainer predicts from its
/// current trees, as upstream does).
enum CBooster {
    /// Loaded from a model file or string (`LGBM_BoosterCreateFromModelfile` / `...FromString`).
    Model(Box<Gbdt>),
    /// Training from a dataset (`LGBM_BoosterCreate`), advanced by `LGBM_BoosterUpdateOneIter`.
    Trainer(Box<GbdtTrainer>),
}

impl CBooster {
    /// Run `f` with a prediction/metadata view of the model. A trainer snapshots its current
    /// trees (upstream predicts from the in-progress model during training).
    fn with_model<R>(&self, f: impl FnOnce(&Gbdt) -> R) -> R {
        match self {
            CBooster::Model(model) => f(model.as_ref()),
            CBooster::Trainer(trainer) => f(&trainer.snapshot()),
        }
    }
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
    let boxed = Box::new(CBooster::Model(Box::new(model)));
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
    // SAFETY: non-null and, per the contract, a live pointer from a booster-create call.
    Ok(unsafe { &*handle.cast::<CBooster>() })
}

/// Borrow the booster a handle points to mutably (for `LGBM_BoosterUpdateOneIter`).
///
/// # Safety
/// `handle` must be a pointer returned by a `LGBM_BoosterCreate*` call, not yet freed, with no
/// other reference to the same booster live for the duration of the borrow.
unsafe fn booster_mut<'a>(handle: *mut c_void) -> Result<&'a mut CBooster, String> {
    if handle.is_null() {
        return Err("booster handle must not be null".into());
    }
    // SAFETY: non-null, a live pointer per the contract; the caller guarantees exclusivity.
    Ok(unsafe { &mut *handle.cast::<CBooster>() })
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
        let value = c_int::try_from(booster.with_model(Gbdt::num_class))
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
        let value = c_int::try_from(booster.with_model(Gbdt::current_iteration))
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
        let per_row = booster
            .with_model(|model| Predictor::new(model, &cfg).map(|p| p.num_pred_one_row()))
            .map_err(|e| e.to_string())?;
        let total = per_row as i64 * i64::from(num_row);
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
        let out = booster
            .with_model(|model| {
                let predictor = Predictor::new(model, &cfg)?;
                predictor.predict_dense(&gathered, nrow, ncol)
            })
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
        let out = booster
            .with_model(|model| {
                let predictor = Predictor::new(model, &cfg)?;
                predictor.predict_csr(&indptr_vec, idx, &values)
            })
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
        let out = booster
            .with_model(|model| {
                let predictor = Predictor::new(model, &cfg)?;
                predictor.predict_csc(&col_ptr_vec, idx, &values, nrow)
            })
            .map_err(|e| e.to_string())?;
        // SAFETY: the caller sized `out_result` by `LGBM_BoosterCalcNumPredict` for these rows.
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), out_result, out.len()) };
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = out.len() as i64 };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterPredictForMatSingleRow`. Predict a single dense row of `ncol` values
/// (`float32`/`float64`) and write `num_pred_one_row` doubles to `out_result`. Equivalent to
/// [`LGBM_BoosterPredictForMat`] with one row.
///
/// # Safety
/// `handle` must be a live booster handle; `data` must point to `ncol` elements of `data_type`;
/// `out_result` must hold `num_pred_one_row` doubles; `out_len` / `parameter` must be valid.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterPredictForMatSingleRow(
    handle: *mut c_void,
    data: *const c_void,
    data_type: c_int,
    ncol: c_int,
    is_row_major: c_int,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    parameter: *const c_char,
    out_len: *mut i64,
    out_result: *mut f64,
) -> c_int {
    // A single row is one row of the dense matrix path; `is_row_major` does not matter for it.
    // SAFETY: the contract matches `LGBM_BoosterPredictForMat` with `nrow == 1`.
    unsafe {
        LGBM_BoosterPredictForMat(
            handle,
            data,
            data_type,
            1,
            ncol,
            is_row_major,
            predict_type,
            start_iteration,
            num_iteration,
            parameter,
            out_len,
            out_result,
        )
    }
}

/// upstream: `LGBM_BoosterGetNumFeature`. The number of features the model expects.
///
/// # Safety
/// `handle` must be a live booster handle; `out_len` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterGetNumFeature(
    handle: *mut c_void,
    out_len: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() {
            return Err("out_len must not be null".into());
        }
        let value = c_int::try_from(booster.with_model(Gbdt::num_features))
            .map_err(|_| "num_features does not fit in an int".to_string())?;
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out_len = value };
        Ok(())
    })
}

/// A dataset handle's contents (upstream's opaque `DatasetHandle`), owned until
/// [`LGBM_DatasetFree`].
struct CDataset {
    dataset: Dataset,
}

/// Borrow the dataset a handle points to.
///
/// # Safety
/// `handle` must be a pointer returned by `LGBM_DatasetCreateFromMat` and not yet freed.
unsafe fn dataset_ref<'a>(handle: *mut c_void) -> Result<&'a CDataset, String> {
    if handle.is_null() {
        return Err("dataset handle must not be null".into());
    }
    // SAFETY: a live `CDataset` pointer per the contract.
    Ok(unsafe { &*handle.cast::<CDataset>() })
}

/// Borrow the dataset a handle points to mutably (for `LGBM_DatasetSetField`).
///
/// # Safety
/// `handle` must be a pointer returned by a dataset-create call, not yet freed, with no other
/// reference to the same dataset live for the duration of the borrow.
unsafe fn dataset_mut<'a>(handle: *mut c_void) -> Result<&'a mut CDataset, String> {
    if handle.is_null() {
        return Err("dataset handle must not be null".into());
    }
    // SAFETY: a live `CDataset` pointer per the contract; the caller guarantees exclusivity.
    Ok(unsafe { &mut *handle.cast::<CDataset>() })
}

/// Parse `categorical_feature` (a comma-separated list of column indices) into indices. The
/// `name:` form upstream also accepts is deferred.
fn parse_categorical(s: &str) -> Result<Vec<i32>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    if s.starts_with("name:") {
        return Err("categorical_feature by name is not supported in the C API yet".into());
    }
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<i32>()
                .map_err(|_| format!("invalid categorical_feature index '{}'", t.trim()))
        })
        .collect()
}

/// upstream: `LGBM_DatasetCreateFromMat`. Bin a dense `nrow x ncol` matrix (`float32`/`float64`,
/// row- or column-major) into a new dataset, using the `parameters` config string. The label and
/// other fields are set separately; `reference` (sharing another dataset's bin mappers) is
/// deferred, so it must be null.
///
/// # Safety
/// `data` must point to `nrow * ncol` elements of `data_type`; `parameters` is null or a C
/// string; `reference` must be null; `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_DatasetCreateFromMat(
    data: *const c_void,
    data_type: c_int,
    nrow: i32,
    ncol: i32,
    is_row_major: c_int,
    parameters: *const c_char,
    reference: *const c_void,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err("out dataset handle pointer must not be null".into());
        }
        if !reference.is_null() {
            return Err("dataset reference (shared bin mappers) is not supported yet".into());
        }
        if nrow < 0 || ncol < 0 {
            return Err("nrow and ncol must not be negative".into());
        }
        let (nrow, ncol) = (nrow as usize, ncol as usize);
        let n = nrow * ncol;
        if n != 0 && data.is_null() {
            return Err("data must not be null".into());
        }
        let params = if parameters.is_null() {
            ""
        } else {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            unsafe { cstr_to_str(parameters, "parameters") }?
        };
        let cfg = Config::from_str_params(params).map_err(|e| e.to_string())?;
        let categorical = parse_categorical(&cfg.categorical_feature)?;
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
        // The label is set separately in the C API; a zero placeholder keeps the length check
        // happy and does not affect the bin mappers.
        let label = vec![0.0f64; nrow];
        let dataset = Dataset::from_dense(&gathered, nrow, ncol, &label, &cfg, &categorical)
            .map_err(|e| e.to_string())?;
        let boxed = Box::new(CDataset { dataset });
        // SAFETY: `out` is non-null (checked) and writable per the contract.
        unsafe { *out = Box::into_raw(boxed).cast::<c_void>() };
        Ok(())
    })
}

/// upstream: `LGBM_DatasetCreateFromCSR`. Bin a CSR matrix (`nindptr` row pointers of
/// `indptr_type`, `nelem` `int32` column indices and `data_type` values, over `num_col` columns)
/// into a new dataset, using the `parameters` config string. The sparse rows are binned without
/// densifying the matrix (core's `Dataset::from_csr`). The label and other fields are set
/// separately; `reference` (sharing another dataset's bin mappers) is deferred, so it must be null.
///
/// # Safety
/// `indptr` must point to `nindptr` elements of `indptr_type`; `indices` and `data` to `nelem`
/// elements each (`int32` and `data_type`); `parameters` is null or a C string; `reference` must
/// be null; `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_DatasetCreateFromCSR(
    indptr: *const c_void,
    indptr_type: c_int,
    indices: *const i32,
    data: *const c_void,
    data_type: c_int,
    nindptr: i64,
    nelem: i64,
    num_col: i64,
    parameters: *const c_char,
    reference: *const c_void,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err("out dataset handle pointer must not be null".into());
        }
        if !reference.is_null() {
            return Err("dataset reference (shared bin mappers) is not supported yet".into());
        }
        if nindptr < 1 {
            return Err("nindptr must be at least 1".into());
        }
        if nelem < 0 || num_col < 0 {
            return Err("nelem and num_col must not be negative".into());
        }
        let params = if parameters.is_null() {
            ""
        } else {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            unsafe { cstr_to_str(parameters, "parameters") }?
        };
        let cfg = Config::from_str_params(params).map_err(|e| e.to_string())?;
        let categorical = parse_categorical(&cfg.categorical_feature)?;
        let (nindptr, nelem, ncol) = (nindptr as usize, nelem as usize, num_col as usize);
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
        let m = CsrMatrix::new(&indptr_vec, idx, &values, ncol).map_err(|e| e.to_string())?;
        // The label is set separately in the C API; a zero placeholder keeps the length check
        // happy and does not affect the bin mappers.
        let label = vec![0.0f64; m.nrow()];
        let dataset =
            Dataset::from_csr(&m, &label, &cfg, &categorical).map_err(|e| e.to_string())?;
        let boxed = Box::new(CDataset { dataset });
        // SAFETY: `out` is non-null (checked) and writable per the contract.
        unsafe { *out = Box::into_raw(boxed).cast::<c_void>() };
        Ok(())
    })
}

/// upstream: `LGBM_DatasetCreateFromCSC`. Bin a CSC matrix (`ncol_ptr` column pointers of
/// `col_ptr_type`, `nelem` `int32` row indices and `data_type` values, over `num_row` rows) into
/// a new dataset. The columns are gathered into CSR rows in upstream's column-walk order
/// (`csc_to_csr`) and binned without densifying (core's `Dataset::from_csr`), using the
/// `parameters` config string. The label and other fields are set separately; `reference`
/// (sharing another dataset's bin mappers) is deferred, so it must be null.
///
/// # Safety
/// `col_ptr` must point to `ncol_ptr` elements of `col_ptr_type`; `indices` and `data` to `nelem`
/// elements each (`int32` and `data_type`); `parameters` is null or a C string; `reference` must
/// be null; `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_DatasetCreateFromCSC(
    col_ptr: *const c_void,
    col_ptr_type: c_int,
    indices: *const i32,
    data: *const c_void,
    data_type: c_int,
    ncol_ptr: i64,
    nelem: i64,
    num_row: i64,
    parameters: *const c_char,
    reference: *const c_void,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err("out dataset handle pointer must not be null".into());
        }
        if !reference.is_null() {
            return Err("dataset reference (shared bin mappers) is not supported yet".into());
        }
        if ncol_ptr < 1 {
            return Err("ncol_ptr must be at least 1".into());
        }
        if nelem < 0 || num_row < 0 {
            return Err("nelem and num_row must not be negative".into());
        }
        let params = if parameters.is_null() {
            ""
        } else {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            unsafe { cstr_to_str(parameters, "parameters") }?
        };
        let cfg = Config::from_str_params(params).map_err(|e| e.to_string())?;
        let categorical = parse_categorical(&cfg.categorical_feature)?;
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
        // A CSC matrix is the CSR form of its transpose: gather the columns into CSR rows, keeping
        // upstream's column-walk order, then bin with the CSR path. `num_row` fixes the row count,
        // and the column count is `ncol_ptr - 1`.
        let ncol = ncol_ptr - 1;
        let (indptr, csr_idx, csr_vals) =
            csc_to_csr(&col_ptr_vec, idx, &values, nrow).map_err(|e| e.to_string())?;
        let m = CsrMatrix::new(&indptr, &csr_idx, &csr_vals, ncol).map_err(|e| e.to_string())?;
        // The label is set separately in the C API; a zero placeholder keeps the length check
        // happy and does not affect the bin mappers.
        let label = vec![0.0f64; nrow];
        let dataset =
            Dataset::from_csr(&m, &label, &cfg, &categorical).map_err(|e| e.to_string())?;
        let boxed = Box::new(CDataset { dataset });
        // SAFETY: `out` is non-null (checked) and writable per the contract.
        unsafe { *out = Box::into_raw(boxed).cast::<c_void>() };
        Ok(())
    })
}

/// upstream: `LGBM_DatasetFree`. Releases a handle from a dataset-create call
/// (`LGBM_DatasetCreateFromMat`, `LGBM_DatasetCreateFromCSR` or `LGBM_DatasetCreateFromCSC`).
///
/// # Safety
/// `handle` must be a live pointer from a dataset-create call, freed at most once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_DatasetFree(handle: *mut c_void) -> c_int {
    guard(|| {
        if handle.is_null() {
            return Err("dataset handle must not be null".into());
        }
        // SAFETY: a live `CDataset` pointer, freed at most once per the contract.
        drop(unsafe { Box::from_raw(handle.cast::<CDataset>()) });
        Ok(())
    })
}

/// upstream: `LGBM_DatasetGetNumData`. The number of rows in the dataset.
///
/// # Safety
/// `handle` must be a live dataset handle; `out` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_DatasetGetNumData(handle: *mut c_void, out: *mut c_int) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let ds = unsafe { dataset_ref(handle) }?;
        if out.is_null() {
            return Err("out must not be null".into());
        }
        let value = c_int::try_from(ds.dataset.num_data())
            .map_err(|_| "num_data does not fit in an int".to_string())?;
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out = value };
        Ok(())
    })
}

/// upstream: `LGBM_DatasetGetNumFeature`. The number of features in the dataset.
///
/// # Safety
/// `handle` must be a live dataset handle; `out` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_DatasetGetNumFeature(handle: *mut c_void, out: *mut c_int) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let ds = unsafe { dataset_ref(handle) }?;
        if out.is_null() {
            return Err("out must not be null".into());
        }
        let value = c_int::try_from(ds.dataset.layout.num_total_features)
            .map_err(|_| "num_feature does not fit in an int".to_string())?;
        // SAFETY: checked non-null just above; writable per the contract.
        unsafe { *out = value };
        Ok(())
    })
}

/// upstream: `LGBM_DatasetSetField`. Set a metadata field from a C buffer of `num_element`
/// elements. The field name selects the array and the expected dtype:
/// `label`/`target` and `weight`/`weights` take `float32`; `init_score` takes `float64`;
/// `group`/`query` (per-query sizes, turned into boundaries) and `position` take `int32`.
/// The dtype must match the field, as upstream.
///
/// # Safety
/// `handle` must be a live dataset handle; `field_name` a valid C string; `field_data` must point
/// to `num_element` elements of `field_type`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_DatasetSetField(
    handle: *mut c_void,
    field_name: *const c_char,
    field_data: *const c_void,
    num_element: c_int,
    field_type: c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let ds = unsafe { dataset_mut(handle) }?;
        // SAFETY: a valid C string per the contract (null is rejected inside).
        let name = unsafe { cstr_to_str(field_name, "field_name") }?.trim();
        if num_element < 0 {
            return Err("num_element must not be negative".into());
        }
        let n = num_element as usize;
        if n != 0 && field_data.is_null() {
            return Err("field_data must not be null when num_element != 0".into());
        }
        match (field_type, name) {
            (C_API_DTYPE_FLOAT32, "label" | "target") => {
                // SAFETY: `field_data` points to `n` f32 values (checked non-null when `n != 0`).
                let src = unsafe { raw_slice::<f32>(field_data, n) };
                let label: Vec<f64> = src.iter().map(|&x| f64::from(x)).collect();
                ds.dataset.set_label(&label).map_err(|e| e.to_string())
            }
            (C_API_DTYPE_FLOAT32, "weight" | "weights") => {
                // SAFETY: `field_data` points to `n` f32 values (checked non-null when `n != 0`).
                let src = unsafe { raw_slice::<f32>(field_data, n) };
                ds.dataset.set_weights(src).map_err(|e| e.to_string())
            }
            (C_API_DTYPE_FLOAT64, "init_score") => {
                // SAFETY: `field_data` points to `n` f64 values (checked non-null when `n != 0`).
                let src = unsafe { raw_slice::<f64>(field_data, n) };
                ds.dataset.set_init_score(src).map_err(|e| e.to_string())
            }
            (C_API_DTYPE_INT32, "group" | "query") => {
                // SAFETY: `field_data` points to `n` i32 values (checked non-null when `n != 0`).
                let src = unsafe { raw_slice::<i32>(field_data, n) };
                ds.dataset.set_group(src).map_err(|e| e.to_string())
            }
            (C_API_DTYPE_INT32, "position") => {
                // SAFETY: `field_data` points to `n` i32 values (checked non-null when `n != 0`).
                let src = unsafe { raw_slice::<i32>(field_data, n) };
                ds.dataset.set_positions(src).map_err(|e| e.to_string())
            }
            (_, other) => Err(format!(
                "unknown field '{other}' for dtype {field_type} (supported: label/weight as \
                 float32, init_score as float64, group/query/position as int32)"
            )),
        }
    })
}

/// upstream: `LGBM_DatasetGetField`. Return a borrowed pointer to a metadata field, its length and
/// its dtype. The pointer stays valid until the field is set again or the dataset is freed. An
/// unset (empty) field yields a null pointer and length 0, as upstream. Field names and dtypes
/// mirror [`LGBM_DatasetSetField`]; `group`/`query` returns the query boundaries
/// (`num_queries + 1` values), not the per-query sizes.
///
/// # Safety
/// `handle` must be a live dataset handle; `field_name` a valid C string; `out_len`, `out_ptr` and
/// `out_type` must be valid writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_DatasetGetField(
    handle: *mut c_void,
    field_name: *const c_char,
    out_len: *mut c_int,
    out_ptr: *mut *const c_void,
    out_type: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let ds = unsafe { dataset_ref(handle) }?;
        // SAFETY: a valid C string per the contract (null is rejected inside).
        let name = unsafe { cstr_to_str(field_name, "field_name") }?.trim();
        if out_len.is_null() || out_ptr.is_null() || out_type.is_null() {
            return Err("out_len, out_ptr and out_type must not be null".into());
        }
        let md = &ds.dataset.metadata;
        // Each arm names the field's dtype and the raw byte pointer + element count.
        let (ty, ptr, len): (c_int, *const c_void, usize) = match name {
            "label" | "target" => (
                C_API_DTYPE_FLOAT32,
                md.label.as_ptr().cast::<c_void>(),
                md.label.len(),
            ),
            "weight" | "weights" => (
                C_API_DTYPE_FLOAT32,
                md.weights.as_ptr().cast::<c_void>(),
                md.weights.len(),
            ),
            "init_score" => (
                C_API_DTYPE_FLOAT64,
                md.init_score.as_ptr().cast::<c_void>(),
                md.init_score.len(),
            ),
            "group" | "query" => (
                C_API_DTYPE_INT32,
                md.query_boundaries.as_ptr().cast::<c_void>(),
                md.query_boundaries.len(),
            ),
            "position" => (
                C_API_DTYPE_INT32,
                md.positions.as_ptr().cast::<c_void>(),
                md.positions.len(),
            ),
            other => return Err(format!("unknown field '{other}'")),
        };
        // An empty field reports a null pointer and length 0 (upstream normalizes this way), since
        // an empty Vec's pointer is dangling.
        let (ptr, len) = if len == 0 {
            (std::ptr::null::<c_void>(), 0)
        } else {
            let len = c_int::try_from(len)
                .map_err(|_| "field length does not fit in an int".to_string())?;
            (ptr, len)
        };
        // SAFETY: all three checked non-null above; writable per the contract.
        unsafe {
            *out_type = ty;
            *out_ptr = ptr;
            *out_len = len;
        }
        Ok(())
    })
}

/// Read `n` elements of type `T` from a C buffer as a slice (empty when `n == 0`).
///
/// # Safety
/// `ptr` must point to `n` readable `T` values, or `n` must be 0.
unsafe fn raw_slice<'a, T>(ptr: *const c_void, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        // SAFETY: `ptr` points to `n` readable `T` per the contract.
        unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), n) }
    }
}

/// upstream: `LGBM_BoosterCreate`. Build a training booster from a dataset handle and a config
/// `parameters` string. The dataset's binned data, labels and other fields are copied into the
/// booster, so the dataset handle may be freed afterwards and later field changes do not affect
/// it (as upstream builds the booster from the dataset at create time). Advance training with
/// [`LGBM_BoosterUpdateOneIter`]; predict with the same handle.
///
/// # Safety
/// `train_data` must be a live dataset handle; `parameters` is null or a C string; `out` must be
/// a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterCreate(
    train_data: *const c_void,
    parameters: *const c_char,
    out: *mut *mut c_void,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err("out booster handle pointer must not be null".into());
        }
        // SAFETY: a live dataset handle per the contract (const handle, borrowed read-only).
        let ds = unsafe { dataset_ref(train_data.cast_mut()) }?;
        let params = if parameters.is_null() {
            ""
        } else {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            unsafe { cstr_to_str(parameters, "parameters") }?
        };
        let cfg = Config::from_str_params(params).map_err(|e| e.to_string())?;
        let trainer = GbdtTrainer::new(Arc::new(cfg), Arc::new(ds.dataset.clone()))
            .map_err(|e| e.to_string())?;
        let boxed = Box::new(CBooster::Trainer(Box::new(trainer)));
        // SAFETY: `out` is non-null (checked) and writable per the contract.
        unsafe { *out = Box::into_raw(boxed).cast::<c_void>() };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterUpdateOneIter`. Run one boosting iteration on a training booster.
/// `produced_empty_tree` (when non-null) is set to 1 when the iteration added no splits, which
/// usually means training is finished (core's `GbdtTrainer::update_one_iter`). A booster loaded
/// from a model has no training state and is rejected.
///
/// # Safety
/// `handle` must be a live training booster handle; `produced_empty_tree` is null or a valid
/// writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterUpdateOneIter(
    handle: *mut c_void,
    produced_empty_tree: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_mut(handle) }?;
        match booster {
            CBooster::Trainer(trainer) => {
                let empty = trainer.update_one_iter();
                if !produced_empty_tree.is_null() {
                    // SAFETY: non-null per the branch; writable per the contract.
                    unsafe { *produced_empty_tree = c_int::from(empty) };
                }
                Ok(())
            }
            CBooster::Model(_) => {
                Err("booster was loaded from a model and cannot be trained".into())
            }
        }
    })
}

/// upstream: `LGBM_BoosterRollbackOneIter`. Drop the last boosting iteration's trees and their
/// contribution to the training scores (core's `GbdtTrainer::rollback_one_iter`). A booster loaded
/// from a model has no training state and is rejected.
///
/// # Safety
/// `handle` must be a live training booster handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterRollbackOneIter(handle: *mut c_void) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_mut(handle) }?;
        match booster {
            CBooster::Trainer(trainer) => trainer.rollback_one_iter().map_err(|e| e.to_string()),
            CBooster::Model(_) => {
                Err("booster was loaded from a model and cannot be trained".into())
            }
        }
    })
}

/// upstream: `LGBM_BoosterPredictForFile`. Load a text data file, predict every row, and write the
/// predictions to `result_filename` — one row per line, the row's values tab-separated, each in its
/// shortest round-trippable form. The file is parsed by `lgbm-core`'s loader with the model's
/// feature count, so a leading label column is detected from the column count. `predict_type`,
/// `start_iteration` and `num_iteration` match [`LGBM_BoosterPredictForMat`]. The prediction
/// `parameter` string is not supported yet and must be empty or null.
///
/// # Safety
/// `handle` must be a live booster handle; `data_filename` and `result_filename` valid C strings;
/// `parameter` null or a valid C string.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterPredictForFile(
    handle: *mut c_void,
    data_filename: *const c_char,
    data_has_header: c_int,
    predict_type: c_int,
    start_iteration: c_int,
    num_iteration: c_int,
    parameter: *const c_char,
    result_filename: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        // SAFETY: forwarded from this function's safety contract.
        let data_path = unsafe { cstr_to_str(data_filename, "data_filename") }?;
        // SAFETY: forwarded from this function's safety contract.
        let result_path = unsafe { cstr_to_str(result_filename, "result_filename") }?;
        if !parameter.is_null() {
            // SAFETY: non-null per the branch; a valid C string per the contract.
            let param = unsafe { cstr_to_str(parameter, "parameter") }?;
            if !param.trim().is_empty() {
                return Err("non-empty prediction parameter string is not supported yet".into());
            }
        }
        let cfg = predict_config(predict_type, start_iteration, num_iteration)?;
        let content = std::fs::read_to_string(Path::new(data_path)).map_err(|e| e.to_string())?;
        let body = booster.with_model(|model| -> Result<String, String> {
            let opts = TextLoadOptions {
                header: data_has_header != 0,
                label_column: "",
                weight_column: "",
                group_column: "",
                ignore_column: "",
                precise_float_parser: false,
                num_features: model.num_features(),
            };
            let loaded = load_text(&content, &opts).map_err(|e| e.to_string())?;
            let predictor = Predictor::new(model, &cfg).map_err(|e| e.to_string())?;
            let preds = predictor
                .predict_dense(&loaded.features, loaded.num_data, loaded.num_features)
                .map_err(|e| e.to_string())?;
            let k = predictor.num_pred_one_row();
            if k == 0 {
                // No value per row (e.g. leaf index with no iterations left): an empty line each.
                return Ok("\n".repeat(loaded.num_data));
            }
            let mut out = String::with_capacity(preds.len() * 8);
            for chunk in preds.chunks(k) {
                let line: Vec<String> = chunk.iter().map(f64::to_string).collect();
                out.push_str(&line.join("\t"));
                out.push('\n');
            }
            Ok(out)
        })?;
        std::fs::write(Path::new(result_path), body).map_err(|e| e.to_string())?;
        Ok(())
    })
}

/// Names of the training metrics a booster reports, in order. A booster loaded from a model has no
/// training data and so reports none (upstream: `training_metrics_` is empty without `train_data_`).
fn booster_eval_names(booster: &CBooster) -> Result<Vec<String>, String> {
    match booster {
        CBooster::Trainer(trainer) => trainer.eval_names().map_err(|e| e.to_string()),
        CBooster::Model(_) => Ok(Vec::new()),
    }
}

/// upstream: `LGBM_BoosterGetEvalCounts`. The number of metric values [`LGBM_BoosterGetEval`]
/// reports for each dataset (the configured `metric` list, with multi-valued metrics expanded).
///
/// # Safety
/// `handle` must be a live booster handle and `out_len` a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterGetEvalCounts(
    handle: *mut c_void,
    out_len: *mut c_int,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() {
            return Err("out_len must not be null".into());
        }
        let count =
            c_int::try_from(booster_eval_names(booster)?.len()).map_err(|_| "too many metrics")?;
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = count };
        Ok(())
    })
}

/// upstream: `LGBM_BoosterGetEvalNames`. Write the training metric names into the caller's buffers.
/// `out_len` is always set to the number of names and `out_buffer_len` to the length of the longest
/// name including its NUL terminator; the names themselves are written only for the first `len`
/// buffers, each truncated to `buffer_len` bytes (NUL-terminated). Call once with `len = 0` to size
/// the buffers, then again with them allocated.
///
/// # Safety
/// `handle` must be a live booster handle; `out_len` and `out_buffer_len` valid writable pointers;
/// when `len > 0`, `out_strs` must point to `len` writable `char*` slots, each with `buffer_len`
/// writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterGetEvalNames(
    handle: *mut c_void,
    len: c_int,
    out_len: *mut c_int,
    buffer_len: usize,
    out_buffer_len: *mut usize,
    out_strs: *mut *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() || out_buffer_len.is_null() {
            return Err("out_len and out_buffer_len must not be null".into());
        }
        let names = booster_eval_names(booster)?;
        let count = c_int::try_from(names.len()).map_err(|_| "too many metrics")?;
        // The longest name including its NUL terminator.
        let needed = names.iter().map(|n| n.len() + 1).max().unwrap_or(0);
        // SAFETY: checked non-null above; writable per the contract.
        unsafe {
            *out_len = count;
            *out_buffer_len = needed;
        }
        let writable = (len.max(0) as usize).min(names.len());
        if writable == 0 {
            return Ok(());
        }
        if out_strs.is_null() {
            return Err("out_strs must not be null when len > 0".into());
        }
        if buffer_len == 0 {
            return Err("buffer_len must be positive when writing names".into());
        }
        // SAFETY: the caller provides `len` writable `char*` slots; we touch the first `writable`.
        let slots = unsafe { std::slice::from_raw_parts(out_strs, writable) };
        for (name, &slot) in names.iter().zip(slots) {
            if slot.is_null() {
                return Err("out_strs entries must not be null".into());
            }
            let n = name.len().min(buffer_len - 1);
            // SAFETY: `slot` is writable for `buffer_len` bytes; `n < buffer_len`, so the NUL fits.
            unsafe {
                std::ptr::copy_nonoverlapping(name.as_ptr().cast::<c_char>(), slot, n);
                *slot.add(n) = 0;
            }
        }
        Ok(())
    })
}

/// upstream: `LGBM_BoosterSaveModelToString`. Serialize the model (loaded or in training) to its
/// text form, following the two-call buffer protocol: `out_len` is always set to the length
/// needed including the trailing NUL, and the text is written to `out_str` only when
/// `buffer_len >= out_len`. `num_iteration <= 0` saves all iterations; `feature_importance_type`
/// is `C_API_FEATURE_IMPORTANCE_SPLIT` (0) or `..._GAIN` (1).
///
/// # Safety
/// `handle` must be a live booster handle; `out_len` a valid writable pointer; `out_str` must be
/// writable for `buffer_len` bytes (or null when `buffer_len` is 0).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn LGBM_BoosterSaveModelToString(
    handle: *mut c_void,
    start_iteration: c_int,
    num_iteration: c_int,
    feature_importance_type: c_int,
    buffer_len: i64,
    out_len: *mut i64,
    out_str: *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        if out_len.is_null() {
            return Err("out_len must not be null".into());
        }
        let text = booster
            .with_model(|model| {
                model.save_model_to_string(start_iteration, num_iteration, feature_importance_type)
            })
            .map_err(|e| e.to_string())?;
        let needed =
            i64::try_from(text.len() + 1).map_err(|_| "model text too large".to_string())?;
        // SAFETY: checked non-null above; writable per the contract.
        unsafe { *out_len = needed };
        if buffer_len >= needed && !out_str.is_null() {
            // SAFETY: `out_str` is writable for `buffer_len >= needed` bytes; the NUL fits.
            unsafe {
                std::ptr::copy_nonoverlapping(text.as_ptr().cast::<c_char>(), out_str, text.len());
                *out_str.add(text.len()) = 0;
            }
        }
        Ok(())
    })
}

/// upstream: `LGBM_BoosterSaveModel`. Serialize the model (loaded or in training) to its text form
/// and write it to `filename`. `num_iteration <= 0` saves all iterations; `feature_importance_type`
/// is `C_API_FEATURE_IMPORTANCE_SPLIT` (0) or `..._GAIN` (1). The file holds the same bytes
/// [`LGBM_BoosterSaveModelToString`] produces, without a trailing NUL.
///
/// # Safety
/// `handle` must be a live booster handle and `filename` a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LGBM_BoosterSaveModel(
    handle: *mut c_void,
    start_iteration: c_int,
    num_iteration: c_int,
    feature_importance_type: c_int,
    filename: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: forwarded from this function's safety contract.
        let booster = unsafe { booster_ref(handle) }?;
        // SAFETY: forwarded from this function's safety contract.
        let path = unsafe { cstr_to_str(filename, "filename") }?;
        let text = booster
            .with_model(|model| {
                model.save_model_to_string(start_iteration, num_iteration, feature_importance_type)
            })
            .map_err(|e| e.to_string())?;
        std::fs::write(Path::new(path), text).map_err(|e| e.to_string())?;
        Ok(())
    })
}
