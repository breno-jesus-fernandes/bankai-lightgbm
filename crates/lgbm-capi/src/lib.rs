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

/// upstream: `API_END()` returns `0` on success.
const C_API_SUCCESS: c_int = 0;
/// upstream: `API_END()` returns `-1` after storing the error.
const C_API_ERROR: c_int = -1;

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
