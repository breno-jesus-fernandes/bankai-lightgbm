//! `LGBM_*` model-load and metadata entry points match `lgbm-core` and upstream's C API
//! contract: `0`/`-1` return codes, a thread-local last error, and panic-free boundaries.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree, LGBM_BoosterGetCurrentIteration,
    LGBM_BoosterGetNumClasses, LGBM_BoosterLoadModelFromString, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    let c = unsafe { CStr::from_ptr(LGBM_GetLastError()) };
    c.to_string_lossy().into_owned()
}

#[test]
fn create_from_model_file_reports_iterations_and_classes() {
    let path = fixtures_dir().join("binary/model.txt");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let reference = Gbdt::from_model_file(&path).unwrap();

    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, 0, "load failed: {}", last_error());
    assert!(!handle.is_null());
    assert_eq!(num_iterations as usize, reference.current_iteration());
    assert!(num_iterations > 0);

    let mut num_classes: c_int = -1;
    // SAFETY: live handle and writable out-pointer.
    assert_eq!(
        unsafe { LGBM_BoosterGetNumClasses(handle, &mut num_classes) },
        0
    );
    assert_eq!(num_classes as usize, reference.num_class());
    assert_eq!(num_classes, 1);

    let mut current: c_int = -1;
    // SAFETY: live handle and writable out-pointer.
    assert_eq!(
        unsafe { LGBM_BoosterGetCurrentIteration(handle, &mut current) },
        0
    );
    assert_eq!(current, num_iterations);

    // SAFETY: handle came from the create call above and is freed exactly once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn load_from_string_matches_core_for_multiclass() {
    let path = fixtures_dir().join("multiclass/model.txt");
    let text = std::fs::read_to_string(&path).unwrap();
    let reference = Gbdt::from_model_string(&text).unwrap();
    let c_text = CString::new(text).unwrap();

    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterLoadModelFromString(c_text.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, 0, "load failed: {}", last_error());

    let mut num_classes: c_int = -1;
    // SAFETY: live handle and writable out-pointer.
    assert_eq!(
        unsafe { LGBM_BoosterGetNumClasses(handle, &mut num_classes) },
        0
    );
    assert_eq!(num_classes as usize, reference.num_class());
    assert_eq!(num_classes, 3);
    assert_eq!(num_iterations as usize, reference.current_iteration());

    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn null_and_bad_inputs_error_like_upstream() {
    let mut num_iterations: c_int = 0;
    let mut handle: *mut c_void = ptr::null_mut();
    // Null filename: returns -1 and records a message, never panics or crosses the boundary.
    // SAFETY: null filename is an explicitly handled error path; out-pointers are writable.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(ptr::null::<c_char>(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, -1);
    assert!(handle.is_null());
    assert!(last_error().contains("null"), "got: {}", last_error());

    // A non-existent file is an error, not a crash.
    let missing = CString::new("/no/such/bankai/model.txt").unwrap();
    // SAFETY: valid C string; writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(missing.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, -1);

    // Null handle to a getter is rejected.
    let mut out: c_int = 0;
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetNumClasses(ptr::null_mut(), &mut out) },
        -1
    );
    assert!(last_error().contains("null"));
}
