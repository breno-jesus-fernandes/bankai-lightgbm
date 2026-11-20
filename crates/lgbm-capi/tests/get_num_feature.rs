//! `LGBM_BoosterGetNumFeature` matches the core model's feature count (SPEC-ERR-5).
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree, LGBM_BoosterGetNumFeature, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn get_num_feature_matches_core() {
    let path = fixtures_dir().join("binary/model.txt");
    let reference = Gbdt::from_model_file(&path).unwrap();
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, 0, "load failed: {}", last_error());

    let mut num_feature: c_int = -1;
    // SAFETY: live handle and writable out-pointer.
    assert_eq!(
        unsafe { LGBM_BoosterGetNumFeature(handle, &mut num_feature) },
        0
    );
    assert_eq!(num_feature as usize, reference.num_features());
    assert!(num_feature > 0);

    // SAFETY: handle from the load call, freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn get_num_feature_rejects_null() {
    let mut out: c_int = 0;
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetNumFeature(ptr::null_mut(), &mut out) },
        -1
    );
    assert!(last_error().contains("null"));
}
