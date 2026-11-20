//! `LGBM_DatasetCreateFromMat` and the dataset metadata/free entry points match `lgbm-core`'s
//! `Dataset::from_dense` and upstream's C API contract (SPEC-ERR-5).
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::ptr;

use lgbm_capi::{
    LGBM_DatasetCreateFromMat, LGBM_DatasetFree, LGBM_DatasetGetNumData, LGBM_DatasetGetNumFeature,
    LGBM_GetLastError,
};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 31 % 97) as f64) * 0.1 - 4.0)
        .collect()
}

#[test]
fn create_from_mat_matches_core_shape() {
    let (nrow, ncol) = (40usize, 5usize);
    let data = matrix(nrow, ncol);
    let params = CString::new("max_bin=15").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: data has nrow*ncol f64; params is a valid C string; reference null; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1, // row-major
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "create failed: {}", last_error());
    assert!(!handle.is_null());

    // Core reference built the same way (label is set separately in the C API).
    let cfg = Config::from_str_params("max_bin=15").unwrap();
    let reference = Dataset::from_dense(&data, nrow, ncol, &vec![0.0; nrow], &cfg, &[]).unwrap();

    let mut n: c_int = -1;
    // SAFETY: live handle, writable out-pointer.
    assert_eq!(unsafe { LGBM_DatasetGetNumData(handle, &mut n) }, 0);
    assert_eq!(n as usize, reference.num_data());
    assert_eq!(n as usize, nrow);

    let mut nf: c_int = -1;
    // SAFETY: live handle, writable out-pointer.
    assert_eq!(unsafe { LGBM_DatasetGetNumFeature(handle, &mut nf) }, 0);
    assert_eq!(nf as usize, reference.layout.num_total_features);
    assert_eq!(nf as usize, ncol);

    // SAFETY: handle from the create call, freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn create_from_mat_accepts_float32_and_column_major() {
    let (nrow, ncol) = (24usize, 4usize);
    let row_major = matrix(nrow, ncol);
    let f32_data: Vec<f32> = row_major.iter().map(|&x| x as f32).collect();
    let params = CString::new("").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: f32 data has nrow*ncol values; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            f32_data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT32,
            nrow as i32,
            ncol as i32,
            1,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "float32 create failed: {}", last_error());
    let mut nf: c_int = 0;
    // SAFETY: live handle, writable out-pointer.
    unsafe { LGBM_DatasetGetNumFeature(handle, &mut nf) };
    assert_eq!(nf as usize, ncol);
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn create_from_mat_rejects_reference_and_nulls() {
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let params = CString::new("").unwrap();
    let mut handle: *mut c_void = ptr::null_mut();

    // A non-null reference is not supported yet: error, not a crash. Point at a real local so
    // the pointer is non-null and not dangling.
    let anchor = 0u8;
    let dummy_ref = std::ptr::from_ref(&anchor).cast::<c_void>();
    // SAFETY: data valid; reference is an explicitly handled unsupported path.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            2,
            2,
            1,
            params.as_ptr(),
            dummy_ref,
            &mut handle,
        )
    };
    assert_eq!(rc, -1);
    assert!(handle.is_null());

    // Null out-pointer.
    // SAFETY: null out is an explicitly handled error path.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            2,
            2,
            1,
            params.as_ptr(),
            ptr::null(),
            ptr::null_mut::<*mut c_void>(),
        )
    };
    assert_eq!(rc, -1);

    // Null handle to a getter.
    let mut out: c_int = 0;
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_DatasetGetNumData(ptr::null_mut(), &mut out) },
        -1
    );
    assert!(last_error().contains("null"));
}
