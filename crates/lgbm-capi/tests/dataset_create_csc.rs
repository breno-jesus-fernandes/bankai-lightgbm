//! `LGBM_DatasetCreateFromCSC` matches `lgbm-core`'s CSC-to-CSR path and upstream's C API
//! contract (SPEC-ERR-5). A CSC matrix is gathered into CSR rows (keeping upstream's
//! column-walk order) and binned by `Dataset::from_csr` without densifying; the oracle is a
//! core dataset built the same way.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::ptr;

use lgbm_capi::{
    LGBM_DatasetCreateFromCSC, LGBM_DatasetFree, LGBM_DatasetGetNumData, LGBM_DatasetGetNumFeature,
    LGBM_GetLastError,
};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::sparse::{CsrMatrix, csc_to_csr};

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_DTYPE_INT32: c_int = 2;
const C_API_DTYPE_INT64: c_int = 3;

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
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

/// The core dataset the wrapper is expected to build: CSC gathered to CSR, then `from_csr`.
fn core_reference(
    col_ptr: &[i64],
    indices: &[i32],
    values: &[f64],
    nrow: usize,
    ncol: usize,
    params: &str,
) -> Dataset {
    let (indptr, idx, vals) = csc_to_csr(col_ptr, indices, values, nrow).unwrap();
    let m = CsrMatrix::new(&indptr, &idx, &vals, ncol).unwrap();
    let cfg = Config::from_str_params(params).unwrap();
    Dataset::from_csr(&m, &vec![0.0; nrow], &cfg, &[]).unwrap()
}

#[test]
fn create_from_csc_matches_core_shape() {
    let (nrow, ncol) = (40usize, 5usize);
    let (col_ptr, indices, values) = csc(nrow, ncol);
    let params = CString::new("max_bin=15").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: col_ptr has ncol+1 i64; indices/values have nelem each; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSC(
            col_ptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            col_ptr.len() as i64,
            values.len() as i64,
            nrow as i64,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "create failed: {}", last_error());
    assert!(!handle.is_null());

    let reference = core_reference(&col_ptr, &indices, &values, nrow, ncol, "max_bin=15");

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
fn create_from_csc_accepts_int32_col_ptr_and_float32() {
    let (nrow, ncol) = (24usize, 4usize);
    let (col_ptr, indices, values) = csc(nrow, ncol);
    let col_ptr32: Vec<i32> = col_ptr.iter().map(|&x| x as i32).collect();
    let values32: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    let params = CString::new("").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: col_ptr32 has ncol+1 i32; values32 has nelem f32; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSC(
            col_ptr32.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT32,
            indices.as_ptr(),
            values32.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT32,
            col_ptr32.len() as i64,
            values32.len() as i64,
            nrow as i64,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "float32 create failed: {}", last_error());

    let mut nd: c_int = 0;
    // SAFETY: live handle, writable out-pointer.
    unsafe { LGBM_DatasetGetNumData(handle, &mut nd) };
    assert_eq!(nd as usize, nrow);
    let mut nf: c_int = 0;
    // SAFETY: live handle, writable out-pointer.
    unsafe { LGBM_DatasetGetNumFeature(handle, &mut nf) };
    assert_eq!(nf as usize, ncol);

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn create_from_csc_rejects_reference_and_nulls() {
    let col_ptr = [0i64, 1, 2];
    let indices = [0i32, 1];
    let values = [1.0f64, 2.0];
    let params = CString::new("").unwrap();
    let mut handle: *mut c_void = ptr::null_mut();

    // A non-null reference is not supported yet: error, not a crash. Point at a real local so the
    // pointer is non-null and not dangling.
    let anchor = 0u8;
    let dummy_ref = std::ptr::from_ref(&anchor).cast::<c_void>();
    // SAFETY: arrays valid; reference is an explicitly handled unsupported path.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSC(
            col_ptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            col_ptr.len() as i64,
            values.len() as i64,
            2,
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
        LGBM_DatasetCreateFromCSC(
            col_ptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            col_ptr.len() as i64,
            values.len() as i64,
            2,
            params.as_ptr(),
            ptr::null(),
            ptr::null_mut::<*mut c_void>(),
        )
    };
    assert_eq!(rc, -1);

    // ncol_ptr < 1 is rejected.
    // SAFETY: an explicitly handled error path; pointers are valid.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSC(
            col_ptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            0,
            values.len() as i64,
            2,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, -1);
    assert!(last_error().contains("ncol_ptr"));
}
