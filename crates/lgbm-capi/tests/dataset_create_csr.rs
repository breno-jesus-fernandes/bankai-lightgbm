//! `LGBM_DatasetCreateFromCSR` matches `lgbm-core`'s `Dataset::from_csr` and upstream's C API
//! contract (SPEC-ERR-5). It bins the sparse rows without densifying the matrix; the oracle is a
//! core dataset built from the same CSR arrays.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_int, c_void};
use std::ptr;

use lgbm_capi::{
    LGBM_DatasetCreateFromCSR, LGBM_DatasetFree, LGBM_DatasetGetNumData, LGBM_DatasetGetNumFeature,
    LGBM_GetLastError,
};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::io::sparse::CsrMatrix;

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

/// A CSR matrix with some structural zeros: `(indptr, indices, values)`.
fn csr(nrow: usize, ncol: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..nrow {
        for c in 0..ncol {
            if (r + c) % 3 != 0 {
                indices.push(c as i32);
                values.push(((r * 7 + c * 3) % 50) as f64 * 0.2 - 4.0);
            }
        }
        indptr.push(indices.len() as i64);
    }
    (indptr, indices, values)
}

fn core_reference(
    indptr: &[i64],
    indices: &[i32],
    values: &[f64],
    ncol: usize,
    params: &str,
) -> Dataset {
    let m = CsrMatrix::new(indptr, indices, values, ncol).unwrap();
    let cfg = Config::from_str_params(params).unwrap();
    let nrow = m.nrow();
    Dataset::from_csr(&m, &vec![0.0; nrow], &cfg, &[]).unwrap()
}

#[test]
fn create_from_csr_matches_core_shape() {
    let (nrow, ncol) = (40usize, 5usize);
    let (indptr, indices, values) = csr(nrow, ncol);
    let params = CString::new("max_bin=15").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: indptr has nrow+1 i64; indices/values have nelem each; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSR(
            indptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            indptr.len() as i64,
            values.len() as i64,
            ncol as i64,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "create failed: {}", last_error());
    assert!(!handle.is_null());

    let reference = core_reference(&indptr, &indices, &values, ncol, "max_bin=15");

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
fn create_from_csr_accepts_int32_indptr_and_float32() {
    let (nrow, ncol) = (24usize, 4usize);
    let (indptr, indices, values) = csr(nrow, ncol);
    let indptr32: Vec<i32> = indptr.iter().map(|&x| x as i32).collect();
    let values32: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    let params = CString::new("").unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: indptr32 has nrow+1 i32; values32 has nelem f32; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSR(
            indptr32.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT32,
            indices.as_ptr(),
            values32.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT32,
            indptr32.len() as i64,
            values32.len() as i64,
            ncol as i64,
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
fn create_from_csr_rejects_reference_and_nulls() {
    let indptr = [0i64, 1, 2];
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
        LGBM_DatasetCreateFromCSR(
            indptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            indptr.len() as i64,
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
        LGBM_DatasetCreateFromCSR(
            indptr.as_ptr().cast::<c_void>(),
            C_API_DTYPE_INT64,
            indices.as_ptr(),
            values.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            indptr.len() as i64,
            values.len() as i64,
            2,
            params.as_ptr(),
            ptr::null(),
            ptr::null_mut::<*mut c_void>(),
        )
    };
    assert_eq!(rc, -1);

    // nindptr < 1 is rejected.
    // SAFETY: an explicitly handled error path; pointers are valid.
    let rc = unsafe {
        LGBM_DatasetCreateFromCSR(
            indptr.as_ptr().cast::<c_void>(),
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
    assert!(last_error().contains("nindptr"));
}
