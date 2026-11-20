//! `LGBM_DatasetSetField` / `LGBM_DatasetGetField` round-trip the dataset metadata fields and
//! match `lgbm-core`'s setters and upstream's C API contract (SPEC-ERR-5). The oracle is the core
//! `Dataset` built the same way, whose `metadata` fields are public.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;

use lgbm_capi::{
    LGBM_DatasetCreateFromMat, LGBM_DatasetFree, LGBM_DatasetGetField, LGBM_DatasetSetField,
    LGBM_GetLastError,
};
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_DTYPE_INT32: c_int = 2;

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

fn create(nrow: usize, ncol: usize) -> *mut c_void {
    let data = matrix(nrow, ncol);
    let params = CString::new("max_bin=15").unwrap();
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: data has nrow*ncol f64; params valid; reference null; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1,
            params.as_ptr(),
            ptr::null(),
            &mut handle,
        )
    };
    assert_eq!(rc, 0, "create failed: {}", last_error());
    handle
}

fn set_field(handle: *mut c_void, name: &str, data: *const c_void, n: usize, ty: c_int) -> c_int {
    let field = CString::new(name).unwrap();
    // SAFETY: `data` points to `n` elements of type `ty`; field is a valid C string.
    unsafe { LGBM_DatasetSetField(handle, field.as_ptr(), data, n as c_int, ty) }
}

/// `(out_len, out_ptr, out_type)` for a field.
fn get_field(handle: *mut c_void, name: &str) -> (c_int, *const c_void, c_int) {
    let field = CString::new(name).unwrap();
    let mut out_len: c_int = -1;
    let mut out_ptr: *const c_void = ptr::null();
    let mut out_type: c_int = -1;
    // SAFETY: live handle; out pointers writable; field is a valid C string.
    let rc = unsafe {
        LGBM_DatasetGetField(
            handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_ptr,
            &mut out_type,
        )
    };
    assert_eq!(rc, 0, "get {name} failed: {}", last_error());
    (out_len, out_ptr, out_type)
}

fn read_f32(ptr: *const c_void, n: usize) -> Vec<f32> {
    // SAFETY: the caller guarantees `ptr` points to `n` readable f32.
    unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), n) }.to_vec()
}
fn read_f64(ptr: *const c_void, n: usize) -> Vec<f64> {
    // SAFETY: the caller guarantees `ptr` points to `n` readable f64.
    unsafe { std::slice::from_raw_parts(ptr.cast::<f64>(), n) }.to_vec()
}
fn read_i32(ptr: *const c_void, n: usize) -> Vec<i32> {
    // SAFETY: the caller guarantees `ptr` points to `n` readable i32.
    unsafe { std::slice::from_raw_parts(ptr.cast::<i32>(), n) }.to_vec()
}

#[test]
fn label_present_after_create_and_round_trips() {
    let (nrow, ncol) = (40usize, 5usize);
    let handle = create(nrow, ncol);

    // The label is always present (zero placeholder from create).
    let (len, p, ty) = get_field(handle, "label");
    assert_eq!(ty, C_API_DTYPE_FLOAT32);
    assert_eq!(len as usize, nrow);
    assert_eq!(read_f32(p, nrow), vec![0.0f32; nrow]);

    // Set a real label and read it back.
    let label: Vec<f32> = (0..nrow).map(|i| (i as f32) * 0.25 - 3.0).collect();
    assert_eq!(
        set_field(
            handle,
            "label",
            label.as_ptr().cast::<c_void>(),
            nrow,
            C_API_DTYPE_FLOAT32
        ),
        0,
        "set label failed: {}",
        last_error()
    );
    let (len, p, ty) = get_field(handle, "label");
    assert_eq!((len as usize, ty), (nrow, C_API_DTYPE_FLOAT32));
    assert_eq!(read_f32(p, nrow), label);

    // The "target" alias reads the same field.
    let (_, p2, _) = get_field(handle, "target");
    assert_eq!(read_f32(p2, nrow), label);

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn weight_init_score_round_trip_and_empty_before_set() {
    let (nrow, ncol) = (30usize, 4usize);
    let handle = create(nrow, ncol);

    // Unset weight is empty: null pointer, length 0.
    let (len, p, ty) = get_field(handle, "weight");
    assert_eq!((len, ty), (0, C_API_DTYPE_FLOAT32));
    assert!(p.is_null());

    let weight: Vec<f32> = (0..nrow).map(|i| 0.5 + (i % 4) as f32).collect();
    assert_eq!(
        set_field(
            handle,
            "weight",
            weight.as_ptr().cast::<c_void>(),
            nrow,
            C_API_DTYPE_FLOAT32
        ),
        0
    );
    let (len, p, _) = get_field(handle, "weight");
    assert_eq!(len as usize, nrow);
    assert_eq!(read_f32(p, nrow), weight);

    // init_score is float64.
    let (len, p, ty) = get_field(handle, "init_score");
    assert_eq!((len, ty), (0, C_API_DTYPE_FLOAT64));
    assert!(p.is_null());
    let init: Vec<f64> = (0..nrow).map(|i| (i as f64) * 0.1).collect();
    assert_eq!(
        set_field(
            handle,
            "init_score",
            init.as_ptr().cast::<c_void>(),
            nrow,
            C_API_DTYPE_FLOAT64
        ),
        0
    );
    let (len, p, ty) = get_field(handle, "init_score");
    assert_eq!((len as usize, ty), (nrow, C_API_DTYPE_FLOAT64));
    assert_eq!(read_f64(p, nrow), init);

    // Cross-check weight against the core setter (avoid_inf transform).
    let cfg = Config::from_str_params("max_bin=15").unwrap();
    let mut reference =
        Dataset::from_dense(&matrix(nrow, ncol), nrow, ncol, &vec![0.0; nrow], &cfg, &[]).unwrap();
    reference.set_weights(&weight).unwrap();
    assert_eq!(read_f32(get_field(handle, "weight").1, nrow), reference.metadata.weights);

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn group_sizes_become_boundaries_and_position_round_trips() {
    let (nrow, ncol) = (40usize, 3usize);
    let handle = create(nrow, ncol);

    // group takes per-query sizes and GetField returns the boundaries (num_queries + 1).
    let sizes = [10i32, 10, 20];
    assert_eq!(
        set_field(
            handle,
            "group",
            sizes.as_ptr().cast::<c_void>(),
            sizes.len(),
            C_API_DTYPE_INT32
        ),
        0,
        "set group failed: {}",
        last_error()
    );
    let (len, p, ty) = get_field(handle, "group");
    assert_eq!(ty, C_API_DTYPE_INT32);
    assert_eq!(len as usize, sizes.len() + 1);
    assert_eq!(read_i32(p, len as usize), vec![0, 10, 20, 40]);
    // The "query" alias reads the same boundaries.
    assert_eq!(get_field(handle, "query").0, len);

    // position is int32, one per row.
    let pos: Vec<i32> = (0..nrow).map(|i| (i % 3) as i32).collect();
    assert_eq!(
        set_field(
            handle,
            "position",
            pos.as_ptr().cast::<c_void>(),
            nrow,
            C_API_DTYPE_INT32
        ),
        0
    );
    let (len, p, ty) = get_field(handle, "position");
    assert_eq!((len as usize, ty), (nrow, C_API_DTYPE_INT32));
    assert_eq!(read_i32(p, nrow), pos);

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

#[test]
fn rejects_unknown_field_type_mismatch_and_nulls() {
    let handle = create(10, 2);

    // Unknown field name.
    let data = [0.0f32; 10];
    assert_eq!(
        set_field(
            handle,
            "nonsense",
            data.as_ptr().cast::<c_void>(),
            10,
            C_API_DTYPE_FLOAT32
        ),
        -1
    );

    // Type mismatch: label is float32, not int32.
    let ints = [0i32; 10];
    assert_eq!(
        set_field(
            handle,
            "label",
            ints.as_ptr().cast::<c_void>(),
            10,
            C_API_DTYPE_INT32
        ),
        -1
    );

    // GetField on an unknown field.
    let field = CString::new("nonsense").unwrap();
    let mut out_len: c_int = -1;
    let mut out_ptr: *const c_void = ptr::null();
    let mut out_type: c_int = -1;
    // SAFETY: live handle; out pointers writable; field valid.
    let rc = unsafe {
        LGBM_DatasetGetField(
            handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_ptr,
            &mut out_type,
        )
    };
    assert_eq!(rc, -1);

    // Null handle.
    let label = CString::new("label").unwrap();
    // SAFETY: null handle is an explicitly handled error path.
    let rc = unsafe {
        LGBM_DatasetSetField(
            ptr::null_mut(),
            label.as_ptr(),
            data.as_ptr().cast::<c_void>(),
            10,
            C_API_DTYPE_FLOAT32,
        )
    };
    assert_eq!(rc, -1);
    assert!(last_error().contains("null"));

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}

/// A null `field_name` is an error, not a crash.
#[test]
fn rejects_null_field_name() {
    let handle = create(10, 2);
    let data = [0.0f32; 10];
    let null_name: *const c_char = ptr::null();
    // SAFETY: null field name is an explicitly handled error path.
    let rc = unsafe {
        LGBM_DatasetSetField(
            handle,
            null_name,
            data.as_ptr().cast::<c_void>(),
            10,
            C_API_DTYPE_FLOAT32,
        )
    };
    assert_eq!(rc, -1);
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_DatasetFree(handle) }, 0);
}
