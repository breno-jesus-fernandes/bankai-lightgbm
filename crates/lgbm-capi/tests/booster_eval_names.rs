//! `LGBM_BoosterGetEvalCounts` and `LGBM_BoosterGetEvalNames` report the configured metrics of a
//! training booster. Oracle: the metric list resolved from the booster's `metric` parameter
//! (`metric=l2,l1` yields exactly `l2` then `l1`). A booster loaded from a model has no training
//! metrics and reports none.
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterGetEvalCounts, LGBM_BoosterGetEvalNames, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::config::Config;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;

const PARAMS: &str = "objective=regression metric=l2,l1 max_bin=31 num_iterations=4 num_leaves=7 \
                      learning_rate=0.1 min_data_in_leaf=1 deterministic=true force_col_wise=true \
                      num_threads=1 seed=42 verbosity=-1";

fn last_error() -> String {
    // SAFETY: the pointer is this thread's valid last-error C string.
    unsafe { CStr::from_ptr(LGBM_GetLastError()) }
        .to_string_lossy()
        .into_owned()
}

fn matrix(nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow * ncol)
        .map(|i| ((i * 37 % 101) as f64) * 0.1 - 5.0)
        .collect()
}

fn labels(data: &[f64], nrow: usize, ncol: usize) -> Vec<f64> {
    (0..nrow)
        .map(|r| {
            let row = &data[r * ncol..(r + 1) * ncol];
            row[0] * 2.0 - row[ncol - 1] + (r % 5) as f64 * 0.3
        })
        .collect()
}

/// Build a dataset handle and a training booster over it. Returns (booster, dataset) to free.
fn trained(data: &[f64], label: &[f64], nrow: usize, ncol: usize) -> (*mut c_void, *mut c_void) {
    let params = CString::new(PARAMS).unwrap();
    let mut ds: *mut c_void = ptr::null_mut();
    // SAFETY: data has nrow*ncol f64; params valid; out writable.
    let rc = unsafe {
        LGBM_DatasetCreateFromMat(
            data.as_ptr().cast::<c_void>(),
            C_API_DTYPE_FLOAT64,
            nrow as i32,
            ncol as i32,
            1,
            params.as_ptr(),
            ptr::null(),
            &mut ds,
        )
    };
    assert_eq!(rc, 0, "dataset create failed: {}", last_error());

    let label32: Vec<f32> = label.iter().map(|&x| x as f32).collect();
    let field = CString::new("label").unwrap();
    // SAFETY: label32 has nrow f32; field valid.
    assert_eq!(
        unsafe {
            LGBM_DatasetSetField(
                ds,
                field.as_ptr(),
                label32.as_ptr().cast::<c_void>(),
                nrow as c_int,
                C_API_DTYPE_FLOAT32,
            )
        },
        0
    );

    let mut booster: *mut c_void = ptr::null_mut();
    // SAFETY: live dataset; params valid; out writable.
    assert_eq!(
        unsafe { LGBM_BoosterCreate(ds, params.as_ptr(), &mut booster) },
        0,
        "booster create failed: {}",
        last_error()
    );
    (booster, ds)
}

/// Read all eval names through the two-call string-array protocol.
fn eval_names(handle: *mut c_void) -> Vec<String> {
    let mut count: c_int = -1;
    // SAFETY: live handle; out writable.
    assert_eq!(
        unsafe { LGBM_BoosterGetEvalCounts(handle, &mut count) },
        0,
        "eval counts failed: {}",
        last_error()
    );
    assert!(count >= 0);
    let count = count as usize;

    // First call: ask for the sizes (no output buffers).
    let mut out_len: c_int = -1;
    let mut out_buffer_len: usize = 0;
    // SAFETY: live handle; out pointers writable; out_strs null when len is 0.
    assert_eq!(
        unsafe {
            LGBM_BoosterGetEvalNames(
                handle,
                0,
                &mut out_len,
                0,
                &mut out_buffer_len,
                ptr::null_mut(),
            )
        },
        0,
        "eval names size query failed: {}",
        last_error()
    );
    assert_eq!(out_len as usize, count, "counts and names must agree");
    if count == 0 {
        return Vec::new();
    }
    assert!(out_buffer_len > 0);

    // Second call: allocate `count` buffers of `out_buffer_len` bytes and fill them.
    let mut buffers: Vec<Vec<u8>> = (0..count).map(|_| vec![0u8; out_buffer_len]).collect();
    let mut ptrs: Vec<*mut c_char> = buffers
        .iter_mut()
        .map(|b| b.as_mut_ptr().cast::<c_char>())
        .collect();
    let mut out_len2: c_int = -1;
    let mut out_buffer_len2: usize = 0;
    // SAFETY: live handle; `count` buffers of `out_buffer_len` bytes; out pointers writable.
    assert_eq!(
        unsafe {
            LGBM_BoosterGetEvalNames(
                handle,
                count as c_int,
                &mut out_len2,
                out_buffer_len,
                &mut out_buffer_len2,
                ptrs.as_mut_ptr(),
            )
        },
        0,
        "eval names failed: {}",
        last_error()
    );
    assert_eq!(out_len2 as usize, count);
    assert_eq!(out_buffer_len2, out_buffer_len);

    buffers
        .iter()
        .map(|b| {
            // SAFETY: the buffer holds a NUL-terminated C string written by the call.
            unsafe { CStr::from_ptr(b.as_ptr().cast::<c_char>()) }
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

#[test]
fn eval_counts_and_names_follow_the_metric_parameter() {
    let (nrow, ncol) = (40usize, 4usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    // Oracle: the resolved metric list from the same parameter string.
    let cfg = Config::from_str_params(PARAMS).unwrap();
    assert_eq!(cfg.metric, vec!["l2".to_string(), "l1".to_string()]);

    let names = eval_names(booster);
    assert_eq!(names, vec!["l2".to_string(), "l1".to_string()]);

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn a_loaded_model_reports_no_training_metrics() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/binary/model.txt");
    let c_src = CString::new(src.to_str().unwrap()).unwrap();
    let mut ni: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    assert_eq!(
        unsafe { LGBM_BoosterCreateFromModelfile(c_src.as_ptr(), &mut ni, &mut handle) },
        0,
        "load failed: {}",
        last_error()
    );

    let mut count: c_int = -1;
    // SAFETY: live handle; out writable.
    assert_eq!(unsafe { LGBM_BoosterGetEvalCounts(handle, &mut count) }, 0);
    assert_eq!(count, 0, "a loaded model has no training metrics");
    assert!(eval_names(handle).is_empty());

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn rejects_null_handle_and_null_out() {
    let mut count: c_int = -1;
    // SAFETY: null handle is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetEvalCounts(ptr::null_mut(), &mut count) },
        -1
    );
    assert!(last_error().contains("null"));

    let (nrow, ncol) = (20usize, 3usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    // Null out_len for counts.
    // SAFETY: null out is an explicitly handled error path.
    assert_eq!(
        unsafe { LGBM_BoosterGetEvalCounts(booster, ptr::null_mut()) },
        -1
    );

    // Null out_len for names.
    let mut out_buffer_len: usize = 0;
    // SAFETY: null out_len is an explicitly handled error path.
    assert_eq!(
        unsafe {
            LGBM_BoosterGetEvalNames(
                booster,
                0,
                ptr::null_mut(),
                0,
                &mut out_buffer_len,
                ptr::null_mut(),
            )
        },
        -1
    );

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}
