//! `LGBM_BoosterSaveModelToString` writes the same model text as `lgbm-core`'s
//! `Gbdt::save_model_to_string` and follows upstream's two-call buffer protocol (SPEC-ERR-5).
#![allow(clippy::unwrap_used)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr;

use lgbm_capi::{
    LGBM_BoosterCreate, LGBM_BoosterCreateFromModelfile, LGBM_BoosterFree,
    LGBM_BoosterSaveModelToString, LGBM_BoosterUpdateOneIter, LGBM_DatasetCreateFromMat,
    LGBM_DatasetFree, LGBM_DatasetSetField, LGBM_GetLastError,
};
use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::dataset::Dataset;

const C_API_DTYPE_FLOAT32: c_int = 0;
const C_API_DTYPE_FLOAT64: c_int = 1;
const C_API_FEATURE_IMPORTANCE_SPLIT: c_int = 0;

const PARAMS: &str = "objective=regression max_bin=31 num_iterations=8 num_leaves=7 \
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

/// Train a booster through the C API and return its handle (plus the dataset handle to free).
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
    let cfg = Config::from_str_params(PARAMS).unwrap();
    for _ in 0..cfg.num_iterations {
        let mut empty: c_int = -1;
        // SAFETY: live booster; out writable.
        assert_eq!(unsafe { LGBM_BoosterUpdateOneIter(booster, &mut empty) }, 0);
        if empty == 1 {
            break;
        }
    }
    (booster, ds)
}

/// The two-call protocol: ask for the length, allocate, then fill. Returns the model string.
fn save_to_string(handle: *mut c_void) -> String {
    let mut needed: i64 = -1;
    // First call: buffer_len 0, null buffer -> just reports the length.
    // SAFETY: live handle; out_len writable; out_str may be null when buffer_len is 0.
    let rc = unsafe {
        LGBM_BoosterSaveModelToString(
            handle,
            0,
            0,
            C_API_FEATURE_IMPORTANCE_SPLIT,
            0,
            &mut needed,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "length query failed: {}", last_error());
    assert!(needed > 0);

    let mut buf = vec![0u8; needed as usize];
    let mut written: i64 = -1;
    // SAFETY: buf holds `needed` bytes; out_len writable.
    let rc = unsafe {
        LGBM_BoosterSaveModelToString(
            handle,
            0,
            0,
            C_API_FEATURE_IMPORTANCE_SPLIT,
            needed,
            &mut written,
            buf.as_mut_ptr().cast::<c_char>(),
        )
    };
    assert_eq!(rc, 0, "save failed: {}", last_error());
    assert_eq!(written, needed);
    // The buffer is NUL-terminated; drop the terminator.
    assert_eq!(*buf.last().unwrap(), 0);
    String::from_utf8(buf[..buf.len() - 1].to_vec()).unwrap()
}

#[test]
fn save_model_to_string_matches_core() {
    let (nrow, ncol) = (60usize, 5usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let got = save_to_string(booster);

    // Oracle: core trains the same way and serializes the model.
    let cfg = Config::from_str_params(PARAMS).unwrap();
    let reference_ds = Dataset::from_dense(&data, nrow, ncol, &label, &cfg, &[]).unwrap();
    let model = Gbdt::train(&cfg, &reference_ds).unwrap();
    let want = model
        .save_model_to_string(0, 0, C_API_FEATURE_IMPORTANCE_SPLIT)
        .unwrap();

    assert_eq!(got, want);

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn save_model_to_string_works_on_a_loaded_model() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/binary/model.txt");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut num_iterations: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    let rc = unsafe {
        LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut num_iterations, &mut handle)
    };
    assert_eq!(rc, 0, "load failed: {}", last_error());

    // The serialized model reloads into an equivalent core model.
    let got = save_to_string(handle);
    assert!(
        Gbdt::from_model_string(&got).is_ok(),
        "reserialized model did not parse"
    );

    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}

#[test]
fn too_small_buffer_reports_length_without_writing() {
    let (nrow, ncol) = (20usize, 3usize);
    let data = matrix(nrow, ncol);
    let label = labels(&data, nrow, ncol);
    let (booster, ds) = trained(&data, &label, nrow, ncol);

    let mut needed: i64 = -1;
    let mut small = [0xAAu8; 4];
    // A 4-byte buffer cannot hold the model: out_len is set, nothing is written, rc is 0.
    // SAFETY: small holds 4 bytes; out_len writable.
    let rc = unsafe {
        LGBM_BoosterSaveModelToString(
            booster,
            0,
            0,
            C_API_FEATURE_IMPORTANCE_SPLIT,
            small.len() as i64,
            &mut needed,
            small.as_mut_ptr().cast::<c_char>(),
        )
    };
    assert_eq!(rc, 0, "query failed: {}", last_error());
    assert!(needed > small.len() as i64);
    assert_eq!(
        small, [0xAAu8; 4],
        "buffer must be untouched when too small"
    );

    // SAFETY: handles freed once each.
    assert_eq!(unsafe { LGBM_BoosterFree(booster) }, 0);
    assert_eq!(unsafe { LGBM_DatasetFree(ds) }, 0);
}

#[test]
fn rejects_null_handle_and_null_out_len() {
    let mut out_len: i64 = -1;
    let mut buf = [0u8; 8];
    // Null handle.
    // SAFETY: null handle is an explicitly handled error path.
    let rc = unsafe {
        LGBM_BoosterSaveModelToString(
            ptr::null_mut(),
            0,
            0,
            C_API_FEATURE_IMPORTANCE_SPLIT,
            buf.len() as i64,
            &mut out_len,
            buf.as_mut_ptr().cast::<c_char>(),
        )
    };
    assert_eq!(rc, -1);
    assert!(last_error().contains("null"));

    // Null out_len on a loaded model.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/binary/model.txt");
    let c_path = CString::new(path.to_str().unwrap()).unwrap();
    let mut ni: c_int = -1;
    let mut handle: *mut c_void = ptr::null_mut();
    // SAFETY: valid C string and writable out-pointers.
    assert_eq!(
        unsafe { LGBM_BoosterCreateFromModelfile(c_path.as_ptr(), &mut ni, &mut handle) },
        0
    );
    // SAFETY: null out_len is an explicitly handled error path.
    let rc = unsafe {
        LGBM_BoosterSaveModelToString(
            handle,
            0,
            0,
            C_API_FEATURE_IMPORTANCE_SPLIT,
            buf.len() as i64,
            ptr::null_mut(),
            buf.as_mut_ptr().cast::<c_char>(),
        )
    };
    assert_eq!(rc, -1);
    // SAFETY: freed once.
    assert_eq!(unsafe { LGBM_BoosterFree(handle) }, 0);
}
