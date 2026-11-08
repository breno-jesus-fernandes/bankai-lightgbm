//! Model text parsing must never panic on any input (AGENTS.md section 4, fuzz layer).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        // Errors are expected for malformed models; only a panic is a bug.
        let _ = lgbm_core::boosting::Gbdt::from_model_string(text);
    }
});
