//! CSV, TSV and LibSVM line parsing must never panic on any input (AGENTS.md section 4).
#![no_main]

use lgbm_core::io::parser::{detect_data_type, parse_line};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let lines: Vec<&str> = text.lines().collect();
    let Ok(ty) = detect_data_type(&lines) else {
        return;
    };
    for line in &lines {
        for precise in [true, false] {
            // Errors are expected for malformed rows; only a panic is a bug.
            let _ = parse_line(ty, line, 0, precise);
        }
    }
});
