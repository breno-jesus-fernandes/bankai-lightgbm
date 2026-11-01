//! `dump_model` JSON must match upstream's `LGBM_BoosterDumpModel` exactly.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use lgbm_core::boosting::Gbdt;
use serde_json::Value;
use support::*;

/// First JSON path where two values differ (for readable failures).
fn first_diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for (k, v) in x {
                match y.get(k) {
                    Some(w) => {
                        if let Some(d) = first_diff(v, w, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    None => return Some(format!("{path}.{k} missing in want")),
                }
            }
            y.keys()
                .find(|k| !x.contains_key(*k))
                .map(|k| format!("{path}.{k} missing in ours"))
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: len {} vs {}", x.len(), y.len()));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .find_map(|(i, (v, w))| first_diff(v, w, &format!("{path}[{i}]")))
        }
        (Value::Number(x), Value::Number(y)) => {
            (x.as_f64() != y.as_f64()).then(|| format!("{path}: {x} vs {y}"))
        }
        _ => (a != b).then(|| format!("{path}: {a} vs {b}")),
    }
}

#[test]
fn dump_model_matches_upstream_json() {
    for case in cases() {
        let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
        let ours: Value = serde_json::from_str(&model.dump_model(0, -1, 0).unwrap())
            .unwrap_or_else(|e| panic!("{}: invalid JSON: {e}", case.display()));
        let mut want: Value =
            serde_json::from_str(&std::fs::read_to_string(case.join("model.json")).unwrap())
                .unwrap();
        // Added by the Python layer, not by the C++ dump.
        want.as_object_mut().unwrap().remove("pandas_categorical");
        if let Some(d) = first_diff(&ours, &want, "$") {
            panic!("{}: {d}", case.display());
        }
    }
}
