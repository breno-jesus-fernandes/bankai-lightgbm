//! Config parity: upstream's `parameters:` block must round-trip through
//! `Config::set` + `Config::to_string` byte-for-byte.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::HashMap;

use lgbm_core::boosting::Gbdt;
use lgbm_core::config::{Config, PARAMETER_TYPES};
use support::*;

fn params_from_block(block: &str) -> HashMap<String, String> {
    block
        .lines()
        .filter_map(|l| l.strip_prefix('[')?.strip_suffix(']')?.split_once(": "))
        .map(|(k, v)| {
            // The block prints bools as 0/1 (C++ stream); parameters take true/false.
            let is_bool = PARAMETER_TYPES.iter().any(|(n, t)| *n == k && *t == "bool");
            let v = match (is_bool, v) {
                (true, "1") => "true",
                (true, "0") => "false",
                _ => v,
            };
            (k.to_string(), v.to_string())
        })
        .collect()
}

#[test]
fn parameters_block_round_trips() {
    for case in cases() {
        let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
        let block = model.loaded_parameter();
        let mut params = params_from_block(block);
        // `num_class=1` with multiclass metrics would be rejected; keep upstream's view.
        params.retain(|_, v| !v.is_empty());
        let cfg =
            Config::from_params(&params).unwrap_or_else(|e| panic!("{}: {e}", case.display()));
        let ours = cfg.to_string();
        if ours != block {
            let (i, (a, b)) = ours
                .lines()
                .zip(block.lines())
                .enumerate()
                .find(|(_, (a, b))| a != b)
                .unwrap_or((0, ("", "")));
            panic!("{}: line {i}: ours {a:?} want {b:?}", case.display());
        }
    }
}
