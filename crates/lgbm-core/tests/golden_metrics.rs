//! P4 gate: every metric matches upstream bit for bit at identical scores
//! (`scripts/gen_metric_fixtures.py` evaluates upstream before the first iteration, so
//! its internal score equals the init score we load here).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lgbm_core::config::Config;
use lgbm_core::io::dataset::Metadata;
use lgbm_core::metric::create_metric;
use lgbm_core::objective::LoadedObjective;
use lgbm_core::objective::train::create_objective;
use support::*;

fn metric_cases() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures_dir().join("metrics"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("expected.json").exists())
        .collect();
    out.sort();
    assert!(
        out.len() >= 24,
        "expected metric fixtures; run scripts/gen_metric_fixtures.py"
    );
    out
}

fn flat_params(case: &Path) -> HashMap<String, String> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(case.join("params.json")).unwrap()).unwrap();
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, val)| {
            let s = match val {
                serde_json::Value::Array(a) => a
                    .iter()
                    .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_string))
                    .collect::<Vec<_>>()
                    .join(","),
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (k.clone(), s)
        })
        .collect()
}

#[test]
fn metrics_match_upstream() {
    let mut failures = Vec::new();
    for case in metric_cases() {
        let cfg = Config::from_params(&flat_params(&case)).unwrap();
        let (label, _) = load_npy(&case.join("label.npy"));
        let (score, _) = load_npy(&case.join("score.npy"));
        let weights: Vec<f32> = if case.join("weight.npy").exists() {
            load_npy(&case.join("weight.npy"))
                .0
                .iter()
                .map(|&w| w as f32)
                .collect()
        } else {
            Vec::new()
        };
        let query_boundaries: Vec<i32> = if case.join("group.npy").exists() {
            let (g, _) = load_npy(&case.join("group.npy"));
            std::iter::once(0)
                .chain(g.iter().scan(0i32, |acc, &n| {
                    *acc += n as i32;
                    Some(*acc)
                }))
                .collect()
        } else {
            Vec::new()
        };
        let md = Metadata {
            label: label.iter().map(|&l| l as f32).collect(),
            weights,
            query_boundaries,
            init_score: Vec::new(),
        };
        // Ranking objectives are not ported yet; their metrics ignore the objective.
        let objective = create_objective(&cfg, &md).ok().and_then(|o| {
            LoadedObjective::from_model_str(&o.to_model_string())
                .ok()
                .flatten()
        });

        let want: Vec<(String, f64)> = serde_json::from_str::<Vec<(String, f64)>>(
            &std::fs::read_to_string(case.join("expected.json")).unwrap(),
        )
        .unwrap();
        let mut got: Vec<(String, f64)> = Vec::new();
        for name in &cfg.metric {
            let Some(m) = create_metric(name, &cfg, &md).unwrap() else {
                continue;
            };
            let vals = m.eval(&score, objective.as_ref());
            got.extend(m.names().iter().cloned().zip(vals));
        }
        let names_got: Vec<&String> = got.iter().map(|(n, _)| n).collect();
        let names_want: Vec<&String> = want.iter().map(|(n, _)| n).collect();
        if names_got != names_want {
            failures.push(format!(
                "{}: names {names_got:?} != {names_want:?}",
                case.display()
            ));
            continue;
        }
        for ((n, g), (_, w)) in got.iter().zip(&want) {
            let max_ulps = if on_fixture_platform() { 0 } else { 64 };
            let same = g.to_bits() == w.to_bits()
                || (max_ulps > 0 && (g - w).abs() <= 1e-12 * w.abs().max(1.0));
            if !same {
                failures.push(format!(
                    "{}: {n}\n  ours {g:e} ({:#x})\n  want {w:e} ({:#x})",
                    case.file_name().unwrap().to_string_lossy(),
                    g.to_bits(),
                    w.to_bits()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
