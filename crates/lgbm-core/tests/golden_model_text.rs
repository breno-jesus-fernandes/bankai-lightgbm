//! Model text written by bankai must be byte-identical to upstream's `save_model`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use lgbm_core::boosting::Gbdt;
use support::*;

/// Python's `save_model` appends `\npandas_categorical:<json>\n` after the C++ string.
fn with_pandas_trailer(model: &Gbdt, s: String) -> String {
    match model.pandas_categorical() {
        Some(p) => format!("{s}\npandas_categorical:{p}\n"),
        None => s,
    }
}

#[test]
fn load_then_save_is_byte_identical() {
    for case in cases() {
        let text = std::fs::read_to_string(case.join("model.txt")).unwrap();
        let model = Gbdt::from_model_string(&text).unwrap();
        let out = with_pandas_trailer(&model, model.save_model_to_string(0, -1, 0).unwrap());
        if out != text {
            let line = out
                .lines()
                .zip(text.lines())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            panic!(
                "{}: first differing line {line}:\n ours: {:?}\n want: {:?}",
                case.display(),
                out.lines().nth(line),
                text.lines().nth(line)
            );
        }
    }
}

#[test]
fn saved_iteration_slice_reloads_with_same_predictions() {
    let case = fixtures_dir().join("multiclass");
    let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
    let sliced = Gbdt::from_model_string(&model.save_model_to_string(2, 4, 0).unwrap()).unwrap();
    assert_eq!(sliced.current_iteration(), 4);
    for (a, b) in sliced.trees().iter().zip(&model.trees()[2 * 3..6 * 3]) {
        assert_eq!(a.leaf_value, b.leaf_value);
        assert_eq!(a.threshold, b.threshold);
    }
}

#[test]
fn feature_importance_split_and_gain() {
    let case = fixtures_dir().join("binary");
    let model = Gbdt::from_model_file(case.join("model.txt")).unwrap();
    let split = model.feature_importance(-1, 0).unwrap();
    let total_splits: usize = model.trees().iter().map(|t| t.num_leaves - 1).sum();
    assert_eq!(split.iter().sum::<f64>() as usize, total_splits);
    let gain = model.feature_importance(-1, 1).unwrap();
    assert!(gain.iter().all(|g| *g >= 0.0));
    assert!(model.feature_importance(-1, 2).is_err());
}
