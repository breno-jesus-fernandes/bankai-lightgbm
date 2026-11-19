//! Oracle: scripts/gen_quantized_grad_fixtures.py, lightgbm==4.7.0.
//! This tests the discretization foundation, not quantized split selection.
#![allow(clippy::unwrap_used)]

use lgbm_core::treelearner::gradient_discretizer::{GradientDiscretizer, histogram_bits};
use serde_json::Value;

struct Case {
    constant_hessian: bool,
    bins: u8,
    gradients: Vec<f32>,
    hessians: Vec<f32>,
    left_rows: Vec<usize>,
    left_output: f64,
    right_output: f64,
    left_weight: f64,
    right_weight: f64,
    split_gain: f64,
}

impl Case {
    fn from_json(v: &Value) -> Self {
        let floats = |name: &str| {
            v[name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        Self {
            constant_hessian: v["constant_hessian"].as_bool().unwrap(),
            bins: v["bins"].as_u64().unwrap() as u8,
            gradients: floats("gradients"),
            hessians: floats("hessians"),
            left_rows: v["left_rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect(),
            left_output: v["left_output"].as_f64().unwrap(),
            right_output: v["right_output"].as_f64().unwrap(),
            left_weight: v["left_weight"].as_f64().unwrap(),
            right_weight: v["right_weight"].as_f64().unwrap(),
            split_gain: v["split_gain"].as_f64().unwrap(),
        }
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn quantized_nearest_statistics_match_upstream_leaves() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/quantized_gradients.json"
    ))
    .unwrap();
    assert_eq!(f["upstream"], "4.7.0");
    assert_eq!(f["cases"].as_array().unwrap().len(), 16);
    for value in f["cases"].as_array().unwrap() {
        let c = Case::from_json(value);
        let mut q =
            GradientDiscretizer::new(c.gradients.len(), c.bins, c.constant_hessian).unwrap();
        q.discretize_nearest(&c.gradients, &c.hessians).unwrap();
        let mut left = (0i64, 0i64);
        let mut right = (0i64, 0i64);
        for (row, pair) in q.packed_gradients_hessians().iter().enumerate() {
            let sum = if c.left_rows.contains(&row) {
                &mut left
            } else {
                &mut right
            };
            sum.0 += i64::from((pair >> 8) as i8);
            sum.1 += i64::from(pair & 0xff);
        }
        let (lg, lh) = (
            left.0 as f64 * q.gradient_scale(),
            left.1 as f64 * q.hessian_scale(),
        );
        let (rg, rh) = (
            right.0 as f64 * q.gradient_scale(),
            right.1 as f64 * q.hessian_scale(),
        );
        close(lh, c.left_weight);
        close(rh, c.right_weight);
        close(-lg / lh, c.left_output);
        close(-rg / rh, c.right_output);
        let gain = lg * lg / lh + rg * rg / rh - (lg + rg).powi(2) / (lh + rh);
        // Tree::Split stores the gain as score_t (f32), also in the JSON oracle.
        close(f64::from(gain as f32), c.split_gain);
    }
}

#[test]
fn quantized_histogram_width_boundaries_match_upstream() {
    assert_eq!(histogram_bits(63, 4), 8);
    assert_eq!(histogram_bits(64, 4), 16);
    assert_eq!(histogram_bits(16383, 4), 16);
    assert_eq!(histogram_bits(16384, 4), 32);
    assert_eq!(histogram_bits(i32::MAX as usize, 127), 32);
}

#[test]
fn quantized_nearest_reuses_buffer_and_recomputes_scales() {
    let mut q = GradientDiscretizer::new(5, 4, false).unwrap();
    q.discretize_nearest(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[1.0; 5])
        .unwrap();
    let address = q.packed_gradients_hessians().as_ptr();
    assert_eq!(q.packed_gradients_hessians(), &[-508, -252, 4, 260, 516]);
    assert_eq!(q.gradient_scale(), 1.0);
    assert_eq!(q.hessian_scale(), 0.25);
    q.discretize_nearest(&[-4.0, -1.0, 0.0, 1.0, 4.0], &[2.0; 5])
        .unwrap();
    assert_eq!(q.packed_gradients_hessians().as_ptr(), address);
    assert_eq!(q.packed_gradients_hessians(), &[-508, -252, 4, 260, 516]);
    assert_eq!(q.gradient_scale(), 2.0);
    assert_eq!(q.hessian_scale(), 0.5);
}

#[test]
fn quantized_nearest_checks_shape_and_representable_bins() {
    assert!(GradientDiscretizer::new(0, 4, false).is_err());
    for bins in [0, 1, 128, 255] {
        assert!(GradientDiscretizer::new(2, bins, false).is_err());
    }
    let mut q = GradientDiscretizer::new(2, 4, false).unwrap();
    assert!(q.discretize_nearest(&[1.0], &[1.0; 2]).is_err());
    assert!(q.discretize_nearest(&[1.0; 2], &[1.0]).is_err());
    assert!(q.discretize_nearest(&[f32::NAN; 2], &[1.0; 2]).is_err());
    assert!(q.discretize_nearest(&[1.0; 2], &[-1.0; 2]).is_err());
}

#[test]
fn quantized_zero_maxima_refuse_undefined_upstream_casts() {
    let mut q = GradientDiscretizer::new(2, 4, false).unwrap();
    q.discretize_nearest(&[-2.0, 2.0], &[1.0; 2]).unwrap();
    let previous = q.packed_gradients_hessians().to_vec();
    assert!(q.discretize_nearest(&[0.0; 2], &[1.0; 2]).is_err());
    assert!(q.discretize_nearest(&[-2.0, 2.0], &[0.0; 2]).is_err());
    assert_eq!(q.packed_gradients_hessians(), previous);
    assert_eq!(q.gradient_scale(), 1.0);
    assert_eq!(q.hessian_scale(), 0.25);
}

#[test]
fn quantized_constant_zero_hessians_do_not_need_an_integer_cast() {
    let mut q = GradientDiscretizer::new(2, 4, true).unwrap();
    q.discretize_nearest(&[-2.0, 2.0], &[0.0; 2]).unwrap();
    assert_eq!(q.packed_gradients_hessians(), &[-511, 513]);
    assert_eq!(q.gradient_scale(), 1.0);
    assert_eq!(q.hessian_scale(), 0.0);
}
