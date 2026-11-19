//! Integer (quantized) numerical split search parity with a brute-force reference (T2.7).
//!
//! upstream: src/treelearner/feature_histogram.hpp (v4.7.0)
//! `FindBestThresholdSequentiallyInt` / `BeforeNumericalInt`. The integer path accumulates
//! exact integer gradient/Hessian prefix sums, estimates the row count of a side from its
//! integer Hessian sum (`cnt_factor = num_data / total_int_hessian`), and recovers the double
//! sums by multiplying once by `grad_scale` / `hess_scale`. `kEpsilon` is added to the child
//! Hessians only inside the gain, never to the stored sums or the leaf-total gain shift.
//!
//! The reference below recomputes the best split independently for the L2-only path (no L1,
//! max-delta-step, smoothing or monotone constraints, no missing bin), so it exercises the
//! loop, the count estimation and the epsilon placement without reusing the implementation.

use lgbm_core::io::bin::BinType;
use lgbm_core::meta::{K_EPSILON, MissingType};
use lgbm_core::treelearner::feature_histogram::{
    FeatureMeta, Mc, SplitParams, find_best_threshold_numerical_int,
};
use lgbm_core::treelearner::split_info::{K_MIN_SCORE, SplitInfo};
use lgbm_core::utils::random::Random;

const EPS: f64 = K_EPSILON as f64;

fn meta(num_bin: i32) -> FeatureMeta {
    FeatureMeta {
        num_bin,
        missing_type: MissingType::None,
        offset: 0,
        default_bin: 0,
        monotone_type: 0,
        penalty: 1.0,
        bin_type: BinType::Numerical,
        rand: Random::new(0),
    }
}

fn params(l2: f64, min_data: i32) -> SplitParams {
    SplitParams {
        lambda_l1: 0.0,
        lambda_l2: l2,
        max_delta_step: 0.0,
        path_smooth: 0.0,
        min_data_in_leaf: min_data,
        min_sum_hessian_in_leaf: 0.0,
        min_gain_to_split: 0.0,
        extra_trees: false,
    }
}

/// Best numerical split for the L2-only path, computed independently of the implementation.
struct Reference {
    is_splittable: bool,
    threshold: u32,
    left_count: i32,
    right_count: i32,
    left_output: f64,
    right_output: f64,
    gain: f64,
}

#[allow(clippy::too_many_arguments)]
fn reference(
    hist: &[(i64, i64)],
    p: &SplitParams,
    int_total_grad: i64,
    int_total_hess: i64,
    grad_scale: f64,
    hess_scale: f64,
    num_data: i32,
) -> Reference {
    let l2 = p.lambda_l2;
    let cnt_factor = f64::from(num_data) / int_total_hess as f64;
    let leaf_output = |g: f64, h: f64| -g / (h + l2);
    let leaf_gain = |g: f64, h: f64| {
        let o = leaf_output(g, h);
        -(2.0 * g * o + (h + l2) * o * o)
    };
    let total_g = int_total_grad as f64 * grad_scale;
    let total_h = int_total_hess as f64 * hess_scale;
    let min_gain_shift = leaf_gain(total_g, total_h) + p.min_gain_to_split;

    let num_bin = hist.len() as i32;
    let mut best = Reference {
        is_splittable: false,
        threshold: num_bin as u32,
        left_count: 0,
        right_count: 0,
        left_output: 0.0,
        right_output: 0.0,
        gain: K_MIN_SCORE,
    };
    // Left is bins [0, s), right is bins [s, num_bin); threshold is s - 1.
    for s in 1..num_bin {
        let mut r_grad = 0i64;
        let mut r_hess = 0i64;
        for bin in &hist[s as usize..] {
            r_grad += bin.0;
            r_hess += bin.1;
        }
        let l_grad = int_total_grad - r_grad;
        let l_hess = int_total_hess - r_hess;
        let right_count = (r_hess as f64 * cnt_factor + 0.5) as i32;
        let left_count = num_data - right_count;
        let sum_right_hessian = r_hess as f64 * hess_scale;
        let sum_left_hessian = l_hess as f64 * hess_scale;
        if right_count < p.min_data_in_leaf || sum_right_hessian < p.min_sum_hessian_in_leaf {
            continue;
        }
        if left_count < p.min_data_in_leaf || sum_left_hessian < p.min_sum_hessian_in_leaf {
            continue;
        }
        let sum_left_gradient = l_grad as f64 * grad_scale;
        let sum_right_gradient = r_grad as f64 * grad_scale;
        let gain = leaf_gain(sum_left_gradient, sum_left_hessian + EPS)
            + leaf_gain(sum_right_gradient, sum_right_hessian + EPS);
        if gain <= min_gain_shift {
            continue;
        }
        best.is_splittable = true;
        if gain > best.gain {
            // Tail counts are recomputed from the integer sums, like upstream.
            let left_count_tail = (l_hess as f64 * cnt_factor + 0.5) as i32;
            let right_count_tail = (r_hess as f64 * cnt_factor + 0.5) as i32;
            best.gain = gain;
            best.threshold = (s - 1) as u32;
            best.left_count = left_count_tail;
            best.right_count = right_count_tail;
            best.left_output = leaf_output(sum_left_gradient, sum_left_hessian);
            best.right_output = leaf_output(sum_right_gradient, sum_right_hessian);
        }
    }
    if best.is_splittable {
        best.gain -= min_gain_shift;
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn check(hist: &[(i64, i64)], grad_scale: f64, hess_scale: f64, l2: f64, min_data: i32) {
    let int_total_grad: i64 = hist.iter().map(|b| b.0).sum();
    let int_total_hess: i64 = hist.iter().map(|b| b.1).sum();
    let num_data = int_total_hess as i32; // constant-ish: row count tracked via Hessian sum
    let p = params(l2, min_data);
    let mut m = meta(hist.len() as i32);
    let mut out = SplitInfo::default();
    let splittable = find_best_threshold_numerical_int(
        hist,
        &mut m,
        &p,
        int_total_grad,
        int_total_hess,
        grad_scale,
        hess_scale,
        num_data,
        0.0,
        Mc::Off,
        &mut out,
    );
    let r = reference(
        hist,
        &p,
        int_total_grad,
        int_total_hess,
        grad_scale,
        hess_scale,
        num_data,
    );
    assert_eq!(splittable, r.is_splittable, "splittable mismatch");
    if !r.is_splittable {
        return;
    }
    assert_eq!(out.threshold, r.threshold, "threshold mismatch");
    assert_eq!(out.left_count, r.left_count, "left_count mismatch");
    assert_eq!(out.right_count, r.right_count, "right_count mismatch");
    assert!(
        (out.gain - r.gain).abs() <= 1e-12 * r.gain.abs().max(1.0),
        "gain mismatch: {} vs {}",
        out.gain,
        r.gain
    );
    assert!(
        (out.left_output - r.left_output).abs() <= 1e-12,
        "left_output mismatch: {} vs {}",
        out.left_output,
        r.left_output
    );
    assert!(
        (out.right_output - r.right_output).abs() <= 1e-12,
        "right_output mismatch: {} vs {}",
        out.right_output,
        r.right_output
    );
}

#[test]
fn constant_hessian_counts_are_exact() {
    // Hessian 1 per row: the integer Hessian sum is the exact row count (cnt_factor == 1).
    let hist = [(3i64, 10i64), (-5, 20), (8, 15), (-2, 25), (6, 30)];
    check(&hist, 1.0, 1.0, 0.0, 1);
    check(&hist, 0.5, 1.0, 0.1, 5);
}

#[test]
fn variable_hessian_estimates_counts() {
    // Hessian varies per bin, so the row count of a side is estimated from its Hessian sum.
    let hist = [
        (12i64, 7i64),
        (-9, 31),
        (4, 12),
        (-15, 44),
        (20, 19),
        (-1, 8),
    ];
    check(&hist, 0.25, 0.125, 0.0, 1);
    check(&hist, 2.0, 0.5, 1.0, 10);
}

#[test]
fn no_valid_split_is_not_splittable() {
    // min_data_in_leaf larger than any achievable side leaves the feature unsplittable.
    let hist = [(3i64, 4i64), (-5, 5), (8, 6)];
    check(&hist, 1.0, 1.0, 0.0, 100);
}
