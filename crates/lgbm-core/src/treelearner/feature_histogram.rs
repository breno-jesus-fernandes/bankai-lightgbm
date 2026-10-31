//! Per-feature histogram split search. upstream: src/treelearner/feature_histogram.hpp
//!
//! A feature histogram is a `[grad0, hess0, grad1, hess1, ...]` slice of
//! `num_bin - offset` bins (bin 0 is omitted when it is the most frequent bin).

use super::split_info::{K_MIN_SCORE, SplitInfo};
use crate::config::Config;
use crate::io::bin::BinType;
use crate::meta::{K_EPSILON, MissingType};
use crate::utils::random::Random;

/// `kEpsilon` (a float constant) as used in double arithmetic.
pub const EPS: f64 = K_EPSILON as f64;

/// upstream: `FeatureMetainfo`
#[derive(Debug, Clone)]
pub struct FeatureMeta {
    pub num_bin: i32,
    pub missing_type: MissingType,
    pub offset: i32,
    pub default_bin: u32,
    pub monotone_type: i8,
    pub penalty: f64,
    pub bin_type: BinType,
    pub rand: Random,
}

/// Split-search hyper-parameters copied out of [`Config`].
#[derive(Debug, Clone, Copy)]
pub struct SplitParams {
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub max_delta_step: f64,
    pub path_smooth: f64,
    pub min_data_in_leaf: i32,
    pub min_sum_hessian_in_leaf: f64,
    pub min_gain_to_split: f64,
    pub extra_trees: bool,
}

impl SplitParams {
    pub fn from_config(cfg: &Config) -> Self {
        SplitParams {
            lambda_l1: cfg.lambda_l1,
            lambda_l2: cfg.lambda_l2,
            max_delta_step: cfg.max_delta_step,
            path_smooth: cfg.path_smooth,
            min_data_in_leaf: cfg.min_data_in_leaf,
            min_sum_hessian_in_leaf: cfg.min_sum_hessian_in_leaf,
            min_gain_to_split: cfg.min_gain_to_split,
            extra_trees: cfg.extra_trees,
        }
    }
}

/// upstream: `Common::Sign` (int result)
#[inline(always)]
fn sign(x: f64) -> f64 {
    f64::from(i32::from(x > 0.0) - i32::from(x < 0.0))
}

/// upstream: `Common::RoundInt` — `static_cast<int>(x + 0.5f)`
#[inline(always)]
fn round_int(x: f64) -> i32 {
    (x + 0.5) as i32
}

/// upstream: `FeatureHistogram::ThresholdL1`
#[inline(always)]
pub fn threshold_l1(s: f64, l1: f64) -> f64 {
    let reg_s = (s.abs() - l1).max(0.0);
    sign(s) * reg_s
}

/// upstream: `FeatureHistogram::CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn calculate_splitted_leaf_output<const L1: bool, const MAX_OUT: bool, const SMOOTH: bool>(
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
    smoothing: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    let mut ret = if L1 {
        -threshold_l1(sum_gradients, l1) / (sum_hessians + l2)
    } else {
        -sum_gradients / (sum_hessians + l2)
    };
    if MAX_OUT && max_delta_step > 0.0 && ret.abs() > max_delta_step {
        ret = sign(ret) * max_delta_step;
    }
    if SMOOTH {
        let n = f64::from(num_data);
        ret = ret * (n / smoothing) / (n / smoothing + 1.0) + parent_output / (n / smoothing + 1.0);
    }
    ret
}

/// upstream `CalculateSplittedLeafOutput<USE_MC=true, USE_L1=true, USE_MAX_OUTPUT=true,
/// USE_SMOOTHING=false>(..., BasicConstraint(), ...)`: the root leaf / parent output.
/// The default `BasicConstraint` is `[-DBL_MAX, DBL_MAX]`.
pub fn leaf_output_root(p: &SplitParams, sum_g: f64, sum_h: f64, num_data: i32) -> f64 {
    let ret = calculate_splitted_leaf_output::<true, true, false>(
        sum_g,
        sum_h,
        p.lambda_l1,
        p.lambda_l2,
        p.max_delta_step,
        p.path_smooth,
        num_data,
        0.0,
    );
    // NaN passes through, like upstream's comparisons.
    ret.clamp(-f64::MAX, f64::MAX)
}

/// upstream: `GetLeafGainGivenOutput<USE_L1>`
#[inline(always)]
fn leaf_gain_given_output<const L1: bool>(
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    output: f64,
) -> f64 {
    let g = if L1 {
        threshold_l1(sum_gradients, l1)
    } else {
        sum_gradients
    };
    -(2.0 * g * output + (sum_hessians + l2) * output * output)
}

/// upstream: `GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn leaf_gain<const L1: bool, const MAX_OUT: bool, const SMOOTH: bool>(
    sum_gradients: f64,
    sum_hessians: f64,
    l1: f64,
    l2: f64,
    max_delta_step: f64,
    smoothing: f64,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    if !MAX_OUT && !SMOOTH {
        if L1 {
            let sg = threshold_l1(sum_gradients, l1);
            (sg * sg) / (sum_hessians + l2)
        } else {
            (sum_gradients * sum_gradients) / (sum_hessians + l2)
        }
    } else {
        let output = calculate_splitted_leaf_output::<L1, MAX_OUT, SMOOTH>(
            sum_gradients,
            sum_hessians,
            l1,
            l2,
            max_delta_step,
            smoothing,
            num_data,
            parent_output,
        );
        leaf_gain_given_output::<L1>(sum_gradients, sum_hessians, l1, l2, output)
    }
}

/// Result of a threshold search on one feature histogram.
pub struct SearchResult {
    pub is_splittable: bool,
}

/// upstream: `FeatureHistogram::FindBestThreshold` for numerical features.
/// Returns whether the feature is splittable at this leaf.
#[allow(clippy::too_many_arguments)]
pub fn find_best_threshold_numerical(
    hist: &[f64],
    meta: &mut FeatureMeta,
    p: &SplitParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
    output: &mut SplitInfo,
) -> bool {
    output.default_left = true;
    output.gain = K_MIN_SCORE;
    let sum_hessian = sum_hessian + 2.0 * EPS;
    let use_l1 = p.lambda_l1 > 0.0;
    let use_max = p.max_delta_step > 0.0;
    let use_smooth = p.path_smooth > EPS;
    macro_rules! go {
        ($r:literal, $l1:literal, $mx:literal, $sm:literal) => {
            numerical::<$r, $l1, $mx, $sm>(
                hist,
                meta,
                p,
                sum_gradient,
                sum_hessian,
                num_data,
                parent_output,
                output,
            )
        };
    }
    let splittable = match (p.extra_trees, use_l1, use_max, use_smooth) {
        (false, false, false, false) => go!(false, false, false, false),
        (false, false, false, true) => go!(false, false, false, true),
        (false, false, true, false) => go!(false, false, true, false),
        (false, false, true, true) => go!(false, false, true, true),
        (false, true, false, false) => go!(false, true, false, false),
        (false, true, false, true) => go!(false, true, false, true),
        (false, true, true, false) => go!(false, true, true, false),
        (false, true, true, true) => go!(false, true, true, true),
        (true, false, false, false) => go!(true, false, false, false),
        (true, false, false, true) => go!(true, false, false, true),
        (true, false, true, false) => go!(true, false, true, false),
        (true, false, true, true) => go!(true, false, true, true),
        (true, true, false, false) => go!(true, true, false, false),
        (true, true, false, true) => go!(true, true, false, true),
        (true, true, true, false) => go!(true, true, true, false),
        (true, true, true, true) => go!(true, true, true, true),
    };
    output.gain *= meta.penalty;
    splittable
}

/// upstream: the `find_best_threshold_fun_` lambdas built by `FuncForNumricalL3`.
#[allow(clippy::too_many_arguments)]
fn numerical<const RAND: bool, const L1: bool, const MAX_OUT: bool, const SMOOTH: bool>(
    hist: &[f64],
    meta: &mut FeatureMeta,
    p: &SplitParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
    output: &mut SplitInfo,
) -> bool {
    // upstream: BeforeNumerical
    output.monotone_type = meta.monotone_type;
    let gain_shift = leaf_gain::<L1, MAX_OUT, SMOOTH>(
        sum_gradient,
        sum_hessian,
        p.lambda_l1,
        p.lambda_l2,
        p.max_delta_step,
        p.path_smooth,
        num_data,
        parent_output,
    );
    let mut rand_threshold = 0;
    if RAND && meta.num_bin - 2 > 0 {
        rand_threshold = meta.rand.next_int(0, meta.num_bin - 2);
    }
    let min_gain_shift = gain_shift + p.min_gain_to_split;
    let mut splittable = false;
    let args = (
        sum_gradient,
        sum_hessian,
        num_data,
        min_gain_shift,
        rand_threshold,
        parent_output,
    );
    macro_rules! seq {
        ($rev:literal, $skip:literal, $na:literal) => {
            splittable |= sequential::<RAND, L1, MAX_OUT, SMOOTH, $rev, $skip, $na>(
                hist, meta, p, args, output,
            )
        };
    }
    if meta.num_bin > 2 && meta.missing_type != MissingType::None {
        if meta.missing_type == MissingType::Zero {
            seq!(true, true, false);
            seq!(false, true, false);
        } else {
            seq!(true, false, true);
            seq!(false, false, true);
        }
    } else {
        seq!(true, false, false);
        if meta.missing_type == MissingType::NaN {
            output.default_left = false;
        }
    }
    splittable
}

/// upstream: `FeatureHistogram::FindBestThresholdSequentially` (without monotone constraints).
/// Returns whether any candidate passed the gain/data constraints.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sequential<
    const RAND: bool,
    const L1: bool,
    const MAX_OUT: bool,
    const SMOOTH: bool,
    const REVERSE: bool,
    const SKIP_DEFAULT_BIN: bool,
    const NA_AS_MISSING: bool,
>(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    (sum_gradient, sum_hessian, num_data, min_gain_shift, rand_threshold, parent_output): (
        f64,
        f64,
        i32,
        f64,
        i32,
        f64,
    ),
    output: &mut SplitInfo,
) -> bool {
    let offset = meta.offset;
    let mut is_splittable = false;
    let mut best_sum_left_gradient = f64::NAN;
    let mut best_sum_left_hessian = f64::NAN;
    let mut best_gain = K_MIN_SCORE;
    let mut best_left_count: i32 = 0;
    let mut best_threshold = meta.num_bin as u32;
    let cnt_factor = f64::from(num_data) / sum_hessian;
    let grad = |t: i32| hist[(t as usize) << 1];
    let hess = |t: i32| hist[((t as usize) << 1) + 1];
    let gains = |slg: f64, slh: f64, srg: f64, srh: f64, lc: i32, rc: i32| {
        leaf_gain::<L1, MAX_OUT, SMOOTH>(
            slg,
            slh,
            p.lambda_l1,
            p.lambda_l2,
            p.max_delta_step,
            p.path_smooth,
            lc,
            parent_output,
        ) + leaf_gain::<L1, MAX_OUT, SMOOTH>(
            srg,
            srh,
            p.lambda_l1,
            p.lambda_l2,
            p.max_delta_step,
            p.path_smooth,
            rc,
            parent_output,
        )
    };

    if REVERSE {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = EPS;
        let mut right_count: i32 = 0;
        let mut t = meta.num_bin - 1 - offset - i32::from(NA_AS_MISSING);
        let t_end = 1 - offset;
        while t >= t_end {
            if SKIP_DEFAULT_BIN && (t + offset) == meta.default_bin as i32 {
                t -= 1;
                continue;
            }
            let g = grad(t);
            let h = hess(t);
            let cnt = round_int(h * cnt_factor);
            sum_right_gradient += g;
            sum_right_hessian += h;
            right_count += cnt;
            if right_count < p.min_data_in_leaf || sum_right_hessian < p.min_sum_hessian_in_leaf {
                t -= 1;
                continue;
            }
            let left_count = num_data - right_count;
            if left_count < p.min_data_in_leaf {
                break;
            }
            let sum_left_hessian = sum_hessian - sum_right_hessian;
            if sum_left_hessian < p.min_sum_hessian_in_leaf {
                break;
            }
            let sum_left_gradient = sum_gradient - sum_right_gradient;
            if RAND && t - 1 + offset != rand_threshold {
                t -= 1;
                continue;
            }
            let current_gain = gains(
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                left_count,
                right_count,
            );
            if current_gain <= min_gain_shift {
                t -= 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_left_count = left_count;
                best_sum_left_gradient = sum_left_gradient;
                best_sum_left_hessian = sum_left_hessian;
                best_threshold = (t - 1 + offset) as u32;
                best_gain = current_gain;
            }
            t -= 1;
        }
    } else {
        let mut sum_left_gradient = 0.0f64;
        let mut sum_left_hessian = EPS;
        let mut left_count: i32 = 0;
        let mut t: i32 = 0;
        let t_end = meta.num_bin - 2 - offset;
        if NA_AS_MISSING && offset == 1 {
            sum_left_gradient = sum_gradient;
            sum_left_hessian = sum_hessian - EPS;
            left_count = num_data;
            for i in 0..meta.num_bin - offset {
                let g = grad(i);
                let h = hess(i);
                let cnt = round_int(h * cnt_factor);
                sum_left_gradient -= g;
                sum_left_hessian -= h;
                left_count -= cnt;
            }
            t = -1;
        }
        while t <= t_end {
            if SKIP_DEFAULT_BIN && (t + offset) == meta.default_bin as i32 {
                t += 1;
                continue;
            }
            if t >= 0 {
                sum_left_gradient += grad(t);
                sum_left_hessian += hess(t);
                left_count += round_int(hess(t) * cnt_factor);
            }
            if left_count < p.min_data_in_leaf || sum_left_hessian < p.min_sum_hessian_in_leaf {
                t += 1;
                continue;
            }
            let right_count = num_data - left_count;
            if right_count < p.min_data_in_leaf {
                break;
            }
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            if sum_right_hessian < p.min_sum_hessian_in_leaf {
                break;
            }
            let sum_right_gradient = sum_gradient - sum_left_gradient;
            if RAND && t + offset != rand_threshold {
                t += 1;
                continue;
            }
            let current_gain = gains(
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                left_count,
                right_count,
            );
            if current_gain <= min_gain_shift {
                t += 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_left_count = left_count;
                best_sum_left_gradient = sum_left_gradient;
                best_sum_left_hessian = sum_left_hessian;
                best_threshold = (t + offset) as u32;
                best_gain = current_gain;
            }
            t += 1;
        }
    }

    // upstream reads the member `is_splittable_`, which may already be true from the
    // other direction's scan; the caller ORs the results, and we check both here.
    if is_splittable && best_gain > output.gain + min_gain_shift {
        let out = |sg: f64, sh: f64, n: i32| {
            calculate_splitted_leaf_output::<L1, MAX_OUT, SMOOTH>(
                sg,
                sh,
                p.lambda_l1,
                p.lambda_l2,
                p.max_delta_step,
                p.path_smooth,
                n,
                parent_output,
            )
        };
        output.threshold = best_threshold;
        output.left_output = out(
            best_sum_left_gradient,
            best_sum_left_hessian,
            best_left_count,
        );
        output.left_count = best_left_count;
        output.left_sum_gradient = best_sum_left_gradient;
        output.left_sum_hessian = best_sum_left_hessian - EPS;
        output.right_output = out(
            sum_gradient - best_sum_left_gradient,
            sum_hessian - best_sum_left_hessian,
            num_data - best_left_count,
        );
        output.right_count = num_data - best_left_count;
        output.right_sum_gradient = sum_gradient - best_sum_left_gradient;
        output.right_sum_hessian = sum_hessian - best_sum_left_hessian - EPS;
        output.gain = best_gain - min_gain_shift;
        output.default_left = REVERSE;
    }
    is_splittable
}
