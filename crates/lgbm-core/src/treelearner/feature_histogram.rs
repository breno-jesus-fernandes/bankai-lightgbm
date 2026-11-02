//! Per-feature histogram split search. upstream: src/treelearner/feature_histogram.hpp
//!
//! A feature histogram is a `[grad0, hess0, grad1, hess1, ...]` slice of
//! `num_bin - offset` bins (bin 0 is omitted when it is the most frequent bin).

use super::split_info::{K_MIN_SCORE, SplitInfo};
use crate::config::Config;
use crate::io::bin::BinType;
use crate::meta::{K_EPSILON, MissingType};
use crate::utils::common::madd;
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
    // upstream: `-(2.0 * g * output + (sum_hessians + l2) * output * output)`. Upstream builds
    // on arm64 contract the add into a fused multiply-add (see `madd`), which can decide a
    // near-tie between the two default directions of a missing value.
    -madd(2.0 * g, output, (sum_hessians + l2) * output * output)
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
    mc: Option<(f64, f64)>,
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
                mc,
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
    mc: Option<(f64, f64)>,
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
                hist, meta, p, args, mc, output,
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
    mc: Option<(f64, f64)>,
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
    let mono = meta.monotone_type;
    let mc_invalid = mc.is_some_and(|(lo, hi)| lo > hi);
    let gains = |slg: f64, slh: f64, srg: f64, srh: f64, lc: i32, rc: i32| {
        if let Some((mn, mx)) = mc {
            // upstream: GetSplitGains<USE_MC>: clamped outputs, monotone direction check.
            let lo = clamp_to(
                calculate_splitted_leaf_output::<L1, MAX_OUT, SMOOTH>(
                    slg,
                    slh,
                    p.lambda_l1,
                    p.lambda_l2,
                    p.max_delta_step,
                    p.path_smooth,
                    lc,
                    parent_output,
                ),
                mn,
                mx,
            );
            let ro = clamp_to(
                calculate_splitted_leaf_output::<L1, MAX_OUT, SMOOTH>(
                    srg,
                    srh,
                    p.lambda_l1,
                    p.lambda_l2,
                    p.max_delta_step,
                    p.path_smooth,
                    rc,
                    parent_output,
                ),
                mn,
                mx,
            );
            if (mono > 0 && lo > ro) || (mono < 0 && lo < ro) {
                return 0.0;
            }
            return leaf_gain_given_output::<L1>(slg, slh, p.lambda_l1, p.lambda_l2, lo)
                + leaf_gain_given_output::<L1>(srg, srh, p.lambda_l1, p.lambda_l2, ro);
        }
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
            if current_gain > best_gain && !mc_invalid {
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
            if current_gain > best_gain && !mc_invalid {
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
            let v = calculate_splitted_leaf_output::<L1, MAX_OUT, SMOOTH>(
                sg,
                sh,
                p.lambda_l1,
                p.lambda_l2,
                p.max_delta_step,
                p.path_smooth,
                n,
                parent_output,
            );
            clamp_opt(v, mc)
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

/// Categorical split parameters (upstream: `Config::cat_*`, `max_cat_*`, `min_data_per_group`).
#[derive(Debug, Clone, Copy)]
pub struct CatParams {
    pub cat_smooth: f64,
    pub cat_l2: f64,
    pub max_cat_threshold: i32,
    pub max_cat_to_onehot: i32,
    pub min_data_per_group: i32,
}

impl CatParams {
    pub fn from_config(cfg: &crate::config::Config) -> Self {
        CatParams {
            cat_smooth: cfg.cat_smooth,
            cat_l2: cfg.cat_l2,
            max_cat_threshold: cfg.max_cat_threshold,
            max_cat_to_onehot: cfg.max_cat_to_onehot,
            min_data_per_group: cfg.min_data_per_group,
        }
    }
}

/// upstream: `CalculateSplittedLeafOutput<USE_MC>` clamps to the leaf's bounds.
#[inline(always)]
fn clamp_to(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline(always)]
fn clamp_opt(v: f64, mc: Option<(f64, f64)>) -> f64 {
    match mc {
        Some((lo, hi)) => clamp_to(v, lo, hi),
        None => v,
    }
}

/// Dispatch a const-generic gain function on the runtime `(l1, max_delta_step, smoothing)`
/// switches, exactly as upstream's `FuncForCategoricalL2` does.
macro_rules! by_flags {
    ($l1:expr, $mx:expr, $sm:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match ($l1, $mx, $sm) {
            (false, false, false) => $f::<false, false, false>($($arg),*),
            (false, false, true) => $f::<false, false, true>($($arg),*),
            (false, true, false) => $f::<false, true, false>($($arg),*),
            (false, true, true) => $f::<false, true, true>($($arg),*),
            (true, false, false) => $f::<true, false, false>($($arg),*),
            (true, false, true) => $f::<true, false, true>($($arg),*),
            (true, true, false) => $f::<true, true, false>($($arg),*),
            (true, true, true) => $f::<true, true, true>($($arg),*),
        }
    };
}

/// upstream: `GetLeafGain<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`, runtime flags.
#[allow(clippy::too_many_arguments)]
fn leaf_gain_rt(
    sg: f64,
    sh: f64,
    p: &SplitParams,
    smoothing: f64,
    num_data: i32,
    parent_output: f64,
    flags: (bool, bool, bool),
) -> f64 {
    by_flags!(
        flags.0,
        flags.1,
        flags.2,
        leaf_gain(
            sg,
            sh,
            p.lambda_l1,
            p.lambda_l2,
            p.max_delta_step,
            smoothing,
            num_data,
            parent_output
        )
    )
}

/// upstream: `CalculateSplittedLeafOutput<false, USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`
/// with an explicit `l2` (categorical searches use `lambda_l2 + cat_l2` when sorting).
#[allow(clippy::too_many_arguments)]
fn output_rt(
    sg: f64,
    sh: f64,
    p: &SplitParams,
    l2: f64,
    smoothing: f64,
    num_data: i32,
    parent_output: f64,
    flags: (bool, bool, bool),
) -> f64 {
    by_flags!(
        flags.0,
        flags.1,
        flags.2,
        calculate_splitted_leaf_output(
            sg,
            sh,
            p.lambda_l1,
            l2,
            p.max_delta_step,
            smoothing,
            num_data,
            parent_output
        )
    )
}

/// upstream: `GetSplitGains`. With bounds (`mc`), outputs are clamped to the leaf's
/// constraint and the gain is taken from the clamped outputs.
#[allow(clippy::too_many_arguments)]
fn split_gains_rt(
    sl_g: f64,
    sl_h: f64,
    sr_g: f64,
    sr_h: f64,
    p: &SplitParams,
    l2: f64,
    lc: i32,
    rc: i32,
    parent_output: f64,
    flags: (bool, bool, bool),
    mc: Option<(f64, f64)>,
) -> f64 {
    if mc.is_some() {
        let lp = SplitParams {
            lambda_l2: l2,
            ..*p
        };
        let lo = clamp_opt(
            output_rt(sl_g, sl_h, &lp, l2, p.path_smooth, lc, parent_output, flags),
            mc,
        );
        let ro = clamp_opt(
            output_rt(sr_g, sr_h, &lp, l2, p.path_smooth, rc, parent_output, flags),
            mc,
        );
        let l1 = flags.0;
        return if l1 {
            leaf_gain_given_output::<true>(sl_g, sl_h, p.lambda_l1, l2, lo)
                + leaf_gain_given_output::<true>(sr_g, sr_h, p.lambda_l1, l2, ro)
        } else {
            leaf_gain_given_output::<false>(sl_g, sl_h, p.lambda_l1, l2, lo)
                + leaf_gain_given_output::<false>(sr_g, sr_h, p.lambda_l1, l2, ro)
        };
    }
    let smooth = p.path_smooth;
    leaf_gain_rt(
        sl_g,
        sl_h,
        &SplitParams {
            lambda_l2: l2,
            ..*p
        },
        smooth,
        lc,
        parent_output,
        flags,
    ) + leaf_gain_rt(
        sr_g,
        sr_h,
        &SplitParams {
            lambda_l2: l2,
            ..*p
        },
        smooth,
        rc,
        parent_output,
        flags,
    )
}

/// upstream: `FeatureHistogram::FindBestThresholdCategoricalInner` (no monotone constraints,
/// no quantized gradients). Returns whether the feature is splittable at this leaf.
#[allow(clippy::too_many_arguments)]
pub fn find_best_threshold_categorical(
    hist: &[f64],
    meta: &mut FeatureMeta,
    p: &SplitParams,
    c: &CatParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
    mc: Option<(f64, f64)>,
    output: &mut SplitInfo,
) -> bool {
    // upstream: FindBestThreshold adds 2 * kEpsilon before dispatching
    let sum_hessian = sum_hessian + 2.0 * EPS;
    output.default_left = false;
    output.gain = K_MIN_SCORE;
    let use_l1 = p.lambda_l1 > 0.0;
    let use_max = p.max_delta_step > 0.0;
    let use_smooth = p.path_smooth > EPS;
    let flags = (use_l1, use_max, use_smooth);
    let rand = p.extra_trees;

    let gain_shift = if use_smooth {
        if use_l1 {
            leaf_gain_given_output::<true>(
                sum_gradient,
                sum_hessian,
                p.lambda_l1,
                p.lambda_l2,
                parent_output,
            )
        } else {
            leaf_gain_given_output::<false>(
                sum_gradient,
                sum_hessian,
                p.lambda_l1,
                p.lambda_l2,
                parent_output,
            )
        }
    } else {
        // upstream: no smoothing keeps `lambda_l2` for the shift and `parent_output = 0`.
        leaf_gain_rt(
            sum_gradient,
            sum_hessian,
            p,
            0.0,
            num_data,
            0.0,
            (use_l1, use_max, false),
        )
    };
    let min_gain_shift = gain_shift + p.min_gain_to_split;

    let offset = meta.offset;
    let bin_start = 1 - offset;
    let bin_end = meta.num_bin - offset;
    let grad = |t: i32| hist[(t as usize) << 1];
    let hess = |t: i32| hist[((t as usize) << 1) + 1];
    let cnt_factor = f64::from(num_data) / sum_hessian;
    let use_onehot = meta.num_bin <= c.max_cat_to_onehot;
    let mut l2 = p.lambda_l2;
    let mut is_splittable = false;
    let mut best_gain = K_MIN_SCORE;
    let mut best_threshold: i32 = -1;
    let mut best_dir: i32 = 1;
    let mut best_left_count: i32 = 0;
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    let mut rand_threshold: i32 = 0;
    let mut used_bin: i32 = -1;
    let mut sorted_idx: Vec<i32> = Vec::new();

    if use_onehot {
        if rand && bin_end - bin_start > 0 {
            rand_threshold = meta.rand.next_int(bin_start, bin_end);
        }
        for t in bin_start..bin_end {
            let (g, h) = (grad(t), hess(t));
            let cnt = round_int(h * cnt_factor);
            // data not enough, or sum hessian too small
            if cnt < p.min_data_in_leaf || h < p.min_sum_hessian_in_leaf {
                continue;
            }
            let other_count = num_data - cnt;
            if other_count < p.min_data_in_leaf {
                continue;
            }
            let sum_other_hessian = sum_hessian - h - EPS;
            if sum_other_hessian < p.min_sum_hessian_in_leaf {
                continue;
            }
            let sum_other_gradient = sum_gradient - g;
            if rand && t != rand_threshold {
                continue;
            }
            let current_gain = split_gains_rt(
                sum_other_gradient,
                sum_other_hessian,
                g,
                h + EPS,
                p,
                l2,
                other_count,
                cnt,
                parent_output,
                flags,
                mc,
            );
            if current_gain <= min_gain_shift {
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_threshold = t;
                best_sum_left_gradient = g;
                best_sum_left_hessian = h + EPS;
                best_left_count = cnt;
                best_gain = current_gain;
            }
        }
    } else {
        for i in bin_start..bin_end {
            if f64::from(round_int(hess(i) * cnt_factor)) >= c.cat_smooth {
                sorted_idx.push(i);
            }
        }
        used_bin = sorted_idx.len() as i32;
        l2 += c.cat_l2;
        let ctr = |sg: f64, sh: f64| sg / (sh + c.cat_smooth);
        sorted_idx.sort_by(|&i, &j| {
            let (a, b) = (ctr(grad(i), hess(i)), ctr(grad(j), hess(j)));
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let directions = [1i32, -1];
        let start_positions = [0i32, used_bin - 1];
        let max_num_cat = c.max_cat_threshold.min((used_bin + 1) / 2);
        let max_threshold = (max_num_cat.min(used_bin) - 1).max(0);
        if rand && max_threshold > 0 {
            rand_threshold = meta.rand.next_int(0, max_threshold);
        }
        for (dir, start) in directions.iter().zip(start_positions) {
            let mut start_pos = start;
            let min_data_per_group = c.min_data_per_group;
            let mut cnt_cur_group = 0i32;
            let mut sum_left_gradient = 0.0f64;
            let mut sum_left_hessian = EPS;
            let mut left_count = 0i32;
            let mut i = 0i32;
            while i < used_bin && i < max_num_cat {
                let t = sorted_idx[start_pos as usize];
                start_pos += dir;
                let (g, h) = (grad(t), hess(t));
                let cnt = round_int(h * cnt_factor);
                sum_left_gradient += g;
                sum_left_hessian += h;
                left_count += cnt;
                cnt_cur_group += cnt;
                let skip =
                    left_count < p.min_data_in_leaf || sum_left_hessian < p.min_sum_hessian_in_leaf;
                if skip {
                    i += 1;
                    continue;
                }
                let right_count = num_data - left_count;
                if right_count < p.min_data_in_leaf || right_count < min_data_per_group {
                    break;
                }
                let sum_right_hessian = sum_hessian - sum_left_hessian;
                if sum_right_hessian < p.min_sum_hessian_in_leaf {
                    break;
                }
                if cnt_cur_group < min_data_per_group {
                    i += 1;
                    continue;
                }
                cnt_cur_group = 0;
                let sum_right_gradient = sum_gradient - sum_left_gradient;
                if rand && i != rand_threshold {
                    i += 1;
                    continue;
                }
                let current_gain = split_gains_rt(
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    p,
                    l2,
                    left_count,
                    right_count,
                    parent_output,
                    flags,
                    mc,
                );
                if current_gain > min_gain_shift {
                    is_splittable = true;
                    if current_gain > best_gain {
                        best_left_count = left_count;
                        best_sum_left_gradient = sum_left_gradient;
                        best_sum_left_hessian = sum_left_hessian;
                        best_threshold = i;
                        best_gain = current_gain;
                        best_dir = *dir;
                    }
                }
                i += 1;
            }
        }
    }

    if is_splittable {
        let num_data_left = best_left_count;
        output.left_output = clamp_opt(
            output_rt(
                best_sum_left_gradient,
                best_sum_left_hessian,
                p,
                l2,
                p.path_smooth,
                num_data_left,
                parent_output,
                flags,
            ),
            mc,
        );
        output.left_count = best_left_count;
        output.left_sum_gradient = best_sum_left_gradient;
        output.left_sum_hessian = best_sum_left_hessian - EPS;
        output.right_output = clamp_opt(
            output_rt(
                sum_gradient - best_sum_left_gradient,
                sum_hessian - best_sum_left_hessian,
                p,
                l2,
                p.path_smooth,
                num_data - best_left_count,
                parent_output,
                flags,
            ),
            mc,
        );
        output.right_count = num_data - best_left_count;
        output.right_sum_gradient = sum_gradient - best_sum_left_gradient;
        output.right_sum_hessian = sum_hessian - best_sum_left_hessian - EPS;
        output.gain = best_gain - min_gain_shift;
        if use_onehot {
            output.num_cat_threshold = 1;
            output.cat_threshold = vec![(best_threshold + offset) as u32];
        } else {
            output.num_cat_threshold = best_threshold + 1;
            output.cat_threshold = (0..=best_threshold)
                .map(|i| {
                    let pos = if best_dir == 1 { i } else { used_bin - 1 - i };
                    (sorted_idx[pos as usize] + offset) as u32
                })
                .collect();
        }
        output.monotone_type = 0;
    }
    output.gain *= meta.penalty;
    is_splittable
}
