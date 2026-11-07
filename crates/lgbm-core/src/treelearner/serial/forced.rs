//! Forced splits (`forcedsplits_filename`): a JSON tree of splits that every tree starts with.
//! upstream: src/treelearner/serial_tree_learner.cpp (`ForceSplits`, `FindAllForceFeatures`),
//! src/treelearner/feature_histogram.hpp (`GatherInfoForThresholdNumerical`)

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::error::{Error, Result};
use crate::io::bin::BinType;
use crate::meta::{K_EPSILON, MissingType};
use crate::treelearner::feature_histogram::{
    calculate_splitted_leaf_output, leaf_gain, leaf_gain_given_output,
};

/// The forced-split tree and the inner features it uses.
pub(super) struct ForcedSplits {
    root: Value,
    /// Inner feature indices used by any forced split (upstream: `FindAllForceFeatures`).
    pub(super) features: Vec<bool>,
}

impl ForcedSplits {
    /// Reads and validates the file. `None` when no file is configured.
    pub(super) fn load(path: &str, ds: &Dataset, metas: &[FeatureMeta]) -> Result<Option<Self>> {
        if path.is_empty() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(path)?;
        let root: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Parse(format!("forced splits file {path}: {e}")))?;
        let mut features = vec![false; metas.len()];
        let mut queue = VecDeque::from([&root]);
        while let Some(node) = queue.pop_front() {
            let real = node.get("feature").and_then(Value::as_i64).unwrap_or(0);
            let inner = usize::try_from(real)
                .ok()
                .and_then(|r| ds.layout.used_feature_map.get(r))
                .copied()
                .filter(|&i| i >= 0)
                .map(|i| i as usize)
                .ok_or_else(|| {
                    Error::InvalidParameter(format!(
                        "Forced split on feature {real}, which the dataset does not use"
                    ))
                })?;
            if metas[inner].bin_type == BinType::Categorical {
                return Err(Error::Unsupported(
                    "forced split on a categorical feature".into(),
                ));
            }
            features[inner] = true;
            for side in ["left", "right"] {
                if let Some(child) = node.get(side) {
                    queue.push_back(child);
                }
            }
        }
        Ok(Some(ForcedSplits { root, features }))
    }
}

/// True when a forced-split node names both a feature and a threshold; only those nodes
/// are queued (upstream: `ForceSplits`).
fn is_complete(node: &Value) -> bool {
    node.get("feature").is_some() && node.get("threshold").is_some()
}

impl SerialTreeLearner {
    /// upstream: `SerialTreeLearner::ForceSplits`. Applies the forced splits breadth-first and
    /// returns how many splits were made; the normal loop grows the rest of the tree.
    pub(super) fn force_splits(
        &mut self,
        forced: &Arc<ForcedSplits>,
        tree: &mut Tree,
        left_leaf: &mut i32,
        right_leaf: &mut i32,
        gradients: &[f32],
        hessians: &[f32],
    ) -> usize {
        let mut result = 0usize;
        *left_leaf = 0;
        let mut queue: VecDeque<(Value, i32)> = VecDeque::from([(forced.root.clone(), 0)]);
        // The specs of the two children of the most recent split, as upstream's `left`/`right`.
        let mut left: Option<Value> = Some(forced.root.clone());
        let mut right: Option<Value> = None;
        let mut left_smaller = true;
        let mut force_map: HashMap<i32, SplitInfo> = HashMap::new();
        let mut abort_last = false;
        while !queue.is_empty() {
            if self.before_find_best_split(tree, *left_leaf, *right_leaf) {
                self.find_best_splits(tree, gradients, hessians, Some(&forced.features));
            }
            // upstream: the histogram of the smaller leaf is the left child's when it is the
            // left child that is smaller, and the larger leaf's otherwise.
            let (left_slot, left_ls, right_slot, right_ls) = if left_smaller {
                (
                    Some(self.smaller_slot),
                    self.smaller,
                    self.larger_slot,
                    self.larger,
                )
            } else {
                (
                    self.larger_slot,
                    self.larger,
                    Some(self.smaller_slot),
                    self.smaller,
                )
            };
            if let Some(spec) = &left {
                // A split always leaves a larger-leaf slot once it has been made, so the
                // `None` arm cannot be reached; it aborts forced splitting if it ever is.
                let Some(slot) = left_slot else {
                    abort_last = true;
                    break;
                };
                let info = self.gather_forced(spec, slot, &left_ls);
                store_or_drop(&mut force_map, *left_leaf, info);
            }
            if let Some(spec) = &right {
                let Some(slot) = right_slot else {
                    abort_last = true;
                    break;
                };
                let info = self.gather_forced(spec, slot, &right_ls);
                store_or_drop(&mut force_map, *right_leaf, info);
            }

            let Some((spec, current)) = queue.pop_front() else {
                break;
            };
            // The split info exists because the queue is breadth-first: the parent queued
            // this leaf after gathering its split.
            let Some(info) = force_map.get(&current).cloned() else {
                abort_last = true;
                break;
            };
            self.best_split_per_leaf[current as usize] = info;
            let (l, r) = self.split(tree, current as usize);
            *left_leaf = l as i32;
            *right_leaf = r as i32;
            let done = &self.best_split_per_leaf[current as usize];
            left_smaller = done.left_count < done.right_count;

            left = None;
            right = None;
            if let Some(child) = spec.get("left") {
                left = Some(child.clone());
                if is_complete(child) {
                    queue.push_back((child.clone(), *left_leaf));
                }
            }
            if let Some(child) = spec.get("right") {
                right = Some(child.clone());
                if is_complete(child) {
                    queue.push_back((child.clone(), *right_leaf));
                }
            }
            result += 1;
        }
        if abort_last {
            let best = argmax_split(&self.best_split_per_leaf);
            if self.best_split_per_leaf[best].gain <= 0.0 {
                log::warn!(
                    "No further splits with positive gain, best gain: {}",
                    self.best_split_per_leaf[best].gain
                );
                return self.cfg.num_leaves as usize;
            }
            let (l, r) = self.split(tree, best);
            *left_leaf = l as i32;
            *right_leaf = r as i32;
            result += 1;
        }
        result
    }

    /// upstream: `SerialTreeLearner::ForceSplits`, the per-child gather
    /// (`GatherInfoForThreshold` on the child's histogram with the child's sums).
    fn gather_forced(&self, spec: &Value, slot: usize, ls: &LeafSplits) -> SplitInfo {
        let real = spec.get("feature").and_then(Value::as_i64).unwrap_or(0);
        let value = spec.get("threshold").and_then(Value::as_f64).unwrap_or(0.0);
        let l = &self.ds.layout;
        // Validated when the file was loaded.
        let inner = l.used_feature_map[real as usize] as usize;
        let group = &l.groups[l.feature2group[inner] as usize];
        let bin = group.bin_mappers[l.feature2subfeature[inner] as usize].value_to_bin(value);
        let start = self.feature_hist_offsets[inner] * 2;
        let len = self.feature_hist_len[inner] * 2;
        let hist = &self.pool.slots[slot][start..start + len];
        let sums = LeafSums {
            sum_gradient: ls.sum_gradients,
            sum_hessian: ls.sum_hessians,
            num_data: ls.num_data_in_leaf,
            parent_output: ls.weight,
        };
        let mut info = gather_numerical(hist, &self.metas[inner], &self.params, sums, bin);
        info.feature = real as i32;
        info
    }
}

/// A forced split is kept unless its gain is negative (upstream: `forceSplitMap` erase).
fn store_or_drop(map: &mut HashMap<i32, SplitInfo>, leaf: i32, info: SplitInfo) {
    if info.gain < 0.0 {
        map.remove(&leaf);
    } else {
        map.insert(leaf, info);
    }
}

/// Sums and output of the leaf a forced threshold is evaluated on.
#[derive(Clone, Copy)]
struct LeafSums {
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
}

/// upstream: `FeatureHistogram::GatherInfoForThresholdNumerical`. Evaluates one threshold on
/// a numerical histogram. The gain is `-inf` when the split is not an improvement.
fn gather_numerical(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    sums: LeafSums,
    threshold: u32,
) -> SplitInfo {
    if p.path_smooth > f64::from(K_EPSILON) {
        gather_numerical_inner::<true>(hist, meta, p, sums, threshold)
    } else {
        gather_numerical_inner::<false>(hist, meta, p, sums, threshold)
    }
}

fn gather_numerical_inner<const SMOOTH: bool>(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    sums: LeafSums,
    threshold: u32,
) -> SplitInfo {
    let LeafSums {
        sum_gradient,
        sum_hessian,
        num_data,
        parent_output,
    } = sums;
    let mut out = SplitInfo::default();
    let gain_shift = leaf_gain_given_output::<true>(
        sum_gradient,
        sum_hessian,
        p.lambda_l1,
        p.lambda_l2,
        parent_output,
    );
    let min_gain_shift = gain_shift + p.min_gain_to_split;
    let offset = meta.offset;
    let mut sum_right_gradient = 0.0f64;
    let mut sum_right_hessian = f64::from(K_EPSILON);
    let mut right_count = 0i32;
    let use_na_as_missing = meta.missing_type == MissingType::NaN;
    let skip_default_bin = meta.missing_type == MissingType::Zero;
    let cnt_factor = f64::from(num_data) / sum_hessian;
    // From the right end down to bin 1: bin 0 is never a right-side bin.
    let mut t = meta.num_bin - 1 - offset - i32::from(use_na_as_missing);
    let t_end = 1 - offset;
    while t >= t_end {
        if (t + offset) as u32 <= threshold {
            break;
        }
        if !(skip_default_bin && (t + offset) as u32 == meta.default_bin) {
            let idx = (t as usize) << 1;
            let (grad, hess) = (hist[idx], hist[idx + 1]);
            // upstream: `Common::RoundInt(hess * cnt_factor)`
            let cnt = (hess * cnt_factor + 0.5) as i32;
            sum_right_gradient += grad;
            sum_right_hessian += hess;
            right_count += cnt;
        }
        t -= 1;
    }
    let sum_left_gradient = sum_gradient - sum_right_gradient;
    let sum_left_hessian = sum_hessian - sum_right_hessian;
    let left_count = num_data - right_count;
    let (l1, l2, mds, sm) = (p.lambda_l1, p.lambda_l2, p.max_delta_step, p.path_smooth);
    let current_gain = leaf_gain::<true, true, SMOOTH>(
        sum_left_gradient,
        sum_left_hessian,
        l1,
        l2,
        mds,
        sm,
        left_count,
        parent_output,
    ) + leaf_gain::<true, true, SMOOTH>(
        sum_right_gradient,
        sum_right_hessian,
        l1,
        l2,
        mds,
        sm,
        right_count,
        parent_output,
    );
    // upstream: "'Forced Split' will be ignored since the gain getting worse."
    if current_gain.is_nan() || current_gain <= min_gain_shift {
        out.gain = K_MIN_SCORE;
        log::warn!("'Forced Split' will be ignored since the gain getting worse.");
        return out;
    }
    out.threshold = threshold;
    out.left_output = calculate_splitted_leaf_output::<true, true, SMOOTH>(
        sum_left_gradient,
        sum_left_hessian,
        l1,
        l2,
        mds,
        sm,
        left_count,
        parent_output,
    );
    out.left_count = left_count;
    out.left_sum_gradient = sum_left_gradient;
    out.left_sum_hessian = sum_left_hessian - f64::from(K_EPSILON);
    out.right_output = calculate_splitted_leaf_output::<true, true, SMOOTH>(
        sum_gradient - sum_left_gradient,
        sum_hessian - sum_left_hessian,
        l1,
        l2,
        mds,
        sm,
        right_count,
        parent_output,
    );
    out.right_count = num_data - left_count;
    out.right_sum_gradient = sum_gradient - sum_left_gradient;
    out.right_sum_hessian = sum_hessian - sum_left_hessian - f64::from(K_EPSILON);
    out.gain = current_gain - min_gain_shift;
    out.default_left = true;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SplitParams {
        SplitParams {
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            max_delta_step: 0.0,
            path_smooth: 0.0,
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            min_gain_to_split: 0.0,
            extra_trees: false,
        }
    }

    fn meta(num_bin: i32) -> FeatureMeta {
        FeatureMeta {
            num_bin,
            missing_type: MissingType::None,
            offset: 0,
            default_bin: 0,
            monotone_type: 0,
            penalty: 1.0,
            bin_type: BinType::Numerical,
            rand: crate::utils::random::Random::new(0),
        }
    }

    #[test]
    fn gather_counts_the_right_side_from_the_top_bin_down() {
        // Bins (grad, hess): (1, 1), (2, 2), (-3, 3); leaf sums (0, 6), 6 rows.
        let hist = [1.0, 1.0, 2.0, 2.0, -3.0, 3.0];
        let sums = LeafSums {
            sum_gradient: 0.0,
            sum_hessian: 6.0,
            num_data: 6,
            parent_output: 0.0,
        };
        // Threshold bin 0: bins 1 and 2 go right.
        let info = gather_numerical(&hist, &meta(3), &params(), sums, 0);
        assert_eq!(info.left_count, 1);
        assert_eq!(info.right_count, 5);
        assert_eq!(info.threshold, 0);
        // Left: g=1, h=1 -> gain 1. Right: g=-1, h=5 -> gain 0.2.
        assert!((info.gain - 1.2).abs() < 1e-9, "gain {}", info.gain);
        assert!((info.left_output + 1.0).abs() < 1e-9);
        assert!(info.default_left);
    }

    #[test]
    fn gather_rejects_a_threshold_that_does_not_improve_the_gain() {
        // Zero gradients: both sides have zero gain, so no split improves on the parent.
        let hist = [0.0, 1.0, 0.0, 1.0];
        let sums = LeafSums {
            sum_gradient: 0.0,
            sum_hessian: 2.0,
            num_data: 2,
            parent_output: 0.0,
        };
        let info = gather_numerical(&hist, &meta(2), &params(), sums, 0);
        assert_eq!(info.gain, K_MIN_SCORE);
    }
}
