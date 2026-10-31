//! Leaf-wise tree learner. upstream: src/treelearner/serial_tree_learner.{h,cpp},
//! src/io/dataset.cpp (`ConstructHistogramsInner`, `FixHistogram`),
//! src/treelearner/feature_histogram.hpp (`HistogramPool`)

use super::data_partition::{DataPartition, SplitRule};
use super::feature_histogram::{
    FeatureMeta, SplitParams, find_best_threshold_numerical, leaf_output_root,
};
use super::histogram;
use super::split_info::{K_MIN_SCORE, SplitInfo, argmax_split};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::bin::BinType;
use crate::io::bin_data::BinData;
use crate::io::dataset::{Dataset, GroupData};
use crate::tree::Tree;
use crate::utils::random::Random;

/// upstream: `LeafSplits` (sums for one leaf being split)
#[derive(Debug, Clone, Copy)]
struct LeafSplits {
    leaf_index: i32,
    num_data_in_leaf: i32,
    sum_gradients: f64,
    sum_hessians: f64,
    weight: f64,
}

impl LeafSplits {
    fn empty() -> Self {
        LeafSplits {
            leaf_index: -1,
            num_data_in_leaf: 0,
            sum_gradients: 0.0,
            sum_hessians: 0.0,
            weight: 0.0,
        }
    }
}

/// Per-leaf histogram storage. upstream: `HistogramPool` with a full cache
/// (`histogram_pool_size <= 0`): every leaf keeps its histogram.
struct HistogramPool {
    slots: Vec<Vec<f64>>,
    splittable: Vec<Vec<bool>>,
    leaf_to_slot: Vec<Option<usize>>,
    slot_to_leaf: Vec<Option<usize>>,
    next_free: usize,
}

impl HistogramPool {
    fn new(num_leaves: usize, total_bins: usize, num_features: usize) -> Self {
        HistogramPool {
            slots: vec![vec![0.0; total_bins * 2]; num_leaves],
            splittable: vec![vec![true; num_features]; num_leaves],
            leaf_to_slot: vec![None; num_leaves],
            slot_to_leaf: vec![None; num_leaves],
            next_free: 0,
        }
    }

    fn reset_map(&mut self) {
        self.leaf_to_slot.fill(None);
        self.slot_to_leaf.fill(None);
        self.next_free = 0;
    }

    /// Returns `(slot, had_data)` like upstream `HistogramPool::Get`.
    fn get(&mut self, leaf: usize) -> (usize, bool) {
        if let Some(s) = self.leaf_to_slot[leaf] {
            return (s, true);
        }
        let s = self.next_free;
        self.next_free += 1;
        self.leaf_to_slot[leaf] = Some(s);
        self.slot_to_leaf[s] = Some(leaf);
        (s, false)
    }

    /// upstream: `HistogramPool::Move`
    fn move_leaf(&mut self, src: usize, dst: usize) {
        if let Some(s) = self.leaf_to_slot[src].take() {
            if let Some(old) = self.leaf_to_slot[dst] {
                self.slot_to_leaf[old] = None;
            }
            self.leaf_to_slot[dst] = Some(s);
            self.slot_to_leaf[s] = Some(dst);
        }
    }
}

pub struct SerialTreeLearner<'a> {
    cfg: &'a Config,
    params: SplitParams,
    ds: &'a Dataset,
    num_data: usize,
    num_features: usize,
    metas: Vec<FeatureMeta>,
    /// Start (in bins) of each inner feature's histogram slice.
    feature_hist_offsets: Vec<usize>,
    /// Length in bins of each feature's histogram slice (`num_bin - offset`).
    feature_hist_len: Vec<usize>,
    pool: HistogramPool,
    pub partition: DataPartition,
    best_split_per_leaf: Vec<SplitInfo>,
    smaller: LeafSplits,
    larger: LeafSplits,
    smaller_slot: usize,
    larger_slot: Option<usize>,
    parent_slot: Option<usize>,
    is_constant_hessian: bool,
    ordered_gradients: Vec<f32>,
    ordered_hessians: Vec<f32>,
}

impl<'a> SerialTreeLearner<'a> {
    /// upstream: `SerialTreeLearner::Init` (col-wise histogram layout).
    pub fn new(cfg: &'a Config, ds: &'a Dataset, is_constant_hessian: bool) -> Result<Self> {
        let l = &ds.layout;
        if l.groups.iter().any(|g| g.is_multi_val) {
            return Err(Error::Unsupported(
                "training with multi-value (sparse) feature groups".into(),
            ));
        }
        if cfg.histogram_pool_size > 0.0 {
            return Err(Error::Unsupported(
                "histogram_pool_size > 0 (LRU histogram cache)".into(),
            ));
        }
        for (gi, g) in l.groups.iter().enumerate() {
            let GroupData::Single(BinData::Dense(bin)) = &ds.data[gi] else {
                return Err(Error::Unsupported("sparse single-value group".into()));
            };
            let nb = *g.bin_offsets.last().unwrap_or(&0) as usize;
            if !histogram::validate_bins(bin, ds.num_data(), nb) {
                return Err(Error::Parse(format!(
                    "feature group {gi} has a bin outside its range"
                )));
            }
        }
        let num_features = l.num_features;
        let mut metas = Vec::with_capacity(num_features);
        let mut feature_hist_offsets = Vec::with_capacity(num_features);
        let mut feature_hist_len = Vec::with_capacity(num_features);
        // upstream TrainingShareStates::CalcBinOffsets (col-wise, single-value groups)
        let mut hist_cur_num_bin = 0usize;
        for g in &l.groups {
            for i in 0..g.bin_mappers.len() {
                feature_hist_offsets.push(hist_cur_num_bin + g.bin_offsets[i] as usize);
            }
            hist_cur_num_bin += *g.bin_offsets.last().unwrap_or(&0) as usize;
        }
        for f in 0..num_features {
            let g = &l.groups[l.feature2group[f] as usize];
            let m = &g.bin_mappers[l.feature2subfeature[f] as usize];
            if m.bin_type == BinType::Categorical {
                return Err(Error::Unsupported(
                    "training with categorical features".into(),
                ));
            }
            let offset = i32::from(m.most_freq_bin == 0);
            let real = l.real_feature_idx[f] as usize;
            metas.push(FeatureMeta {
                num_bin: m.num_bin,
                missing_type: m.missing_type,
                offset,
                default_bin: m.default_bin,
                monotone_type: cfg.monotone_constraints.get(real).copied().unwrap_or(0),
                penalty: cfg.feature_contri.get(real).copied().unwrap_or(1.0),
                bin_type: m.bin_type,
                rand: Random::new(cfg.extra_seed + f as i32),
            });
            feature_hist_len.push((m.num_bin - offset) as usize);
        }
        if !cfg.monotone_constraints.is_empty() {
            return Err(Error::Unsupported("monotone_constraints".into()));
        }
        let num_leaves = cfg.num_leaves as usize;
        Ok(SerialTreeLearner {
            cfg,
            params: SplitParams::from_config(cfg),
            ds,
            num_data: ds.num_data(),
            num_features,
            metas,
            feature_hist_offsets,
            feature_hist_len,
            pool: HistogramPool::new(num_leaves, hist_cur_num_bin.max(1), num_features),
            partition: DataPartition::new(ds.num_data(), num_leaves),
            best_split_per_leaf: vec![SplitInfo::default(); num_leaves],
            smaller: LeafSplits::empty(),
            larger: LeafSplits::empty(),
            smaller_slot: 0,
            larger_slot: None,
            parent_slot: None,
            is_constant_hessian,
            ordered_gradients: Vec::with_capacity(ds.num_data()),
            ordered_hessians: Vec::with_capacity(ds.num_data()),
        })
    }

    /// upstream: `SerialTreeLearner::Train`
    pub fn train(&mut self, gradients: &[f32], hessians: &[f32]) -> Tree {
        self.before_train(gradients, hessians);
        let num_leaves = self.cfg.num_leaves as usize;
        let mut tree = Tree::new_training(num_leaves);
        let root = leaf_output_root(
            &self.params,
            self.smaller.sum_gradients,
            self.smaller.sum_hessians,
            self.num_data as i32,
        );
        tree.set_leaf_output(0, root);

        let mut left_leaf: i32 = 0;
        let mut right_leaf: i32 = -1;
        for _ in 0..num_leaves.saturating_sub(1) {
            if self.before_find_best_split(&tree, left_leaf, right_leaf) {
                self.find_best_splits(&tree, gradients, hessians);
            }
            let best_leaf = argmax_split(&self.best_split_per_leaf);
            if self.best_split_per_leaf[best_leaf].gain <= 0.0 {
                log::warn!(
                    "No further splits with positive gain, best gain: {}",
                    self.best_split_per_leaf[best_leaf].gain
                );
                break;
            }
            let (l, r) = self.split(&mut tree, best_leaf);
            left_leaf = l as i32;
            right_leaf = r as i32;
        }
        tree
    }

    /// upstream: `SerialTreeLearner::BeforeTrain` (no column sampling, no bagging)
    fn before_train(&mut self, gradients: &[f32], hessians: &[f32]) {
        self.pool.reset_map();
        self.partition.init();
        for s in &mut self.best_split_per_leaf {
            s.reset();
        }
        // upstream LeafSplits::Init(gradients, hessians): sequential double sums
        let mut sg = 0.0f64;
        let mut sh = 0.0f64;
        for i in 0..self.num_data {
            sg += f64::from(gradients[i]);
            sh += f64::from(hessians[i]);
        }
        self.smaller = LeafSplits {
            leaf_index: 0,
            num_data_in_leaf: self.num_data as i32,
            sum_gradients: sg,
            sum_hessians: sh,
            weight: 0.0,
        };
        self.larger = LeafSplits::empty();
    }

    /// upstream: `SerialTreeLearner::BeforeFindBestSplit`
    fn before_find_best_split(&mut self, tree: &Tree, left_leaf: i32, right_leaf: i32) -> bool {
        let cfg = self.cfg;
        if cfg.max_depth > 0 && tree.leaf_depth[left_leaf as usize] >= cfg.max_depth {
            self.best_split_per_leaf[left_leaf as usize].gain = K_MIN_SCORE;
            if right_leaf >= 0 {
                self.best_split_per_leaf[right_leaf as usize].gain = K_MIN_SCORE;
            }
            return false;
        }
        let count = |leaf: i32| {
            if leaf >= 0 {
                self.partition.leaf_count(leaf as usize) as i32
            } else {
                0
            }
        };
        let num_left = count(left_leaf);
        let num_right = count(right_leaf);
        if num_right < cfg.min_data_in_leaf * 2 && num_left < cfg.min_data_in_leaf * 2 {
            self.best_split_per_leaf[left_leaf as usize].gain = K_MIN_SCORE;
            if right_leaf >= 0 {
                self.best_split_per_leaf[right_leaf as usize].gain = K_MIN_SCORE;
            }
            return false;
        }
        self.parent_slot = None;
        if right_leaf < 0 {
            self.smaller_slot = self.pool.get(left_leaf as usize).0;
            self.larger_slot = None;
        } else if num_left < num_right {
            let (larger, had) = self.pool.get(left_leaf as usize);
            if had {
                self.parent_slot = Some(larger);
            }
            self.larger_slot = Some(larger);
            self.pool.move_leaf(left_leaf as usize, right_leaf as usize);
            self.smaller_slot = self.pool.get(left_leaf as usize).0;
        } else {
            let (larger, had) = self.pool.get(left_leaf as usize);
            if had {
                self.parent_slot = Some(larger);
            }
            self.larger_slot = Some(larger);
            self.smaller_slot = self.pool.get(right_leaf as usize).0;
        }
        true
    }

    /// upstream: `SerialTreeLearner::FindBestSplits`
    fn find_best_splits(&mut self, tree: &Tree, gradients: &[f32], hessians: &[f32]) {
        let nf = self.num_features;
        let mut is_feature_used = vec![false; nf];
        for f in 0..nf {
            if let Some(p) = self.parent_slot
                && !self.pool.splittable[p][f]
            {
                self.pool.splittable[self.smaller_slot][f] = false;
                continue;
            }
            is_feature_used[f] = true;
        }
        let use_subtract = self.parent_slot.is_some();
        // smaller leaf histogram
        let smaller_leaf = self.smaller.leaf_index as usize;
        self.construct_histograms(
            &is_feature_used,
            Some(smaller_leaf),
            self.smaller_slot,
            gradients,
            hessians,
        );
        if let Some(ls) = self.larger_slot
            && !use_subtract
            && self.larger.leaf_index >= 0
        {
            let larger_leaf = self.larger.leaf_index as usize;
            self.construct_histograms(&is_feature_used, Some(larger_leaf), ls, gradients, hessians);
        }
        self.find_best_splits_from_histograms(&is_feature_used, use_subtract, tree);
    }

    /// upstream: `Dataset::ConstructHistograms` (col-wise, dense single-value groups).
    /// With constant hessians, the hessian slot counts rows and is scaled by `hessians[0]`.
    fn construct_histograms(
        &mut self,
        is_feature_used: &[bool],
        leaf: Option<usize>,
        slot: usize,
        gradients: &[f32],
        hessians: &[f32],
    ) {
        let l = &self.ds.layout;
        let leaf_cnt = leaf
            .map(|lf| self.partition.leaf_count(lf))
            .unwrap_or(self.num_data);
        let indices: Option<&[i32]> = match leaf {
            Some(lf) if leaf_cnt < self.num_data => Some(self.partition.indices_on_leaf(lf)),
            _ => None,
        };
        let const_hess = self.is_constant_hessian;
        // upstream gathers gradients into leaf order once, then streams them per group.
        let (og, oh): (&[f32], &[f32]) = match indices {
            Some(ix) => {
                self.ordered_gradients.clear();
                self.ordered_gradients
                    .extend(ix.iter().map(|&i| gradients[i as usize]));
                if !const_hess {
                    self.ordered_hessians.clear();
                    self.ordered_hessians
                        .extend(ix.iter().map(|&i| hessians[i as usize]));
                }
                (
                    &self.ordered_gradients,
                    if const_hess {
                        hessians
                    } else {
                        &self.ordered_hessians
                    },
                )
            }
            None => (&gradients[..leaf_cnt], &hessians[..leaf_cnt]),
        };
        let hist = &mut self.pool.slots[slot];
        let mut group_start = 0usize;
        for (gi, g) in l.groups.iter().enumerate() {
            let num_bin = *g.bin_offsets.last().unwrap_or(&0) as usize;
            let fs = l.group_feature_start[gi] as usize;
            let fc = l.group_feature_cnt[gi] as usize;
            if (fs..fs + fc).any(|f| is_feature_used[f]) {
                let out = &mut hist[group_start * 2..(group_start + num_bin) * 2];
                out.fill(0.0);
                let GroupData::Single(BinData::Dense(bin)) = &self.ds.data[gi] else {
                    unreachable!("checked in new()")
                };
                histogram::construct(bin, indices, og, oh, const_hess, out);
                if const_hess {
                    let h0 = f64::from(hessians[0]);
                    for i in (0..num_bin * 2).step_by(2) {
                        out[i + 1] *= h0;
                    }
                }
            }
            group_start += num_bin;
        }
    }

    /// upstream: `Dataset::FixHistogram`
    fn fix_histogram(&mut self, slot: usize, feature: usize, sum_gradient: f64, sum_hessian: f64) {
        let l = &self.ds.layout;
        let g = &l.groups[l.feature2group[feature] as usize];
        let m = &g.bin_mappers[l.feature2subfeature[feature] as usize];
        let mfb = m.most_freq_bin as usize;
        if mfb > 0 {
            let start = self.feature_hist_offsets[feature] * 2;
            let data = &mut self.pool.slots[slot][start..start + (m.num_bin as usize) * 2];
            data[mfb << 1] = sum_gradient;
            data[(mfb << 1) + 1] = sum_hessian;
            for i in 0..m.num_bin as usize {
                if i != mfb {
                    data[mfb << 1] -= data[i << 1];
                    data[(mfb << 1) + 1] -= data[(i << 1) + 1];
                }
            }
        }
    }

    /// upstream: `SerialTreeLearner::FindBestSplitsFromHistograms`
    fn find_best_splits_from_histograms(
        &mut self,
        is_feature_used: &[bool],
        use_subtract: bool,
        tree: &Tree,
    ) {
        let mut smaller_best = SplitInfo::default();
        let mut larger_best = SplitInfo::default();
        let smaller_parent_output = self.parent_output(tree, &self.smaller);
        let larger_parent_output = if self.larger.leaf_index >= 0 {
            self.parent_output(tree, &self.larger)
        } else {
            0.0
        };
        for f in 0..self.num_features {
            if !is_feature_used[f] {
                continue;
            }
            let (sg, sh) = (self.smaller.sum_gradients, self.smaller.sum_hessians);
            self.fix_histogram(self.smaller_slot, f, sg, sh);
            let real = self.ds.layout.real_feature_idx[f];
            let s = self.smaller;
            self.compute_best_split_for_feature(
                self.smaller_slot,
                f,
                real,
                &s,
                smaller_parent_output,
                &mut smaller_best,
            );

            if self.larger.leaf_index < 0 {
                continue;
            }
            let Some(ls) = self.larger_slot else { continue };
            if use_subtract {
                let start = self.feature_hist_offsets[f] * 2;
                let len = self.feature_hist_len[f] * 2;
                let (small, large) = two_slots(&mut self.pool.slots, self.smaller_slot, ls);
                for i in start..start + len {
                    large[i] -= small[i];
                }
            } else {
                let (lg, lh) = (self.larger.sum_gradients, self.larger.sum_hessians);
                self.fix_histogram(ls, f, lg, lh);
            }
            let lsplits = self.larger;
            self.compute_best_split_for_feature(
                ls,
                f,
                real,
                &lsplits,
                larger_parent_output,
                &mut larger_best,
            );
        }
        self.best_split_per_leaf[self.smaller.leaf_index as usize] = smaller_best;
        if self.larger.leaf_index >= 0 {
            self.best_split_per_leaf[self.larger.leaf_index as usize] = larger_best;
        }
    }

    /// upstream: `SerialTreeLearner::GetParentOutput`
    fn parent_output(&self, tree: &Tree, ls: &LeafSplits) -> f64 {
        if tree.num_leaves == 1 {
            leaf_output_root(
                &self.params,
                ls.sum_gradients,
                ls.sum_hessians,
                ls.num_data_in_leaf,
            )
        } else {
            ls.weight
        }
    }

    /// upstream: `SerialTreeLearner::ComputeBestSplitForFeature`
    fn compute_best_split_for_feature(
        &mut self,
        slot: usize,
        feature: usize,
        real_fidx: i32,
        ls: &LeafSplits,
        parent_output: f64,
        best: &mut SplitInfo,
    ) {
        let start = self.feature_hist_offsets[feature] * 2;
        let len = self.feature_hist_len[feature] * 2;
        let hist = &self.pool.slots[slot][start..start + len];
        let mut new_split = SplitInfo::default();
        let splittable = find_best_threshold_numerical(
            hist,
            &mut self.metas[feature],
            &self.params,
            ls.sum_gradients,
            ls.sum_hessians,
            ls.num_data_in_leaf,
            parent_output,
            &mut new_split,
        );
        self.pool.splittable[slot][feature] = splittable;
        new_split.feature = real_fidx;
        if new_split.better_than(best) {
            *best = new_split;
        }
    }

    /// upstream: `SerialTreeLearner::SplitInner` (numerical). Returns (left, right) leaves.
    fn split(&mut self, tree: &mut Tree, best_leaf: usize) -> (usize, usize) {
        let mut info = self.best_split_per_leaf[best_leaf].clone();
        let ds = self.ds;
        let l = &ds.layout;
        let inner = l.used_feature_map[info.feature as usize] as usize;
        let g = &l.groups[l.feature2group[inner] as usize];
        let m = &g.bin_mappers[l.feature2subfeature[inner] as usize];
        let next_leaf = tree.num_leaves;
        let threshold_double = m.bin_to_value(info.threshold);
        let rule = SplitRule::numerical(ds, inner, info.threshold, info.default_left);
        self.partition.split(best_leaf, &rule, next_leaf);
        info.left_count = self.partition.leaf_count(best_leaf) as i32;
        info.right_count = self.partition.leaf_count(next_leaf) as i32;
        let right = tree.split_numerical(
            best_leaf,
            inner as i32,
            info.feature,
            info.threshold,
            threshold_double,
            info.left_output,
            info.right_output,
            info.left_count,
            info.right_count,
            info.left_sum_hessian,
            info.right_sum_hessian,
            (info.gain + self.cfg.min_gain_to_split) as f32,
            m.missing_type,
            info.default_left,
        );
        let left_ls = LeafSplits {
            leaf_index: best_leaf as i32,
            num_data_in_leaf: info.left_count,
            sum_gradients: info.left_sum_gradient,
            sum_hessians: info.left_sum_hessian,
            weight: info.left_output,
        };
        let right_ls = LeafSplits {
            leaf_index: right as i32,
            num_data_in_leaf: info.right_count,
            sum_gradients: info.right_sum_gradient,
            sum_hessians: info.right_sum_hessian,
            weight: info.right_output,
        };
        if info.left_count < info.right_count {
            self.smaller = left_ls;
            self.larger = right_ls;
        } else {
            self.smaller = right_ls;
            self.larger = left_ls;
        }
        self.best_split_per_leaf[best_leaf] = info;
        (best_leaf, right)
    }

    /// upstream: `SerialTreeLearner::AddPredictionToScore`
    pub fn add_prediction_to_score(&self, tree: &Tree, score: &mut [f64]) {
        if tree.num_leaves <= 1 {
            return;
        }
        for leaf in 0..tree.num_leaves {
            let out = tree.leaf_value[leaf];
            for &idx in self.partition.indices_on_leaf(leaf) {
                score[idx as usize] += out;
            }
        }
    }
}

fn two_slots(slots: &mut [Vec<f64>], a: usize, b: usize) -> (&Vec<f64>, &mut Vec<f64>) {
    assert_ne!(a, b);
    if a < b {
        let (x, y) = slots.split_at_mut(b);
        (&x[a], &mut y[0])
    } else {
        let (x, y) = slots.split_at_mut(a);
        (&y[0], &mut x[b])
    }
}
