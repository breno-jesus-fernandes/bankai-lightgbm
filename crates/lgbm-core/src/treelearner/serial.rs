//! Leaf-wise tree learner. upstream: src/treelearner/serial_tree_learner.{h,cpp},
//! src/io/dataset.cpp (`ConstructHistogramsInner`, `FixHistogram`),
//! src/treelearner/feature_histogram.hpp (`HistogramPool`)

mod forced;
mod linear;

use self::forced::ForcedSplits;
use super::col_sampler::ColSampler;
use super::data_partition::{DataPartition, SplitRule};
use super::feature_histogram::{
    CatParams, FeatureMeta, SplitParams, find_best_threshold_categorical,
    find_best_threshold_numerical, leaf_output_root,
};
use super::histogram;
use super::split_info::{K_MIN_SCORE, SplitInfo, argmax_split};
use crate::utils::spin_pool::SpinPool;
use std::sync::Arc;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::bin::BinType;
use crate::io::bin_data::BinData;
use crate::io::dataset::{Dataset, GroupData};
use crate::tree::Tree;
use crate::treelearner::multi_val::MultiValBin;
use crate::utils::common::construct_bitset;
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

/// One numerical split of the tree being trained (replayed for out-of-bag rows).
#[derive(Debug, Clone)]
pub(crate) struct SplitRecord {
    leaf: usize,
    right: usize,
    inner: usize,
    kind: SplitKind,
}

#[derive(Debug, Clone)]
pub(crate) enum SplitKind {
    Numerical {
        threshold: u32,
        default_left: bool,
    },
    /// Bins that go left (`cat_bitset_inner`).
    Categorical(Vec<u32>),
}

pub struct SerialTreeLearner {
    cfg: Arc<Config>,
    params: SplitParams,
    ds: Arc<Dataset>,
    num_data: usize,
    num_features: usize,
    metas: Vec<FeatureMeta>,
    /// Start (in bins) of each inner feature's histogram slice.
    feature_hist_offsets: Vec<usize>,
    /// Length in bins of each feature's histogram slice (`num_bin - offset`).
    feature_hist_len: Vec<usize>,
    pool: HistogramPool,
    /// One verified dense column per single-value feature group (see
    /// `histogram::CheckedBins`); `None` for the multi-value group.
    checked_bins: Vec<Option<histogram::BinsCheck>>,
    /// The multi-value group (at most one) and its row-wise bins.
    multi_val: Option<(usize, MultiValBin)>,
    /// Per-block histograms of the multi-value group, reused across leaves.
    multi_val_scratch: Vec<f64>,
    col_sampler: ColSampler,
    splits: Vec<SplitRecord>,
    pub partition: DataPartition,
    best_split_per_leaf: Vec<SplitInfo>,
    /// Forced splits applied at the start of every tree (`forcedsplits_filename`).
    forced: Option<Arc<ForcedSplits>>,
    /// Per-dataset facts for linear leaves (`linear_tree`); `None` otherwise.
    linear: Option<linear::LinearState>,
    smaller: LeafSplits,
    larger: LeafSplits,
    smaller_slot: usize,
    larger_slot: Option<usize>,
    parent_slot: Option<usize>,
    is_constant_hessian: bool,
    ordered_gradients: Vec<f32>,
    ordered_hessians: Vec<f32>,
    cat_params: CatParams,
    /// upstream `Tree::branch_features`: real feature indices on the path to each leaf.
    branch_features: Vec<Vec<i32>>,
    /// upstream `BasicLeafConstraints`: output bounds of each leaf (`monotone_constraints`).
    monotone_active: bool,
    monotone_penalization: f64,
    monotone_bounds: Vec<(f64, f64)>,
    leaf_depth: Vec<i32>,
    /// Threads for the split search and the multi-value histograms (upstream `num_threads_`).
    num_threads: usize,
    /// Persistent spinning workers for the parallel regions of one tree (`None` when serial).
    spin: Option<SpinPool>,
}

impl SerialTreeLearner {
    /// upstream: SerialTreeLearner::ResetIsConstantHessian after objective reset.
    pub fn reset_constant_hessian(&mut self, value: bool) {
        self.is_constant_hessian = value;
    }

    /// upstream: `SerialTreeLearner::Init` (col-wise histogram layout).
    pub fn new(cfg: Arc<Config>, ds: Arc<Dataset>, is_constant_hessian: bool) -> Result<Self> {
        let l = &ds.layout;
        if cfg.histogram_pool_size > 0.0 {
            return Err(Error::Unsupported(
                "histogram_pool_size > 0 (LRU histogram cache)".into(),
            ));
        }
        let mut checked_bins = Vec::with_capacity(l.groups.len());
        let mut multi_val = None;
        for (gi, g) in l.groups.iter().enumerate() {
            if g.is_multi_val {
                // upstream: GetMultiBinFromSparseFeatures ("There should be only one
                // multi-val group").
                let GroupData::Multi(cols) = &ds.data[gi] else {
                    return Err(Error::Parse(format!(
                        "feature group {gi} is multi-value but has no per-feature columns"
                    )));
                };
                if multi_val.is_some() {
                    return Err(Error::Parse(
                        "more than one multi-value feature group".into(),
                    ));
                }
                multi_val = Some((gi, MultiValBin::from_group(g, cols, ds.num_data())));
                checked_bins.push(None);
                continue;
            }
            let GroupData::Single(BinData::Dense(bin)) = &ds.data[gi] else {
                return Err(Error::Unsupported("sparse single-value group".into()));
            };
            let nb = *g.bin_offsets.last().unwrap_or(&0) as usize;
            let Some(checked) = histogram::BinsCheck::new(bin, ds.num_data(), nb) else {
                return Err(Error::Parse(format!(
                    "feature group {gi} has a bin outside its range"
                )));
            };
            checked_bins.push(Some(checked));
        }
        let num_features = l.num_features;
        // upstream: SerialTreeLearner::Init and FeatureHistogram::DynamicChangeSize (v4.7.0)
        log::info!(
            "Total Bins {}",
            (0..num_features)
                .map(|i| ds.feature_num_bin(i))
                .sum::<i32>()
        );
        log::info!(
            "Number of data points in the train set: {}, number of used features: {}",
            ds.num_data(),
            num_features
        );
        let num_threads = if cfg.num_threads > 0 {
            cfg.num_threads as usize
        } else {
            rayon::current_num_threads()
        };
        let spin = if num_threads > 1 {
            Some(SpinPool::new(num_threads)?)
        } else {
            None
        };
        let mut metas = Vec::with_capacity(num_features);
        let mut feature_hist_offsets = Vec::with_capacity(num_features);
        let mut feature_hist_len = Vec::with_capacity(num_features);
        // upstream TrainingShareStates::CalcBinOffsets (col-wise). A dense multi-value
        // group keeps bin 0 of features whose most frequent bin is 0, so their histograms
        // start one bin later.
        let mut hist_cur_num_bin = 0usize;
        for g in &l.groups {
            for i in 0..g.bin_mappers.len() {
                let skip_zero =
                    usize::from(g.is_dense_multi_val && g.bin_mappers[i].most_freq_bin == 0);
                feature_hist_offsets.push(hist_cur_num_bin + g.bin_offsets[i] as usize + skip_zero);
            }
            hist_cur_num_bin += *g.bin_offsets.last().unwrap_or(&0) as usize;
        }
        for f in 0..num_features {
            let g = &l.groups[l.feature2group[f] as usize];
            let m = &g.bin_mappers[l.feature2subfeature[f] as usize];
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
        if !cfg.monotone_constraints.is_empty() && cfg.monotone_constraints_method != "basic" {
            return Err(Error::Unsupported(format!(
                "monotone_constraints_method={}",
                cfg.monotone_constraints_method
            )));
        }
        let num_leaves = cfg.num_leaves as usize;
        let params = SplitParams::from_config(&cfg);
        let cat_params = CatParams::from_config(&cfg);
        let col_sampler = ColSampler::new(&cfg, &ds)?;
        let num_data = ds.num_data();
        let monotone_active = !cfg.monotone_constraints.is_empty();
        let monotone_penalization = cfg.monotone_penalty;
        let forced = ForcedSplits::load(&cfg.forcedsplits_filename, &ds, &metas)?.map(Arc::new);
        let linear = cfg
            .linear_tree
            .then(|| linear::LinearState::new(&ds, &metas));
        Ok(SerialTreeLearner {
            cfg,
            params,
            ds,
            num_data,
            num_features,
            metas,
            feature_hist_offsets,
            feature_hist_len,
            pool: HistogramPool::new(num_leaves, hist_cur_num_bin.max(1), num_features),
            checked_bins,
            multi_val,
            multi_val_scratch: Vec::new(),
            col_sampler,
            splits: Vec::new(),
            partition: DataPartition::new(num_data, num_leaves),
            best_split_per_leaf: vec![SplitInfo::default(); num_leaves],
            forced,
            linear,
            smaller: LeafSplits::empty(),
            larger: LeafSplits::empty(),
            smaller_slot: 0,
            larger_slot: None,
            parent_slot: None,
            is_constant_hessian,
            ordered_gradients: Vec::with_capacity(num_data),
            ordered_hessians: Vec::with_capacity(num_data),
            cat_params,
            branch_features: vec![Vec::new(); num_leaves],
            monotone_active,
            monotone_penalization,
            monotone_bounds: vec![(-f64::MAX, f64::MAX); num_leaves],
            leaf_depth: vec![0; num_leaves],
            num_threads,
            spin,
        })
    }

    /// upstream: `SerialTreeLearner::Train`
    /// `rows` restricts the tree to a bagged subset (ascending row ids); `None` uses all rows.
    pub fn train(&mut self, gradients: &[f32], hessians: &[f32], rows: Option<&[i32]>) -> Tree {
        self.grow_tree(gradients, hessians, rows)
    }

    fn grow_tree(&mut self, gradients: &[f32], hessians: &[f32], rows: Option<&[i32]>) -> Tree {
        self.before_train(gradients, hessians, rows);
        self.splits.clear();
        let num_leaves = self.cfg.num_leaves as usize;
        let mut tree = Tree::new_training(num_leaves);
        let root = leaf_output_root(
            &self.params,
            self.smaller.sum_gradients,
            self.smaller.sum_hessians,
            self.smaller.num_data_in_leaf,
        );
        tree.set_leaf_output(0, root);

        let mut left_leaf: i32 = 0;
        let mut right_leaf: i32 = -1;
        // upstream: `SerialTreeLearner::Train` applies the forced splits first; the normal
        // loop then makes the remaining `num_leaves - 1 - init_splits` splits.
        let init_splits = match self.forced.clone() {
            Some(forced) => self.force_splits(
                &forced,
                &mut tree,
                &mut left_leaf,
                &mut right_leaf,
                gradients,
                hessians,
            ),
            None => 0,
        };
        for _ in init_splits..num_leaves.saturating_sub(1) {
            if self.before_find_best_split(&tree, left_leaf, right_leaf) {
                self.find_best_splits(&tree, gradients, hessians, None);
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
    fn before_train(&mut self, gradients: &[f32], hessians: &[f32], rows: Option<&[i32]>) {
        self.pool.reset_map();
        for b in &mut self.branch_features {
            b.clear();
        }
        self.monotone_bounds.fill((-f64::MAX, f64::MAX));
        self.leaf_depth.fill(0);
        self.col_sampler.reset_by_tree();
        match rows {
            Some(r) => self.partition.init_with(r),
            None => self.partition.init(),
        }
        for s in &mut self.best_split_per_leaf {
            s.reset();
        }
        // upstream LeafSplits::Init(gradients, hessians): sequential double sums, over the
        // bagged rows when a subset is used (`LeafSplits::Init(leaf, partition, ...)`).
        let mut sg = 0.0f64;
        let mut sh = 0.0f64;
        let count = match rows {
            Some(r) => {
                for &i in r {
                    sg += f64::from(gradients[i as usize]);
                    sh += f64::from(hessians[i as usize]);
                }
                r.len()
            }
            None => {
                for i in 0..self.num_data {
                    sg += f64::from(gradients[i]);
                    sh += f64::from(hessians[i]);
                }
                self.num_data
            }
        };
        self.smaller = LeafSplits {
            leaf_index: 0,
            num_data_in_leaf: count as i32,
            sum_gradients: sg,
            sum_hessians: sh,
            weight: 0.0,
        };
        self.larger = LeafSplits::empty();
    }

    /// upstream: `SerialTreeLearner::BeforeFindBestSplit`
    fn before_find_best_split(&mut self, tree: &Tree, left_leaf: i32, right_leaf: i32) -> bool {
        let cfg = Arc::clone(&self.cfg);
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
    /// `forced` lists the inner features that must get histograms even when the by-tree
    /// sampler skips them (upstream: `FindBestSplits(tree, force_features)`).
    fn find_best_splits(
        &mut self,
        tree: &Tree,
        gradients: &[f32],
        hessians: &[f32],
        forced: Option<&[bool]>,
    ) {
        let nf = self.num_features;
        // upstream: the by-tree mask gates histogram construction and split search.
        let bytree = self.col_sampler.is_feature_used_bytree().to_vec();
        let mut is_feature_used = vec![false; nf];
        for f in 0..nf {
            let is_forced = forced.is_some_and(|m| m[f]);
            if !bytree[f] && !is_forced {
                continue;
            }
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
        let ds = Arc::clone(&self.ds);
        let l = &ds.layout;
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
        let rows = indices.and_then(|ix| histogram::CheckedRows::new(ix, self.num_data));
        assert!(
            indices.is_none() || rows.is_some(),
            "partition produced an invalid row index"
        );
        // Dense single-value groups: each writes its own slice of the histogram, so they
        // can be built as independent tasks (upstream: parallel over feature groups).
        let mut jobs: Vec<(usize, usize, usize)> = Vec::new(); // (group, start bin, bins)
        let mut group_start = 0usize;
        let mut multi_val_start = None;
        for (gi, g) in l.groups.iter().enumerate() {
            let num_bin = *g.bin_offsets.last().unwrap_or(&0) as usize;
            let fs = l.group_feature_start[gi] as usize;
            let fc = l.group_feature_cnt[gi] as usize;
            let used = (fs..fs + fc).any(|f| is_feature_used[f]);
            if g.is_multi_val {
                if used {
                    multi_val_start = Some(group_start);
                }
            } else if used {
                jobs.push((gi, group_start, num_bin));
            }
            group_start += num_bin;
        }
        let checked_bins = &self.checked_bins;
        let build = |gi: usize, out: &mut [f64]| {
            let (GroupData::Single(BinData::Dense(bin)), Some(check)) =
                (&ds.data[gi], checked_bins[gi])
            else {
                unreachable!("checked in new()")
            };
            let checked = histogram::CheckedBins::with(bin, check);
            out.fill(0.0);
            histogram::construct(&checked, rows.as_ref(), og, oh, const_hess, out);
            if const_hess {
                let h0 = f64::from(hessians[0]);
                let num_bin = out.len() / 2;
                for i in (0..num_bin * 2).step_by(2) {
                    out[i + 1] *= h0;
                }
            }
        };
        let hist = &mut self.pool.slots[slot];
        let parallel_done = match self.spin.as_ref() {
            Some(pool) if jobs.len() > 1 && leaf_cnt * jobs.len() >= PARALLEL_HIST_MIN_WORK => {
                // Groups own disjoint, ascending bin ranges.
                let ranges: Vec<(usize, usize)> = jobs
                    .iter()
                    .map(|&(_, gs, nb)| (gs * 2, (gs + nb) * 2))
                    .collect();
                pool.for_each_range_mut(&mut hist[..], &ranges, |j, out| build(jobs[j].0, out))
                    .is_ok()
            }
            _ => false,
        };
        if !parallel_done {
            for &(gi, gs, nb) in &jobs {
                build(gi, &mut hist[gs * 2..(gs + nb) * 2]);
            }
        }
        // upstream: ConstructHistogramsMultiVal after the dense groups. Values are the same
        // whether upstream reads gathered (`ORDERED`) or indexed derivatives, so position
        // `i` of the gathered arrays is used. It always accumulates the actual hessians:
        // unlike dense groups there is no count-times-`hessians[0]` path.
        if let (Some(start), Some((_, mv))) = (multi_val_start, self.multi_val.as_ref()) {
            let num_threads = if self.cfg.num_threads > 0 {
                self.cfg.num_threads as usize
            } else {
                rayon::current_num_threads()
            };
            let h0 = hessians.first().copied().unwrap_or(0.0);
            let grad_at = |i: usize, _row: usize| og[i];
            let hess_at = |i: usize, _row: usize| if const_hess { h0 } else { oh[i] };
            let out = &mut hist[start * 2..];
            mv.construct(
                indices,
                leaf_cnt,
                num_threads,
                self.spin.as_ref(),
                &grad_at,
                &hess_at,
                out,
                &mut self.multi_val_scratch,
            );
        }
    }

    /// upstream: `Dataset::FixHistogram`
    fn fix_histogram(&mut self, slot: usize, feature: usize, sum_gradient: f64, sum_hessian: f64) {
        let l = &self.ds.layout;
        let g = &l.groups[l.feature2group[feature] as usize];
        let m = &g.bin_mappers[l.feature2subfeature[feature] as usize];
        let mfb = m.most_freq_bin as usize;
        if mfb > 0 {
            // Only a nonzero most-frequent bin is fixed, so the slice is taken only then:
            // for bin 0 the range can extend past the end of the histogram.
            let num_bin = m.num_bin as usize;
            let start = self.feature_hist_offsets[feature] * 2;
            let data = &mut self.pool.slots[slot][start..start + num_bin * 2];
            fix_slice(data, mfb, num_bin, sum_gradient, sum_hessian);
        }
    }

    /// Output bounds of a leaf when monotone constraints are active (upstream: the
    /// `USE_MC` template instantiations).
    fn mc_bounds(&self, leaf: i32) -> Option<(f64, f64)> {
        if self.monotone_active {
            Some(self.monotone_bounds[leaf as usize])
        } else {
            None
        }
    }

    /// upstream: `SerialTreeLearner::FindBestSplitsFromHistograms`
    ///
    /// Two phases. Phase 1 is sequential and cheap: each used feature's smaller-leaf
    /// histogram is fixed and the larger leaf's is derived, touching only that feature's
    /// bins. Phase 2 searches thresholds feature by feature, in parallel once the work is
    /// large enough. A search reads histograms and mutates only its own `FeatureMeta`
    /// (including its random stream), so each candidate is the same for any thread count.
    /// The candidates are folded in feature order with upstream's `better_than`, so ties
    /// keep the lowest feature, as `ArgMax` over features does.
    fn find_best_splits_from_histograms(
        &mut self,
        is_feature_used: &[bool],
        use_subtract: bool,
        tree: &Tree,
    ) {
        // upstream draws the by-node sample for the smaller leaf, then the larger leaf.
        let smaller_mask = self
            .col_sampler
            .get_by_node(&self.branch_features[self.smaller.leaf_index as usize]);
        let larger_mask = if self.larger.leaf_index >= 0 {
            self.col_sampler
                .get_by_node(&self.branch_features[self.larger.leaf_index as usize])
        } else {
            Vec::new()
        };
        let smaller_parent_output = self.parent_output(tree, &self.smaller);
        let larger_parent_output = if self.larger.leaf_index >= 0 {
            self.parent_output(tree, &self.larger)
        } else {
            0.0
        };
        let larger = if self.larger.leaf_index >= 0 {
            self.larger_slot
        } else {
            None
        };
        let (smaller_slot, smaller, larger_leaf) = (self.smaller_slot, self.smaller, self.larger);
        let mc_smaller = self.mc_bounds(smaller.leaf_index);
        let mc_larger = match larger {
            Some(_) => self.mc_bounds(larger_leaf.leaf_index),
            None => None,
        };

        // Phase 2: threshold searches. Parallel only when the work outweighs a fork/join.
        let searched_bins: usize = (0..self.num_features)
            .filter(|&f| is_feature_used[f])
            .map(|f| self.feature_hist_len[f])
            .sum::<usize>()
            * if larger.is_some() { 2 } else { 1 };
        let parallel = self.num_threads > 1 && searched_bins >= PARALLEL_SEARCH_MIN_BINS;
        let inp = SearchInputs {
            is_feature_used,
            use_subtract,
            smaller_slot,
            smaller,
            smaller_parent_output,
            mc_smaller,
            larger: larger.map(|slot| (slot, larger_leaf)),
            larger_parent_output,
            mc_larger,
        };
        // Parallel: fused per-feature phases (see `fused_search`). Otherwise, or when the
        // histogram extents overlap, the two sequential phases below.
        let fused = if parallel {
            self.fused_search(&inp)
        } else {
            None
        };
        let outcomes: Vec<Option<FeatureOutcome>> = match fused {
            Some(outcomes) => outcomes,
            None => {
                // Phase 1 (sequential).
                for f in 0..self.num_features {
                    if !is_feature_used[f] {
                        continue;
                    }
                    self.fix_histogram(
                        smaller_slot,
                        f,
                        smaller.sum_gradients,
                        smaller.sum_hessians,
                    );
                    if let Some(ls) = larger {
                        if use_subtract {
                            let start = self.feature_hist_offsets[f] * 2;
                            let len = self.feature_hist_len[f] * 2;
                            let (small, large) = two_slots(&mut self.pool.slots, smaller_slot, ls);
                            for i in start..start + len {
                                large[i] -= small[i];
                            }
                        } else {
                            self.fix_histogram(
                                ls,
                                f,
                                larger_leaf.sum_gradients,
                                larger_leaf.sum_hessians,
                            );
                        }
                    }
                }

                let ctx = SearchCtx {
                    params: &self.params,
                    cat_params: &self.cat_params,
                    monotone_active: self.monotone_active,
                    monotone_penalization: self.monotone_penalization,
                    leaf_depth: &self.leaf_depth,
                };
                let slots = &self.pool.slots;
                let offsets = &self.feature_hist_offsets;
                let lens = &self.feature_hist_len;
                let real_idx = &self.ds.layout.real_feature_idx;
                let search = |f: usize, meta: &mut FeatureMeta| -> Option<FeatureOutcome> {
                    if !is_feature_used[f] {
                        return None;
                    }
                    let range = |slot: usize| {
                        let start = offsets[f] * 2;
                        &slots[slot][start..start + lens[f] * 2]
                    };
                    let real = real_idx[f];
                    let small = search_feature(
                        &ctx,
                        range(smaller_slot),
                        meta,
                        real,
                        &smaller,
                        smaller_parent_output,
                        mc_smaller,
                    );
                    let large = larger.map(|ls| {
                        search_feature(
                            &ctx,
                            range(ls),
                            meta,
                            real,
                            &larger_leaf,
                            larger_parent_output,
                            mc_larger,
                        )
                    });
                    Some(FeatureOutcome { small, large })
                };
                match self.spin.as_ref() {
                    Some(pool) if parallel => {
                        let mut work: Vec<(&mut FeatureMeta, Option<FeatureOutcome>)> =
                            self.metas.iter_mut().map(|m| (m, None)).collect();
                        pool.for_each_mut(&mut work, |f, item| {
                            item.1 = search(f, &mut *item.0);
                        });
                        work.into_iter().map(|(_, o)| o).collect()
                    }
                    _ => self
                        .metas
                        .iter_mut()
                        .enumerate()
                        .map(|(f, meta)| search(f, meta))
                        .collect(),
                }
            }
        };

        // Fold in feature order, as the sequential search did.
        let mut smaller_best = SplitInfo::default();
        let mut larger_best = SplitInfo::default();
        for (f, outcome) in outcomes.into_iter().enumerate() {
            let Some(o) = outcome else { continue };
            let (splittable, candidate) = o.small;
            self.pool.splittable[smaller_slot][f] = splittable;
            if smaller_mask[f] && candidate.better_than(&smaller_best) {
                smaller_best = candidate;
            }
            if let (Some((splittable, candidate)), Some(ls)) = (o.large, larger) {
                self.pool.splittable[ls][f] = splittable;
                if larger_mask[f] && candidate.better_than(&larger_best) {
                    larger_best = candidate;
                }
            }
        }
        self.best_split_per_leaf[smaller.leaf_index as usize] = smaller_best;
        if larger_leaf.leaf_index >= 0 {
            self.best_split_per_leaf[larger_leaf.leaf_index as usize] = larger_best;
        }
    }

    /// Phases 1 and 2 for the parallel path, fused per feature. Each used feature's fix (and
    /// the subtraction for the larger leaf) runs in the same task as its threshold search, so
    /// the thread that writes a feature's histogram also reads it, and phase 1 runs in
    /// parallel. Every feature owns disjoint histogram slices, and each sum keeps the order of
    /// the sequential code, so the results are identical. Returns `None` when the features'
    /// histogram extents overlap; the caller then runs the sequential order.
    fn fused_search(&mut self, inp: &SearchInputs<'_>) -> Option<Vec<Option<FeatureOutcome>>> {
        let pool = self.spin.as_ref()?;
        // (start, extent, feature, most-frequent bin, num_bin, searched length, real index)
        // in histogram entries (two per bin). The fix writes `num_bin` bins, the search reads
        // `feature_hist_len` bins, and the extent covers both.
        let mut spans: Vec<(usize, usize, usize, usize, usize, usize, i32)> = Vec::new();
        for f in 0..self.num_features {
            if !inp.is_feature_used[f] {
                continue;
            }
            let l = &self.ds.layout;
            let m = &l.groups[l.feature2group[f] as usize].bin_mappers
                [l.feature2subfeature[f] as usize];
            let num_bin = m.num_bin as usize;
            let len_bins = self.feature_hist_len[f];
            let mfb = m.most_freq_bin as usize;
            // A feature with bin 0 as most frequent is not fixed, so it reads only `len_bins`.
            let extent_bins = if mfb > 0 {
                num_bin.max(len_bins)
            } else {
                len_bins
            };
            spans.push((
                self.feature_hist_offsets[f] * 2,
                extent_bins * 2,
                f,
                mfb,
                num_bin,
                len_bins * 2,
                l.real_feature_idx[f],
            ));
        }
        spans.sort_unstable_by_key(|s| s.0);
        let mut end = 0usize;
        for s in &spans {
            if s.0 < end {
                return None;
            }
            end = s.0 + s.1;
        }
        if end > self.pool.slots[inp.smaller_slot].len()
            || inp
                .larger
                .is_some_and(|(ls, _)| end > self.pool.slots[ls].len())
        {
            return None;
        }

        let (small_buf, large_buf) = match inp.larger {
            Some((ls, _)) => {
                let (a, b) = two_slots_mut(&mut self.pool.slots, inp.smaller_slot, ls);
                (a, Some(b))
            }
            None => (&mut self.pool.slots[inp.smaller_slot], None),
        };
        let spans_iter = || spans.iter().map(|s| (s.0, s.1));
        let small_pieces = carve_mut(small_buf.as_mut_slice(), spans_iter());
        let large_pieces: Vec<Option<&mut [f64]>> = match large_buf {
            Some(b) => carve_mut(b.as_mut_slice(), spans_iter())
                .into_iter()
                .map(Some)
                .collect(),
            None => spans.iter().map(|_| None).collect(),
        };
        let mut metas: Vec<Option<&mut FeatureMeta>> = self.metas.iter_mut().map(Some).collect();
        let mut work: Vec<(Option<FeatureWork<'_>>, Option<FeatureOutcome>)> =
            (0..self.num_features).map(|_| (None, None)).collect();
        for ((s, small), large) in spans.iter().zip(small_pieces).zip(large_pieces) {
            let meta = metas[s.2].take()?;
            work[s.2].0 = Some(FeatureWork {
                meta,
                small,
                large,
                mfb: s.3,
                num_bin: s.4,
                search_len: s.5,
                real: s.6,
            });
        }

        let ctx = SearchCtx {
            params: &self.params,
            cat_params: &self.cat_params,
            monotone_active: self.monotone_active,
            monotone_penalization: self.monotone_penalization,
            leaf_depth: &self.leaf_depth,
        };
        pool.for_each_mut(&mut work, |_f, item| {
            if let Some(w) = item.0.as_mut() {
                item.1 = Some(run_feature(&ctx, inp, w));
            }
        });
        Some(work.into_iter().map(|(_, outcome)| outcome).collect())
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

    /// upstream: `SerialTreeLearner::SplitInner` (numerical). Returns (left, right) leaves.
    fn split(&mut self, tree: &mut Tree, best_leaf: usize) -> (usize, usize) {
        let mut info = self.best_split_per_leaf[best_leaf].clone();
        let ds = Arc::clone(&self.ds);
        let l = &ds.layout;
        let inner = l.used_feature_map[info.feature as usize] as usize;
        let g = &l.groups[l.feature2group[inner] as usize];
        let m = &g.bin_mappers[l.feature2subfeature[inner] as usize];
        let next_leaf = tree.num_leaves;
        // upstream: `Tree::Split` keeps the path features of both children.
        let path = self.branch_features[best_leaf].clone();
        self.branch_features[next_leaf] = path;
        self.branch_features[next_leaf].push(info.feature);
        self.branch_features[best_leaf].push(info.feature);
        let gain = (info.gain + self.cfg.min_gain_to_split) as f32;
        let right = if m.bin_type == BinType::Categorical {
            // upstream: `SerialTreeLearner::SplitInner` (categorical). The tree stores the
            // category values; partitioning uses the bins.
            let num = info.num_cat_threshold as usize;
            let bins = info.cat_threshold[..num].to_vec();
            let real: Vec<u32> = bins
                .iter()
                .map(|&b| m.bin_2_categorical[b as usize] as u32)
                .collect();
            let bitset_real = construct_bitset(&real);
            let bitset_bin = construct_bitset(&bins);
            let rule = SplitRule::categorical(&ds, inner, &bitset_bin);
            self.partition
                .split(best_leaf, &rule, next_leaf, self.spin.as_ref());
            self.splits.push(SplitRecord {
                leaf: best_leaf,
                right: next_leaf,
                inner,
                kind: SplitKind::Categorical(bitset_bin),
            });
            info.left_count = self.partition.leaf_count(best_leaf) as i32;
            info.right_count = self.partition.leaf_count(next_leaf) as i32;
            tree.split_categorical(
                best_leaf,
                inner as i32,
                info.feature,
                &bitset_real,
                info.left_output,
                info.right_output,
                info.left_count,
                info.right_count,
                info.left_sum_hessian,
                info.right_sum_hessian,
                gain,
                m.missing_type,
            )
        } else {
            let threshold_double = m.bin_to_value(info.threshold);
            let rule = SplitRule::numerical(&ds, inner, info.threshold, info.default_left);
            self.partition
                .split(best_leaf, &rule, next_leaf, self.spin.as_ref());
            self.splits.push(SplitRecord {
                leaf: best_leaf,
                right: next_leaf,
                inner,
                kind: SplitKind::Numerical {
                    threshold: info.threshold,
                    default_left: info.default_left,
                },
            });
            info.left_count = self.partition.leaf_count(best_leaf) as i32;
            info.right_count = self.partition.leaf_count(next_leaf) as i32;
            tree.split_numerical(
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
            )
        };
        // upstream: BasicLeafConstraints::Update. The new leaf starts from the parent's
        // bounds; numerical monotone splits then clamp the children at the midpoint.
        let mono = self.metas[inner].monotone_type;
        self.monotone_bounds[next_leaf] = self.monotone_bounds[best_leaf];
        if self.metas[inner].bin_type != BinType::Categorical {
            let mid = (info.left_output + info.right_output) / 2.0;
            if mono < 0 {
                let b = &mut self.monotone_bounds[best_leaf];
                b.0 = b.0.max(mid);
                let n = &mut self.monotone_bounds[next_leaf];
                n.1 = n.1.min(mid);
            } else if mono > 0 {
                let b = &mut self.monotone_bounds[best_leaf];
                b.1 = b.1.min(mid);
                let n = &mut self.monotone_bounds[next_leaf];
                n.0 = n.0.max(mid);
            }
        }
        self.leaf_depth[next_leaf] = self.leaf_depth[best_leaf] + 1;
        self.leaf_depth[best_leaf] += 1;
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
            if tree.is_linear {
                for &idx in self.partition.indices_on_leaf(leaf) {
                    score[idx as usize] += self.linear_leaf_output(tree, leaf, idx as usize);
                }
                continue;
            }
            let out = tree.leaf_value[leaf];
            for &idx in self.partition.indices_on_leaf(leaf) {
                score[idx as usize] += out;
            }
        }
    }

    /// upstream: `GBDT::UpdateScore` for out-of-bag rows (`AddScore(tree, indices)`). The
    /// last tree's splits are replayed on `rows` with the same bin routing used in training,
    /// so each row lands in the leaf the tree would have sent it to.
    pub fn add_out_of_bag_prediction(&self, tree: &Tree, rows: &[i32], score: &mut [f64]) {
        if tree.num_leaves <= 1 {
            return;
        }
        self.add_prediction_for_rows(tree, &self.splits, rows, score);
    }

    /// The splits of the last tree trained (for later replays on any rows).
    pub(crate) fn last_splits(&self) -> Vec<SplitRecord> {
        self.splits.clone()
    }

    /// Adds `tree` to `score` for `rows`, routing them through `splits` with the training
    /// routing (upstream: `Tree::AddPredictionToScore` on the given rows).
    pub(crate) fn add_prediction_for_rows(
        &self,
        tree: &Tree,
        splits: &[SplitRecord],
        rows: &[i32],
        score: &mut [f64],
    ) {
        if rows.is_empty() {
            return;
        }
        let mut part = DataPartition::new(self.num_data, tree.num_leaves.max(1));
        part.init_with(rows);
        for s in splits {
            let rule = match &s.kind {
                SplitKind::Numerical {
                    threshold,
                    default_left,
                } => SplitRule::numerical(&self.ds, s.inner, *threshold, *default_left),
                SplitKind::Categorical(bins) => SplitRule::categorical(&self.ds, s.inner, bins),
            };
            part.split(s.leaf, &rule, s.right, None);
        }
        for leaf in 0..tree.num_leaves {
            if tree.is_linear {
                for &idx in part.indices_on_leaf(leaf) {
                    score[idx as usize] += self.linear_leaf_output(tree, leaf, idx as usize);
                }
                continue;
            }
            let out = tree.leaf_value[leaf];
            for &idx in part.indices_on_leaf(leaf) {
                score[idx as usize] += out;
            }
        }
    }
}

/// Inputs shared by every feature's search in one `find_best_splits_from_histograms` call.
struct SearchInputs<'a> {
    is_feature_used: &'a [bool],
    use_subtract: bool,
    smaller_slot: usize,
    smaller: LeafSplits,
    smaller_parent_output: f64,
    mc_smaller: Option<(f64, f64)>,
    /// Slot and leaf of the larger leaf, when it is being searched.
    larger: Option<(usize, LeafSplits)>,
    larger_parent_output: f64,
    mc_larger: Option<(f64, f64)>,
}

/// One feature's disjoint histogram slices and its metadata, for the fused search.
struct FeatureWork<'a> {
    meta: &'a mut FeatureMeta,
    small: &'a mut [f64],
    large: Option<&'a mut [f64]>,
    mfb: usize,
    num_bin: usize,
    search_len: usize,
    real: i32,
}

/// Phases 1 and 2 for one feature, in the sequential order: fix the smaller leaf, derive the
/// larger leaf (subtraction, or its own fix), then search both leaves.
fn run_feature(
    ctx: &SearchCtx<'_>,
    inp: &SearchInputs<'_>,
    w: &mut FeatureWork<'_>,
) -> FeatureOutcome {
    fix_slice(
        w.small,
        w.mfb,
        w.num_bin,
        inp.smaller.sum_gradients,
        inp.smaller.sum_hessians,
    );
    if let Some(large) = w.large.as_deref_mut() {
        if inp.use_subtract {
            for (l, s) in large[..w.search_len]
                .iter_mut()
                .zip(&w.small[..w.search_len])
            {
                *l -= *s;
            }
        } else if let Some((_, leaf)) = inp.larger {
            fix_slice(
                large,
                w.mfb,
                w.num_bin,
                leaf.sum_gradients,
                leaf.sum_hessians,
            );
        }
    }
    let small = search_feature(
        ctx,
        &w.small[..w.search_len],
        w.meta,
        w.real,
        &inp.smaller,
        inp.smaller_parent_output,
        inp.mc_smaller,
    );
    let large = match (w.large.as_deref(), inp.larger) {
        (Some(large), Some((_, leaf))) => Some(search_feature(
            ctx,
            &large[..w.search_len],
            w.meta,
            w.real,
            &leaf,
            inp.larger_parent_output,
            inp.mc_larger,
        )),
        _ => None,
    };
    FeatureOutcome { small, large }
}

/// upstream: `Dataset::FixHistogram` on one feature's `num_bin` bins (the body of
/// `SerialTreeLearner::fix_histogram`): the most frequent bin takes the leaf totals minus the rest.
fn fix_slice(data: &mut [f64], mfb: usize, num_bin: usize, sum_gradient: f64, sum_hessian: f64) {
    if mfb > 0 {
        let data = &mut data[..num_bin * 2];
        data[mfb << 1] = sum_gradient;
        data[(mfb << 1) + 1] = sum_hessian;
        for i in 0..num_bin {
            if i != mfb {
                data[mfb << 1] -= data[i << 1];
                data[(mfb << 1) + 1] -= data[(i << 1) + 1];
            }
        }
    }
}

/// Splits `buf` into the given ascending, disjoint `(start, len)` spans, in order. Spans are
/// validated by the caller, so the splits cannot go out of range.
fn carve_mut(mut buf: &mut [f64], spans: impl Iterator<Item = (usize, usize)>) -> Vec<&mut [f64]> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    for (start, len) in spans {
        let (_skip, after) = buf.split_at_mut(start - pos);
        let (piece, tail) = after.split_at_mut(len);
        out.push(piece);
        buf = tail;
        pos = start + len;
    }
    out
}

/// Two distinct slots, both mutable.
fn two_slots_mut(slots: &mut [Vec<f64>], a: usize, b: usize) -> (&mut Vec<f64>, &mut Vec<f64>) {
    assert_ne!(a, b);
    if a < b {
        let (x, y) = slots.split_at_mut(b);
        (&mut x[a], &mut y[0])
    } else {
        let (x, y) = slots.split_at_mut(a);
        (&mut y[0], &mut x[b])
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

/// upstream: `LeafConstraintsBase::ComputeMonotoneSplitGainPenalty`. Applied to every split
/// on a monotone feature, so with the default `monotone_penalty = 0` it is `1 + kEpsilon`.
fn monotone_split_gain_penalty(depth: i32, penalization: f64) -> f64 {
    let eps = f64::from(crate::meta::K_EPSILON);
    let d = f64::from(depth);
    if penalization >= d + 1.0 {
        return eps;
    }
    if penalization <= 1.0 {
        return 1.0 - penalization / 2f64.powf(d) + eps;
    }
    1.0 - 2f64.powf(penalization - 1.0 - d) + eps
}

/// Rows x dense groups one histogram pass must cover before its groups run in parallel.
const PARALLEL_HIST_MIN_WORK: usize = 1 << 12;

/// Histogram bins one split search must cover before its feature searches run in parallel.
/// Below this, the fork/join of a parallel region costs more than the searches themselves.
const PARALLEL_SEARCH_MIN_BINS: usize = 1 << 10;

/// Read-only settings shared by every feature of one split search.
struct SearchCtx<'a> {
    params: &'a SplitParams,
    cat_params: &'a CatParams,
    monotone_active: bool,
    monotone_penalization: f64,
    leaf_depth: &'a [i32],
}

/// One used feature's results: `(splittable, candidate)` for the smaller leaf and, when
/// there is a larger leaf, for that leaf too.
struct FeatureOutcome {
    small: (bool, SplitInfo),
    large: Option<(bool, SplitInfo)>,
}

/// upstream: `SerialTreeLearner::ComputeBestSplitForFeature`, without the best-split update.
/// Returns whether the feature stays splittable for the children, and the candidate split.
/// The caller keeps the candidate only when the by-node mask allows the feature; the
/// threshold search always runs because it sets the splittable flag.
fn search_feature(
    ctx: &SearchCtx<'_>,
    hist: &[f64],
    meta: &mut FeatureMeta,
    real_fidx: i32,
    ls: &LeafSplits,
    parent_output: f64,
    mc: Option<(f64, f64)>,
) -> (bool, SplitInfo) {
    let mut new_split = SplitInfo::default();
    let splittable = if meta.bin_type == BinType::Categorical {
        find_best_threshold_categorical(
            hist,
            meta,
            ctx.params,
            ctx.cat_params,
            ls.sum_gradients,
            ls.sum_hessians,
            ls.num_data_in_leaf,
            parent_output,
            mc,
            &mut new_split,
        )
    } else {
        find_best_threshold_numerical(
            hist,
            meta,
            ctx.params,
            ls.sum_gradients,
            ls.sum_hessians,
            ls.num_data_in_leaf,
            parent_output,
            mc,
            &mut new_split,
        )
    };
    // upstream: ComputeBestSplitForFeature, monotone_penalty (applied even at 0)
    if ctx.monotone_active && meta.monotone_type != 0 {
        let depth = ctx.leaf_depth[ls.leaf_index as usize];
        new_split.gain *= monotone_split_gain_penalty(depth, ctx.monotone_penalization);
    }
    new_split.feature = real_fidx;
    (splittable, new_split)
}
