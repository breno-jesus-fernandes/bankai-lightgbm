//! Cost-efficient gradient boosting (CEGB).
//!
//! upstream: src/treelearner/cost_effective_gradient_boosting.hpp (v4.7.0)
//!
//! CEGB subtracts a cost penalty from each candidate split's gain before the split is chosen,
//! so costlier features and splits are used only when they pay for themselves. The penalty has
//! three parts, all scaled by `cegb_tradeoff`: `cegb_penalty_split` per data point in the leaf,
//! a one-off `cegb_penalty_feature_coupled[f]` the first time feature `f` is used in any split,
//! and a lazy `cegb_penalty_feature_lazy[f]` per data point in the leaf that has not yet used
//! feature `f`.

use super::split_info::{K_MIN_SCORE, SplitInfo};
use crate::config::Config;
use crate::error::{Error, Result};

/// CEGB state for one tree learner. Feature penalties index by *real* feature; the per-leaf and
/// per-data bookkeeping indexes by *inner* (used) feature.
pub(crate) struct Cegb {
    tradeoff: f64,
    penalty_split: f64,
    /// Per real feature; empty when `cegb_penalty_feature_coupled` is unset.
    coupled: Vec<f64>,
    /// Per real feature; empty when `cegb_penalty_feature_lazy` is unset.
    lazy: Vec<f64>,
    /// Number of inner (used) features.
    num_features: usize,
    num_data: usize,
    /// The last split seen for each `(leaf, inner feature)`, `num_leaves * num_features`.
    /// upstream: `splits_per_leaf_`.
    splits_per_leaf: Vec<SplitInfo>,
    /// Whether each inner feature has been used in a split yet. upstream: `is_feature_used_in_split_`.
    is_feature_used_in_split: Vec<bool>,
    /// `inner feature * num_data + data index`; empty unless `lazy` is set.
    /// upstream: `feature_used_in_data_` (a bitset there; a `bool` vector here, same behaviour).
    feature_used_in_data: Vec<bool>,
}

impl Cegb {
    /// upstream: `CostEfficientGradientBoosting::IsEnable`.
    pub fn is_enabled(cfg: &Config) -> bool {
        !(cfg.cegb_tradeoff >= 1.0
            && cfg.cegb_penalty_split <= 0.0
            && cfg.cegb_penalty_feature_coupled.is_empty()
            && cfg.cegb_penalty_feature_lazy.is_empty())
    }

    /// Build the state, validating the per-feature vectors against the total feature count, as
    /// upstream's `Init` does (a `Log::Fatal` there). Returns `None` when CEGB is not enabled.
    pub fn new(
        cfg: &Config,
        num_features: usize,
        num_total_features: usize,
        num_leaves: usize,
        num_data: usize,
    ) -> Result<Option<Self>> {
        if !Self::is_enabled(cfg) {
            return Ok(None);
        }
        if !cfg.cegb_penalty_feature_coupled.is_empty()
            && cfg.cegb_penalty_feature_coupled.len() != num_total_features
        {
            return Err(Error::InvalidParameter(
                "cegb_penalty_feature_coupled should be the same size as feature number.".into(),
            ));
        }
        if !cfg.cegb_penalty_feature_lazy.is_empty()
            && cfg.cegb_penalty_feature_lazy.len() != num_total_features
        {
            return Err(Error::InvalidParameter(
                "cegb_penalty_feature_lazy should be the same size as feature number.".into(),
            ));
        }
        let lazy_len = if cfg.cegb_penalty_feature_lazy.is_empty() {
            0
        } else {
            num_features * num_data
        };
        Ok(Some(Self {
            tradeoff: cfg.cegb_tradeoff,
            penalty_split: cfg.cegb_penalty_split,
            coupled: cfg.cegb_penalty_feature_coupled.clone(),
            lazy: cfg.cegb_penalty_feature_lazy.clone(),
            num_features,
            num_data,
            splits_per_leaf: vec![SplitInfo::default(); num_leaves * num_features],
            is_feature_used_in_split: vec![false; num_features],
            feature_used_in_data: vec![false; lazy_len],
        }))
    }

    /// Clear the per-leaf splits before each tree. upstream: `BeforeTrain`.
    pub fn before_train(&mut self) {
        for s in &mut self.splits_per_leaf {
            s.reset();
        }
    }

    /// The gain penalty for a candidate split on `inner_feature` (real index `real_feature`) of
    /// `leaf`, which has `num_data_in_leaf` rows with indices `leaf_indices`. Records the split
    /// (with its pre-penalty gain) for the later coupled refund, as upstream's `DeltaGain` does.
    pub fn delta_gain(
        &mut self,
        inner_feature: usize,
        real_feature: usize,
        leaf: usize,
        num_data_in_leaf: i32,
        split: &SplitInfo,
        leaf_indices: &[i32],
    ) -> f64 {
        let mut delta = self.tradeoff * self.penalty_split * f64::from(num_data_in_leaf);
        if !self.coupled.is_empty() && !self.is_feature_used_in_split[inner_feature] {
            delta += self.tradeoff * self.coupled[real_feature];
        }
        if !self.lazy.is_empty() {
            delta += self.tradeoff * self.ondemand_cost(inner_feature, real_feature, leaf_indices);
        }
        self.splits_per_leaf[leaf * self.num_features + inner_feature] = split.clone();
        delta
    }

    /// The lazy per-data cost: one `lazy[real]` for each row in the leaf that has not yet used the
    /// feature. upstream: `CalculateOndemandCosts`.
    fn ondemand_cost(
        &self,
        inner_feature: usize,
        real_feature: usize,
        leaf_indices: &[i32],
    ) -> f64 {
        if self.lazy.is_empty() {
            return 0.0;
        }
        let penalty = self.lazy[real_feature];
        let base = inner_feature * self.num_data;
        let mut total = 0.0;
        for &di in leaf_indices {
            if self.feature_used_in_data[base + di as usize] {
                continue;
            }
            total += penalty;
        }
        total
    }

    /// After the best split for `best_leaf` is chosen (inner feature `inner_feature`, real index
    /// `real_feature`), refund the coupled penalty to the other leaves' stored splits now that the
    /// feature is paid for, and mark the leaf's rows as having used the feature. upstream:
    /// `UpdateLeafBestSplits`. `num_leaves` is the tree's current leaf count.
    pub fn update_leaf_best_splits(
        &mut self,
        best_leaf: usize,
        inner_feature: usize,
        real_feature: usize,
        num_leaves: usize,
        best_split_per_leaf: &mut [SplitInfo],
        best_leaf_indices: &[i32],
    ) {
        if !self.coupled.is_empty() && !self.is_feature_used_in_split[inner_feature] {
            self.is_feature_used_in_split[inner_feature] = true;
            let refund = self.tradeoff * self.coupled[real_feature];
            for i in 0..num_leaves {
                if i == best_leaf {
                    continue;
                }
                let split = &mut self.splits_per_leaf[i * self.num_features + inner_feature];
                split.gain += refund;
                if best_split_per_leaf[i].gain > K_MIN_SCORE
                    && split.better_than(&best_split_per_leaf[i])
                {
                    best_split_per_leaf[i] = split.clone();
                }
            }
        }
        if !self.lazy.is_empty() {
            let base = inner_feature * self.num_data;
            for &di in best_leaf_indices {
                self.feature_used_in_data[base + di as usize] = true;
            }
        }
    }
}
