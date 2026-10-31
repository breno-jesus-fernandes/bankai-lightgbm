//! Feature sub-sampling per tree and per node. upstream: src/treelearner/col_sampler.hpp
//!
//! The draws come from one LCG stream seeded by `feature_fraction_seed`, consumed in the
//! same order as upstream (by-tree once per tree, then by-node for the smaller leaf and
//! then the larger leaf), so the sampled features match bit for bit.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Dataset;
use crate::utils::random::Random;

/// upstream: `ColSampler::GetCnt`
pub fn get_cnt(total: usize, fraction: f64) -> usize {
    let min = 1.min(total);
    // upstream: `Common::RoundInt(x)` is `static_cast<int>(x + 0.5f)`
    let used = (total as f64 * fraction + 0.5) as usize;
    used.max(min)
}

pub struct ColSampler {
    fraction_bynode: f64,
    random: Random,
    /// Real feature indices of the used (non-trivial) features, in real-index order.
    valid_feature_indices: Vec<usize>,
    /// Inner index of each valid feature.
    valid_inner: Vec<usize>,
    num_features: usize,
    need_reset_bytree: bool,
    used_cnt_bytree: usize,
    is_feature_used: Vec<bool>,
    used_feature_indices: Vec<usize>,
}

impl ColSampler {
    /// upstream: `ColSampler::ColSampler` + `SetTrainingData`
    pub fn new(cfg: &Config, ds: &Dataset) -> Result<Self> {
        if !cfg.interaction_constraints.is_empty() {
            return Err(Error::Unsupported("interaction_constraints".into()));
        }
        let l = &ds.layout;
        let mut valid_feature_indices = Vec::new();
        let mut valid_inner = Vec::new();
        for real in 0..l.num_total_features {
            let inner = l.used_feature_map[real];
            if inner >= 0 {
                valid_feature_indices.push(real);
                valid_inner.push(inner as usize);
            }
        }
        let fraction_bytree = cfg.feature_fraction;
        let (need_reset_bytree, used_cnt_bytree) = if fraction_bytree >= 1.0 {
            (false, valid_feature_indices.len())
        } else {
            (true, get_cnt(valid_feature_indices.len(), fraction_bytree))
        };
        let mut s = ColSampler {
            fraction_bynode: cfg.feature_fraction_bynode,
            random: Random::new(cfg.feature_fraction_seed),
            valid_feature_indices,
            valid_inner,
            num_features: l.num_features,
            need_reset_bytree,
            used_cnt_bytree,
            is_feature_used: vec![true; l.num_features],
            used_feature_indices: Vec::new(),
        };
        s.reset_by_tree();
        Ok(s)
    }

    /// upstream: `ColSampler::ResetByTree`
    pub fn reset_by_tree(&mut self) {
        if self.need_reset_bytree {
            self.is_feature_used.fill(false);
            self.used_feature_indices = self
                .random
                .sample(
                    self.valid_feature_indices.len() as i32,
                    self.used_cnt_bytree as i32,
                )
                .into_iter()
                .map(|i| i as usize)
                .collect();
            for &i in &self.used_feature_indices {
                self.is_feature_used[self.valid_inner[i]] = true;
            }
        }
    }

    /// Features used by the current tree (inner indices).
    pub fn is_feature_used_bytree(&self) -> &[bool] {
        &self.is_feature_used
    }

    /// upstream: `ColSampler::GetByNode` without interaction constraints. Returns a mask
    /// over inner feature indices.
    pub fn get_by_node(&mut self) -> Vec<bool> {
        let mut ret = vec![false; self.num_features];
        if self.fraction_bynode >= 1.0 {
            return vec![true; self.num_features];
        }
        if self.need_reset_bytree {
            let cnt = get_cnt(self.used_feature_indices.len(), self.fraction_bynode);
            let picks = self
                .random
                .sample(self.used_feature_indices.len() as i32, cnt as i32);
            for p in picks {
                let vi = self.used_feature_indices[p as usize];
                ret[self.valid_inner[vi]] = true;
            }
        } else {
            let cnt = get_cnt(self.valid_feature_indices.len(), self.fraction_bynode);
            let picks = self
                .random
                .sample(self.valid_feature_indices.len() as i32, cnt as i32);
            for p in picks {
                ret[self.valid_inner[p as usize]] = true;
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::get_cnt;

    #[test]
    fn count_rounds_half_up_and_keeps_at_least_one() {
        assert_eq!(get_cnt(10, 0.5), 5);
        assert_eq!(get_cnt(10, 0.25), 3); // 2.5 + 0.5 -> 3
        assert_eq!(get_cnt(3, 0.01), 1);
        assert_eq!(get_cnt(0, 0.5), 0);
    }
}
