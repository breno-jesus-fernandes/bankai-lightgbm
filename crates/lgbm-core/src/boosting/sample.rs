//! Row sampling per iteration: bagging (also balanced) and GOSS.
//! upstream: src/boosting/sample_strategy.cpp, bagging.hpp, goss.hpp
//!
//! Single-threaded semantics. Upstream draws one random number per row from a generator
//! per block of [`RAND_BLOCK`] rows (seeded `seed + block`), so the draws do not depend on
//! the thread count. With one thread, the bagged rows come out in ascending row order,
//! which is the order histograms are summed in.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::utils::array_args::arg_max_at_k;
use crate::utils::random::Random;

/// upstream: `SampleStrategy::bagging_rand_block_`
const RAND_BLOCK: usize = 1024;

/// Bagging over all rows (`bagging_fraction`, `bagging_freq`), or balanced by label
/// (`pos_bagging_fraction`, `neg_bagging_fraction`).
pub(crate) struct Bagging {
    freq: i32,
    fraction: f64,
    pos_fraction: f64,
    neg_fraction: f64,
    balanced: bool,
    positive: Vec<bool>,
    rands: Vec<Random>,
    /// upstream `need_re_bagging_`: true until the first draw.
    need_rebag: bool,
    bag: Vec<i32>,
    /// Row boundaries of each query (`num_queries + 1` entries) when `bagging_by_query` is
    /// on; empty otherwise. Whole queries are kept or dropped together.
    query_boundaries: Vec<i32>,
}

/// GOSS: keep the `top_rate` rows with the largest gradients, sample `other_rate` of the
/// rest and amplify their gradients. upstream: `GOSSStrategy`
pub(crate) struct Goss {
    top_rate: f64,
    other_rate: f64,
    /// Iterations before sampling starts (`1 / learning_rate`).
    warmup: i32,
    rands: Vec<Random>,
    bag: Vec<i32>,
    num_tree_per_iteration: usize,
}

pub enum SampleStrategy {
    /// All rows, every iteration.
    Full,
    Bagging(Bagging),
    Goss(Goss),
}

fn block_rands(seed: i32, n: usize) -> Vec<Random> {
    (0..n.div_ceil(RAND_BLOCK))
        .map(|i| Random::new(seed.wrapping_add(i as i32)))
        .collect()
}

impl SampleStrategy {
    /// upstream: `SampleStrategy::CreateSampleStrategy` + `ResetSampleConfig`
    pub fn new(cfg: &Config, md: &Metadata, num_tree_per_iteration: usize) -> Result<Self> {
        let n = md.label.len();
        if cfg.data_sample_strategy == "goss" {
            if cfg.bagging_freq > 0 && cfg.bagging_fraction != 1.0 {
                return Err(Error::InvalidParameter("Cannot use bagging in GOSS".into()));
            }
            if cfg.top_rate + cfg.other_rate > 1.0 || cfg.top_rate <= 0.0 || cfg.other_rate <= 0.0 {
                return Err(Error::InvalidParameter(
                    "top_rate + other_rate must be <= 1, both positive".into(),
                ));
            }
            return Ok(SampleStrategy::Goss(Goss {
                top_rate: cfg.top_rate,
                other_rate: cfg.other_rate,
                warmup: (1.0 / cfg.learning_rate) as i32,
                rands: block_rands(cfg.bagging_seed, n),
                bag: Vec::new(),
                num_tree_per_iteration,
            }));
        }
        if cfg.data_sample_strategy != "bagging" {
            return Err(Error::Unsupported(format!(
                "data_sample_strategy={}",
                cfg.data_sample_strategy
            )));
        }
        if cfg.bagging_by_query {
            return Self::new_query_bagging(cfg, md, num_tree_per_iteration);
        }
        // upstream: positives are counted by the binary objective only.
        let positive: Vec<bool> = md.label.iter().map(|&l| l > 0.0).collect();
        let num_pos = if cfg.objective == "binary" {
            positive.iter().filter(|&&p| p).count()
        } else {
            0
        };
        let balanced =
            (cfg.pos_bagging_fraction < 1.0 || cfg.neg_bagging_fraction < 1.0) && num_pos > 0;
        let active = (cfg.bagging_fraction < 1.0 || balanced) && cfg.bagging_freq > 0;
        if !active {
            return Ok(SampleStrategy::Full);
        }
        Ok(SampleStrategy::Bagging(Bagging {
            freq: cfg.bagging_freq,
            fraction: cfg.bagging_fraction,
            pos_fraction: cfg.pos_bagging_fraction,
            neg_fraction: cfg.neg_bagging_fraction,
            balanced,
            positive,
            rands: block_rands(cfg.bagging_seed, n),
            need_rebag: true,
            bag: Vec::new(),
            query_boundaries: Vec::new(),
        }))
    }

    /// upstream: `BaggingSampleStrategy` with `bagging_by_query`. Only the ranking objective
    /// is supported: `rank_xendcg` draws a random number per query, and upstream computes
    /// gradients for the sampled queries only, which this port does not yet mirror.
    fn new_query_bagging(cfg: &Config, md: &Metadata, _ntpi: usize) -> Result<Self> {
        if matches!(
            cfg.objective.as_str(),
            "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart"
        ) {
            return Err(Error::Unsupported(
                "bagging_by_query with rank_xendcg".into(),
            ));
        }
        if md.query_boundaries.is_empty() {
            return Err(Error::InvalidParameter(
                "bagging_by_query requires query or group information".into(),
            ));
        }
        let n = md.label.len();
        Ok(SampleStrategy::Bagging(Bagging {
            freq: cfg.bagging_freq,
            fraction: cfg.bagging_fraction,
            pos_fraction: 1.0,
            neg_fraction: 1.0,
            balanced: false,
            positive: Vec::new(),
            rands: block_rands(cfg.bagging_seed, n),
            need_rebag: true,
            bag: Vec::new(),
            query_boundaries: md.query_boundaries.clone(),
        }))
    }

    /// upstream: `SampleStrategy::IsHessianChange`
    pub fn changes_hessians(&self) -> bool {
        matches!(self, SampleStrategy::Goss(_))
    }

    /// upstream: `SampleStrategy::Bagging` (and `GOSSStrategy::Bagging`) for iteration
    /// `iter`. May rescale `gradients`/`hessians` in place (GOSS). Returns the rows the tree
    /// is trained on, or `None` when all rows are used.
    pub fn sample(
        &mut self,
        iter: i32,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Option<&[i32]> {
        match self {
            SampleStrategy::Full => None,
            SampleStrategy::Bagging(b) => {
                let n = b.num_rows();
                let bag_cnt = b.bag.len();
                if (bag_cnt < n && iter % b.freq == 0) || b.need_rebag {
                    b.need_rebag = false;
                    b.bag = b.draw(n);
                }
                (b.bag.len() < n).then_some(b.bag.as_slice())
            }
            SampleStrategy::Goss(g) => {
                let n = gradients.len() / g.num_tree_per_iteration;
                if iter < g.warmup {
                    return None;
                }
                g.bag = g.draw(n, gradients, hessians);
                (g.bag.len() < n).then_some(g.bag.as_slice())
            }
        }
    }
}

impl Bagging {
    fn num_rows(&self) -> usize {
        match self.query_boundaries.last() {
            Some(&end) => end as usize,
            None => self.positive.len(),
        }
    }

    /// upstream: `BaggingHelper` / `BalancedBaggingHelper`, one draw per row in row order.
    /// With query boundaries, upstream: one draw per query (`BaggingHelper` over
    /// `num_queries_`), keeping every row of the queries that are drawn.
    fn draw(&mut self, n: usize) -> Vec<i32> {
        let mut bag = Vec::new();
        if !self.query_boundaries.is_empty() {
            for (q, span) in self.query_boundaries.windows(2).enumerate() {
                let r = f64::from(self.rands[q / RAND_BLOCK].next_float());
                if r < self.fraction {
                    bag.extend(span[0]..span[1]);
                }
            }
            return bag;
        }
        for i in 0..n {
            let r = f64::from(self.rands[i / RAND_BLOCK].next_float());
            let threshold = if self.balanced {
                if self.positive[i] {
                    self.pos_fraction
                } else {
                    self.neg_fraction
                }
            } else {
                self.fraction
            };
            if r < threshold {
                bag.push(i as i32);
            }
        }
        bag
    }
}

impl Goss {
    /// upstream: `GOSSStrategy::Helper` over all rows (one block).
    fn draw(&mut self, n: usize, gradients: &mut [f32], hessians: &mut [f32]) -> Vec<i32> {
        let ntpi = self.num_tree_per_iteration;
        let mut bag = Vec::new();
        if n == 0 {
            return bag;
        }
        // Per-row sum over trees of |g * h|, accumulated in f32 like upstream's score_t.
        let mut tmp: Vec<f32> = (0..n)
            .map(|i| {
                let mut s = 0.0f32;
                for k in 0..ntpi {
                    let idx = k * n + i;
                    s += (gradients[idx] * hessians[idx]).abs();
                }
                s
            })
            .collect();
        let top_k = ((n as f64 * self.top_rate) as usize).max(1);
        let other_k = (n as f64 * self.other_rate) as usize;
        arg_max_at_k(&mut tmp, 0, n as isize, top_k as isize - 1);
        let threshold = tmp[top_k - 1];
        // upstream: `static_cast<score_t>(cnt - top_k) / other_k` (float division)
        let multiply = (n - top_k) as f32 / other_k as f32;
        let mut big_weight_cnt = 0usize;
        for i in 0..n {
            let mut grad = 0.0f32;
            for k in 0..ntpi {
                let idx = k * n + i;
                grad += (gradients[idx] * hessians[idx]).abs();
            }
            if grad >= threshold {
                bag.push(i as i32);
                big_weight_cnt += 1;
            } else {
                let sampled = bag.len() - big_weight_cnt;
                let rest_need = other_k as i64 - sampled as i64;
                let rest_all = (n - i) as i64 - (top_k as i64 - big_weight_cnt as i64);
                let prob = rest_need as f64 / rest_all as f64;
                if f64::from(self.rands[i / RAND_BLOCK].next_float()) < prob {
                    bag.push(i as i32);
                    for k in 0..ntpi {
                        let idx = k * n + i;
                        gradients[idx] *= multiply;
                        hessians[idx] *= multiply;
                    }
                }
            }
        }
        bag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bagging_draws_are_ascending_and_block_seeded() {
        let mut b = Bagging {
            freq: 1,
            fraction: 0.5,
            pos_fraction: 1.0,
            neg_fraction: 1.0,
            balanced: false,
            positive: vec![false; 2500],
            rands: block_rands(3, 2500),
            need_rebag: true,
            bag: Vec::new(),
            query_boundaries: Vec::new(),
        };
        let bag = b.draw(2500);
        assert!(bag.windows(2).all(|w| w[0] < w[1]));
        assert!(bag.len() > 900 && bag.len() < 1600);
        // The same seed reproduces the same draw.
        let mut b2 = Bagging {
            rands: block_rands(3, 2500),
            ..b
        };
        assert_eq!(b2.draw(2500), bag);
    }

    #[test]
    fn query_bagging_keeps_whole_queries_in_ascending_order() {
        // Four queries of sizes 3, 2, 4, 3.
        let bounds = vec![0, 3, 5, 9, 12];
        let mut b = Bagging {
            freq: 1,
            fraction: 0.5,
            pos_fraction: 1.0,
            neg_fraction: 1.0,
            balanced: false,
            positive: Vec::new(),
            rands: block_rands(3, 12),
            need_rebag: true,
            bag: Vec::new(),
            query_boundaries: bounds.clone(),
        };
        assert_eq!(b.num_rows(), 12);
        let bag = b.draw(12);
        assert!(bag.windows(2).all(|w| w[0] < w[1]));
        // Each query is either fully in the bag or fully out of it.
        for span in bounds.windows(2) {
            let inside = bag.iter().filter(|&&r| r >= span[0] && r < span[1]).count();
            assert!(inside == 0 || inside == (span[1] - span[0]) as usize);
        }
        // The same seed reproduces the same draw.
        let mut b2 = Bagging {
            rands: block_rands(3, 12),
            ..b
        };
        assert_eq!(b2.draw(12), bag);
    }
}
