//! Ranking objectives. upstream: src/objective/rank_objective.hpp

use std::sync::Mutex;

use super::train::Objective;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::metric::rank_support::{
    cal_max_dcg, check_label_range, check_rank_labels, default_label_gain, discounts,
};
use crate::utils::common::softmax;
use crate::utils::random::Random;

/// upstream: `kMinScore`
const K_MIN_SCORE: f64 = f64::NEG_INFINITY;
/// upstream: `_sigmoid_bins`
const SIGMOID_BINS: usize = 1024 * 1024;

/// Shared state of `RankingObjective`.
struct Queries {
    label: Vec<f32>,
    weights: Vec<f32>,
    boundaries: Vec<i32>,
}

impl Queries {
    fn new(md: &Metadata) -> Result<Self> {
        if md.query_boundaries.is_empty() {
            return Err(Error::InvalidParameter(
                "Ranking tasks require query information".into(),
            ));
        }
        Ok(Queries {
            label: md.label.clone(),
            weights: md.weights.clone(),
            boundaries: md.query_boundaries.clone(),
        })
    }

    fn num_queries(&self) -> usize {
        self.boundaries.len() - 1
    }

    /// upstream: `RankingObjective::GetGradients`
    fn gradients(
        &self,
        score: &[f64],
        g: &mut [f32],
        h: &mut [f32],
        mut one_query: impl FnMut(usize, &[f32], &[f64], &mut [f32], &mut [f32]),
    ) {
        for q in 0..self.num_queries() {
            let (lo, hi) = (self.boundaries[q] as usize, self.boundaries[q + 1] as usize);
            one_query(
                q,
                &self.label[lo..hi],
                &score[lo..hi],
                &mut g[lo..hi],
                &mut h[lo..hi],
            );
            if !self.weights.is_empty() {
                for j in lo..hi {
                    g[j] *= self.weights[j];
                    h[j] *= self.weights[j];
                }
            }
        }
    }
}

/// upstream: `LambdarankNDCG`
pub struct LambdarankNdcg {
    queries: Queries,
    sigmoid: f64,
    norm: bool,
    truncation_level: i32,
    label_gain: Vec<f64>,
    discount: Vec<f64>,
    inverse_max_dcgs: Vec<f64>,
    sigmoid_table: Vec<f64>,
    min_sigmoid_input: f64,
    max_sigmoid_input: f64,
    sigmoid_table_idx_factor: f64,
}

impl LambdarankNdcg {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        if cfg.sigmoid <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "Sigmoid param {} should be greater than zero",
                cfg.sigmoid
            )));
        }
        let label_gain = default_label_gain(&cfg.label_gain);
        let queries = Queries::new(md)?;
        check_label_range(&md.query_boundaries)?;
        check_rank_labels(&md.label, label_gain.len())?;
        let discount = discounts();
        let trunc = cfg.lambdarank_truncation_level.max(0) as usize;
        let inverse_max_dcgs = md
            .query_boundaries
            .windows(2)
            .map(|b| {
                let mut v = [0.0f64];
                cal_max_dcg(
                    &[trunc],
                    &md.label[b[0] as usize..b[1] as usize],
                    &label_gain,
                    &discount,
                    &mut v,
                );
                if v[0] > 0.0 { 1.0 / v[0] } else { v[0] }
            })
            .collect();
        // upstream: ConstructSigmoidTable
        let min_in = -50.0 / cfg.sigmoid / 2.0;
        let max_in = -min_in;
        let factor = SIGMOID_BINS as f64 / (max_in - min_in);
        let table = (0..SIGMOID_BINS)
            .map(|i| {
                let score = i as f64 / factor + min_in;
                1.0 / (1.0 + (score * cfg.sigmoid).exp())
            })
            .collect();
        Ok(LambdarankNdcg {
            queries,
            sigmoid: cfg.sigmoid,
            norm: cfg.lambdarank_norm,
            truncation_level: cfg.lambdarank_truncation_level,
            label_gain,
            discount,
            inverse_max_dcgs,
            sigmoid_table: table,
            min_sigmoid_input: min_in,
            max_sigmoid_input: max_in,
            sigmoid_table_idx_factor: factor,
        })
    }

    #[inline]
    fn get_sigmoid(&self, score: f64) -> f64 {
        if score <= self.min_sigmoid_input {
            self.sigmoid_table[0]
        } else if score >= self.max_sigmoid_input {
            self.sigmoid_table[SIGMOID_BINS - 1]
        } else {
            self.sigmoid_table
                [((score - self.min_sigmoid_input) * self.sigmoid_table_idx_factor) as usize]
        }
    }

    /// upstream: `LambdarankNDCG::GetGradientsForOneQuery`
    fn one_query(
        &self,
        query_id: usize,
        label: &[f32],
        score: &[f64],
        lambdas: &mut [f32],
        hessians: &mut [f32],
    ) {
        let cnt = label.len();
        let inverse_max_dcg = self.inverse_max_dcgs[query_id];
        lambdas.fill(0.0);
        hessians.fill(0.0);
        if cnt == 0 {
            return;
        }
        let mut sorted: Vec<usize> = (0..cnt).collect();
        sorted.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best_score = score[sorted[0]];
        let mut worst_idx = cnt - 1;
        if worst_idx > 0 && score[sorted[worst_idx]] == K_MIN_SCORE {
            worst_idx -= 1;
        }
        let worst_score = score[sorted[worst_idx]];
        let mut sum_lambdas = 0.0f64;
        let mut i = 0usize;
        while i + 1 < cnt && (i as i32) < self.truncation_level {
            if score[sorted[i]] != K_MIN_SCORE {
                for j in i + 1..cnt {
                    if score[sorted[j]] == K_MIN_SCORE || label[sorted[i]] == label[sorted[j]] {
                        continue;
                    }
                    let (high_rank, low_rank) = if label[sorted[i]] > label[sorted[j]] {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    let high = sorted[high_rank];
                    let low = sorted[low_rank];
                    let high_label_gain = self.label_gain[label[high] as usize];
                    let low_label_gain = self.label_gain[label[low] as usize];
                    let high_discount = self.discount[high_rank];
                    let low_discount = self.discount[low_rank];
                    let delta_score = score[high] - score[low];
                    let dcg_gap = high_label_gain - low_label_gain;
                    let paired_discount = (high_discount - low_discount).abs();
                    let mut delta_pair_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
                    if self.norm && best_score != worst_score {
                        delta_pair_ndcg /= f64::from(0.01f32) + delta_score.abs();
                    }
                    let mut p_lambda = self.get_sigmoid(delta_score);
                    let mut p_hessian = p_lambda * (1.0 - p_lambda);
                    p_lambda *= -self.sigmoid * delta_pair_ndcg;
                    p_hessian *= self.sigmoid * self.sigmoid * delta_pair_ndcg;
                    lambdas[low] -= p_lambda as f32;
                    hessians[low] += p_hessian as f32;
                    lambdas[high] += p_lambda as f32;
                    hessians[high] += p_hessian as f32;
                    sum_lambdas -= 2.0 * p_lambda;
                }
            }
            i += 1;
        }
        if self.norm && sum_lambdas > 0.0 {
            let norm_factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
            for i in 0..cnt {
                lambdas[i] = (f64::from(lambdas[i]) * norm_factor) as f32;
                hessians[i] = (f64::from(hessians[i]) * norm_factor) as f32;
            }
        }
    }
}

impl Objective for LambdarankNdcg {
    fn name(&self) -> &'static str {
        "lambdarank"
    }

    fn to_model_string(&self) -> String {
        "lambdarank".into()
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        self.queries.gradients(score, g, h, |q, label, sc, gg, hh| {
            self.one_query(q, label, sc, gg, hh);
        });
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        0.0
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}

/// upstream: `Common::Pow<int>(2, power)`: the base is squared in `int` arithmetic.
fn pow_int(base: i32, power: i32) -> f64 {
    if power < 0 {
        1.0 / pow_int(base, -power)
    } else if power == 0 {
        1.0
    } else if power % 2 == 0 {
        pow_int(base.wrapping_mul(base), power / 2)
    } else if power % 3 == 0 {
        pow_int(base.wrapping_mul(base).wrapping_mul(base), power / 3)
    } else {
        f64::from(base) * pow_int(base, power - 1)
    }
}

/// upstream: `RankXENDCG`
pub struct RankXendcg {
    queries: Queries,
    rands: Mutex<Vec<Random>>,
}

impl RankXendcg {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let queries = Queries::new(md)?;
        let rands = (0..queries.num_queries())
            .map(|i| Random::new(cfg.objective_seed.wrapping_add(i as i32)))
            .collect();
        Ok(RankXendcg {
            queries,
            rands: Mutex::new(rands),
        })
    }
}

impl Objective for RankXendcg {
    fn name(&self) -> &'static str {
        "rank_xendcg"
    }

    fn to_model_string(&self) -> String {
        "rank_xendcg".into()
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        let Ok(mut rands) = self.rands.lock() else {
            return;
        };
        self.queries
            .gradients(score, g, h, |q, label, sc, lambdas, hessians| {
                let cnt = label.len();
                if cnt <= 1 {
                    lambdas.fill(0.0);
                    hessians.fill(0.0);
                    return;
                }
                let mut rho = sc.to_vec();
                softmax(&mut rho);
                let mut params = vec![0.0f64; cnt];
                let mut inv_denominator = 0.0f64;
                for i in 0..cnt {
                    let phi = pow_int(2, label[i] as i32) - f64::from(rands[q].next_float());
                    params[i] = phi;
                    inv_denominator += phi;
                }
                inv_denominator = 1.0 / f64::from(K_EPSILON).max(inv_denominator);
                let mut sum_l1 = 0.0f64;
                for i in 0..cnt {
                    let term = -params[i] * inv_denominator + rho[i];
                    lambdas[i] = term as f32;
                    params[i] = term / (1.0 - rho[i]);
                    sum_l1 += params[i];
                }
                let mut sum_l2 = 0.0f64;
                for i in 0..cnt {
                    let term = rho[i] * (sum_l1 - params[i]);
                    lambdas[i] += term as f32;
                    params[i] = term / (1.0 - rho[i]);
                    sum_l2 += params[i];
                }
                for i in 0..cnt {
                    lambdas[i] += (rho[i] * (sum_l2 - params[i])) as f32;
                    hessians[i] = (rho[i] * (1.0 - rho[i])) as f32;
                }
            });
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        0.0
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}
