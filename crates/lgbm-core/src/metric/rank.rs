//! Ranking metrics. upstream: src/metric/rank_metric.hpp, map_metric.hpp,
//! dcg_calculator.cpp

use super::{Metric, query_weights};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::objective::LoadedObjective;
use crate::utils::common::madd;

/// upstream: `DCGCalculator::kMaxPosition`
const MAX_POSITION: usize = 10_000;

/// upstream: `DCGCalculator::DefaultEvalAt`
pub(crate) fn default_eval_at(eval_at: &[i32]) -> Result<Vec<usize>> {
    if eval_at.is_empty() {
        return Ok((1..=5).collect());
    }
    eval_at
        .iter()
        .map(|&k| {
            usize::try_from(k)
                .ok()
                .filter(|&k| k > 0)
                .ok_or_else(|| Error::InvalidParameter(format!("eval_at must be positive: {k}")))
        })
        .collect()
}

/// upstream: `DCGCalculator::DefaultLabelGain`
pub(crate) fn default_label_gain(label_gain: &[f64]) -> Vec<f64> {
    if !label_gain.is_empty() {
        return label_gain.to_vec();
    }
    let mut g = vec![0.0];
    g.extend((1..31).map(|i| f64::from((1u32 << i) - 1)));
    g
}

/// upstream: `DCGCalculator::discount_`
pub(crate) fn discounts() -> Vec<f64> {
    (0..MAX_POSITION)
        .map(|i| 1.0 / (2.0 + i as f64).log2())
        .collect()
}

/// upstream: `DCGCalculator::CheckLabel`
pub(crate) fn check_rank_labels(label: &[f32], label_gain_len: usize) -> Result<()> {
    for &l in label {
        if (l - (l as i32) as f32).abs() > K_EPSILON {
            return Err(Error::InvalidParameter(format!(
                "label should be int type (met {l}) for ranking task, for the gain of label, \
                 please set the label_gain parameter"
            )));
        }
        if l < 0.0 {
            return Err(Error::InvalidParameter(format!(
                "Label should be non-negative (met {l}) for ranking task"
            )));
        }
        if l as usize >= label_gain_len {
            return Err(Error::InvalidParameter(format!(
                "Label {} is not less than the number of label mappings ({label_gain_len})",
                l as usize
            )));
        }
    }
    Ok(())
}

/// upstream: `DCGCalculator::CalMaxDCG`
pub(crate) fn cal_max_dcg(
    ks: &[usize],
    label: &[f32],
    gain: &[f64],
    discount: &[f64],
    out: &mut [f64],
) {
    let mut label_cnt = vec![0usize; gain.len()];
    for &l in label {
        label_cnt[l as usize] += 1;
    }
    let mut cur_result = 0.0f64;
    let mut cur_left = 0usize;
    let mut top_label = gain.len() as isize - 1;
    for (i, &k) in ks.iter().enumerate() {
        let cur_k = k.min(label.len());
        for j in cur_left..cur_k {
            while top_label > 0 && label_cnt[top_label as usize] == 0 {
                top_label -= 1;
            }
            if top_label < 0 {
                break;
            }
            cur_result = madd(discount[j], gain[top_label as usize], cur_result);
            label_cnt[top_label as usize] -= 1;
        }
        out[i] = cur_result;
        cur_left = cur_k;
    }
}

/// upstream: `DCGCalculator::CalDCG`
pub(crate) fn cal_dcg(
    ks: &[usize],
    label: &[f32],
    score: &[f64],
    gain: &[f64],
    discount: &[f64],
    out: &mut [f64],
) {
    let mut sorted: Vec<usize> = (0..label.len()).collect();
    sorted.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cur_result = 0.0f64;
    let mut cur_left = 0usize;
    for (i, &k) in ks.iter().enumerate() {
        let cur_k = k.min(label.len());
        for j in cur_left..cur_k {
            let idx = sorted[j];
            cur_result = madd(gain[label[idx] as usize], discount[j], cur_result);
        }
        out[i] = cur_result;
        cur_left = cur_k;
    }
}

fn check_queries(md: &Metadata, what: &str) -> Result<()> {
    if md.query_boundaries.is_empty() {
        return Err(Error::InvalidParameter(format!(
            "The {what} metric requires query information"
        )));
    }
    Ok(())
}

/// upstream: `NDCGMetric`
pub struct Ndcg {
    names: Vec<String>,
    eval_at: Vec<usize>,
    label: Vec<f32>,
    boundaries: Vec<i32>,
    qw: Vec<f32>,
    sum_query_weights: f64,
    gain: Vec<f64>,
    discount: Vec<f64>,
    inverse_max_dcgs: Vec<Vec<f64>>,
}

impl Ndcg {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let eval_at = default_eval_at(&cfg.eval_at)?;
        let gain = default_label_gain(&cfg.label_gain);
        check_queries(md, "NDCG")?;
        check_rank_labels(&md.label, gain.len())?;
        let num_queries = md.query_boundaries.len() - 1;
        for b in md.query_boundaries.windows(2) {
            let rows = (b[1] - b[0]) as usize;
            if rows > MAX_POSITION {
                return Err(Error::InvalidParameter(format!(
                    "Number of rows {rows} exceeds upper limit of {MAX_POSITION} for a query"
                )));
            }
        }
        let qw = query_weights(md);
        let sum_query_weights = if qw.is_empty() {
            num_queries as f64
        } else {
            qw.iter().fold(0.0f64, |s, &w| s + f64::from(w))
        };
        let discount = discounts();
        let inverse_max_dcgs = md
            .query_boundaries
            .windows(2)
            .map(|b| {
                let mut v = vec![0.0f64; eval_at.len()];
                cal_max_dcg(
                    &eval_at,
                    &md.label[b[0] as usize..b[1] as usize],
                    &gain,
                    &discount,
                    &mut v,
                );
                for x in &mut v {
                    *x = if *x > 0.0 { 1.0 / *x } else { -1.0 };
                }
                v
            })
            .collect();
        Ok(Ndcg {
            names: eval_at.iter().map(|k| format!("ndcg@{k}")).collect(),
            eval_at,
            label: md.label.clone(),
            boundaries: md.query_boundaries.clone(),
            qw,
            sum_query_weights,
            gain,
            discount,
            inverse_max_dcgs,
        })
    }
}

impl Metric for Ndcg {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], _objective: Option<&LoadedObjective>) -> Vec<f64> {
        let width = self.eval_at.len();
        let mut dcg = vec![0.0f64; width];
        let mut result = vec![0.0f64; width];
        for q in 0..self.boundaries.len() - 1 {
            let (lo, hi) = (self.boundaries[q] as usize, self.boundaries[q + 1] as usize);
            let inv = &self.inverse_max_dcgs[q];
            if inv[0] <= 0.0 {
                for r in &mut result {
                    *r += 1.0;
                }
                continue;
            }
            cal_dcg(
                &self.eval_at,
                &self.label[lo..hi],
                &score[lo..hi],
                &self.gain,
                &self.discount,
                &mut dcg,
            );
            for j in 0..width {
                result[j] = if self.qw.is_empty() {
                    madd(dcg[j], inv[j], result[j])
                } else {
                    madd(dcg[j] * inv[j], f64::from(self.qw[q]), result[j])
                };
            }
        }
        for r in &mut result {
            *r /= self.sum_query_weights;
        }
        result
    }
}

/// upstream: `MapMetric`
pub struct MapMetric {
    names: Vec<String>,
    eval_at: Vec<usize>,
    label: Vec<f32>,
    boundaries: Vec<i32>,
    qw: Vec<f32>,
    sum_query_weights: f64,
    npos_per_query: Vec<usize>,
}

impl MapMetric {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let eval_at = default_eval_at(&cfg.eval_at)?;
        check_queries(md, "MAP")?;
        let num_queries = md.query_boundaries.len() - 1;
        let qw = query_weights(md);
        let sum_query_weights = if qw.is_empty() {
            num_queries as f64
        } else {
            qw.iter().fold(0.0f64, |s, &w| s + f64::from(w))
        };
        let npos_per_query = md
            .query_boundaries
            .windows(2)
            .map(|b| {
                md.label[b[0] as usize..b[1] as usize]
                    .iter()
                    .filter(|&&l| l > 0.5)
                    .count()
            })
            .collect();
        Ok(MapMetric {
            names: eval_at.iter().map(|k| format!("map@{k}")).collect(),
            eval_at,
            label: md.label.clone(),
            boundaries: md.query_boundaries.clone(),
            qw,
            sum_query_weights,
            npos_per_query,
        })
    }

    /// upstream: `MapMetric::CalMapAtK`
    fn cal_map_at_k(&self, npos: usize, label: &[f32], score: &[f64], out: &mut [f64]) {
        let mut sorted: Vec<usize> = (0..label.len()).collect();
        sorted.sort_by(|&a, &b| {
            score[b]
                .partial_cmp(&score[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (mut num_hit, mut sum_ap, mut cur_left) = (0i32, 0.0f64, 0usize);
        for (i, &k) in self.eval_at.iter().enumerate() {
            let cur_k = k.min(label.len());
            for j in cur_left..cur_k {
                if label[sorted[j]] > 0.5 {
                    num_hit += 1;
                    sum_ap += f64::from(num_hit) / f64::from((j as f32) + 1.0);
                }
            }
            out[i] = if npos > 0 {
                sum_ap / npos.min(cur_k) as f64
            } else {
                1.0
            };
            cur_left = cur_k;
        }
    }
}

impl Metric for MapMetric {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], _objective: Option<&LoadedObjective>) -> Vec<f64> {
        let width = self.eval_at.len();
        let mut tmp = vec![0.0f64; width];
        let mut result = vec![0.0f64; width];
        for q in 0..self.boundaries.len() - 1 {
            let (lo, hi) = (self.boundaries[q] as usize, self.boundaries[q + 1] as usize);
            self.cal_map_at_k(
                self.npos_per_query[q],
                &self.label[lo..hi],
                &score[lo..hi],
                &mut tmp,
            );
            for j in 0..width {
                result[j] = if self.qw.is_empty() {
                    result[j] + tmp[j]
                } else {
                    madd(tmp[j], f64::from(self.qw[q]), result[j])
                };
            }
        }
        for r in &mut result {
            *r /= self.sum_query_weights;
        }
        result
    }
}
