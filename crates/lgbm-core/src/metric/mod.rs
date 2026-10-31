//! Evaluation metrics. upstream: src/metric/*.hpp, include/LightGBM/metric.h
//!
//! Sums run serially in row order, which is what upstream's OpenMP reductions do with a
//! single thread. `x += a * b` expressions use [`madd`], mirroring clang's contraction.

mod binary;
mod multiclass;
mod rank;
/// DCG helpers shared with the ranking objectives (upstream: `DCGCalculator`).
pub(crate) mod rank_support {
    pub(crate) use super::rank::{
        cal_max_dcg, check_label_range, check_rank_labels, default_label_gain, discounts,
    };
}
mod regression;
mod xentropy;

use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::objective::LoadedObjective;
use crate::utils::common::madd;

/// upstream: `Metric`
pub trait Metric: Send + Sync {
    /// Names of the values returned by [`Metric::eval`] (`ndcg@1`, `l2`, ...).
    fn names(&self) -> &[String];
    /// `-1.0` when a smaller value is better, `1.0` otherwise.
    fn factor_to_bigger_better(&self) -> f64;
    /// `score` holds raw scores, class-major for multiclass. With an objective the scores
    /// are converted first (upstream `ConvertOutput`).
    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64>;
}

/// upstream: `Metric::CreateMetric` followed by `Init(metadata, num_data)`.
/// Unknown names yield `Ok(None)` (upstream returns `nullptr`, which is skipped).
pub fn create_metric(name: &str, cfg: &Config, md: &Metadata) -> Result<Option<Box<dyn Metric>>> {
    Ok(Some(match name {
        "l2" | "rmse" | "l1" | "quantile" | "huber" | "fair" | "poisson" | "mape" | "gamma"
        | "gamma_deviance" | "tweedie" => Box::new(regression::Pointwise::new(name, cfg, md)?),
        "r2" => Box::new(regression::R2::new(md)),
        "binary_logloss" | "binary_error" => Box::new(binary::BinaryPointwise::new(name, md)),
        "auc" => Box::new(binary::Auc::new(md)),
        "average_precision" => Box::new(binary::AveragePrecision::new(md)),
        "auc_mu" => Box::new(multiclass::AucMu::new(cfg, md)?),
        "multi_logloss" | "multi_error" => Box::new(multiclass::MultiPointwise::new(name, cfg, md)),
        "ndcg" => Box::new(rank::Ndcg::new(cfg, md)?),
        "map" => Box::new(rank::MapMetric::new(cfg, md)?),
        "cross_entropy" => Box::new(xentropy::CrossEntropyMetric::new(md)?),
        "cross_entropy_lambda" => Box::new(xentropy::CrossEntropyLambdaMetric::new(md)?),
        "kullback_leibler" => Box::new(xentropy::KullbackLeibler::new(md)?),
        _ => return Ok(None),
    }))
}

/// Sum of weights, or the row count without weights (shared `Init` logic).
pub(crate) fn sum_weights(md: &Metadata) -> f64 {
    if md.weights.is_empty() {
        md.label.len() as f64
    } else {
        let mut s = 0.0f64;
        for &w in &md.weights {
            s += f64::from(w);
        }
        s
    }
}

/// `sum += loss(i) [* weight[i]]` for a pointwise loss, mirroring upstream's four-way
/// branch on (objective, weights).
pub(crate) fn weighted_loss_sum(
    n: usize,
    weights: &[f32],
    mut loss: impl FnMut(usize) -> f64,
) -> f64 {
    let mut sum = 0.0f64;
    if weights.is_empty() {
        for i in 0..n {
            sum += loss(i);
        }
    } else {
        for i in 0..n {
            sum = madd(loss(i), f64::from(weights[i]), sum);
        }
    }
    sum
}

/// Convert one raw score to the objective's output space (identity without objective).
#[inline]
pub(crate) fn convert1(objective: Option<&LoadedObjective>, raw: f64) -> f64 {
    match objective {
        None => raw,
        Some(o) => {
            let mut t = [raw];
            o.convert_output(&mut t);
            t[0]
        }
    }
}

pub(crate) fn check_positive_labels(label: &[f32], name: &str) -> Result<()> {
    match label.iter().find(|&&l| l <= 0.0) {
        Some(l) => Err(Error::InvalidParameter(format!(
            "[{name}]: label {l} must be positive"
        ))),
        None => Ok(()),
    }
}

/// `Metadata::query_weights`: mean sample weight per query, accumulated in `f32` as
/// upstream does. Empty without weights or query boundaries.
pub(crate) fn query_weights(md: &Metadata) -> Vec<f32> {
    if md.weights.is_empty() || md.query_boundaries.is_empty() {
        return Vec::new();
    }
    md.query_boundaries
        .windows(2)
        .map(|b| {
            let mut s = 0.0f32;
            for &w in &md.weights[b[0] as usize..b[1] as usize] {
                s += w;
            }
            s / (b[1] - b[0]) as f32
        })
        .collect()
}
