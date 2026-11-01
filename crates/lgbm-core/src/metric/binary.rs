//! Binary-classification metrics. upstream: src/metric/binary_metric.hpp

use super::{Metric, convert1, sum_weights, weighted_loss_sum};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::objective::LoadedObjective;
use crate::utils::common::madd;

/// upstream: `BinaryLoglossMetric` / `BinaryErrorMetric`
pub struct BinaryPointwise {
    logloss: bool,
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
}

impl BinaryPointwise {
    pub fn new(name: &str, md: &Metadata) -> Self {
        BinaryPointwise {
            logloss: name == "binary_logloss",
            names: vec![name.to_string()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sum_weights(md),
        }
    }

    fn loss(&self, label: f32, prob: f64) -> f64 {
        let eps = f64::from(K_EPSILON);
        if self.logloss {
            if label <= 0.0 {
                if 1.0 - prob > eps {
                    return -(1.0 - prob).ln();
                }
            } else if prob > eps {
                return -prob.ln();
            }
            -eps.ln()
        } else if prob <= 0.5 {
            f64::from(u8::from(label > 0.0))
        } else {
            f64::from(u8::from(label <= 0.0))
        }
    }
}

impl Metric for BinaryPointwise {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let sum = weighted_loss_sum(self.label.len(), &self.weights, |i| {
            self.loss(self.label[i], convert1(objective, score[i]))
        });
        vec![sum / self.sum_weights]
    }
}

/// Row indices sorted by descending score (ties keep row order).
fn sorted_desc(score: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    idx
}

/// upstream: `AUCMetric`
pub struct Auc {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
}

impl Auc {
    pub fn new(md: &Metadata) -> Self {
        Auc {
            names: vec!["auc".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sum_weights(md),
        }
    }
}

impl Metric for Auc {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], _objective: Option<&LoadedObjective>) -> Vec<f64> {
        let n = self.label.len();
        let idx = sorted_desc(score, n);
        let (mut cur_pos, mut sum_pos, mut accum, mut cur_neg) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut threshold = score[idx[0]];
        for &r in &idx {
            let cur_score = score[r];
            if cur_score != threshold {
                threshold = cur_score;
                accum = madd(cur_neg, madd(cur_pos, 0.5, sum_pos), accum);
                sum_pos += cur_pos;
                cur_neg = 0.0;
                cur_pos = 0.0;
            }
            let w = if self.weights.is_empty() {
                1.0
            } else {
                f64::from(self.weights[r])
            };
            if self.label[r] <= 0.0 {
                cur_neg += w;
            } else {
                cur_pos += w;
            }
        }
        accum = madd(cur_neg, madd(cur_pos, 0.5, sum_pos), accum);
        sum_pos += cur_pos;
        let mut auc = 1.0;
        if sum_pos > 0.0 && sum_pos != self.sum_weights {
            auc = accum / (sum_pos * (self.sum_weights - sum_pos));
        }
        vec![auc]
    }
}

/// upstream: `AveragePrecisionMetric`
pub struct AveragePrecision {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
}

impl AveragePrecision {
    pub fn new(md: &Metadata) -> Self {
        AveragePrecision {
            names: vec!["average_precision".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sum_weights(md),
        }
    }
}

impl Metric for AveragePrecision {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], _objective: Option<&LoadedObjective>) -> Vec<f64> {
        let n = self.label.len();
        let idx = sorted_desc(score, n);
        let (mut cur_pos, mut sum_pos, mut sum_pred, mut accum, mut cur_neg) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut threshold = score[idx[0]];
        for &r in &idx {
            let cur_score = score[r];
            if cur_score != threshold {
                threshold = cur_score;
                sum_pos += cur_pos;
                sum_pred += cur_pos + cur_neg;
                let accum_prec = sum_pos / sum_pred;
                accum = madd(cur_pos, accum_prec, accum);
                cur_neg = 0.0;
                cur_pos = 0.0;
            }
            let w = if self.weights.is_empty() {
                1.0
            } else {
                f64::from(self.weights[r])
            };
            if self.label[r] <= 0.0 {
                cur_neg += w;
            } else {
                cur_pos += w;
            }
        }
        sum_pos += cur_pos;
        sum_pred += cur_pos + cur_neg;
        let accum_prec = sum_pos / sum_pred;
        accum = madd(cur_pos, accum_prec, accum);
        let mut ap = 1.0;
        if sum_pos > 0.0 && sum_pos != self.sum_weights {
            ap = accum / sum_pos;
        }
        vec![ap]
    }
}
