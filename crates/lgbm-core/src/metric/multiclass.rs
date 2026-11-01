//! Multiclass metrics. upstream: src/metric/multiclass_metric.hpp

use super::{Metric, sum_weights};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::objective::LoadedObjective;
use crate::utils::common::madd;

/// upstream: `MultiErrorMetric` / `MultiSoftmaxLoglossMetric`
pub struct MultiPointwise {
    logloss: bool,
    top_k: i32,
    num_class: usize,
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
}

impl MultiPointwise {
    pub fn new(name: &str, cfg: &Config, md: &Metadata) -> Self {
        let logloss = name == "multi_logloss";
        let name = if logloss {
            "multi_logloss".to_string()
        } else if cfg.multi_error_top_k == 1 {
            "multi_error".to_string()
        } else {
            format!("multi_error@{}", cfg.multi_error_top_k)
        };
        MultiPointwise {
            logloss,
            top_k: cfg.multi_error_top_k,
            num_class: cfg.num_class as usize,
            names: vec![name],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sum_weights(md),
        }
    }

    fn loss(&self, label: f32, rec: &[f64]) -> f64 {
        let k = label as usize;
        if self.logloss {
            let eps = f64::from(K_EPSILON);
            if rec[k] > eps {
                -rec[k].ln()
            } else {
                -eps.ln()
            }
        } else {
            let mut num_larger = 0;
            for &r in rec {
                if r >= rec[k] {
                    num_larger += 1;
                }
                if num_larger > self.top_k {
                    return 1.0;
                }
            }
            0.0
        }
    }
}

impl Metric for MultiPointwise {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let n = self.label.len();
        let nc = self.num_class;
        let mut rec = vec![0.0f64; nc];
        let mut sum = 0.0f64;
        for i in 0..n {
            for (k, r) in rec.iter_mut().enumerate() {
                *r = score[n * k + i];
            }
            if let Some(o) = objective {
                o.convert_output(&mut rec);
            }
            let loss = self.loss(self.label[i], &rec);
            if self.weights.is_empty() {
                sum += loss;
            } else {
                sum = madd(loss, f64::from(self.weights[i]), sum);
            }
        }
        vec![sum / self.sum_weights]
    }
}

/// upstream: `AucMuMetric`
pub struct AucMu {
    names: Vec<String>,
    num_class: usize,
    class_weights: Vec<Vec<f64>>,
    label: Vec<f32>,
    weights: Vec<f32>,
    class_data_weights: Vec<f64>,
    class_sizes: Vec<usize>,
    sorted_data_idx: Vec<usize>,
}

impl AucMu {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let nc = cfg.num_class as usize;
        let mut sorted: Vec<usize> = (0..md.label.len()).collect();
        sorted.sort_by(|&a, &b| md.label[a].total_cmp(&md.label[b]));
        let mut class_sizes = vec![0usize; nc];
        let mut class_data_weights = vec![0.0f64; nc];
        for (i, &l) in md.label.iter().enumerate() {
            let c = l as usize;
            if c >= nc {
                return Err(Error::InvalidParameter(format!(
                    "auc_mu: label {c} is not less than num_class {nc}"
                )));
            }
            class_sizes[c] += 1;
            if !md.weights.is_empty() {
                class_data_weights[c] += f64::from(md.weights[i]);
            }
        }
        Ok(AucMu {
            names: vec!["auc_mu".into()],
            num_class: nc,
            class_weights: cfg.auc_mu_weights_matrix.clone(),
            label: md.label.clone(),
            weights: md.weights.clone(),
            class_data_weights,
            class_sizes,
            sorted_data_idx: sorted,
        })
    }
}

impl Metric for AucMu {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], _objective: Option<&LoadedObjective>) -> Vec<f64> {
        let n = self.label.len();
        let nc = self.num_class;
        let eps = f64::from(K_EPSILON);
        let mut s = vec![vec![0.0f64; nc]; nc];
        let mut i_start = 0usize;
        for i in 0..nc {
            let mut j_start = i_start + self.class_sizes[i];
            for j in i + 1..nc {
                let curr_v: Vec<f64> = (0..nc)
                    .map(|k| self.class_weights[i][k] - self.class_weights[j][k])
                    .collect();
                let t1 = curr_v[i] - curr_v[j];
                let mut idx: Vec<usize> =
                    self.sorted_data_idx[i_start..i_start + self.class_sizes[i]].to_vec();
                idx.extend_from_slice(
                    &self.sorted_data_idx[j_start..j_start + self.class_sizes[j]],
                );
                let mut dist: Vec<(usize, f64)> = idx
                    .iter()
                    .map(|&a| {
                        let mut v_a = 0.0f64;
                        for (m, &cv) in curr_v.iter().enumerate() {
                            v_a = madd(cv, score[n * m + a], v_a);
                        }
                        (a, t1 * v_a)
                    })
                    .collect();
                dist.sort_by(|a, b| {
                    use std::cmp::Ordering::{Equal, Greater, Less};
                    if (a.1 - b.1).abs() < eps {
                        self.label[b.0].total_cmp(&self.label[a.0])
                    } else if a.1 < b.1 {
                        Less
                    } else if b.1 < a.1 {
                        Greater
                    } else {
                        Equal
                    }
                });
                let (mut num_j, mut last_j_dist, mut num_current_j) = (0.0f64, 0.0f64, 0.0f64);
                for &(a, curr_dist) in &dist {
                    let w = if self.weights.is_empty() {
                        1.0
                    } else {
                        f64::from(self.weights[a])
                    };
                    if self.label[a] as usize == i {
                        let close = (curr_dist - last_j_dist).abs() < eps;
                        let inner = if close {
                            madd(-0.5, num_current_j, num_j)
                        } else {
                            num_j
                        };
                        s[i][j] = if self.weights.is_empty() {
                            s[i][j] + inner
                        } else {
                            madd(w, inner, s[i][j])
                        };
                    } else {
                        num_j += w;
                        if (curr_dist - last_j_dist).abs() < eps {
                            num_current_j += w;
                        } else {
                            last_j_dist = curr_dist;
                            num_current_j = w;
                        }
                    }
                }
                j_start += self.class_sizes[j];
            }
            i_start += self.class_sizes[i];
        }
        let mut ans = 0.0f64;
        for i in 0..nc {
            for j in i + 1..nc {
                ans += if self.weights.is_empty() {
                    (s[i][j] / self.class_sizes[i] as f64) / self.class_sizes[j] as f64
                } else {
                    (s[i][j] / self.class_data_weights[i]) / self.class_data_weights[j]
                };
            }
        }
        vec![(2.0 * ans / nc as f64) / (nc as f64 - 1.0)]
    }
}
