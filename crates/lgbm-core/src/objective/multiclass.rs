//! Multiclass objectives. upstream: src/objective/multiclass_objective.hpp

use super::binary::BinaryLogloss;
use super::train::Objective;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::utils::common::softmax;
use crate::utils::format::format_g;

/// upstream: `MulticlassSoftmax`
pub struct MulticlassSoftmax {
    num_class: usize,
    factor: f64,
    label_int: Vec<usize>,
    weights: Vec<f32>,
    class_init_probs: Vec<f64>,
}

impl MulticlassSoftmax {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let num_class = cfg.num_class as usize;
        let factor = f64::from(cfg.num_class) / (f64::from(cfg.num_class) - 1.0);
        let mut label_int = Vec::with_capacity(md.label.len());
        let mut class_init_probs = vec![0.0f64; num_class];
        let mut sum_weight = 0.0f64;
        for (i, &l) in md.label.iter().enumerate() {
            let li = l as i32;
            if li < 0 || li >= cfg.num_class {
                return Err(Error::InvalidParameter(format!(
                    "Label must be in [0, {num_class}), but found {li} in label"
                )));
            }
            label_int.push(li as usize);
            if md.weights.is_empty() {
                class_init_probs[li as usize] += 1.0;
            } else {
                class_init_probs[li as usize] += f64::from(md.weights[i]);
                sum_weight += f64::from(md.weights[i]);
            }
        }
        if md.weights.is_empty() {
            sum_weight = md.label.len() as f64;
        }
        for p in &mut class_init_probs {
            *p /= sum_weight;
        }
        Ok(MulticlassSoftmax {
            num_class,
            factor,
            label_int,
            weights: md.weights.clone(),
            class_init_probs,
        })
    }
}

impl Objective for MulticlassSoftmax {
    fn name(&self) -> &'static str {
        "multiclass"
    }

    fn to_model_string(&self) -> String {
        format!("multiclass num_class:{}", self.num_class)
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        let n = self.label_int.len();
        let mut rec = vec![0.0f64; self.num_class];
        for i in 0..n {
            for (k, r) in rec.iter_mut().enumerate() {
                *r = score[n * k + i];
            }
            softmax(&mut rec);
            for (k, &p) in rec.iter().enumerate() {
                let idx = n * k + i;
                let gi = if self.label_int[i] == k { p - 1.0 } else { p };
                let hi = self.factor * p * (1.0 - p);
                if self.weights.is_empty() {
                    g[idx] = gi as f32;
                    h[idx] = hi as f32;
                } else {
                    let w = f64::from(self.weights[i]);
                    g[idx] = (gi * w) as f32;
                    h[idx] = (hi * w) as f32;
                }
            }
        }
    }

    fn boost_from_score(&self, class_id: usize) -> f64 {
        f64::from(K_EPSILON)
            .max(self.class_init_probs[class_id])
            .ln()
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }

    fn num_model_per_iteration(&self) -> usize {
        self.num_class
    }

    fn class_need_train(&self, class_id: usize) -> bool {
        let p = self.class_init_probs[class_id].abs();
        let eps = f64::from(K_EPSILON);
        !(p <= eps || p >= 1.0 - eps)
    }
}

/// upstream: `MulticlassOVA` (one `BinaryLogloss` per class)
pub struct MulticlassOva {
    sigmoid: f64,
    num_data: usize,
    binary: Vec<BinaryLogloss>,
}

impl MulticlassOva {
    pub fn new(cfg: &Config, md: &Metadata) -> Result<Self> {
        let binary = (0..cfg.num_class)
            .map(|i| BinaryLogloss::new(cfg, md, move |l| l as i32 == i))
            .collect::<Result<Vec<_>>>()?;
        Ok(MulticlassOva {
            sigmoid: cfg.sigmoid,
            num_data: md.label.len(),
            binary,
        })
    }
}

impl Objective for MulticlassOva {
    fn name(&self) -> &'static str {
        "multiclassova"
    }

    fn to_model_string(&self) -> String {
        format!(
            "multiclassova num_class:{} sigmoid:{}",
            self.binary.len(),
            format_g(self.sigmoid, 6)
        )
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        let n = self.num_data;
        for (k, b) in self.binary.iter().enumerate() {
            let r = k * n..(k + 1) * n;
            b.get_gradients(&score[r.clone()], &mut g[r.clone()], &mut h[r]);
        }
    }

    fn boost_from_score(&self, class_id: usize) -> f64 {
        self.binary[class_id].boost_from_score(0)
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }

    fn num_model_per_iteration(&self) -> usize {
        self.binary.len()
    }

    fn class_need_train(&self, class_id: usize) -> bool {
        self.binary[class_id].class_need_train(0)
    }
}
