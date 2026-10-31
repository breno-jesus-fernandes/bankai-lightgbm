//! Cross-entropy objectives. upstream: src/objective/xentropy_objective.hpp

use super::train::{Objective, check_labels_in};
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::utils::common::{madd, obtain_min_max_sum};

/// upstream: `CrossEntropy`
pub struct CrossEntropy {
    label: Vec<f32>,
    weights: Vec<f32>,
}

impl CrossEntropy {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_labels_in(&md.label, 0.0, 1.0, "cross_entropy")?;
        if !md.weights.is_empty() {
            let (minw, _, sumw) = obtain_min_max_sum(&md.weights);
            if minw < 0.0 {
                return Err(Error::InvalidParameter(
                    "[cross_entropy]: at least one weight is negative".into(),
                ));
            }
            if sumw == 0.0 {
                return Err(Error::InvalidParameter(
                    "[cross_entropy]: sum of weights is zero".into(),
                ));
            }
        }
        Ok(CrossEntropy {
            label: md.label.clone(),
            weights: md.weights.clone(),
        })
    }
}

/// (Weighted) mean label, as summed by upstream `BoostFromScore`.
fn mean_label(label: &[f32], weights: &[f32]) -> f64 {
    let mut suml = 0.0f64;
    if weights.is_empty() {
        for &l in label {
            suml += f64::from(l);
        }
        suml / label.len() as f64
    } else {
        let mut sumw = 0.0f64;
        for (&l, &w) in label.iter().zip(weights) {
            suml = madd(f64::from(l), f64::from(w), suml);
            sumw += f64::from(w);
        }
        suml / sumw
    }
}

impl Objective for CrossEntropy {
    fn name(&self) -> &'static str {
        "cross_entropy"
    }

    fn to_model_string(&self) -> String {
        "cross_entropy".into()
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        for (i, &l) in self.label.iter().enumerate() {
            let (gi, hi) = if score[i] > -37.0 {
                let e = (-score[i]).exp();
                // `(1.0f - l) - l * e`: float subtraction, then a fused multiply-subtract.
                (
                    madd(-f64::from(l), e, f64::from(1.0f32 - l)) / (1.0 + e),
                    e / ((1.0 + e) * (1.0 + e)),
                )
            } else {
                let e = score[i].exp();
                (e - f64::from(l), e)
            };
            if self.weights.is_empty() {
                g[i] = gi as f32;
                h[i] = hi as f32;
            } else {
                let w = f64::from(self.weights[i]);
                g[i] = (gi * w) as f32;
                h[i] = (hi * w) as f32;
            }
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        let eps = f64::from(K_EPSILON);
        let pavg = mean_label(&self.label, &self.weights)
            .min(1.0 - eps)
            .max(eps);
        (pavg / (1.0 - pavg)).ln()
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}

/// upstream: `CrossEntropyLambda`
pub struct CrossEntropyLambda {
    label: Vec<f32>,
    weights: Vec<f32>,
}

impl CrossEntropyLambda {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_labels_in(&md.label, 0.0, 1.0, "cross_entropy_lambda")?;
        if md.weights.iter().any(|&w| w <= 0.0) {
            return Err(Error::InvalidParameter(
                "[cross_entropy_lambda]: at least one weight is non-positive".into(),
            ));
        }
        Ok(CrossEntropyLambda {
            label: md.label.clone(),
            weights: md.weights.clone(),
        })
    }
}

impl Objective for CrossEntropyLambda {
    fn name(&self) -> &'static str {
        "cross_entropy_lambda"
    }

    fn to_model_string(&self) -> String {
        "cross_entropy_lambda".into()
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        if self.weights.is_empty() {
            for (i, &l) in self.label.iter().enumerate() {
                let z = 1.0 / (1.0 + (-score[i]).exp());
                g[i] = (z - f64::from(l)) as f32;
                h[i] = (z * (1.0 - z)) as f32;
            }
            return;
        }
        for (i, &l) in self.label.iter().enumerate() {
            let w = f64::from(self.weights[i]);
            let y = f64::from(l);
            let epf = score[i].exp();
            let hhat = epf.ln_1p();
            let z = 1.0 - (-w * hhat).exp();
            let enf = 1.0 / epf;
            g[i] = ((1.0 - y / z) * w / (1.0 + enf)) as f32;
            let c = 1.0 / (1.0 - z);
            let mut d = 1.0 + epf;
            let a = w * epf / (d * d);
            d = c - 1.0;
            // `1.0f + w * epf - c` and `1.0f + y * b` contract into fused multiply-adds.
            let b = (c / (d * d)) * (madd(w, epf, 1.0) - c);
            h[i] = (a * madd(y, b, 1.0)) as f32;
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        mean_label(&self.label, &self.weights).exp_m1().ln()
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}
