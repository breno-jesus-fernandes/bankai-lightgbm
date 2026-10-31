//! Cross-entropy objectives. upstream: src/objective/xentropy_objective.hpp

use super::train::{Objective, check_labels_in};
use crate::error::Result;
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::utils::common::madd;

/// upstream: `CrossEntropy` (unweighted)
pub struct CrossEntropy {
    label: Vec<f32>,
}

impl CrossEntropy {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_labels_in(&md.label, 0.0, 1.0, "cross_entropy")?;
        Ok(CrossEntropy {
            label: md.label.clone(),
        })
    }
}

/// Mean label (upstream `BoostFromScore` sum, unweighted).
fn mean_label(label: &[f32]) -> f64 {
    let mut suml = 0.0f64;
    for &l in label {
        suml += f64::from(l);
    }
    suml / label.len() as f64
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
            if score[i] > -37.0 {
                let e = (-score[i]).exp();
                // `(1.0f - l) - l * e`: float subtraction, then a fused multiply-subtract.
                g[i] = (madd(-f64::from(l), e, f64::from(1.0f32 - l)) / (1.0 + e)) as f32;
                h[i] = (e / ((1.0 + e) * (1.0 + e))) as f32;
            } else {
                let e = score[i].exp();
                g[i] = (e - f64::from(l)) as f32;
                h[i] = e as f32;
            }
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        let eps = f64::from(K_EPSILON);
        let pavg = mean_label(&self.label).min(1.0 - eps).max(eps);
        (pavg / (1.0 - pavg)).ln()
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}

/// upstream: `CrossEntropyLambda` (unweighted)
pub struct CrossEntropyLambda {
    label: Vec<f32>,
}

impl CrossEntropyLambda {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_labels_in(&md.label, 0.0, 1.0, "cross_entropy_lambda")?;
        Ok(CrossEntropyLambda {
            label: md.label.clone(),
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
        for (i, &l) in self.label.iter().enumerate() {
            let z = 1.0 / (1.0 + (-score[i]).exp());
            g[i] = (z - f64::from(l)) as f32;
            h[i] = (z * (1.0 - z)) as f32;
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        mean_label(&self.label).exp_m1().ln()
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }
}
