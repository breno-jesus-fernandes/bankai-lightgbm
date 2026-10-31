//! Training-side objectives (gradients, init scores). upstream: src/objective/*.hpp

use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;

/// upstream: `ObjectiveFunction` (training interface)
pub trait Objective: Send + Sync {
    fn name(&self) -> &'static str;
    /// upstream `ObjectiveFunction::ToString` (the model file's `objective=` line).
    fn to_model_string(&self) -> String;
    fn get_gradients(&self, score: &[f64], gradients: &mut [f32], hessians: &mut [f32]);
    fn boost_from_score(&self, class_id: usize) -> f64;
    fn is_constant_hessian(&self) -> bool;
    fn num_model_per_iteration(&self) -> usize {
        1
    }
    fn is_renew_tree_output(&self) -> bool {
        false
    }
}

/// upstream: `ObjectiveFunction::CreateObjectiveFunction(const std::string&, const Config&)`
pub fn create_objective(cfg: &Config, metadata: &Metadata) -> Result<Box<dyn Objective>> {
    match cfg.objective.as_str() {
        "regression" => Ok(Box::new(RegressionL2::new(cfg, metadata))),
        other => Err(Error::Unsupported(format!(
            "training with objective={other}"
        ))),
    }
}

/// upstream: `Common::Sign`
fn sign(x: f64) -> f64 {
    f64::from(i32::from(x > 0.0) - i32::from(x < 0.0))
}

/// upstream: `RegressionL2loss`
pub struct RegressionL2 {
    sqrt: bool,
    label: Vec<f32>,
    weights: Vec<f32>,
}

impl RegressionL2 {
    pub fn new(cfg: &Config, md: &Metadata) -> Self {
        let sqrt = cfg.reg_sqrt;
        let label = if sqrt {
            md.label
                .iter()
                .map(|&l| (sign(f64::from(l)) * f64::from(l).abs().sqrt()) as f32)
                .collect()
        } else {
            md.label.clone()
        };
        RegressionL2 {
            sqrt,
            label,
            weights: md.weights.clone(),
        }
    }
}

impl Objective for RegressionL2 {
    fn name(&self) -> &'static str {
        "regression"
    }

    fn to_model_string(&self) -> String {
        if self.sqrt {
            "regression sqrt".into()
        } else {
            "regression".into()
        }
    }

    fn get_gradients(&self, score: &[f64], gradients: &mut [f32], hessians: &mut [f32]) {
        if self.weights.is_empty() {
            for i in 0..self.label.len() {
                gradients[i] = (score[i] - f64::from(self.label[i])) as f32;
                hessians[i] = 1.0;
            }
        } else {
            for i in 0..self.label.len() {
                gradients[i] = ((score[i] - f64::from(self.label[i])) as f32) * self.weights[i];
                hessians[i] = self.weights[i];
            }
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        let mut suml = 0.0f64;
        let sumw;
        if self.weights.is_empty() {
            sumw = self.label.len() as f64;
            for &l in &self.label {
                suml += f64::from(l);
            }
        } else {
            let mut w = 0.0f64;
            for (&l, &wt) in self.label.iter().zip(&self.weights) {
                suml = crate::utils::common::madd(f64::from(l), f64::from(wt), suml);
                w += f64::from(wt);
            }
            sumw = w;
        }
        suml / sumw
    }

    fn is_constant_hessian(&self) -> bool {
        self.weights.is_empty()
    }
}
