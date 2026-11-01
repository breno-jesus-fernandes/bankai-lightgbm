//! Cross-entropy metrics. upstream: src/metric/xentropy_metric.hpp

use super::{Metric, convert1, weighted_loss_sum};
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::objective::LoadedObjective;
use crate::utils::common::{madd, obtain_min_max_sum};

/// upstream: `XentLoss`
fn xent_loss(label: f32, prob: f64) -> f64 {
    let eps = 1.0e-12;
    let mut a = f64::from(label);
    a *= if prob > eps { prob.ln() } else { f64::ln(eps) };
    // `1.0f - label` is a float subtraction.
    let mut b = f64::from(1.0f32 - label);
    b *= if 1.0 - prob > eps {
        (1.0 - prob).ln()
    } else {
        f64::ln(eps)
    };
    -(a + b)
}

/// upstream: `YentLoss`
fn yent_loss(p: f64) -> f64 {
    let mut hp = 0.0f64;
    if p > 0.0 {
        hp = madd(p, p.ln(), hp);
    }
    let q = 1.0 - p;
    if q > 0.0 {
        hp = madd(q, q.ln(), hp);
    }
    hp
}

fn check_unit_labels(label: &[f32], name: &str) -> Result<()> {
    super::super::objective::train::check_labels_in(label, 0.0, 1.0, name)
}

fn check_weights(md: &Metadata, name: &str) -> Result<f64> {
    let sw = if md.weights.is_empty() {
        md.label.len() as f64
    } else {
        obtain_min_max_sum(&md.weights).2
    };
    if md.weights.iter().any(|&w| w < 0.0) {
        return Err(Error::InvalidParameter(format!(
            "[{name}]: (metric) weights not allowed to be negative"
        )));
    }
    if sw <= 0.0 {
        return Err(Error::InvalidParameter(format!(
            "[{name}]: sum-of-weights = {sw} is non-positive"
        )));
    }
    Ok(sw)
}

/// upstream: `CrossEntropyMetric`
pub struct CrossEntropyMetric {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
}

impl CrossEntropyMetric {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_unit_labels(&md.label, "cross_entropy")?;
        Ok(CrossEntropyMetric {
            names: vec!["cross_entropy".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: check_weights(md, "cross_entropy")?,
        })
    }
}

impl Metric for CrossEntropyMetric {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let sum = weighted_loss_sum(self.label.len(), &self.weights, |i| {
            xent_loss(self.label[i], convert1(objective, score[i]))
        });
        vec![sum / self.sum_weights]
    }
}

/// upstream: `CrossEntropyLambdaMetric`
pub struct CrossEntropyLambdaMetric {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
}

impl CrossEntropyLambdaMetric {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_unit_labels(&md.label, "cross_entropy_lambda")?;
        if md.weights.iter().any(|&w| w <= 0.0) {
            return Err(Error::InvalidParameter(
                "[cross_entropy_lambda]: (metric) all weights must be positive".into(),
            ));
        }
        Ok(CrossEntropyLambdaMetric {
            names: vec!["cross_entropy_lambda".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
        })
    }
}

impl Metric for CrossEntropyLambdaMetric {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let mut sum = 0.0f64;
        for (i, &label) in self.label.iter().enumerate() {
            let hhat = match objective {
                None => score[i].exp().ln_1p(),
                Some(_) => convert1(objective, score[i]),
            };
            let w = if self.weights.is_empty() {
                1.0f32
            } else {
                self.weights[i]
            };
            sum += xent_loss(label, 1.0 - (-f64::from(w) * hhat).exp());
        }
        vec![sum / self.label.len() as f64]
    }
}

/// upstream: `KullbackLeiblerDivergence`
pub struct KullbackLeibler {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
    presum_label_entropy: f64,
}

impl KullbackLeibler {
    pub fn new(md: &Metadata) -> Result<Self> {
        check_unit_labels(&md.label, "kullback_leibler")?;
        let sw = check_weights(md, "kullback_leibler")?;
        let entropy = weighted_loss_sum(md.label.len(), &md.weights, |i| {
            yent_loss(f64::from(md.label[i]))
        });
        Ok(KullbackLeibler {
            names: vec!["kullback_leibler".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sw,
            presum_label_entropy: entropy / sw,
        })
    }
}

impl Metric for KullbackLeibler {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let sum = weighted_loss_sum(self.label.len(), &self.weights, |i| {
            xent_loss(self.label[i], convert1(objective, score[i]))
        });
        vec![self.presum_label_entropy + sum / self.sum_weights]
    }
}
