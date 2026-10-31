//! Regression metrics. upstream: src/metric/regression_metric.hpp

use super::{Metric, check_positive_labels, convert1, sum_weights};
use crate::config::Config;
use crate::error::Result;
use crate::io::dataset::Metadata;
use crate::meta::K_ZERO_THRESHOLD;
use crate::objective::LoadedObjective;
use crate::utils::common::madd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Rmse,
    L2,
    Quantile,
    L1,
    Huber,
    Fair,
    Poisson,
    Mape,
    Gamma,
    GammaDeviance,
    Tweedie,
}

/// upstream: `RegressionMetric<PointWiseLossCalculator>`
pub struct Pointwise {
    kind: Kind,
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    sum_weights: f64,
    alpha: f64,
    fair_c: f64,
    rho: f64,
}

fn safe_log(x: f64) -> f64 {
    if x > 0.0 { x.ln() } else { f64::NEG_INFINITY }
}

fn safe_log_f32(x: f32) -> f32 {
    if x > 0.0 { x.ln() } else { f32::NEG_INFINITY }
}

impl Pointwise {
    pub fn new(name: &str, cfg: &Config, md: &Metadata) -> Result<Self> {
        let kind = match name {
            "rmse" => Kind::Rmse,
            "l2" => Kind::L2,
            "quantile" => Kind::Quantile,
            "l1" => Kind::L1,
            "huber" => Kind::Huber,
            "fair" => Kind::Fair,
            "poisson" => Kind::Poisson,
            "mape" => Kind::Mape,
            "gamma" => Kind::Gamma,
            "gamma_deviance" => Kind::GammaDeviance,
            _ => Kind::Tweedie,
        };
        if matches!(kind, Kind::Gamma | Kind::GammaDeviance) {
            check_positive_labels(&md.label, name)?;
        }
        Ok(Pointwise {
            kind,
            names: vec![name.to_string()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            sum_weights: sum_weights(md),
            alpha: cfg.alpha,
            fair_c: cfg.fair_c,
            rho: cfg.tweedie_variance_power,
        })
    }

    /// upstream: `LossOnPoint`
    #[inline]
    fn loss(&self, label: f32, score: f64) -> f64 {
        let l = f64::from(label);
        match self.kind {
            Kind::Rmse | Kind::L2 => (score - l) * (score - l),
            Kind::Quantile => {
                let delta = l - score;
                if delta < 0.0 {
                    (self.alpha - 1.0) * delta
                } else {
                    self.alpha * delta
                }
            }
            Kind::L1 => (score - l).abs(),
            Kind::Huber => {
                let diff = score - l;
                if diff.abs() <= self.alpha {
                    0.5 * diff * diff
                } else {
                    self.alpha * (diff.abs() - 0.5 * self.alpha)
                }
            }
            Kind::Fair => {
                let x = (score - l).abs();
                let c = self.fair_c;
                // `c * x - c * c * log1p(x / c)`: clang fuses the left product.
                madd(c, x, -(c * c * (x / c).ln_1p()))
            }
            Kind::Poisson => {
                let eps = f64::from(1e-10f32);
                let s = if score < eps { eps } else { score };
                madd(-l, s.ln(), s)
            }
            Kind::Mape => (l - score).abs() / f64::from(1.0f32.max(label.abs())),
            Kind::Gamma => {
                let theta = -1.0 / score;
                let b = -safe_log(-theta);
                // `SafeLog(label)` is instantiated for `float`, so it uses single-precision log.
                let c = safe_log(l) - f64::from(safe_log_f32(label));
                -((madd(l, theta, -b)) / 1.0 + c)
            }
            Kind::GammaDeviance => {
                let tmp = l / (score + 1.0e-9);
                tmp - safe_log(tmp) - 1.0
            }
            Kind::Tweedie => {
                let rho = self.rho;
                let eps = f64::from(1e-10f32);
                let s = if score < eps { eps } else { score };
                let a = l * ((1.0 - rho) * s.ln()).exp() / (1.0 - rho);
                let b = ((2.0 - rho) * s.ln()).exp() / (2.0 - rho);
                -a + b
            }
        }
    }

    fn average(&self, sum_loss: f64) -> f64 {
        match self.kind {
            Kind::Rmse => (sum_loss / self.sum_weights).sqrt(),
            Kind::GammaDeviance => sum_loss * 2.0,
            _ => sum_loss / self.sum_weights,
        }
    }
}

impl Metric for Pointwise {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        -1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let mut sum = 0.0f64;
        for (i, &label) in self.label.iter().enumerate() {
            let loss = self.loss(label, convert1(objective, score[i]));
            if self.weights.is_empty() {
                sum += loss;
            } else {
                sum = madd(loss, f64::from(self.weights[i]), sum);
            }
        }
        vec![self.average(sum)]
    }
}

/// upstream: `R2Metric`
pub struct R2 {
    names: Vec<String>,
    label: Vec<f32>,
    weights: Vec<f32>,
    total_sum_squares: f64,
}

impl R2 {
    pub fn new(md: &Metadata) -> Self {
        let n = md.label.len();
        let (mut sum_label, sum_weights);
        if md.weights.is_empty() {
            sum_weights = n as f64;
            sum_label = 0.0f64;
            for &l in &md.label {
                sum_label += f64::from(l);
            }
        } else {
            let mut sw = 0.0f64;
            sum_label = 0.0;
            for (&l, &w) in md.label.iter().zip(&md.weights) {
                sw += f64::from(w);
                // `label_[i] * weights_[i]` multiplies two floats before widening.
                sum_label += f64::from(l * w);
            }
            sum_weights = sw;
        }
        let mean = sum_label / sum_weights;
        let mut tss = 0.0f64;
        for (i, &l) in md.label.iter().enumerate() {
            let diff = f64::from(l) - mean;
            tss = if md.weights.is_empty() {
                madd(diff, diff, tss)
            } else {
                madd(diff * diff, f64::from(md.weights[i]), tss)
            };
        }
        R2 {
            names: vec!["r2".into()],
            label: md.label.clone(),
            weights: md.weights.clone(),
            total_sum_squares: tss,
        }
    }
}

impl Metric for R2 {
    fn names(&self) -> &[String] {
        &self.names
    }

    fn factor_to_bigger_better(&self) -> f64 {
        1.0
    }

    fn eval(&self, score: &[f64], objective: Option<&LoadedObjective>) -> Vec<f64> {
        let mut rss = 0.0f64;
        for (i, &l) in self.label.iter().enumerate() {
            let diff = f64::from(l) - convert1(objective, score[i]);
            rss = if self.weights.is_empty() {
                madd(diff, diff, rss)
            } else {
                madd(diff * diff, f64::from(self.weights[i]), rss)
            };
        }
        let r2 = 1.0 - rss / self.total_sum_squares;
        if self.total_sum_squares.abs() < K_ZERO_THRESHOLD {
            return vec![if rss.abs() < K_ZERO_THRESHOLD {
                1.0
            } else {
                0.0
            }];
        }
        vec![r2]
    }
}
