//! Binary log-loss. upstream: src/objective/binary_objective.hpp

use super::train::Objective;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;
use crate::meta::K_EPSILON;
use crate::utils::format::format_g;

/// upstream: `BinaryLogloss` (unweighted)
pub struct BinaryLogloss {
    sigmoid: f64,
    is_pos: Vec<bool>,
    label_weights: [f64; 2],
    need_train: bool,
}

impl BinaryLogloss {
    /// `is_pos` decides the positive class (upstream: `label > 0`, or `label == i` for OVA).
    pub fn new(cfg: &Config, md: &Metadata, is_pos: impl Fn(f32) -> bool) -> Result<Self> {
        let sigmoid = cfg.sigmoid;
        if sigmoid <= 0.0 {
            return Err(Error::InvalidParameter(format!(
                "Sigmoid parameter {sigmoid} should be greater than zero"
            )));
        }
        let scale_pos_weight = cfg.scale_pos_weight;
        if cfg.is_unbalance && (scale_pos_weight - 1.0).abs() > 1e-6 {
            return Err(Error::InvalidParameter(
                "Cannot set is_unbalance and scale_pos_weight at the same time".into(),
            ));
        }
        let is_pos: Vec<bool> = md.label.iter().map(|&l| is_pos(l)).collect();
        let cnt_positive = is_pos.iter().filter(|&&p| p).count() as i32;
        let cnt_negative = is_pos.len() as i32 - cnt_positive;
        let need_train = cnt_negative != 0 && cnt_positive != 0;
        if !need_train {
            log::warn!("Contains only one class");
        }
        let mut label_weights = [1.0f64, 1.0];
        if cfg.is_unbalance && cnt_positive > 0 && cnt_negative > 0 {
            if cnt_positive > cnt_negative {
                label_weights[0] = f64::from(cnt_positive) / f64::from(cnt_negative);
            } else {
                label_weights[1] = f64::from(cnt_negative) / f64::from(cnt_positive);
            }
        }
        label_weights[1] *= scale_pos_weight;
        Ok(BinaryLogloss {
            sigmoid,
            is_pos,
            label_weights,
            need_train,
        })
    }
}

impl Objective for BinaryLogloss {
    fn name(&self) -> &'static str {
        "binary"
    }

    fn to_model_string(&self) -> String {
        format!("binary sigmoid:{}", format_g(self.sigmoid, 6))
    }

    fn get_gradients(&self, score: &[f64], g: &mut [f32], h: &mut [f32]) {
        if !self.need_train {
            return;
        }
        let s = self.sigmoid;
        for (i, &pos) in self.is_pos.iter().enumerate() {
            let label = if pos { 1.0 } else { -1.0 };
            let label_weight = self.label_weights[usize::from(pos)];
            let response = -label * s / (1.0 + (label * s * score[i]).exp());
            let abs_response = response.abs();
            g[i] = (response * label_weight) as f32;
            h[i] = (abs_response * (s - abs_response) * label_weight) as f32;
        }
    }

    fn boost_from_score(&self, _class_id: usize) -> f64 {
        let mut suml = 0.0f64;
        for &p in &self.is_pos {
            suml += f64::from(u8::from(p));
        }
        let eps = f64::from(K_EPSILON);
        let pavg = (suml / self.is_pos.len() as f64).min(1.0 - eps).max(eps);
        (pavg / (1.0 - pavg)).ln() / self.sigmoid
    }

    fn is_constant_hessian(&self) -> bool {
        false
    }

    fn class_need_train(&self, _class_id: usize) -> bool {
        self.need_train
    }
}
