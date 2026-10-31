//! Batch prediction over dense rows. upstream: src/application/predictor.hpp,
//! src/boosting/gbdt_prediction.cpp, src/boosting/prediction_early_stop.cpp

use rayon::prelude::*;

use crate::boosting::Gbdt;
use crate::error::{Error, Result};
use crate::meta::K_ZERO_THRESHOLD;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictKind {
    /// Transformed output (probabilities for classification).
    #[default]
    Normal,
    RawScore,
    /// Leaf index per tree (`pred_leaf`).
    LeafIndex,
    /// SHAP feature contributions (`pred_contrib`).
    Contrib,
}

#[derive(Debug, Clone)]
pub struct PredictConfig {
    pub kind: PredictKind,
    pub start_iteration: i32,
    /// `<= 0` means all iterations.
    pub num_iteration: i32,
    pub pred_early_stop: bool,
    pub pred_early_stop_freq: i32,
    pub pred_early_stop_margin: f64,
    /// `0` uses rayon's global pool.
    pub num_threads: usize,
}

impl Default for PredictConfig {
    fn default() -> Self {
        PredictConfig {
            kind: PredictKind::Normal,
            start_iteration: 0,
            num_iteration: -1,
            pred_early_stop: false,
            pred_early_stop_freq: 10,
            pred_early_stop_margin: 10.0,
            num_threads: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EarlyStop {
    None,
    Binary { margin: f64, period: usize },
    Multiclass { margin: f64, period: usize },
}

impl EarlyStop {
    fn period(&self) -> usize {
        match *self {
            EarlyStop::None => usize::MAX,
            EarlyStop::Binary { period, .. } | EarlyStop::Multiclass { period, .. } => period,
        }
    }

    #[inline]
    fn should_stop(&self, pred: &[f64]) -> bool {
        match *self {
            EarlyStop::None => false,
            EarlyStop::Binary { margin, .. } => 2.0 * pred[0].abs() > margin,
            EarlyStop::Multiclass { margin, .. } => {
                let (mut a, mut b) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &v in pred {
                    if v > a {
                        b = a;
                        a = v;
                    } else if v > b {
                        b = v;
                    }
                }
                a - b > margin
            }
        }
    }
}

/// Prepared prediction over a loaded model.
pub struct Predictor<'a> {
    model: &'a Gbdt,
    kind: PredictKind,
    start: usize,
    num: usize,
    early_stop: EarlyStop,
    num_pred_one_row: usize,
    pool: Option<rayon::ThreadPool>,
}

impl<'a> Predictor<'a> {
    pub fn new(model: &'a Gbdt, cfg: &PredictConfig) -> Result<Self> {
        let (start, num) = model.iteration_range(cfg.start_iteration, cfg.num_iteration);
        let mut early_stop = EarlyStop::None;
        if cfg.pred_early_stop && !model.need_accurate_prediction() {
            if cfg.pred_early_stop_freq <= 0 || cfg.pred_early_stop_margin < 0.0 {
                return Err(Error::InvalidParameter(
                    "pred_early_stop_freq must be > 0 and pred_early_stop_margin >= 0".into(),
                ));
            }
            let (margin, period) = (
                cfg.pred_early_stop_margin,
                cfg.pred_early_stop_freq as usize,
            );
            early_stop = if model.num_class() == 1 {
                EarlyStop::Binary { margin, period }
            } else {
                if model.num_tree_per_iteration() < 2 {
                    return Err(Error::InvalidParameter(
                        "Multiclass early stopping needs predictions to be of length two or larger"
                            .into(),
                    ));
                }
                EarlyStop::Multiclass { margin, period }
            };
        }
        if cfg.kind == PredictKind::Contrib && model.is_linear() {
            return Err(Error::Unsupported(
                "Predicting SHAP feature contributions is not implemented for linear trees.".into(),
            ));
        }
        // upstream: GBDT::NumPredictOneRow
        let num_pred_one_row = match cfg.kind {
            PredictKind::LeafIndex => model.num_class() * num,
            PredictKind::Contrib => model.num_tree_per_iteration() * (model.num_features() + 1),
            _ => model.num_class(),
        };
        let pool = if cfg.num_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cfg.num_threads)
                    .build()
                    .map_err(|e| Error::InvalidParameter(format!("thread pool: {e}")))?,
            )
        } else {
            None
        };
        Ok(Predictor {
            model,
            kind: cfg.kind,
            start,
            num,
            early_stop,
            num_pred_one_row,
            pool,
        })
    }

    pub fn num_pred_one_row(&self) -> usize {
        self.num_pred_one_row
    }

    /// Predict a row-major dense matrix of `nrow x ncol` values.
    /// Returns `nrow * num_pred_one_row` values, row-major.
    pub fn predict_dense(&self, data: &[f64], nrow: usize, ncol: usize) -> Result<Vec<f64>> {
        if data.len() != nrow * ncol {
            return Err(Error::InvalidParameter(format!(
                "data has {} values, expected {nrow} x {ncol}",
                data.len()
            )));
        }
        let mut out = vec![0.0; nrow * self.num_pred_one_row];
        if nrow == 0 {
            return Ok(out);
        }
        let k = self.num_pred_one_row;
        let nf = self.model.num_features();
        let run = |out: &mut [f64]| {
            out.par_chunks_mut(k)
                .zip(data.par_chunks(ncol.max(1)))
                .for_each_init(
                    || vec![0.0f64; nf],
                    |buf, (o, row)| {
                        fill_buffer(buf, row);
                        self.predict_one(buf, o);
                    },
                );
        };
        match &self.pool {
            Some(p) => p.install(|| run(&mut out)),
            None => run(&mut out),
        }
        Ok(out)
    }

    /// Predict one row given a dense feature buffer of length `num_features`.
    #[inline]
    pub fn predict_one(&self, features: &[f64], out: &mut [f64]) {
        let m = self.model;
        let ntpi = m.num_tree_per_iteration();
        let trees = m.trees();
        match self.kind {
            PredictKind::LeafIndex => {
                let first = self.start * ntpi;
                for (i, t) in trees[first..first + self.num * ntpi].iter().enumerate() {
                    out[i] = t.predict_leaf_index(features) as f64;
                }
            }
            PredictKind::Contrib => {
                let nf = m.num_features();
                out.fill(0.0);
                for it in self.start..self.start + self.num {
                    for k in 0..ntpi {
                        trees[it * ntpi + k].predict_contrib(
                            features,
                            nf,
                            &mut out[k * (nf + 1)..(k + 1) * (nf + 1)],
                        );
                    }
                }
            }
            PredictKind::RawScore | PredictKind::Normal => {
                // upstream: GBDT::PredictRaw (output has num_tree_per_iteration slots)
                let raw = &mut out[..ntpi];
                raw.fill(0.0);
                let period = self.early_stop.period();
                let mut counter = 0usize;
                for it in self.start..self.start + self.num {
                    for k in 0..ntpi {
                        raw[k] += trees[it * ntpi + k].predict(features);
                    }
                    counter += 1;
                    if counter == period {
                        if self.early_stop.should_stop(raw) {
                            break;
                        }
                        counter = 0;
                    }
                }
                if self.kind == PredictKind::Normal {
                    if m.average_output() {
                        for v in raw.iter_mut() {
                            *v /= self.num as f64;
                        }
                    }
                    if let Some(obj) = m.objective() {
                        obj.convert_output(raw);
                    }
                }
            }
        }
    }
}

/// Copy a dense row into the feature buffer the way upstream's dense-matrix path does:
/// values with `|x| <= kZeroThreshold` (and not NaN) are dropped, i.e. read as `0.0`;
/// columns past `num_features` are ignored.
#[inline]
fn fill_buffer(buf: &mut [f64], row: &[f64]) {
    let n = buf.len().min(row.len());
    for (b, &v) in buf[..n].iter_mut().zip(row) {
        *b = if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
            v
        } else {
            0.0
        };
    }
    buf[n..].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_buffer_zeroes_tiny_values_and_pads() {
        let mut buf = vec![9.0; 4];
        fill_buffer(&mut buf, &[1e-40, -0.0, f64::NAN]);
        assert_eq!(buf[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(buf[1].to_bits(), 0.0f64.to_bits());
        assert!(buf[2].is_nan());
        assert_eq!(buf[3], 0.0);
    }

    #[test]
    fn multiclass_margin_uses_top_two() {
        let es = EarlyStop::Multiclass {
            margin: 1.0,
            period: 1,
        };
        assert!(es.should_stop(&[0.0, 3.0, 1.5]));
        assert!(!es.should_stop(&[0.0, 3.0, 2.5]));
        let es = EarlyStop::Binary {
            margin: 1.0,
            period: 1,
        };
        assert!(es.should_stop(&[-0.6]));
        assert!(!es.should_stop(&[0.4]));
    }
}
