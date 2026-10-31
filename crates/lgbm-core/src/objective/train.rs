//! Training-side objectives (gradients, init scores). upstream: src/objective/*.hpp

use super::binary::BinaryLogloss;
use super::multiclass::{MulticlassOva, MulticlassSoftmax};
use super::rank::{LambdarankNdcg, RankXendcg};
use super::regression::Regression;
use super::xentropy::{CrossEntropy, CrossEntropyLambda};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::dataset::Metadata;

/// upstream: `ObjectiveFunction` (training interface)
pub trait Objective: Send + Sync {
    fn name(&self) -> &'static str;
    /// upstream `ObjectiveFunction::ToString` (the model file's `objective=` line).
    fn to_model_string(&self) -> String;
    /// Gradients for all `num_model_per_iteration()` classes (class-major, like `score`).
    fn get_gradients(&self, score: &[f64], gradients: &mut [f32], hessians: &mut [f32]);
    fn boost_from_score(&self, class_id: usize) -> f64;
    fn is_constant_hessian(&self) -> bool;
    fn num_model_per_iteration(&self) -> usize {
        1
    }
    /// upstream: `ObjectiveFunction::ClassNeedTrain`
    fn class_need_train(&self, _class_id: usize) -> bool {
        true
    }
    fn is_renew_tree_output(&self) -> bool {
        false
    }
    /// upstream: `ObjectiveFunction::RenewTreeOutput`, with the residual getter
    /// `label[i] - score[i]` and no bagging. `indices` are the rows of one leaf.
    fn renew_tree_output(&self, output: f64, _score: &[f64], _indices: &[i32]) -> f64 {
        output
    }
}

/// upstream: `ObjectiveFunction::CreateObjectiveFunction(const std::string&, const Config&)`
/// followed by `Init(metadata, num_data)`.
pub fn create_objective(cfg: &Config, md: &Metadata) -> Result<Box<dyn Objective>> {
    let name = cfg.objective.as_str();
    Ok(match name {
        "regression" | "regression_l1" | "quantile" | "huber" | "fair" | "poisson" | "mape"
        | "gamma" | "tweedie" => Box::new(Regression::new(name, cfg, md)?),
        "binary" => Box::new(BinaryLogloss::new(cfg, md, |l| l > 0.0)?),
        "multiclass" => Box::new(MulticlassSoftmax::new(cfg, md)?),
        "multiclassova" => Box::new(MulticlassOva::new(cfg, md)?),
        "lambdarank" => Box::new(LambdarankNdcg::new(cfg, md)?),
        "rank_xendcg" => Box::new(RankXendcg::new(cfg, md)?),
        "cross_entropy" => Box::new(CrossEntropy::new(md)?),
        "cross_entropy_lambda" => Box::new(CrossEntropyLambda::new(md)?),
        other => {
            return Err(Error::Unsupported(format!(
                "training with objective={other}"
            )));
        }
    })
}

/// upstream: `Common::Sign`
pub(crate) fn sign(x: f64) -> i32 {
    i32::from(x > 0.0) - i32::from(x < 0.0)
}

/// upstream: `Common::CheckElementsIntervalClosed`
pub(crate) fn check_labels_in(label: &[f32], lo: f32, hi: f32, name: &str) -> Result<()> {
    match label.iter().find(|&&l| !(lo..=hi).contains(&l)) {
        Some(l) => Err(Error::InvalidParameter(format!(
            "[{name}]: does not tolerate element [{l}] outside [{lo}, {hi}]"
        ))),
        None => Ok(()),
    }
}
