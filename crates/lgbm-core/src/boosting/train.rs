//! GBDT training loop. upstream: src/boosting/gbdt.cpp (`Init`, `TrainOneIter`,
//! `BoostFromAverage`, `UpdateScore`)

use super::Gbdt;
use std::sync::Arc;

use super::sample::SampleStrategy;
use crate::config::{Config, check_dataset_reset_config};
use crate::error::{Error, Result};
use crate::io::bin::BinType;
use crate::io::dataset::Dataset;
use crate::meta::K_EPSILON;
use crate::objective::LoadedObjective;
use crate::objective::train::{Objective, create_objective};
use crate::tree::Tree;
use crate::treelearner::feature_histogram::calculate_splitted_leaf_output;
use crate::treelearner::serial::{SerialTreeLearner, SplitRecord};
use crate::utils::format::format_g;
use crate::utils::random::Random;
use std::collections::HashMap;

/// upstream: `BinMapper::bin_info_string` (stream precision 17)
fn bin_info_string(m: &crate::io::bin::BinMapper) -> String {
    if m.bin_type == BinType::Categorical {
        m.bin_2_categorical
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(":")
    } else {
        format!("[{}:{}]", format_g(m.min_val, 17), format_g(m.max_val, 17))
    }
}

/// upstream: `Dataset::feature_infos` ("none" for unused features).
fn feature_infos(ds: &Dataset) -> Vec<String> {
    let l = &ds.layout;
    (0..l.num_total_features)
        .map(|i| {
            let inner = l.used_feature_map[i];
            if inner < 0 {
                return "none".to_string();
            }
            let inner = inner as usize;
            let g = &l.groups[l.feature2group[inner] as usize];
            bin_info_string(&g.bin_mappers[l.feature2subfeature[inner] as usize])
        })
        .collect()
}

impl Gbdt {
    /// Train a model on `ds` with `cfg` (upstream `Booster` construction + `num_iterations`
    /// calls to `GBDT::TrainOneIter`). This convenience method clones `ds`; use
    /// [`Gbdt::train_shared`] when the dataset is already shared or large.
    pub fn train(cfg: &Config, ds: &Dataset) -> Result<Gbdt> {
        Self::train_shared(Arc::new(cfg.clone()), Arc::new(ds.clone()))
    }

    /// Train using shared configuration and dataset storage, without copying binned data.
    pub fn train_shared(cfg: Arc<Config>, ds: Arc<Dataset>) -> Result<Gbdt> {
        let mut trainer = GbdtTrainer::new(Arc::clone(&cfg), Arc::clone(&ds))?;
        for _ in 0..cfg.num_iterations {
            if trainer.update_one_iter() {
                break;
            }
        }
        Ok(trainer.into_model())
    }
}

/// Validate a forced-splits file against the dataset's features (upstream: src/boosting/gbdt.cpp
/// (v4.7.0) `GBDT::CheckForcedSplitFeatures`). The learner applies the splits.
pub fn check_forced_split_features(path: &str, max_feature_idx: i32) -> Result<()> {
    let text = std::fs::read_to_string(path)?;
    let root: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Parse(format!("forced splits file {path}: {e}")))?;
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(node) = queue.pop_front() {
        let feature = node
            .get("feature")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        if feature > i64::from(max_feature_idx) {
            return Err(Error::InvalidParameter(format!(
                "Forced splits file includes feature index {feature}, but maximum feature index in dataset is {max_feature_idx}"
            )));
        }
        for side in ["left", "right"] {
            if let Some(child) = node.get(side) {
                queue.push_back(child.clone());
            }
        }
    }
    Ok(())
}

/// A booster that trains one iteration at a time (upstream `Booster::UpdateOneIter`).
/// It owns its configuration and training data, so it can outlive the caller's references.
pub struct GbdtTrainer {
    trainer: Trainer,
}

impl GbdtTrainer {
    /// upstream: `Booster` construction (`GBDT::Init` for the configured boosting type).
    pub fn new(cfg: Arc<Config>, ds: Arc<Dataset>) -> Result<Self> {
        if !matches!(cfg.boosting.as_str(), "gbdt" | "rf" | "dart") {
            return Err(Error::Unsupported(format!("boosting={}", cfg.boosting)));
        }
        if cfg.boosting == "rf" && cfg.objective == "custom" {
            return Err(Error::InvalidParameter(
                "RF mode do not support custom objective function, please use built-in objectives."
                    .into(),
            ));
        }
        if !cfg.forcedsplits_filename.is_empty() {
            let max_feature_idx = ds.layout.num_total_features as i32 - 1;
            check_forced_split_features(&cfg.forcedsplits_filename, max_feature_idx)?;
        }
        let objective = create_objective(&cfg, &ds.metadata)?;
        Ok(GbdtTrainer {
            trainer: Trainer::new(cfg, ds, objective)?,
        })
    }

    /// Continue training after `init`'s trees (upstream `GBDT::MergeFrom`): the initial trees
    /// come first and new trees follow. The dataset's `init_score` must hold `init`'s raw
    /// predictions, as upstream's Python layer sets them; the trainer starts from those scores.
    pub fn with_init_model(cfg: Arc<Config>, ds: Arc<Dataset>, init: &Gbdt) -> Result<Self> {
        if ds.metadata.init_score.is_empty() {
            return Err(Error::InvalidParameter(
                "continued training needs the init model's predictions as init_score".into(),
            ));
        }
        let mut trainer = Self::new(cfg, ds)?;
        let ntpi = trainer.trainer.num_tree_per_iteration;
        if !init.models.len().is_multiple_of(ntpi) {
            return Err(Error::InvalidParameter(
                "init_model tree count does not match the number of classes".into(),
            ));
        }
        trainer.trainer.models = init.models.clone();
        let init_iterations = init.models.len() / ntpi;
        if let Some(state) = trainer.trainer.dart.as_mut() {
            state.init_iterations = init_iterations;
        }
        Ok(trainer)
    }

    /// Refit `init`'s trees to the training data (upstream `GBDT::RefitTree`): every tree keeps
    /// its structure, and its leaf outputs are re-fitted on `ds` from the rows that
    /// `leaf_pred` routes to each leaf. `leaf_pred` is row-major, `ncol` columns (one per tree).
    /// Unlike continued training, the scores start from the dataset's `init_score` only.
    pub fn refit(
        cfg: Arc<Config>,
        ds: Arc<Dataset>,
        init: &Gbdt,
        leaf_pred: &[i32],
        ncol: usize,
    ) -> Result<Self> {
        if matches!(cfg.boosting.as_str(), "rf" | "dart") {
            return Err(Error::Unsupported(format!(
                "refit with boosting={}",
                cfg.boosting
            )));
        }
        let mut trainer = Self::new(cfg, ds)?;
        let ntpi = trainer.trainer.num_tree_per_iteration;
        if !init.models.len().is_multiple_of(ntpi) {
            return Err(Error::InvalidParameter(
                "refit tree count does not match the number of classes".into(),
            ));
        }
        trainer.trainer.models = init.models.clone();
        trainer.trainer.refit_trees(leaf_pred, ncol)?;
        Ok(trainer)
    }

    /// Shuffle the trainer's iterations in `[start, end)` (upstream `GBDT::ShuffleModels`).
    pub fn shuffle_models(&mut self, start: i32, end: i32) {
        self.trainer.shuffle_models(start, end);
    }

    /// Edit one leaf output of the trainer's trees (upstream `GBDT::SetLeafValue`).
    pub fn set_leaf_value(&mut self, tree_idx: usize, leaf_idx: usize, value: f64) -> Result<()> {
        super::gbdt::set_leaf_in(&mut self.trainer.models, tree_idx, leaf_idx, value)
    }

    /// Remove the last iteration's trees and their contribution to the training scores
    /// (upstream `Booster::RollbackOneIter`).
    pub fn rollback_one_iter(&mut self) -> Result<()> {
        self.trainer.rollback_one_iter()
    }

    /// One boosting iteration. Returns `true` when no tree could split any further.
    pub fn update_one_iter(&mut self) -> bool {
        self.trainer.train_one_iter()
    }

    /// Switch to external derivatives, preserving trees and scores.
    /// upstream: src/boosting/gbdt.h (v4.7.0) ResetConfig.
    pub fn reset_custom_objective(&mut self) -> Result<()> {
        let t = &mut self.trainer;
        if t.rf {
            return Err(Error::InvalidParameter(
                "RF mode do not support custom objective function, please use built-in objectives."
                    .into(),
            ));
        }
        let mut cfg = (*t.cfg).clone();
        cfg.objective = "custom".into();
        t.objective = create_objective(&cfg, &t.ds.metadata)?;
        t.cfg = Arc::new(cfg);
        t.class_need_train.fill(true);
        t.learner.reset_constant_hessian(false);
        Ok(())
    }

    /// External derivatives in class-major order, as in upstream GBDT::TrainOneIter.
    pub fn update_custom(&mut self, gradients: &[f32], hessians: &[f32]) -> Result<bool> {
        let t = &mut self.trainer;
        if t.cfg.objective != "custom" {
            return Err(Error::InvalidParameter(
                "custom updates require objective=custom".into(),
            ));
        }
        if gradients.len() != t.gradients.len() || hessians.len() != t.hessians.len() {
            return Err(Error::InvalidParameter(
                "gradient and Hessian lengths must equal rows times classes".into(),
            ));
        }
        t.gradients.copy_from_slice(gradients);
        t.hessians.copy_from_slice(hessians);
        Ok(t.train_one_iter_with(true))
    }

    /// The raw training score, class-major (`num_class * num_data`), including the init
    /// score, as a custom objective receives it (upstream `LGBM_BoosterGetPredict(0)`). With
    /// DART this applies the current iteration's drop first, once; [`Self::update_custom`]
    /// then trains on that dropped score.
    pub fn training_score(&mut self) -> &[f64] {
        self.trainer.training_score()
    }

    /// Completed iterations.
    pub fn iteration(&self) -> usize {
        self.trainer.models.len() / self.trainer.num_tree_per_iteration.max(1)
    }

    /// The model trained so far (a copy; training can continue afterwards).
    pub fn snapshot(&self) -> Gbdt {
        self.trainer.snapshot()
    }

    /// Consume the trainer and return its model without copying the trained trees.
    pub fn into_model(self) -> Gbdt {
        self.trainer.into_model()
    }

    /// Apply new parameters to a trainer that may already have trees (upstream:
    /// `Booster::ResetConfig` through `GBDT::ResetConfig`). Dataset-binning parameters are
    /// checked against the constructed Dataset first. The boosting type, objective and number of
    /// classes cannot change once training has started.
    pub fn reset_parameter(&mut self, params: &HashMap<String, String>) -> Result<()> {
        check_dataset_reset_config(&self.trainer.cfg, params)?;
        let cfg = Config::from_params(params)?;
        self.trainer.reset_config(cfg)
    }
}

/// Parameters the objective is built from (upstream: `ObjectiveFunction::Init` reads them).
const OBJECTIVE_PARAMETERS: &[&str] = &[
    "objective_seed",
    "sigmoid",
    "alpha",
    "fair_c",
    "poisson_max_delta_step",
    "tweedie_variance_power",
    "lambdarank_truncation_level",
    "lambdarank_norm",
    "label_gain",
    "boost_from_average",
    "reg_sqrt",
    "scale_pos_weight",
    "is_unbalance",
    "max_position",
    "lambdarank_position_bias_regularization",
    "multiclass_softmax",
];

/// Parameters that only affect the training loop's bookkeeping, not the sample strategy or the
/// tree learner.
const NON_TRAINING_PARAMETERS: &[&str] = &[
    "learning_rate",
    "metric",
    "metric_freq",
    "is_provide_training_metric",
    "eval_at",
    "verbose",
    "num_iterations",
    "early_stopping_round",
    "first_metric_only",
    "output_model",
    "input_model",
    "snapshot_freq",
];

/// Names of the parameters whose `[name: value]` line differs between two configs.
fn changed_parameters(old: &Config, new: &Config) -> Vec<String> {
    let parse = |cfg: &Config| -> std::collections::BTreeMap<String, String> {
        cfg.to_string()
            .lines()
            .filter_map(|line| {
                let inner = line.strip_prefix('[')?.strip_suffix(']')?;
                let (name, value) = inner.split_once(": ")?;
                Some((name.to_string(), value.to_string()))
            })
            .collect()
    };
    let (old, new) = (parse(old), parse(new));
    let mut names: Vec<String> = Vec::new();
    for (name, value) in &new {
        if old.get(name) != Some(value) {
            names.push(name.clone());
        }
    }
    names
}

struct Trainer {
    cfg: Arc<Config>,
    ds: Arc<Dataset>,
    sampler: SampleStrategy,
    /// upstream `iter_`: completed iterations, used by the sampling schedules.
    iter: i32,
    objective: Box<dyn Objective>,
    learner: SerialTreeLearner,
    num_tree_per_iteration: usize,
    class_need_train: Vec<bool>,
    score: Vec<f64>,
    gradients: Vec<f32>,
    hessians: Vec<f32>,
    models: Vec<Tree>,
    /// Random forest (`boosting=rf`).
    rf: bool,
    /// upstream `init_scores_` of RF: constant residual base per class.
    rf_init: Vec<f64>,
    /// The RF residual base broadcast over rows (class-major).
    rf_pred: Vec<f64>,
    /// DART state (`boosting=dart`).
    dart: Option<DartState>,
    /// Shrinkage of the tree being trained (upstream `shrinkage_rate_`; DART changes it).
    shrinkage: f64,
    /// User configured learning rate, kept separate from DART's temporary shrinkage.
    model_learning_rate: f64,
    /// Splits of every model tree, for replays on all rows (DART only).
    tree_splits: Vec<Vec<SplitRecord>>,
}

/// upstream: `DART` members.
struct DartState {
    tree_weight: Vec<f64>,
    sum_weight: f64,
    drop_index: Vec<usize>,
    rng: Random,
    /// Whether this iteration's trees were already dropped (upstream
    /// `is_update_score_cur_iter_`, which upstream leaves uninitialized; `false` here).
    dropped_this_iter: bool,
    /// Iterations of the init model (upstream `num_init_iteration_`). They come first in the
    /// model list and are never dropped; `tree_weight` covers only the trees trained here.
    init_iterations: usize,
}

/// upstream `std::min(a, b)`: `b` only when strictly smaller.
fn min_upstream(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

impl Trainer {
    fn new(cfg: Arc<Config>, ds: Arc<Dataset>, objective: Box<dyn Objective>) -> Result<Self> {
        let n = ds.num_data();
        let ntpi = objective.num_model_per_iteration();
        let sampler = SampleStrategy::new(&cfg, &ds.metadata, ntpi)?;
        let trainer_boosting = cfg.boosting.clone();
        let (drop_seed, learning_rate) = (cfg.drop_seed, cfg.learning_rate);
        // upstream: `GetIsConstHessian`. GOSS rescales hessians, so they are never constant.
        let constant_hessian = objective.is_constant_hessian() && !sampler.changes_hessians();
        let learner = SerialTreeLearner::new(Arc::clone(&cfg), Arc::clone(&ds), constant_hessian)?;
        let class_need_train = (0..ntpi).map(|k| objective.class_need_train(k)).collect();
        let rf = cfg.boosting == "rf";
        // upstream: ScoreUpdater ctor starts from the init score (class-major).
        let init_scores = &ds.metadata.init_score;
        let mut score = vec![0.0; n * ntpi];
        if !init_scores.is_empty() {
            if init_scores.len() != n * ntpi {
                return Err(Error::InvalidParameter(
                    "Number of class for initial score error".into(),
                ));
            }
            if rf {
                // upstream: src/boosting/rf.hpp (v4.7.0) RF::Init: CHECK_EQ(init_score, nullptr).
                return Err(Error::InvalidParameter(
                    "random forest does not support init_score".into(),
                ));
            }
            score.copy_from_slice(init_scores);
        }
        let mut trainer = Trainer {
            cfg,
            ds,
            sampler,
            iter: 0,
            objective,
            learner,
            num_tree_per_iteration: ntpi,
            class_need_train,
            score,
            gradients: vec![0.0; n * ntpi],
            hessians: vec![0.0; n * ntpi],
            models: Vec::new(),
            rf,
            rf_init: Vec::new(),
            rf_pred: Vec::new(),
            dart: (trainer_boosting == "dart").then(|| DartState {
                tree_weight: Vec::new(),
                sum_weight: 0.0,
                drop_index: Vec::new(),
                rng: Random::new(drop_seed),
                dropped_this_iter: false,
                init_iterations: 0,
            }),
            shrinkage: learning_rate,
            model_learning_rate: learning_rate,
            tree_splits: Vec::new(),
        };
        if rf {
            // upstream: RF::Init. Boosting() runs once: gradients come from the initial
            // scores and are never recomputed.
            let bagging = trainer.cfg.data_sample_strategy == "bagging";
            let bag_ok = trainer.cfg.bagging_freq > 0
                && trainer.cfg.bagging_fraction < 1.0
                && trainer.cfg.bagging_fraction > 0.0;
            let ff_ok = trainer.cfg.feature_fraction < 1.0 && trainer.cfg.feature_fraction > 0.0;
            if bagging && !(bag_ok || ff_ok) {
                return Err(Error::InvalidParameter(
                    "random forest needs bagging (bagging_freq > 0, 0 < bagging_fraction < 1) or feature_fraction < 1".into(),
                ));
            }
            if !bagging && trainer.cfg.data_sample_strategy != "goss" {
                return Err(Error::InvalidParameter(
                    "random forest needs data_sample_strategy bagging or goss".into(),
                ));
            }
            trainer.rf_init = (0..ntpi).map(|k| trainer.automatic_init(k)).collect();
            trainer.rf_pred = (0..ntpi)
                .flat_map(|k| std::iter::repeat_n(trainer.rf_init[k], n))
                .collect();
            trainer.objective.get_gradients(
                &trainer.rf_pred,
                &mut trainer.gradients,
                &mut trainer.hessians,
            );
        }
        Ok(trainer)
    }

    /// upstream: the value `GBDT::BoostFromAverage` returns (0 when not boosting from the
    /// average). Does not change the scores.
    fn automatic_init(&self, class_id: usize) -> f64 {
        // upstream: GBDT::BoostFromAverage skips the automatic score when the training
        // score updater has an init score.
        if self.models.is_empty()
            && self.ds.metadata.init_score.is_empty()
            && (self.cfg.boost_from_average || self.ds.layout.num_features == 0)
        {
            let init = self.objective.boost_from_score(class_id);
            if init.abs() > f64::from(K_EPSILON) {
                return init;
            }
        }
        0.0
    }

    /// upstream: `GBDT::BoostFromAverage` (with `update_scorer = true`)
    fn boost_from_average(&mut self, class_id: usize) -> f64 {
        let init = self.automatic_init(class_id);
        if init != 0.0 {
            let n = self.ds.num_data();
            for s in &mut self.score[class_id * n..(class_id + 1) * n] {
                *s += init;
            }
        }
        init
    }

    /// upstream: `RF::TrainOneIter`. The gradients stay fixed; the score is the running
    /// mean of the trees (`score = score * iter + tree`, then `* 1 / (iter + 1)`).
    fn train_one_iter_rf(&mut self) -> bool {
        let ntpi = self.num_tree_per_iteration;
        let n = self.ds.num_data();
        let iter = self.iter;
        self.iter += 1;
        let rows: Option<Vec<i32>> = self
            .sampler
            .sample(iter, &mut self.gradients, &mut self.hessians)
            .map(<[i32]>::to_vec);
        let oob = out_of_bag(rows.as_deref(), n);
        for k in 0..ntpi {
            let off = k * n;
            let init = self.rf_init[k];
            let mut tree = if self.class_need_train[k] && self.ds.layout.num_features > 0 {
                self.train_one_tree(off, rows.as_deref())
            } else {
                Tree::new_training(2)
            };
            if tree.num_leaves > 1 {
                self.renew_tree_output(&mut tree, &self.rf_pred[off..off + n]);
                if init.abs() > f64::from(K_EPSILON) {
                    tree.add_bias(init);
                }
                let score = &mut self.score[off..off + n];
                scale(score, iter as f64);
                self.learner.add_prediction_to_score(&tree, score);
                if let Some(oob) = &oob {
                    self.learner.add_out_of_bag_prediction(&tree, oob, score);
                }
                scale(score, 1.0 / f64::from(iter + 1));
            } else if self.models.len() < ntpi {
                let output = if self.class_need_train[k] {
                    0.0
                } else {
                    self.objective.boost_from_score(k)
                };
                tree.as_constant_tree(output, n as i32);
                // upstream: the learner adds nothing for a one-leaf tree; only the
                // out-of-bag rows receive the constant.
                let score = &mut self.score[off..off + n];
                scale(score, iter as f64);
                if let Some(oob) = &oob {
                    for &r in oob {
                        score[r as usize] += output;
                    }
                }
                scale(score, 1.0 / f64::from(iter + 1));
            }
            tree.finish_training();
            self.models.push(tree);
        }
        false
    }

    /// upstream: `GBDT::TrainOneIter` (built-in objective). Returns `true` when finished.
    /// The training score (class-major) that a custom objective is given. For DART this
    /// first drops this iteration's trees, once per iteration
    /// (upstream: src/boosting/dart.hpp (v4.7.0) `DART::GetTrainingScore`).
    fn training_score(&mut self) -> &[f64] {
        if self.dart.as_ref().is_some_and(|s| !s.dropped_this_iter) {
            self.dart_drop();
            if let Some(state) = self.dart.as_mut() {
                state.dropped_this_iter = true;
            }
        }
        &self.score
    }

    /// One iteration with the built-in objective (upstream `GBDT::TrainOneIter(nullptr, ..)`).
    fn train_one_iter(&mut self) -> bool {
        self.train_one_iter_with(false)
    }

    /// One iteration; `external` means the gradients were supplied by the caller.
    fn train_one_iter_with(&mut self, external: bool) -> bool {
        if self.rf {
            return self.train_one_iter_rf();
        }
        let ntpi = self.num_tree_per_iteration;
        let n = self.ds.num_data();
        let mut init_scores = vec![0.0; ntpi];
        for (k, init) in init_scores.iter_mut().enumerate() {
            *init = self.boost_from_average(k);
        }
        // upstream: `DART::TrainOneIter` clears `is_update_score_cur_iter_`, then built-in
        // objectives drop inside `Boosting()` (via `GetTrainingScore`). External gradients were
        // computed on a score whose drop already happened in `training_score`; when the caller
        // never fetched it, drop now so the iteration still applies DART.
        if let Some(state) = self.dart.as_mut() {
            let already_dropped = state.dropped_this_iter;
            state.dropped_this_iter = false;
            if !external || !already_dropped {
                self.dart_drop();
            }
        }
        self.objective
            .get_gradients(&self.score, &mut self.gradients, &mut self.hessians);
        let iter = self.iter;
        self.iter += 1;
        // upstream: `Bagging`/GOSS run after gradients; GOSS may rescale them in place.
        let rows: Option<Vec<i32>> = self
            .sampler
            .sample(iter, &mut self.gradients, &mut self.hessians)
            .map(<[i32]>::to_vec);
        let oob = out_of_bag(rows.as_deref(), n);

        let mut should_continue = false;
        for k in 0..ntpi {
            let off = k * n;
            let mut tree = if self.class_need_train[k] && self.ds.layout.num_features > 0 {
                self.train_one_tree(off, rows.as_deref())
            } else {
                Tree::new_training(2)
            };
            if tree.num_leaves > 1 {
                should_continue = true;
                self.renew_tree_output(&mut tree, &self.score[off..off + n]);
                tree.apply_shrinkage(self.shrinkage);
                self.learner
                    .add_prediction_to_score(&tree, &mut self.score[off..off + n]);
                if let Some(oob) = &oob {
                    self.learner.add_out_of_bag_prediction(
                        &tree,
                        oob,
                        &mut self.score[off..off + n],
                    );
                }
                if init_scores[k].abs() > f64::from(K_EPSILON) {
                    tree.add_bias(init_scores[k]);
                }
            } else if self.models.len() < ntpi {
                if !self.cfg.boost_from_average {
                    init_scores[k] = self.objective.boost_from_score(k);
                    for s in &mut self.score[off..off + n] {
                        *s += init_scores[k];
                    }
                }
                tree.as_constant_tree(init_scores[k], n as i32);
            } else {
                tree.as_constant_tree(0.0, n as i32);
            }
            tree.finish_training();
            // Kept for every tree: DART replays them, and rollback subtracts them.
            let splits = if tree.num_leaves > 1 {
                self.learner.last_splits()
            } else {
                Vec::new()
            };
            self.tree_splits.push(splits);
            self.models.push(tree);
        }
        if !should_continue {
            log::warn!(
                "Stopped training because there are no more leaves that meet the split requirements"
            );
            if self.models.len() > ntpi {
                for _ in 0..ntpi {
                    self.models.pop();
                    self.tree_splits.pop();
                }
            }
            return true;
        }
        if self.dart.is_some() {
            self.dart_normalize();
            if let Some(state) = self.dart.as_mut()
                && !self.cfg.uniform_drop
            {
                state.tree_weight.push(self.shrinkage);
                state.sum_weight += self.shrinkage;
            }
        }
        false
    }

    /// upstream: `GBDT::RefitTree` with `SerialTreeLearner::FitByExistingTree`. Gradients are
    /// computed once per iteration on the current scores; each class's tree is then refitted
    /// and its new output added to the scores.
    fn refit_trees(&mut self, leaf_pred: &[i32], ncol: usize) -> Result<()> {
        let n = self.ds.num_data();
        let ntpi = self.num_tree_per_iteration;
        if ncol != self.models.len() || leaf_pred.len() != n * ncol {
            return Err(Error::InvalidParameter(format!(
                "refit needs {n} x {} leaf predictions, got {} values for {} trees",
                self.models.len(),
                leaf_pred.len(),
                self.models.len()
            )));
        }
        let (l1, l2) = (self.cfg.lambda_l1, self.cfg.lambda_l2);
        let (max_delta_step, path_smooth) = (self.cfg.max_delta_step, self.cfg.path_smooth);
        let decay = self.cfg.refit_decay_rate;
        let epsilon = f64::from(K_EPSILON);
        for iter in 0..self.models.len() / ntpi.max(1) {
            self.objective
                .get_gradients(&self.score, &mut self.gradients, &mut self.hessians);
            for k in 0..ntpi {
                let model_index = iter * ntpi + k;
                let off = k * n;
                let num_leaves = self.models[model_index].num_leaves;
                // upstream: `ResetByLeafPred` routes rows to leaves in ascending row order.
                let mut leaf_of = Vec::with_capacity(n);
                for row in 0..n {
                    let raw = leaf_pred[row * ncol + model_index];
                    let leaf = usize::try_from(raw).ok().filter(|&l| l < num_leaves);
                    let Some(leaf) = leaf else {
                        return Err(Error::InvalidParameter(format!(
                            "refit leaf index {raw} is outside tree {model_index} with {num_leaves} leaves"
                        )));
                    };
                    leaf_of.push(leaf);
                }
                let mut sum_grad = vec![0.0_f64; num_leaves];
                let mut sum_hess = vec![epsilon; num_leaves];
                let mut count = vec![0_i32; num_leaves];
                for (row, &leaf) in leaf_of.iter().enumerate() {
                    sum_grad[leaf] += f64::from(self.gradients[off + row]);
                    sum_hess[leaf] += f64::from(self.hessians[off + row]);
                    count[leaf] += 1;
                }
                let tree = &mut self.models[model_index];
                for leaf in 0..num_leaves {
                    // upstream: `FitByExistingTree` passes `tree->leaf_parent(i)`, a node index,
                    // as the parent output of the smoothing; it is ported as written.
                    let output = if path_smooth > epsilon && leaf > 0 {
                        calculate_splitted_leaf_output::<true, true, true>(
                            sum_grad[leaf],
                            sum_hess[leaf],
                            l1,
                            l2,
                            max_delta_step,
                            path_smooth,
                            count[leaf],
                            f64::from(tree.leaf_parent[leaf]),
                        )
                    } else {
                        calculate_splitted_leaf_output::<true, true, false>(
                            sum_grad[leaf],
                            sum_hess[leaf],
                            l1,
                            l2,
                            max_delta_step,
                            path_smooth,
                            count[leaf],
                            0.0,
                        )
                    };
                    let new_output = output * tree.shrinkage;
                    let old_output = tree.leaf_value[leaf];
                    tree.set_leaf_output(leaf, decay * old_output + (1.0 - decay) * new_output);
                }
                // upstream: `ScoreUpdater::AddScore` -> `AddPredictionToScore`, which skips
                // single-leaf trees.
                if num_leaves > 1 {
                    let tree = &self.models[model_index];
                    for (row, &leaf) in leaf_of.iter().enumerate() {
                        self.score[off + row] += tree.leaf_value[leaf];
                    }
                }
            }
        }
        Ok(())
    }

    /// Adds model `idx` (class `k`) to the training score for all rows, through its
    /// recorded splits (upstream: `ScoreUpdater::AddScore(tree, cur_tree_id)`).
    /// Grows one tree on class `k`'s gradients (upstream: `tree_learner_->Train`), then fits
    /// linear leaves when `linear_tree` is set (upstream: `is_first_tree` is the first
    /// `num_tree_per_iteration` trees).
    fn train_one_tree(&mut self, off: usize, rows: Option<&[i32]>) -> Tree {
        let n = self.ds.num_data();
        let is_first_tree = self.models.len() < self.num_tree_per_iteration;
        let mut tree = self.learner.train(
            &self.gradients[off..off + n],
            &self.hessians[off..off + n],
            rows,
        );
        if self.cfg.linear_tree {
            self.learner.fit_linear(
                &mut tree,
                &self.gradients[off..off + n],
                &self.hessians[off..off + n],
                is_first_tree,
            );
        }
        tree
    }

    fn add_model_to_train_score(&mut self, idx: usize, k: usize) {
        let n = self.ds.num_data();
        let rows: Vec<i32> = (0..n as i32).collect();
        // Split records cover only the trees trained here, after the init model's trees.
        let init_trees = self.dart.as_ref().map_or(0, |state| {
            state.init_iterations * self.num_tree_per_iteration
        });
        self.learner.add_prediction_for_rows(
            &self.models[idx],
            &self.tree_splits[idx - init_trees],
            &rows,
            &mut self.score[k * n..(k + 1) * n],
        );
    }

    /// upstream: `GBDT::ResetConfig`. The new settings apply from the next iteration; models and
    /// scores are kept. Only the parts a changed parameter touches are rebuilt: the objective when
    /// an objective parameter changes, and the sample strategy and tree learner when a training
    /// parameter changes. Rebuilding reseeds their random streams, so a learning-rate schedule
    /// (which changes neither) leaves them alone, as upstream does.
    fn reset_config(&mut self, cfg: Config) -> Result<()> {
        if cfg.boosting != self.cfg.boosting {
            return Err(Error::Unsupported(
                "reset_parameter for boosting after training started".into(),
            ));
        }
        if cfg.objective != self.cfg.objective || cfg.num_class != self.cfg.num_class {
            return Err(Error::Unsupported(
                "reset_parameter for objective or num_class after training started".into(),
            ));
        }
        let changed = changed_parameters(&self.cfg, &cfg);
        let ntpi = self.num_tree_per_iteration;
        let cfg = Arc::new(cfg);
        if changed
            .iter()
            .any(|name| OBJECTIVE_PARAMETERS.contains(&name.as_str()))
        {
            self.objective = create_objective(&cfg, &self.ds.metadata)?;
            self.class_need_train = (0..ntpi)
                .map(|k| self.objective.class_need_train(k))
                .collect();
        }
        if changed
            .iter()
            .any(|name| !NON_TRAINING_PARAMETERS.contains(&name.as_str()))
        {
            let sampler = SampleStrategy::new(&cfg, &self.ds.metadata, ntpi)?;
            let constant_hessian =
                self.objective.is_constant_hessian() && !sampler.changes_hessians();
            self.learner =
                SerialTreeLearner::new(Arc::clone(&cfg), Arc::clone(&self.ds), constant_hessian)?;
            self.sampler = sampler;
        }
        if changed.iter().any(|name| name == "learning_rate") {
            self.shrinkage = cfg.learning_rate;
            self.model_learning_rate = cfg.learning_rate;
        }
        self.cfg = cfg;
        Ok(())
    }

    /// Shuffle iterations in `[start, end)` (upstream `GBDT::ShuffleModels`). Split records
    /// move with their trees so rollback and DART keep matching them.
    fn shuffle_models(&mut self, start: i32, end: i32) {
        let ntpi = self.num_tree_per_iteration.max(1);
        // Split records exist only for trees this trainer built; keep them aligned if so.
        let aligned = self.tree_splits.len() == self.models.len();
        let order = super::gbdt::shuffle_order(self.models.len() / ntpi, start, end);
        super::gbdt::permute_iterations(&mut self.models, ntpi, &order);
        if aligned {
            super::gbdt::permute_iterations(&mut self.tree_splits, ntpi, &order);
        }
    }

    /// upstream: `GBDT::RollbackOneIter`. Subtracts the last iteration's trees (leaf values
    /// negated, as upstream's `Shrinkage(-1.0)`) from the training scores and drops them.
    fn rollback_one_iter(&mut self) -> Result<()> {
        if self.rf || self.dart.is_some() {
            return Err(Error::Unsupported(format!(
                "rollback_one_iter with boosting={}",
                self.cfg.boosting
            )));
        }
        let ntpi = self.num_tree_per_iteration;
        if self.iter <= 0 || self.models.len() < ntpi {
            return Ok(());
        }
        if self.tree_splits.len() != self.models.len() {
            return Err(Error::Unsupported(
                "rollback_one_iter of trees loaded from init_model".into(),
            ));
        }
        let n = self.ds.num_data();
        let rows: Vec<i32> = (0..n as i32).collect();
        let first = self.models.len() - ntpi;
        for k in 0..ntpi {
            let idx = first + k;
            let mut negated = self.models[idx].clone();
            for value in &mut negated.leaf_value {
                *value = -*value;
            }
            self.learner.add_prediction_for_rows(
                &negated,
                &self.tree_splits[idx],
                &rows,
                &mut self.score[k * n..(k + 1) * n],
            );
        }
        self.models.truncate(first);
        self.tree_splits.truncate(first);
        self.iter -= 1;
        Ok(())
    }

    /// upstream: `DART::DroppingTrees`. Chooses the trees to drop, removes them from the
    /// training score and sets the shrinkage of the new tree.
    fn dart_drop(&mut self) {
        let ntpi = self.num_tree_per_iteration;
        let iter = self.iter.max(0) as usize;
        let cfg = Arc::clone(&self.cfg);
        let Some(state) = self.dart.as_mut() else {
            return;
        };
        // upstream: `drop_index_.push_back(num_init_iteration_ + i)`; `tree_weight` stays relative.
        let init_iterations = state.init_iterations;
        let mut drop_index: Vec<usize> = Vec::new();
        let is_skip = f64::from(state.rng.next_float()) < cfg.skip_drop;
        if !is_skip {
            let mut drop_rate = cfg.drop_rate;
            // upstream: `static_cast<size_t>(config_->max_drop)` (a negative value wraps)
            let max_drop_size = cfg.max_drop as i64 as u64;
            if !cfg.uniform_drop {
                let inv_average_weight = state.tree_weight.len() as f64 / state.sum_weight;
                if cfg.max_drop > 0 {
                    drop_rate = min_upstream(
                        drop_rate,
                        f64::from(cfg.max_drop) * inv_average_weight / state.sum_weight,
                    );
                }
                for i in 0..iter {
                    let w = state.tree_weight.get(i).copied().unwrap_or(0.0);
                    if f64::from(state.rng.next_float()) < drop_rate * w * inv_average_weight {
                        drop_index.push(init_iterations + i);
                        if drop_index.len() as u64 >= max_drop_size {
                            break;
                        }
                    }
                }
            } else {
                if cfg.max_drop > 0 {
                    drop_rate = min_upstream(drop_rate, f64::from(cfg.max_drop) / iter as f64);
                }
                for i in 0..iter {
                    if f64::from(state.rng.next_float()) < drop_rate {
                        drop_index.push(init_iterations + i);
                        if drop_index.len() as u64 >= max_drop_size {
                            break;
                        }
                    }
                }
            }
        }
        for &i in &drop_index {
            for k in 0..ntpi {
                let idx = i * ntpi + k;
                self.models[idx].apply_shrinkage(-1.0);
                self.add_model_to_train_score(idx, k);
            }
        }
        let num_drop = drop_index.len() as f64;
        self.shrinkage = if !cfg.xgboost_dart_mode {
            cfg.learning_rate / (1.0 + num_drop)
        } else if drop_index.is_empty() {
            cfg.learning_rate
        } else {
            cfg.learning_rate / (cfg.learning_rate + num_drop)
        };
        if let Some(state) = self.dart.as_mut() {
            state.drop_index = drop_index;
        }
    }

    /// upstream: `DART::Normalize`. Restores the dropped trees, rescaled by the number of
    /// dropped trees, and updates their weights.
    fn dart_normalize(&mut self) {
        let ntpi = self.num_tree_per_iteration;
        let cfg = Arc::clone(&self.cfg);
        let drop_index = match self.dart.as_ref() {
            Some(state) => state.drop_index.clone(),
            None => return,
        };
        let k = drop_index.len() as f64;
        let lr = cfg.learning_rate;
        let init_iterations = self.dart.as_ref().map_or(0, |state| state.init_iterations);
        for &i in &drop_index {
            // upstream: `tree_weight_[i - num_init_iteration_]`
            let rel = i - init_iterations;
            for c in 0..ntpi {
                let idx = i * ntpi + c;
                if !cfg.xgboost_dart_mode {
                    self.models[idx].apply_shrinkage(1.0 / (k + 1.0));
                    self.models[idx].apply_shrinkage(-k);
                } else {
                    self.models[idx].apply_shrinkage(self.shrinkage);
                    self.models[idx].apply_shrinkage(-k / lr);
                }
                self.add_model_to_train_score(idx, c);
            }
            if !cfg.uniform_drop
                && let Some(state) = self.dart.as_mut()
            {
                let w = state.tree_weight.get(rel).copied().unwrap_or(0.0);
                if !cfg.xgboost_dart_mode {
                    state.sum_weight -= w * (1.0 / (k + 1.0));
                    if let Some(t) = state.tree_weight.get_mut(rel) {
                        *t *= k / (k + 1.0);
                    }
                } else {
                    state.sum_weight -= w * (1.0 / (k + lr));
                    if let Some(t) = state.tree_weight.get_mut(rel) {
                        *t *= k / (k + lr);
                    }
                }
            }
        }
    }

    /// upstream: `SerialTreeLearner::RenewTreeOutput` (no bagging, single machine)
    fn renew_tree_output(&self, tree: &mut Tree, score: &[f64]) {
        if !self.objective.is_renew_tree_output() {
            return;
        }
        for leaf in 0..tree.num_leaves {
            let indices = self.learner.partition.indices_on_leaf(leaf);
            if !indices.is_empty() {
                let out = self
                    .objective
                    .renew_tree_output(tree.leaf_value[leaf], score, indices);
                tree.set_leaf_output(leaf, out);
            }
        }
    }

    /// The model so far (copies the trees).
    fn snapshot(&self) -> Gbdt {
        self.build_model(self.models.clone())
    }

    fn into_model(self) -> Gbdt {
        let objective_str = self.objective.to_model_string();
        let loaded_parameter = self.model_parameters();
        let ds = self.ds;
        Gbdt {
            models: self.models,
            num_class: self.cfg.num_class as usize,
            num_tree_per_iteration: self.num_tree_per_iteration,
            label_idx: ds.label_idx,
            max_feature_idx: ds.layout.num_total_features as i32 - 1,
            average_output: self.rf,
            feature_names: ds.feature_names.clone(),
            feature_infos: feature_infos(&ds),
            monotone_constraints: self.cfg.monotone_constraints.clone(),
            objective: LoadedObjective::from_model_str(&objective_str)
                .ok()
                .flatten(),
            linear_tree: self.cfg.linear_tree,
            loaded_parameter,
            parser_config: String::new(),
            pandas_categorical: None,
            version: "v4".into(),
        }
    }

    fn build_model(&self, models: Vec<Tree>) -> Gbdt {
        let ds = &self.ds;
        let objective_str = self.objective.to_model_string();
        let loaded_parameter = self.model_parameters();
        Gbdt {
            models,
            num_class: self.cfg.num_class as usize,
            num_tree_per_iteration: self.num_tree_per_iteration,
            label_idx: ds.label_idx,
            max_feature_idx: ds.layout.num_total_features as i32 - 1,
            average_output: self.rf,
            feature_names: ds.feature_names.clone(),
            feature_infos: feature_infos(ds),
            monotone_constraints: self.cfg.monotone_constraints.clone(),
            objective: LoadedObjective::from_model_str(&objective_str)
                .ok()
                .flatten(),
            linear_tree: self.cfg.linear_tree,
            loaded_parameter,
            parser_config: String::new(),
            pandas_categorical: None,
            version: "v4".into(),
        }
    }

    fn model_parameters(&self) -> String {
        let mut parameters = self.cfg.to_string();
        if self.model_learning_rate != self.cfg.learning_rate {
            let original = format!("[learning_rate: {}]\n", format_g(self.cfg.learning_rate, 6));
            let updated = format!(
                "[learning_rate: {}]\n",
                format_g(self.model_learning_rate, 6)
            );
            parameters = parameters.replacen(&original, &updated, 1);
        }
        parameters
    }
}

/// Rows not in the bag, ascending (upstream: the out-of-bag tail of the bagging indices).
fn out_of_bag(rows: Option<&[i32]>, n: usize) -> Option<Vec<i32>> {
    rows.map(|bag| {
        let mut in_bag = vec![false; n];
        for &r in bag {
            in_bag[r as usize] = true;
        }
        (0..n).filter(|&i| !in_bag[i]).map(|i| i as i32).collect()
    })
}

/// Multiply a score array in place (upstream: `ScoreUpdater::MultiplyScore`).
fn scale(score: &mut [f64], val: f64) {
    for s in score {
        *s *= val;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::GbdtTrainer;
    use crate::config::Config;
    use crate::io::dataset::Dataset;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn row_wise_layout_trains_the_same_model_as_column_wise() {
        // upstream: force_row_wise builds histograms row by row; each bin accumulates its rows
        // in the same order as column-wise, so the models must be identical.
        let base = [
            ("objective", "regression"),
            ("num_leaves", "3"),
            ("min_data_in_leaf", "1"),
            ("min_data_in_bin", "1"),
            ("max_bin", "8"),
            ("boost_from_average", "false"),
            ("num_iterations", "3"),
        ];
        let features = (0..8).map(f64::from).collect::<Vec<_>>();
        let labels = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let train = |layout: &str| {
            let mut params = base
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect::<HashMap<_, _>>();
            params.insert(layout.to_owned(), "true".to_owned());
            let config = Arc::new(Config::from_params(&params).unwrap());
            let dataset =
                Arc::new(Dataset::from_dense(&features, 8, 1, &labels, &config, &[]).unwrap());
            super::Gbdt::train_shared(config, dataset)
                .unwrap()
                .save_model_to_string(0, -1, 0)
                .unwrap()
        };
        // Compare the trees only: the trailing parameters block echoes the layout flag itself.
        let trees = |model: String| {
            model
                .split("\nparameters:")
                .next()
                .unwrap_or_default()
                .to_owned()
        };
        assert_eq!(
            trees(train("force_row_wise")),
            trees(train("force_col_wise"))
        );
    }

    #[test]
    fn incremental_training_snapshots_each_completed_iteration() {
        let params = HashMap::from([
            ("objective".to_owned(), "regression".to_owned()),
            ("num_leaves".to_owned(), "3".to_owned()),
            ("min_data_in_leaf".to_owned(), "1".to_owned()),
            ("min_data_in_bin".to_owned(), "1".to_owned()),
            ("max_bin".to_owned(), "8".to_owned()),
            ("boost_from_average".to_owned(), "false".to_owned()),
            ("num_iterations".to_owned(), "2".to_owned()),
        ]);
        let config = Arc::new(Config::from_params(&params).unwrap());
        let features = (0..8).map(f64::from).collect::<Vec<_>>();
        let labels = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let dataset =
            Arc::new(Dataset::from_dense(&features, 8, 1, &labels, &config, &[]).unwrap());
        let mut trainer = GbdtTrainer::new(Arc::clone(&config), Arc::clone(&dataset)).unwrap();

        assert_eq!(trainer.iteration(), 0);
        assert!(!trainer.update_one_iter());
        assert_eq!(trainer.iteration(), 1);
        let first_snapshot = trainer.snapshot();
        assert_eq!(first_snapshot.current_iteration(), 1);

        assert!(!trainer.update_one_iter());
        assert_eq!(trainer.iteration(), 2);
        assert_eq!(first_snapshot.current_iteration(), 1);
        let incremental = trainer.snapshot();
        assert_eq!(incremental.current_iteration(), 2);
        let batch = super::Gbdt::train_shared(config, dataset).unwrap();
        assert_eq!(
            incremental.save_model_to_string(0, -1, 0).unwrap(),
            batch.save_model_to_string(0, -1, 0).unwrap()
        );
    }
}

#[cfg(test)]
mod forced_split_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lgbm-forced-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("forced.json");
        std::fs::write(&path, body).unwrap();
        path
    }

    // Red tests for `check_forced_split_features` (upstream: src/boosting/gbdt.cpp (v4.7.0)
    // `GBDT::CheckForcedSplitFeatures`), written before the function existed.
    #[test]
    fn forced_split_feature_beyond_the_dataset_is_rejected() {
        let path = write("bad", r#"{"feature": 0, "left": {"feature": 2}}"#);
        let err = check_forced_split_features(path.to_str().unwrap(), 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("Forced splits file includes feature index 2")
        );
    }

    #[test]
    fn valid_forced_splits_file_passes_the_check() {
        let path = write("ok", r#"{"feature": 0, "right": {"feature": 1}}"#);
        check_forced_split_features(path.to_str().unwrap(), 1).unwrap();
    }
}

#[cfg(test)]
mod refit_tests {
    #![allow(clippy::unwrap_used)]

    use super::GbdtTrainer;
    use crate::boosting::Gbdt;
    use crate::config::Config;
    use crate::io::dataset::Dataset;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// Trains two regression iterations on one feature and returns the model, the training
    /// dataset and the leaf index of every row in every tree (row-major, one column per tree).
    fn trained(extra: &[(&str, &str)]) -> (Arc<Config>, Arc<Dataset>, Gbdt, Vec<i32>) {
        let mut params = HashMap::from([
            ("objective".to_owned(), "regression".to_owned()),
            ("num_leaves".to_owned(), "4".to_owned()),
            ("min_data_in_leaf".to_owned(), "1".to_owned()),
            ("min_data_in_bin".to_owned(), "1".to_owned()),
            ("max_bin".to_owned(), "8".to_owned()),
            ("num_iterations".to_owned(), "2".to_owned()),
        ]);
        for (k, v) in extra {
            params.insert((*k).to_owned(), (*v).to_owned());
        }
        let config = Arc::new(Config::from_params(&params).unwrap());
        let features = (0..8).map(f64::from).collect::<Vec<_>>();
        let labels = vec![0.0, 1.0, 0.0, 2.0, 5.0, 4.0, 6.0, 5.0];
        let dataset =
            Arc::new(Dataset::from_dense(&features, 8, 1, &labels, &config, &[]).unwrap());
        let model = super::Gbdt::train_shared(Arc::clone(&config), Arc::clone(&dataset)).unwrap();
        let ncol = model.trees().len();
        let mut leaf_pred = Vec::with_capacity(8 * ncol);
        for row in &features {
            for tree in model.trees() {
                leaf_pred.push(tree.predict_leaf_index(&[*row]) as i32);
            }
        }
        (config, dataset, model, leaf_pred)
    }

    fn leaf_values(model: &Gbdt) -> Vec<f64> {
        let mut out = Vec::new();
        for (t, tree) in model.trees().iter().enumerate() {
            for l in 0..tree.num_leaves {
                out.push(model.leaf_value(t, l).unwrap());
            }
        }
        out
    }

    #[test]
    fn refit_with_decay_one_keeps_every_leaf_value() {
        let (config, dataset, model, leaf_pred) = trained(&[("refit_decay_rate", "1.0")]);
        let ncol = model.trees().len();
        let refit = GbdtTrainer::refit(config, dataset, &model, &leaf_pred, ncol)
            .unwrap()
            .snapshot();
        assert_eq!(leaf_values(&refit), leaf_values(&model));
    }

    #[test]
    fn refit_rejects_a_leaf_index_outside_the_tree() {
        let (config, dataset, model, mut leaf_pred) = trained(&[]);
        leaf_pred[0] = 99;
        let ncol = model.trees().len();
        let err = GbdtTrainer::refit(config, dataset, &model, &leaf_pred, ncol)
            .err()
            .unwrap();
        assert!(matches!(err, crate::Error::InvalidParameter(_)));
    }
}
