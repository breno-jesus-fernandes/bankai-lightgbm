//! GBDT training loop. upstream: src/boosting/gbdt.cpp (`Init`, `TrainOneIter`,
//! `BoostFromAverage`, `UpdateScore`)

use super::Gbdt;
use super::sample::SampleStrategy;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::io::bin::BinType;
use crate::io::dataset::Dataset;
use crate::meta::K_EPSILON;
use crate::objective::LoadedObjective;
use crate::objective::train::{Objective, create_objective};
use crate::tree::Tree;
use crate::treelearner::serial::SerialTreeLearner;
use crate::utils::format::format_g;

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
    /// calls to `GBDT::TrainOneIter`).
    pub fn train(cfg: &Config, ds: &Dataset) -> Result<Gbdt> {
        if cfg.boosting != "gbdt" && cfg.boosting != "rf" {
            return Err(Error::Unsupported(format!("boosting={}", cfg.boosting)));
        }
        let objective = create_objective(cfg, &ds.metadata)?;
        let mut trainer = Trainer::new(cfg, ds, objective)?;
        for _ in 0..cfg.num_iterations {
            if trainer.train_one_iter() {
                break;
            }
        }
        Ok(trainer.into_model())
    }
}

struct Trainer<'a> {
    cfg: &'a Config,
    ds: &'a Dataset,
    sampler: SampleStrategy,
    /// upstream `iter_`: completed iterations, used by the sampling schedules.
    iter: i32,
    objective: Box<dyn Objective>,
    learner: SerialTreeLearner<'a>,
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
}

impl<'a> Trainer<'a> {
    fn new(cfg: &'a Config, ds: &'a Dataset, objective: Box<dyn Objective>) -> Result<Self> {
        let n = ds.num_data();
        let ntpi = objective.num_model_per_iteration();
        let sampler = SampleStrategy::new(cfg, &ds.metadata, ntpi)?;
        // upstream: `GetIsConstHessian`. GOSS rescales hessians, so they are never constant.
        let constant_hessian = objective.is_constant_hessian() && !sampler.changes_hessians();
        let learner = SerialTreeLearner::new(cfg, ds, constant_hessian)?;
        let class_need_train = (0..ntpi).map(|k| objective.class_need_train(k)).collect();
        let rf = cfg.boosting == "rf";
        let mut trainer = Trainer {
            cfg,
            ds,
            sampler,
            iter: 0,
            objective,
            learner,
            num_tree_per_iteration: ntpi,
            class_need_train,
            score: vec![0.0; n * ntpi],
            gradients: vec![0.0; n * ntpi],
            hessians: vec![0.0; n * ntpi],
            models: Vec::new(),
            rf,
            rf_init: Vec::new(),
            rf_pred: Vec::new(),
        };
        if rf {
            // upstream: RF::Init. Boosting() runs once: gradients come from the initial
            // scores and are never recomputed.
            let bagging = cfg.data_sample_strategy == "bagging";
            let bag_ok =
                cfg.bagging_freq > 0 && cfg.bagging_fraction < 1.0 && cfg.bagging_fraction > 0.0;
            let ff_ok = cfg.feature_fraction < 1.0 && cfg.feature_fraction > 0.0;
            if bagging && !(bag_ok || ff_ok) {
                return Err(Error::InvalidParameter(
                    "random forest needs bagging (bagging_freq > 0, 0 < bagging_fraction < 1) or feature_fraction < 1".into(),
                ));
            }
            if !bagging && cfg.data_sample_strategy != "goss" {
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
        if self.models.is_empty()
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
                self.learner.train(
                    &self.gradients[off..off + n],
                    &self.hessians[off..off + n],
                    rows.as_deref(),
                )
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
    fn train_one_iter(&mut self) -> bool {
        if self.rf {
            return self.train_one_iter_rf();
        }
        let ntpi = self.num_tree_per_iteration;
        let n = self.ds.num_data();
        let mut init_scores = vec![0.0; ntpi];
        for (k, init) in init_scores.iter_mut().enumerate() {
            *init = self.boost_from_average(k);
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
                self.learner.train(
                    &self.gradients[off..off + n],
                    &self.hessians[off..off + n],
                    rows.as_deref(),
                )
            } else {
                Tree::new_training(2)
            };
            if tree.num_leaves > 1 {
                should_continue = true;
                self.renew_tree_output(&mut tree, &self.score[off..off + n]);
                tree.apply_shrinkage(self.cfg.learning_rate);
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
            self.models.push(tree);
        }
        if !should_continue {
            log::warn!(
                "Stopped training because there are no more leaves that meet the split requirements"
            );
            if self.models.len() > ntpi {
                for _ in 0..ntpi {
                    self.models.pop();
                }
            }
            return true;
        }
        false
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

    fn into_model(self) -> Gbdt {
        let ds = self.ds;
        let objective_str = self.objective.to_model_string();
        Gbdt {
            models: self.models,
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
            loaded_parameter: self.cfg.to_string(),
            parser_config: String::new(),
            pandas_categorical: None,
            version: "v4".into(),
        }
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
