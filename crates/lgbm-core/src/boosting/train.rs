//! GBDT training loop. upstream: src/boosting/gbdt.cpp (`Init`, `TrainOneIter`,
//! `BoostFromAverage`, `UpdateScore`)

use super::Gbdt;
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
        if cfg.boosting != "gbdt" {
            return Err(Error::Unsupported(format!("boosting={}", cfg.boosting)));
        }
        if cfg.data_sample_strategy != "bagging"
            || (cfg.bagging_freq > 0 && cfg.bagging_fraction < 1.0)
        {
            return Err(Error::Unsupported("bagging / GOSS".into()));
        }
        if cfg.feature_fraction < 1.0 || cfg.feature_fraction_bynode < 1.0 {
            return Err(Error::Unsupported("feature_fraction < 1".into()));
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
    objective: Box<dyn Objective>,
    learner: SerialTreeLearner<'a>,
    num_tree_per_iteration: usize,
    class_need_train: Vec<bool>,
    score: Vec<f64>,
    gradients: Vec<f32>,
    hessians: Vec<f32>,
    models: Vec<Tree>,
}

impl<'a> Trainer<'a> {
    fn new(cfg: &'a Config, ds: &'a Dataset, objective: Box<dyn Objective>) -> Result<Self> {
        let n = ds.num_data();
        let ntpi = objective.num_model_per_iteration();
        let learner = SerialTreeLearner::new(cfg, ds, objective.is_constant_hessian())?;
        let class_need_train = (0..ntpi).map(|k| objective.class_need_train(k)).collect();
        Ok(Trainer {
            cfg,
            ds,
            objective,
            learner,
            num_tree_per_iteration: ntpi,
            class_need_train,
            score: vec![0.0; n * ntpi],
            gradients: vec![0.0; n * ntpi],
            hessians: vec![0.0; n * ntpi],
            models: Vec::new(),
        })
    }

    /// upstream: `GBDT::BoostFromAverage`
    fn boost_from_average(&mut self, class_id: usize) -> f64 {
        if self.models.is_empty()
            && (self.cfg.boost_from_average || self.ds.layout.num_features == 0)
        {
            let init = self.objective.boost_from_score(class_id);
            if init.abs() > f64::from(K_EPSILON) {
                let n = self.ds.num_data();
                for s in &mut self.score[class_id * n..(class_id + 1) * n] {
                    *s += init;
                }
                return init;
            }
        }
        0.0
    }

    /// upstream: `GBDT::TrainOneIter` (built-in objective). Returns `true` when finished.
    fn train_one_iter(&mut self) -> bool {
        let ntpi = self.num_tree_per_iteration;
        let n = self.ds.num_data();
        let mut init_scores = vec![0.0; ntpi];
        for (k, init) in init_scores.iter_mut().enumerate() {
            *init = self.boost_from_average(k);
        }
        self.objective
            .get_gradients(&self.score, &mut self.gradients, &mut self.hessians);

        let mut should_continue = false;
        for k in 0..ntpi {
            let off = k * n;
            let mut tree = if self.class_need_train[k] && self.ds.layout.num_features > 0 {
                self.learner
                    .train(&self.gradients[off..off + n], &self.hessians[off..off + n])
            } else {
                Tree::new_training(2)
            };
            if tree.num_leaves > 1 {
                should_continue = true;
                self.renew_tree_output(&mut tree, off..off + n);
                tree.apply_shrinkage(self.cfg.learning_rate);
                self.learner
                    .add_prediction_to_score(&tree, &mut self.score[off..off + n]);
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
    fn renew_tree_output(&self, tree: &mut Tree, class_range: std::ops::Range<usize>) {
        if !self.objective.is_renew_tree_output() {
            return;
        }
        let score = &self.score[class_range];
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
            average_output: false,
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
