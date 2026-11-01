//! Gradient boosted decision trees. upstream: src/boosting/gbdt.{h,cpp}, gbdt_prediction.cpp

use crate::error::{Error, Result};
use crate::objective::LoadedObjective;
use crate::tree::Tree;
use crate::utils::random::Random;

/// A GBDT ensemble (also hosts DART/RF/GOSS models, which share the format).
#[derive(Debug, Clone)]
pub struct Gbdt {
    pub(crate) models: Vec<Tree>,
    pub(crate) num_class: usize,
    pub(crate) num_tree_per_iteration: usize,
    pub(crate) label_idx: i32,
    pub(crate) max_feature_idx: i32,
    pub(crate) average_output: bool,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_infos: Vec<String>,
    pub(crate) monotone_constraints: Vec<i8>,
    pub(crate) objective: Option<LoadedObjective>,
    pub(crate) linear_tree: bool,
    /// Raw `parameters:` block, kept verbatim for round-tripping.
    pub(crate) loaded_parameter: String,
    /// Raw `parser:` block (custom parser config), if present.
    pub(crate) parser_config: String,
    /// Trailing `pandas_categorical:` line value, if present.
    pub(crate) pandas_categorical: Option<String>,
    pub(crate) version: String,
}

/// upstream: `GBDT::SetLeafValue` (`models_[tree_idx]->SetLeafOutput`), indices checked.
pub(crate) fn set_leaf_in(
    models: &mut [Tree],
    tree_idx: usize,
    leaf_idx: usize,
    value: f64,
) -> Result<()> {
    let tree = models
        .get_mut(tree_idx)
        .ok_or_else(|| Error::InvalidParameter("tree index out of range".into()))?;
    if leaf_idx >= tree.num_leaves {
        return Err(Error::InvalidParameter("leaf index out of range".into()));
    }
    tree.leaf_value[leaf_idx] = value;
    Ok(())
}

/// upstream: `GBDT::ShuffleModels`: the iteration order over `[start, end)` drawn with
/// `Random(17)`; iterations outside the range keep their place.
pub(crate) fn shuffle_order(total_iter: usize, start: i32, end: i32) -> Vec<usize> {
    let start = start.max(0) as usize;
    let end = if end <= 0 {
        total_iter
    } else {
        (end as usize).min(total_iter)
    };
    let mut order: Vec<usize> = (0..total_iter).collect();
    let mut rng = Random::new(17);
    for i in start..end.saturating_sub(1) {
        let j = rng.next_short(i as i32 + 1, end as i32) as usize;
        order.swap(i, j);
    }
    order
}

/// Reorder `items` (blocks of `ntpi` per iteration) by `order`.
pub(crate) fn permute_iterations<T>(items: &mut Vec<T>, ntpi: usize, order: &[usize]) {
    let mut taken: Vec<Option<T>> = items.drain(..).map(Some).collect();
    for &iter in order {
        for j in 0..ntpi {
            if let Some(item) = taken[iter * ntpi + j].take() {
                items.push(item);
            }
        }
    }
    items.extend(taken.into_iter().flatten());
}

impl Gbdt {
    /// Shuffle iterations in `[start, end)` (upstream `GBDT::ShuffleModels`).
    pub fn shuffle_models(&mut self, start: i32, end: i32) {
        let ntpi = self.num_tree_per_iteration.max(1);
        let order = shuffle_order(self.models.len() / ntpi, start, end);
        permute_iterations(&mut self.models, ntpi, &order);
    }

    /// Edit one leaf output of the model (upstream `LGBM_BoosterSetLeafValue`).
    pub fn set_leaf_value(&mut self, tree_idx: usize, leaf_idx: usize, value: f64) -> Result<()> {
        set_leaf_in(&mut self.models, tree_idx, leaf_idx, value)
    }
}

impl Gbdt {
    pub fn num_class(&self) -> usize {
        self.num_class
    }
    pub fn num_tree_per_iteration(&self) -> usize {
        self.num_tree_per_iteration
    }
    pub fn num_features(&self) -> usize {
        (self.max_feature_idx + 1) as usize
    }
    pub fn num_trees(&self) -> usize {
        self.models.len()
    }
    /// upstream: `GetCurrentIteration`
    pub fn current_iteration(&self) -> usize {
        self.models.len() / self.num_tree_per_iteration
    }
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    pub fn feature_infos(&self) -> &[String] {
        &self.feature_infos
    }
    pub fn objective(&self) -> Option<&LoadedObjective> {
        self.objective.as_ref()
    }
    pub fn average_output(&self) -> bool {
        self.average_output
    }
    pub fn is_linear(&self) -> bool {
        self.linear_tree || self.models.iter().any(|t| t.is_linear)
    }
    pub fn trees(&self) -> &[Tree] {
        &self.models
    }
    pub fn label_idx(&self) -> i32 {
        self.label_idx
    }
    pub fn pandas_categorical(&self) -> Option<&str> {
        self.pandas_categorical.as_deref()
    }
    pub fn monotone_constraints(&self) -> &[i8] {
        &self.monotone_constraints
    }
    pub fn parser_config(&self) -> &str {
        &self.parser_config
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn loaded_parameter(&self) -> &str {
        &self.loaded_parameter
    }
    /// upstream: `GBDT::NeedAccuratePrediction`
    pub fn need_accurate_prediction(&self) -> bool {
        self.objective
            .as_ref()
            .is_none_or(|o| o.need_accurate_prediction)
    }

    /// upstream: `GBDT::GetUpperBoundValue` (sum of each tree's max leaf value).
    pub fn upper_bound_value(&self) -> f64 {
        let mut v = 0.0;
        for t in &self.models {
            v += t
                .leaf_value
                .iter()
                .copied()
                .skip(1)
                .fold(t.leaf_value[0], |a, b| if b > a { b } else { a });
        }
        v
    }

    /// upstream: `GBDT::GetLowerBoundValue` (sum of each tree's min leaf value).
    pub fn lower_bound_value(&self) -> f64 {
        let mut v = 0.0;
        for t in &self.models {
            v += t
                .leaf_value
                .iter()
                .copied()
                .skip(1)
                .fold(t.leaf_value[0], |a, b| if b < a { b } else { a });
        }
        v
    }

    /// upstream: `GBDT::GetLeafValue`
    pub fn leaf_value(&self, tree_idx: usize, leaf_idx: usize) -> Option<f64> {
        self.models.get(tree_idx)?.leaf_value.get(leaf_idx).copied()
    }

    /// Clamp `(start, num)` the way upstream's `InitPredict` does. `num <= 0` means "all".
    pub fn iteration_range(&self, start_iteration: i32, num_iteration: i32) -> (usize, usize) {
        let total = self.current_iteration() as i64;
        let start = i64::from(start_iteration).clamp(0, total);
        let num = if num_iteration > 0 {
            i64::from(num_iteration).min(total - start)
        } else {
            total - start
        };
        (start as usize, num as usize)
    }
}
