//! Gradient boosted decision trees. upstream: src/boosting/gbdt.{h,cpp}, gbdt_prediction.cpp

use crate::objective::LoadedObjective;
use crate::tree::Tree;

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
