//! Parameters. upstream: include/LightGBM/config.h, src/io/config.cpp
//!
//! The parameter table itself (`auto.rs`) is generated from upstream's `config.h` by
//! `scripts/gen_config.py`; this module holds the hand-written logic from `config.cpp`.

mod auto;

use std::collections::{HashMap, HashSet};
use std::fmt;

pub use auto::{ALIAS_TABLE, Config, PARAMETER_SET, PARAMETER_TO_ALIASES, PARAMETER_TYPES};

use crate::error::{Error, Result};
use crate::meta::{K_EPSILON, K_ZERO_THRESHOLD};
use crate::objective::parse_objective_alias;
use crate::utils::common::atof_legacy;
use crate::utils::format::format_g;
use crate::utils::random::Random;

/// upstream: `TaskType`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskType {
    #[default]
    Train,
    Predict,
    ConvertModel,
    RefitTree,
    SaveBinary,
}

fn warn(msg: &str) {
    log::warn!("{msg}");
}

// ---------------------------------------------------------------------------------------
// Typed getters (upstream `Config::GetString/GetInt/GetDouble/GetBool`).
// ---------------------------------------------------------------------------------------

pub(crate) fn get_string(params: &HashMap<String, String>, name: &str) -> Option<String> {
    params.get(name).filter(|v| !v.is_empty()).cloned()
}

/// upstream `Common::Atoi` + "and check": optional spaces, sign, digits, optional spaces.
fn atoi_and_check(s: &str) -> Option<i64> {
    let t = s.trim_matches(' ');
    let (neg, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut v: i64 = 0;
    for b in digits.bytes() {
        v = v.wrapping_mul(10).wrapping_add(i64::from(b - b'0'));
    }
    Some(if neg { -v } else { v })
}

pub(crate) fn get_int(params: &HashMap<String, String>, name: &str, out: &mut i32) -> Result<bool> {
    match get_string(params, name) {
        None => Ok(false),
        Some(v) => {
            let parsed = atoi_and_check(&v).ok_or_else(|| {
                Error::InvalidParameter(format!(
                    "Parameter {name} should be of type int, got \"{v}\""
                ))
            })?;
            // C++ stores into a 32-bit int.
            *out = parsed as i32;
            Ok(true)
        }
    }
}

pub(crate) fn get_double(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut f64,
) -> Result<bool> {
    match get_string(params, name) {
        None => Ok(false),
        Some(v) => {
            *out = atof_legacy(&v).ok_or_else(|| {
                Error::InvalidParameter(format!(
                    "Parameter {name} should be of type double, got \"{v}\""
                ))
            })?;
            Ok(true)
        }
    }
}

pub(crate) fn get_bool(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut bool,
) -> Result<bool> {
    match get_string(params, name) {
        None => Ok(false),
        Some(v) => {
            *out = match v.to_ascii_lowercase().as_str() {
                "false" | "-" => false,
                "true" | "+" => true,
                _ => {
                    return Err(Error::InvalidParameter(format!(
                        "Parameter {name} should be \"true\"/\"+\" or \"false\"/\"-\", got \"{v}\""
                    )));
                }
            };
            Ok(true)
        }
    }
}

pub(crate) fn get_string_into(
    params: &HashMap<String, String>,
    name: &str,
    out: &mut String,
) -> Result<bool> {
    match get_string(params, name) {
        None => Ok(false),
        Some(v) => {
            *out = v;
            Ok(true)
        }
    }
}

// Generated code calls `get_string(params, name, &mut field)` for scalar strings.
pub(crate) use get_string_into as get_string_field;

pub(crate) fn check(ok: bool, what: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidParameter(format!("Check failed: {what}")))
    }
}

/// upstream `Common::Split(str, ',')` (drops empty tokens).
pub(crate) fn split_strings(s: &str) -> Result<Vec<String>> {
    Ok(s.split(',')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect())
}

/// upstream `Common::StringToArray<double>(str, ',')` (uses `std::stod`).
pub(crate) fn parse_vec_f64(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| {
            stod(t)
                .ok_or_else(|| Error::InvalidParameter(format!("cannot convert {t:?} to double")))
        })
        .collect()
}

/// `std::stod`: skips leading whitespace and parses the longest valid prefix.
fn stod(s: &str) -> Option<f64> {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut end = 0;
    let mut best = None;
    while end < b.len() {
        end += 1;
        if let Ok(v) = t[..end].parse::<f64>() {
            best = Some(v);
        }
    }
    best.or_else(|| match t.to_ascii_lowercase().as_str() {
        x if x.starts_with("nan") => Some(f64::NAN),
        x if x.starts_with("inf") => Some(f64::INFINITY),
        x if x.starts_with("-inf") => Some(f64::NEG_INFINITY),
        _ => None,
    })
}

/// upstream `Common::StringToArray<int>(str, ',')` (lenient `Atoi`).
pub(crate) fn parse_vec_int<T: TryFrom<i64>>(s: &str) -> Result<Vec<T>> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let v = atoi_lenient(t);
            T::try_from(v).map_err(|_| Error::InvalidParameter(format!("value {t:?} out of range")))
        })
        .collect()
}

fn atoi_lenient(s: &str) -> i64 {
    let t = s.trim_start_matches(' ');
    let (neg, rest) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let mut v: i64 = 0;
    for b in rest.bytes().take_while(u8::is_ascii_digit) {
        v = v.wrapping_mul(10).wrapping_add(i64::from(b - b'0'));
    }
    if neg { -v } else { v }
}

/// upstream `Common::Join` for doubles: stream precision 17 (`%.17g`).
pub(crate) fn join_f64_17(v: &[f64]) -> String {
    v.iter()
        .map(|x| format_g(*x, 17))
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) fn join_display<T: fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// upstream `ParseMetricAlias`
pub fn parse_metric_alias(name: &str) -> &str {
    match name {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" => "l2",
        "l2_root" | "root_mean_squared_error" | "rmse" => "rmse",
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => "l1",
        "binary_logloss" | "binary" => "binary_logloss",
        "ndcg" | "lambdarank" | "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart"
        | "xendcg_mart" => "ndcg",
        "map" | "mean_average_precision" => "map",
        "multi_logloss" | "multiclass" | "softmax" | "multiclassova" | "multiclass_ova" | "ova"
        | "ovr" => "multi_logloss",
        "xentropy" | "cross_entropy" => "cross_entropy",
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda",
        "kldiv" | "kullback_leibler" => "kullback_leibler",
        "mean_absolute_percentage_error" | "mape" => "mape",
        "none" | "null" | "custom" | "na" => "custom",
        other => other,
    }
}

fn parse_metrics(value: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for m in value.split(',').filter(|t| !t.is_empty()) {
        let t = parse_metric_alias(m).to_string();
        if seen.insert(t.clone()) {
            out.push(t);
        }
    }
    out
}

/// upstream `Config::SortAlias`: shorter first, then lexicographic.
fn sort_alias(x: &str, y: &str) -> bool {
    x.len() < y.len() || (x.len() == y.len() && x < y)
}

fn alias_of(name: &str) -> Option<&'static str> {
    ALIAS_TABLE
        .iter()
        .find(|(a, _)| *a == name)
        .map(|(_, n)| *n)
}

/// upstream `ParameterAlias::KeyAliasTransform`: rename aliases to canonical names.
/// When several spellings are given, the canonical name wins, then the shortest alias.
pub fn key_alias_transform(params: &mut HashMap<String, String>) {
    let mut tmp: HashMap<&'static str, String> = HashMap::new();
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort(); // deterministic warnings/choices (upstream iterates an unordered_map)
    for key in keys {
        if let Some(canonical) = alias_of(key) {
            match tmp.get(canonical) {
                Some(prev) => {
                    if !sort_alias(prev, key) {
                        warn(&format!(
                            "{canonical} is set with {prev}, will be overridden by {key}"
                        ));
                        tmp.insert(canonical, key.clone());
                    } else {
                        warn(&format!(
                            "{canonical} is set with {prev}, {key} will be ignored"
                        ));
                    }
                }
                None => {
                    tmp.insert(canonical, key.clone());
                }
            }
        } else if !PARAMETER_SET.contains(&key.as_str()) {
            warn(&format!("Unknown parameter: {key}"));
        }
    }
    for (canonical, alias) in tmp {
        if params.contains_key(canonical) {
            warn(&format!("{canonical} is set, {alias} will be ignored"));
        } else if let Some(v) = params.get(&alias).cloned() {
            params.insert(canonical.to_string(), v);
            params.remove(&alias);
        }
    }
}

/// upstream `Config::Str2Map`: parse `"k1=v1 k2=v2"` into canonical-name params.
/// Returns the params and the requested verbosity (if any).
pub fn str2map(parameters: &str) -> Result<HashMap<String, String>> {
    let mut all: HashMap<String, Vec<String>> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    for arg in parameters
        .split([' ', '\t', '\n', '\r'])
        .filter(|a| !a.is_empty())
    {
        let parts: Vec<&str> = arg.split('=').filter(|t| !t.is_empty()).collect();
        if parts.len() == 1 || parts.len() == 2 {
            let clean = |s: &str| s.trim().trim_matches(|c| c == '\'' || c == '"').to_string();
            let key = clean(parts[0]);
            let value = if parts.len() == 2 {
                clean(parts[1])
            } else {
                String::new()
            };
            if !key.is_empty() {
                if !all.contains_key(&key) {
                    order.push(key.clone());
                }
                all.entry(key).or_default().push(value);
            }
        } else {
            warn(&format!("Unknown parameter {arg}"));
        }
    }
    let mut params = HashMap::new();
    for key in order {
        let values = &all[&key];
        for extra in &values[1..] {
            warn(&format!(
                "{key} is set={}, {key}={extra} will be ignored",
                values[0]
            ));
        }
        params.insert(key, values[0].clone());
    }
    key_alias_transform(&mut params);
    Ok(params)
}

impl Config {
    /// Build a config from canonical-name (or alias) params, as upstream `Config::Set`.
    pub fn from_params(params: &HashMap<String, String>) -> Result<Config> {
        let mut cfg = Config::default();
        let mut params = params.clone();
        key_alias_transform(&mut params);
        cfg.set(&params)?;
        Ok(cfg)
    }

    /// Parse a `"key=value ..."` string (upstream `Config::Str2Map` + `Set`).
    pub fn from_str_params(parameters: &str) -> Result<Config> {
        let params = str2map(parameters)?;
        let mut cfg = Config::default();
        cfg.set(&params)?;
        Ok(cfg)
    }

    /// upstream: `Config::Set`
    pub fn set(&mut self, params: &HashMap<String, String>) -> Result<()> {
        if get_int(params, "seed", &mut self.seed)? {
            let mut rand = Random::new(self.seed);
            let int_max = i32::from(i16::MAX);
            self.data_random_seed = rand.next_short(0, int_max);
            self.bagging_seed = rand.next_short(0, int_max);
            self.drop_seed = rand.next_short(0, int_max);
            self.feature_fraction_seed = rand.next_short(0, int_max);
            self.objective_seed = rand.next_short(0, int_max);
            self.extra_seed = rand.next_short(0, int_max);
        }
        if let Some(v) = get_string(params, "task") {
            self.task = match v.to_ascii_lowercase().as_str() {
                "train" | "training" => TaskType::Train,
                "predict" | "prediction" | "test" => TaskType::Predict,
                "convert_model" => TaskType::ConvertModel,
                "refit" | "refit_tree" => TaskType::RefitTree,
                "save_binary" => TaskType::SaveBinary,
                _ => return Err(Error::InvalidParameter(format!("Unknown task type {v}"))),
            };
        }
        if let Some(v) = get_string(params, "boosting") {
            self.boosting = match v.to_ascii_lowercase().as_str() {
                "gbdt" | "gbrt" => "gbdt",
                "dart" => "dart",
                "goss" => "goss",
                "rf" | "random_forest" => "rf",
                _ => {
                    return Err(Error::InvalidParameter(format!(
                        "Unknown boosting type {v}"
                    )));
                }
            }
            .to_string();
        }
        if let Some(v) = get_string(params, "data_sample_strategy") {
            self.data_sample_strategy = match v.to_ascii_lowercase().as_str() {
                "goss" => "goss",
                "bagging" => "bagging",
                _ => {
                    return Err(Error::InvalidParameter(format!(
                        "Unknown sample strategy {v}"
                    )));
                }
            }
            .to_string();
        }
        if let Some(v) = get_string(params, "objective") {
            self.objective = parse_objective_alias(&v.to_ascii_lowercase()).to_string();
        }
        // upstream GetMetricType
        let metric_value = get_string(params, "metric").map(|v| v.to_ascii_lowercase());
        if let Some(v) = &metric_value {
            self.metric = parse_metrics(v);
        }
        if self.metric.is_empty() && metric_value.as_deref().unwrap_or("").is_empty() {
            self.metric = parse_metrics(&self.objective);
        }
        if let Some(v) = get_string(params, "device_type") {
            self.device_type = match v.to_ascii_lowercase().as_str() {
                "cpu" => "cpu",
                "gpu" => "gpu",
                "cuda" => "cuda",
                _ => return Err(Error::InvalidParameter(format!("Unknown device type {v}"))),
            }
            .to_string();
        }
        if let Some(v) = get_string(params, "tree_learner") {
            self.tree_learner = match v.to_ascii_lowercase().as_str() {
                "serial" => "serial",
                "feature" | "feature_parallel" => "feature",
                "data" | "data_parallel" => "data",
                "voting" | "voting_parallel" => "voting",
                _ => {
                    return Err(Error::InvalidParameter(format!(
                        "Unknown tree learner type {v}"
                    )));
                }
            }
            .to_string();
        }

        self.get_members_from_string(params)?;
        self.get_auc_mu_weights()?;
        self.get_interaction_constraints()?;
        self.eval_at.sort_unstable();

        let data = self.data.clone();
        let mut new_valid = Vec::new();
        for v in std::mem::take(&mut self.valid) {
            if v != data {
                new_valid.push(v);
            } else {
                self.is_provide_training_metric = true;
            }
        }
        self.valid = new_valid;
        if self.task == TaskType::SaveBinary && !self.save_binary {
            self.save_binary = true;
        }
        self.check_param_conflict(params)
    }

    /// upstream: `Config::GetAucMuWeights`
    fn get_auc_mu_weights(&mut self) -> Result<()> {
        let n = usize::try_from(self.num_class).unwrap_or(0);
        if self.auc_mu_weights.is_empty() {
            self.auc_mu_weights_matrix = (0..n)
                .map(|i| (0..n).map(|j| if i == j { 0.0 } else { 1.0 }).collect())
                .collect();
            return Ok(());
        }
        if self.auc_mu_weights.len() != n * n {
            return Err(Error::InvalidParameter(format!(
                "auc_mu_weights must have {} elements, but found {}",
                n * n,
                self.auc_mu_weights.len()
            )));
        }
        let mut m = vec![vec![0.0; n]; n];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let w = self.auc_mu_weights[i * n + j];
                if i != j {
                    if w.abs() < K_ZERO_THRESHOLD {
                        return Err(Error::InvalidParameter(format!(
                            "AUC-mu matrix must have non-zero values for non-diagonal entries. \
                             Found zero value in position {} of auc_mu_weights.",
                            i * n + j
                        )));
                    }
                    *cell = w;
                }
            }
        }
        self.auc_mu_weights_matrix = m;
        Ok(())
    }

    /// upstream: `Config::GetInteractionConstraints` (`StringToArrayofArrays<int>(s, '[', ']', ',')`)
    fn get_interaction_constraints(&mut self) -> Result<()> {
        self.interaction_constraints_vector.clear();
        let s = self.interaction_constraints.clone();
        let b = s.as_bytes();
        let (mut i, mut open) = (0usize, false);
        for pos in 0..b.len() {
            if b[pos] == b'[' {
                open = true;
                i = pos + 1;
            } else if b[pos] == b']' && open {
                if i < pos {
                    self.interaction_constraints_vector
                        .push(parse_vec_int::<i32>(&s[i..pos])?);
                }
                open = false;
            }
        }
        Ok(())
    }

    /// upstream: `Config::CheckParamConflict`
    fn check_param_conflict(&mut self, params: &HashMap<String, String>) -> Result<()> {
        let is_multi = |o: &str| o == "multiclass" || o == "multiclassova";
        let num_class = self.num_class;
        let objective_multiclass =
            is_multi(&self.objective) || (self.objective == "custom" && num_class > 1);
        if objective_multiclass {
            if num_class <= 1 {
                return Err(Error::InvalidParameter(
                    "Number of classes should be specified and greater than 1 for multiclass training".into(),
                ));
            }
        } else if self.task == TaskType::Train && num_class != 1 {
            return Err(Error::InvalidParameter(
                "Number of classes must be 1 for non-multiclass training".into(),
            ));
        }
        for m in &self.metric {
            let metric_multiclass = is_multi(m)
                || m == "multi_logloss"
                || m == "multi_error"
                || m == "auc_mu"
                || (m == "custom" && num_class > 1);
            if objective_multiclass != metric_multiclass {
                return Err(Error::InvalidParameter(
                    "Multiclass objective and metrics don't match".into(),
                ));
            }
        }
        if self.num_machines > 1 {
            self.is_parallel = true;
        } else {
            self.is_parallel = false;
            self.tree_learner = "serial".into();
        }
        let single = self.tree_learner == "serial";
        if single {
            self.is_parallel = false;
            self.num_machines = 1;
        }
        if single || self.tree_learner == "feature" {
            self.is_data_based_parallel = false;
        } else if self.tree_learner == "data" || self.tree_learner == "voting" {
            self.is_data_based_parallel = true;
            if self.histogram_pool_size >= 0.0 && self.tree_learner == "data" {
                warn(
                    "Histogram LRU queue was enabled; will disable this to reduce communication costs",
                );
                self.histogram_pool_size = -1.0;
            }
        }
        if self.is_data_based_parallel && !self.forcedsplits_filename.is_empty() {
            return Err(Error::InvalidParameter(format!(
                "Don't support forcedsplits in {} tree learner",
                self.tree_learner
            )));
        }
        let num_leaves_given = params.get("num_leaves").is_some_and(|v| !v.is_empty());
        if self.max_depth > 0 && !num_leaves_given {
            let full = 2f64.powi(self.max_depth);
            if full > f64::from(self.num_leaves) {
                warn(
                    "Provided parameters constrain tree depth (max_depth) without explicitly setting 'num_leaves'.",
                );
            }
            if full < f64::from(self.num_leaves) {
                self.num_leaves = full as i32;
            }
        }
        if self.device_type == "gpu" {
            self.force_col_wise = true;
            self.force_row_wise = false;
            if self.use_quantized_grad {
                warn(
                    "Quantized training is not supported by GPU tree learner. Switch to full precision training.",
                );
                self.use_quantized_grad = false;
            }
        } else if self.device_type == "cuda" {
            self.force_col_wise = false;
            self.force_row_wise = true;
        }
        if self.linear_tree {
            if self.device_type != "cpu" && self.device_type != "gpu" {
                self.device_type = "cpu".into();
                warn("Linear tree learner only works with CPU and GPU. Falling back to CPU now.");
            }
            if self.tree_learner != "serial" {
                self.tree_learner = "serial".into();
                warn("Linear tree learner must be serial.");
            }
            if self.zero_as_missing {
                return Err(Error::InvalidParameter(
                    "zero_as_missing must be false when fitting linear trees.".into(),
                ));
            }
            if self.objective == "regression_l1" {
                return Err(Error::InvalidParameter(
                    "Cannot use regression_l1 objective when fitting linear trees.".into(),
                ));
            }
        }
        if self.path_smooth > f64::from(K_EPSILON) && self.min_data_in_leaf < 2 {
            self.min_data_in_leaf = 2;
            warn(
                "min_data_in_leaf has been increased to 2 because this is required when path smoothing is active.",
            );
        }
        let advanced = self.monotone_constraints_method == "intermediate"
            || self.monotone_constraints_method == "advanced";
        if self.is_parallel && advanced {
            warn(
                "Cannot use \"intermediate\" or \"advanced\" monotone constraints in distributed learning, auto set to \"basic\" method.",
            );
            self.monotone_constraints_method = "basic".into();
        }
        let advanced = self.monotone_constraints_method == "intermediate"
            || self.monotone_constraints_method == "advanced";
        if self.feature_fraction_bynode != 1.0 && advanced {
            warn(
                "Cannot use \"intermediate\" or \"advanced\" monotone constraints with feature fraction different from 1, auto set monotone constraints to \"basic\" method.",
            );
            self.monotone_constraints_method = "basic".into();
        }
        if self.max_depth > 0 && self.monotone_penalty >= f64::from(self.max_depth) {
            warn("Monotone penalty greater than tree depth. Monotone features won't be used.");
        }
        if self.min_data_in_leaf <= 0 && self.min_sum_hessian_in_leaf <= f64::from(K_EPSILON) {
            warn(
                "Cannot set both min_data_in_leaf and min_sum_hessian_in_leaf to 0. Will set min_data_in_leaf to 1.",
            );
            self.min_data_in_leaf = 1;
        }
        if self.boosting == "goss" {
            self.boosting = "gbdt".into();
            self.data_sample_strategy = "goss".into();
            warn(
                "Found boosting=goss. For backwards compatibility reasons, LightGBM interprets this as boosting=gbdt, data_sample_strategy=goss.",
            );
        }
        if self.bagging_by_query && self.data_sample_strategy != "bagging" {
            warn(
                "bagging_by_query=true is only compatible with data_sample_strategy=bagging. Setting bagging_by_query=false.",
            );
            self.bagging_by_query = false;
        }
        Ok(())
    }
}

/// upstream: `Config::ToString` — the `parameters:` block of a saved model.
impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[boosting: {}]", self.boosting)?;
        writeln!(f, "[objective: {}]", self.objective)?;
        writeln!(f, "[metric: {}]", self.metric.join(","))?;
        writeln!(f, "[tree_learner: {}]", self.tree_learner)?;
        writeln!(f, "[device_type: {}]", self.device_type)?;
        f.write_str(&self.save_members_to_string())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn p(kv: &[(&str, &str)]) -> HashMap<String, String> {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn defaults_match_upstream() {
        let c = Config::default();
        assert_eq!(c.num_leaves, 31);
        assert_eq!(c.learning_rate, 0.1);
        assert_eq!(c.max_bin, 255);
        assert_eq!(c.objective, "regression");
        assert_eq!(c.min_sum_hessian_in_leaf, 1e-3);
        assert!(c.use_missing && !c.zero_as_missing);
        assert_eq!(c.output_model, "LightGBM_model.txt");
    }

    #[test]
    fn aliases_resolve_and_canonical_wins() {
        let c = Config::from_params(&p(&[
            ("eta", "0.05"),
            ("num_leaf", "7"),
            ("n_estimators", "12"),
        ]))
        .unwrap();
        assert_eq!(c.learning_rate, 0.05);
        assert_eq!(c.num_leaves, 7);
        assert_eq!(c.num_iterations, 12);
        let c = Config::from_params(&p(&[("learning_rate", "0.2"), ("eta", "0.05")])).unwrap();
        assert_eq!(c.learning_rate, 0.2);
        // shortest alias wins among aliases
        let c = Config::from_params(&p(&[("eta", "0.3"), ("shrinkage_rate", "0.05")])).unwrap();
        assert_eq!(c.learning_rate, 0.3);
    }

    #[test]
    fn checks_are_enforced() {
        assert!(Config::from_params(&p(&[("num_leaves", "1")])).is_err());
        assert!(Config::from_params(&p(&[("learning_rate", "0")])).is_err());
        assert!(Config::from_params(&p(&[("bagging_fraction", "1.5")])).is_err());
        assert!(Config::from_params(&p(&[("num_leaves", "abc")])).is_err());
        assert!(Config::from_params(&p(&[("use_missing", "maybe")])).is_err());
    }

    #[test]
    fn objective_metric_and_conflicts() {
        let c = Config::from_params(&p(&[("objective", "softmax"), ("num_class", "3")])).unwrap();
        assert_eq!(c.objective, "multiclass");
        assert_eq!(c.metric, vec!["multi_logloss"]);
        assert!(Config::from_params(&p(&[("objective", "multiclass")])).is_err());
        assert!(Config::from_params(&p(&[("objective", "binary"), ("num_class", "2")])).is_err());
        let c = Config::from_params(&p(&[("metric", "mse,l2,rmse,auc")])).unwrap();
        assert_eq!(c.metric, vec!["l2", "rmse", "auc"]);
    }

    #[test]
    fn seed_derives_sub_seeds_like_upstream() {
        let c = Config::from_params(&p(&[("seed", "42")])).unwrap();
        let mut r = Random::new(42);
        assert_eq!(c.data_random_seed, r.next_short(0, 32767));
        assert_eq!(c.bagging_seed, r.next_short(0, 32767));
    }

    #[test]
    fn max_depth_caps_default_num_leaves() {
        let c = Config::from_params(&p(&[("max_depth", "3")])).unwrap();
        assert_eq!(c.num_leaves, 8);
        let c = Config::from_params(&p(&[("max_depth", "3"), ("num_leaves", "31")])).unwrap();
        assert_eq!(c.num_leaves, 31);
    }

    #[test]
    fn goss_boosting_is_rewritten() {
        let c = Config::from_params(&p(&[("boosting", "goss")])).unwrap();
        assert_eq!(c.boosting, "gbdt");
        assert_eq!(c.data_sample_strategy, "goss");
    }

    #[test]
    fn str2map_handles_quotes_duplicates_and_aliases() {
        let m = str2map("num_trees=5 'eta'=\"0.3\" num_trees=9 verbose=-1").unwrap();
        assert_eq!(m["num_iterations"], "5");
        assert_eq!(m["learning_rate"], "0.3");
        assert_eq!(m["verbosity"], "-1");
    }

    #[test]
    fn vector_params_and_interaction_constraints() {
        let c = Config::from_params(&p(&[
            ("label_gain", "0,1,3.5"),
            ("eval_at", "5,1,3"),
            ("monotone_constraints", "1,-1,0"),
            ("interaction_constraints", "[0,1],[2, 3]"),
        ]))
        .unwrap();
        assert_eq!(c.label_gain, vec![0.0, 1.0, 3.5]);
        assert_eq!(c.eval_at, vec![1, 3, 5]);
        assert_eq!(c.monotone_constraints, vec![1, -1, 0]);
        assert_eq!(
            c.interaction_constraints_vector,
            vec![vec![0, 1], vec![2, 3]]
        );
    }
}
