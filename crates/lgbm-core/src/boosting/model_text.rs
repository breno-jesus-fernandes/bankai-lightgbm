//! Model text format reader. upstream: src/boosting/gbdt_model_text.cpp

use std::path::Path;

use rayon::prelude::*;

use super::Gbdt;
use crate::error::{Error, Result};
use crate::objective::LoadedObjective;
use crate::tree::Tree;
use crate::tree::avoid_inf;
use crate::utils::common::atof_legacy;
use crate::utils::common::{parse_array_any, parse_scalar};
use crate::utils::format::{format_g, join_display};

/// Cursor over model text that mimics upstream's `GetLine`/`SkipNewLine`.
struct Cursor<'a> {
    s: &'a str,
    p: usize,
}

impl<'a> Cursor<'a> {
    fn done(&self) -> bool {
        self.p >= self.s.len()
    }
    /// Returns the current line (without terminator) and advances past it.
    fn next_line(&mut self) -> &'a str {
        let b = self.s.as_bytes();
        let start = self.p;
        while self.p < b.len() && b[self.p] != b'\r' && b[self.p] != b'\n' {
            self.p += 1;
        }
        let line = &self.s[start..self.p];
        self.skip_newline();
        line
    }
    fn peek_line(&self) -> &'a str {
        let b = self.s.as_bytes();
        let mut e = self.p;
        while e < b.len() && b[e] != b'\r' && b[e] != b'\n' {
            e += 1;
        }
        &self.s[self.p..e]
    }
    fn skip_newline(&mut self) {
        let b = self.s.as_bytes();
        if self.p < b.len() && b[self.p] == b'\r' {
            self.p += 1;
        }
        if self.p < b.len() && b[self.p] == b'\n' {
            self.p += 1;
        }
    }
}

impl Gbdt {
    /// upstream: `GBDT::FeatureImportance`. Note upstream ignores `start_iteration` here
    /// and counts trees from the first one up to `num_iteration`.
    pub fn feature_importance(&self, num_iteration: i32, importance_type: i32) -> Result<Vec<f64>> {
        let mut num_used_model = self.models.len();
        if num_iteration > 0 {
            num_used_model =
                num_used_model.min(num_iteration as usize * self.num_tree_per_iteration);
        }
        let mut imp = vec![0.0; self.num_features()];
        for t in &self.models[..num_used_model] {
            for i in 0..t.num_leaves.saturating_sub(1) {
                if t.split_gain[i] > 0.0 {
                    let f = t.split_feature[i] as usize;
                    match importance_type {
                        0 => imp[f] += 1.0,
                        1 => imp[f] += f64::from(t.split_gain[i]),
                        _ => {}
                    }
                }
            }
        }
        if !(0..=1).contains(&importance_type) {
            return Err(Error::InvalidParameter(
                "Unknown importance type: only support split=0 and gain=1".into(),
            ));
        }
        Ok(imp)
    }

    /// upstream: `GBDT::SaveModelToString`. The Python layer appends the
    /// `pandas_categorical` trailer, as upstream's Python package does.
    pub fn save_model_to_string(
        &self,
        start_iteration: i32,
        num_iteration: i32,
        importance_type: i32,
    ) -> Result<String> {
        use std::fmt::Write;
        let ntpi = self.num_tree_per_iteration;
        let mut s = String::with_capacity(4096 + self.models.len() * 2048);
        let _ = writeln!(s, "tree");
        let _ = writeln!(s, "version=v4");
        let _ = writeln!(s, "num_class={}", self.num_class);
        let _ = writeln!(s, "num_tree_per_iteration={ntpi}");
        let _ = writeln!(s, "label_index={}", self.label_idx);
        let _ = writeln!(s, "max_feature_idx={}", self.max_feature_idx);
        if let Some(o) = &self.objective {
            let _ = writeln!(s, "objective={}", o.to_model_string());
        }
        if self.average_output {
            let _ = writeln!(s, "average_output");
        }
        let _ = writeln!(s, "feature_names={}", self.feature_names.join(" "));
        if !self.monotone_constraints.is_empty() {
            let _ = writeln!(
                s,
                "monotone_constraints={}",
                join_display(&self.monotone_constraints)
            );
        }
        let _ = writeln!(s, "feature_infos={}", self.feature_infos.join(" "));

        let total_iteration = (self.models.len() / ntpi) as i64;
        let start = i64::from(start_iteration).clamp(0, total_iteration) as usize;
        let mut num_used_model = self.models.len();
        if num_iteration > 0 {
            let end = start + num_iteration as usize;
            num_used_model = num_used_model.min(end * ntpi);
        }
        let start_model = (start * ntpi).min(num_used_model);
        let tree_strs: Vec<String> = self.models[start_model..num_used_model]
            .par_iter()
            .enumerate()
            .map(|(idx, t)| format!("Tree={idx}\n{}\n", t.to_model_string()))
            .collect();
        let sizes: Vec<usize> = tree_strs.iter().map(String::len).collect();
        let _ = writeln!(s, "tree_sizes={}", join_display(&sizes));
        s.push('\n');
        for t in &tree_strs {
            s.push_str(t);
        }
        s.push_str("end of trees\n");

        let imp = self.feature_importance(num_iteration, importance_type)?;
        let mut pairs: Vec<(usize, &str)> = imp
            .iter()
            .zip(&self.feature_names)
            .filter_map(|(v, n)| {
                let iv = *v as usize;
                (iv > 0).then_some((iv, n.as_str()))
            })
            .collect();
        pairs.sort_by_key(|p| std::cmp::Reverse(p.0)); // stable, descending
        s.push_str("\nfeature_importances:\n");
        for (v, n) in pairs {
            let _ = writeln!(s, "{n}={v}");
        }
        if !self.loaded_parameter.is_empty() {
            s.push_str("\nparameters:\n");
            s.push_str(&self.loaded_parameter);
            s.push_str("\nend of parameters\n");
        }
        if !self.parser_config.is_empty() {
            s.push_str("\nparser:\n");
            s.push_str(&self.parser_config);
            s.push_str("\nend of parser\n");
        }
        Ok(s)
    }

    /// upstream: `GBDT::DumpModel` (JSON). The Python layer adds `pandas_categorical`.
    pub fn dump_model(
        &self,
        start_iteration: i32,
        num_iteration: i32,
        importance_type: i32,
    ) -> Result<String> {
        use std::fmt::Write;
        let g17 = |v: f64| format_g(v, 17);
        let ntpi = self.num_tree_per_iteration;
        let mut s = String::from("{");
        let _ = writeln!(s, "\"name\":\"tree\",");
        let _ = writeln!(s, "\"version\":\"v4\",");
        let _ = writeln!(s, "\"num_class\":{},", self.num_class);
        let _ = writeln!(s, "\"num_tree_per_iteration\":{ntpi},");
        let _ = writeln!(s, "\"label_index\":{},", self.label_idx);
        let _ = writeln!(s, "\"max_feature_idx\":{},", self.max_feature_idx);
        if let Some(o) = &self.objective {
            let _ = writeln!(s, "\"objective\":\"{}\",", o.to_model_string());
        }
        let _ = writeln!(s, "\"average_output\":{},", self.average_output);
        let _ = writeln!(
            s,
            "\"feature_names\":[\"{}\"],",
            self.feature_names.join("\",\"")
        );
        let mc: Vec<String> = self
            .monotone_constraints
            .iter()
            .map(ToString::to_string)
            .collect();
        let _ = writeln!(s, "\"monotone_constraints\":[{}],", mc.join(","));

        s.push_str("\"feature_infos\":{");
        let mut first = true;
        for (info, name) in self.feature_infos.iter().zip(&self.feature_names) {
            let obj = if let Some(range) = info.strip_prefix('[') {
                let range = range.strip_suffix(']').unwrap_or(range);
                let (lo, hi) = range.split_once(':').unwrap_or((range, ""));
                let parse = |x: &str| atof_legacy(x).unwrap_or(f64::NAN);
                format!(
                    "{{\"min_value\":{},\"max_value\":{},\"values\":[]}}",
                    g17(avoid_inf(parse(lo))),
                    g17(avoid_inf(parse(hi)))
                )
            } else if info != "none" {
                let vals: Vec<i64> = info
                    .split(':')
                    .map(parse_scalar::<i64>)
                    .collect::<Result<_>>()?;
                // upstream ArrayArgs::ArgMin/ArgMax return the first extreme.
                let min = vals.iter().copied().min().unwrap_or(0);
                let max = vals.iter().copied().max().unwrap_or(0);
                let joined: Vec<String> = vals.iter().map(ToString::to_string).collect();
                format!(
                    "{{\"min_value\":{min},\"max_value\":{max},\"values\":[{}]}}",
                    joined.join(",")
                )
            } else {
                continue;
            };
            if !first {
                s.push(',');
            }
            let _ = write!(s, "\"{name}\":{obj}");
            first = false;
        }
        s.push_str("},\n");

        s.push_str("\"tree_info\":[");
        let total_iteration = (self.models.len() / ntpi) as i64;
        let start = i64::from(start_iteration).clamp(0, total_iteration) as usize;
        let mut num_used_model = self.models.len();
        if num_iteration > 0 {
            num_used_model = num_used_model.min((start + num_iteration as usize) * ntpi);
        }
        let start_model = (start * ntpi).min(num_used_model);
        let trees: Vec<String> = self.models[start_model..num_used_model]
            .par_iter()
            .enumerate()
            .map(|(k, t)| format!("{{\"tree_index\":{},{}}}", start_model + k, t.to_json()))
            .collect();
        s.push_str(&trees.join(","));
        s.push_str("],\n");

        // Unlike the text format, the JSON dump does not sort importances.
        let imp = self.feature_importance(num_iteration, importance_type)?;
        s.push_str("\n\"feature_importances\":{");
        let mut first = true;
        for (v, name) in imp.iter().zip(&self.feature_names) {
            let iv = *v as usize;
            if iv > 0 {
                if !first {
                    s.push(',');
                }
                let _ = write!(s, "\"{name}\":{iv}");
                first = false;
            }
        }
        s.push_str("}\n}\n");
        Ok(s)
    }

    pub fn from_model_file(path: impl AsRef<Path>) -> Result<Gbdt> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Gbdt::from_model_string(&text)
    }

    /// upstream: `GBDT::LoadModelFromString`
    pub fn from_model_string(text: &str) -> Result<Gbdt> {
        let mut cur = Cursor { s: text, p: 0 };
        let mut kv: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
        let mut version = String::new();
        let mut first = true;
        while !cur.done() {
            let line = cur.peek_line();
            if line.starts_with("Tree=") {
                break;
            }
            cur.next_line();
            if line.is_empty() {
                continue;
            }
            if first {
                first = false;
                // First line is the submodel name, e.g. "tree".
                if !line.contains('=') {
                    continue;
                }
            }
            match line.split_once('=') {
                None => {
                    kv.insert(line, "");
                }
                Some((k, v)) => {
                    if v.contains('=') && k != "feature_names" && k != "monotone_constraints" {
                        let n = line.len().min(128);
                        return Err(Error::Parse(format!(
                            "Wrong line at model file: {}",
                            &line[..n]
                        )));
                    }
                    kv.insert(k, v);
                }
            }
        }
        if let Some(v) = kv.get("version") {
            version = (*v).to_string();
        }
        let get = |k: &str, msg: &str| {
            kv.get(k)
                .copied()
                .ok_or_else(|| Error::Parse(msg.to_string()))
        };

        let num_class: usize = parse_scalar(get(
            "num_class",
            "Model file doesn't specify the number of classes",
        )?)?;
        let num_tree_per_iteration: usize = match kv.get("num_tree_per_iteration") {
            Some(v) => parse_scalar(v)?,
            None => num_class,
        };
        if num_tree_per_iteration == 0 {
            return Err(Error::Parse(
                "num_tree_per_iteration must be positive".into(),
            ));
        }
        let label_idx: i32 = parse_scalar(get(
            "label_index",
            "Model file doesn't specify the label index",
        )?)?;
        let max_feature_idx: i32 = parse_scalar(get(
            "max_feature_idx",
            "Model file doesn't specify max_feature_idx",
        )?)?;
        let nfeat = usize::try_from(max_feature_idx + 1)
            .map_err(|_| Error::Parse("invalid max_feature_idx".into()))?;
        let average_output = kv.contains_key("average_output");

        let feature_names: Vec<String> =
            get("feature_names", "Model file doesn't contain feature_names")?
                .split(' ')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        if feature_names.len() != nfeat {
            return Err(Error::Parse("Wrong size of feature_names".into()));
        }
        let monotone_constraints: Vec<i8> = match kv.get("monotone_constraints") {
            Some(v) => {
                let m = parse_array_any(v)?;
                if m.len() != nfeat {
                    return Err(Error::Parse("Wrong size of monotone_constraints".into()));
                }
                m
            }
            None => Vec::new(),
        };
        let feature_infos: Vec<String> =
            get("feature_infos", "Model file doesn't contain feature_infos")?
                .split(' ')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        if feature_infos.len() != nfeat {
            return Err(Error::Parse("Wrong size of feature_infos".into()));
        }
        let objective = match kv.get("objective") {
            Some(v) => LoadedObjective::from_model_str(v)?,
            None => None,
        };

        let models = match kv.get("tree_sizes") {
            Some(sizes) => {
                let sizes: Vec<usize> = parse_array_any(sizes)?;
                let base = cur.p;
                let mut bounds = Vec::with_capacity(sizes.len());
                let mut off = base;
                for sz in &sizes {
                    let end = off.checked_add(*sz).filter(|e| *e <= text.len());
                    let end =
                        end.ok_or_else(|| Error::Parse("tree_sizes exceed model length".into()))?;
                    bounds.push((off, end));
                    off = end;
                }
                cur.p = off;
                bounds
                    .par_iter()
                    .map(|&(a, b)| {
                        let mut c = Cursor {
                            s: &text[..b],
                            p: a,
                        };
                        let head = c.next_line();
                        if !head.starts_with("Tree=") {
                            return Err(Error::Parse(format!(
                                "Model format error, expect a tree here. met {head}"
                            )));
                        }
                        Tree::from_model_str(&text[c.p..b]).map(|(t, _)| t)
                    })
                    .collect::<Result<Vec<Tree>>>()?
            }
            None => {
                let mut models = Vec::new();
                while !cur.done() {
                    let line = cur.peek_line();
                    if line.is_empty() {
                        cur.skip_newline();
                        continue;
                    }
                    if !line.starts_with("Tree=") {
                        break;
                    }
                    cur.next_line();
                    let (t, used) = Tree::from_model_str(&text[cur.p..])?;
                    cur.p += used;
                    models.push(t);
                }
                models
            }
        };
        for t in &models {
            if t.max_feature_used() > max_feature_idx {
                return Err(Error::Parse(
                    "tree uses a feature beyond max_feature_idx".into(),
                ));
            }
        }

        // parameters: ... end of parameters
        let mut loaded_parameter = String::new();
        let mut linear_tree = false;
        let mut in_params = false;
        while !cur.done() {
            let line = cur.next_line();
            if line == "parameters:" {
                in_params = true;
            } else if line == "end of parameters" {
                break;
            } else if in_params && !line.is_empty() {
                loaded_parameter.push_str(line);
                loaded_parameter.push('\n');
                if let Some(rest) = line.strip_prefix("[linear_tree: ") {
                    linear_tree = rest.starts_with('1');
                }
            }
        }
        // parser: ... end of parser (optional), then pandas_categorical
        let mut parser_config = String::new();
        let mut pandas_categorical = None;
        let mut in_parser = false;
        while !cur.done() {
            let line = cur.next_line();
            if line == "parser:" {
                in_parser = true;
            } else if line == "end of parser" {
                in_parser = false;
            } else if in_parser && !line.is_empty() {
                parser_config.push_str(line);
                parser_config.push('\n');
            } else if let Some(v) = line.strip_prefix("pandas_categorical:") {
                pandas_categorical = Some(v.to_string());
            }
        }

        Ok(Gbdt {
            models,
            num_class,
            num_tree_per_iteration,
            label_idx,
            max_feature_idx,
            average_output,
            feature_names,
            feature_infos,
            monotone_constraints,
            objective,
            linear_tree,
            loaded_parameter,
            parser_config,
            pandas_categorical,
            version,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    const MODEL: &str = "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n\
max_feature_idx=1\nobjective=binary sigmoid:1\nfeature_names=a b\nfeature_infos=[0:1] [0:1]\n\n\
Tree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=1\nsplit_gain=1\nthreshold=0.5\ndecision_type=2\n\
left_child=-1\nright_child=-2\nleaf_value=-1 1\nleaf_weight=1 1\nleaf_count=5 5\ninternal_value=0\n\
internal_weight=2\ninternal_count=10\nis_linear=0\nshrinkage=1\n\n\nend of trees\n\n\
parameters:\n[boosting: gbdt]\n[linear_tree: 0]\nend of parameters\n\npandas_categorical:null\n";

    #[test]
    fn loads_header_trees_and_trailer() {
        let m = Gbdt::from_model_string(MODEL).unwrap();
        assert_eq!(m.num_trees(), 1);
        assert_eq!(m.feature_names(), ["a", "b"]);
        assert_eq!(m.objective().unwrap().name, "binary");
        assert_eq!(m.loaded_parameter(), "[boosting: gbdt]\n[linear_tree: 0]\n");
        assert_eq!(m.pandas_categorical(), Some("null"));
        assert_eq!(m.version, "v4");
    }

    #[test]
    fn crlf_line_endings_are_accepted() {
        let m = Gbdt::from_model_string(&MODEL.replace('\n', "\r\n")).unwrap();
        assert_eq!(m.num_trees(), 1);
    }

    #[test]
    fn missing_header_field_is_an_error() {
        let bad = MODEL.replace("label_index=0\n", "");
        assert!(Gbdt::from_model_string(&bad).is_err());
    }

    #[test]
    fn feature_out_of_range_is_rejected() {
        let bad = MODEL.replace("split_feature=1", "split_feature=7");
        assert!(Gbdt::from_model_string(&bad).is_err());
    }

    #[test]
    fn iteration_range_clamps_like_upstream() {
        let m = Gbdt::from_model_string(MODEL).unwrap();
        assert_eq!(m.iteration_range(0, -1), (0, 1));
        assert_eq!(m.iteration_range(5, -1), (1, 0));
        assert_eq!(m.iteration_range(-3, 10), (0, 1));
    }
}
