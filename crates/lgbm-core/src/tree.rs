//! Decision tree: storage, text (de)serialization, prediction and TreeSHAP.
//!
//! upstream: include/LightGBM/tree.h, src/io/tree.cpp (v4.7.0)

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::meta::{K_ZERO_THRESHOLD, MissingType};
use crate::utils::common::{
    ParseNum, atof_legacy, find_in_bitset, madd, parse_array, parse_array_fast_f64, parse_scalar,
};
use crate::utils::format::{format_g, join_display, join_g, join_g17};

pub const K_CATEGORICAL_MASK: i8 = 1;
pub const K_DEFAULT_LEFT_MASK: i8 = 2;

/// A trained regression tree. Internal nodes are indexed `0..num_leaves-1`;
/// a negative child `c` refers to leaf `!c`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub num_leaves: usize,
    pub num_cat: usize,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub split_feature: Vec<i32>,
    pub split_gain: Vec<f32>,
    pub threshold: Vec<f64>,
    pub decision_type: Vec<i8>,
    pub cat_boundaries: Vec<i32>,
    pub cat_threshold: Vec<u32>,
    pub leaf_value: Vec<f64>,
    pub leaf_weight: Vec<f64>,
    pub leaf_count: Vec<i32>,
    pub internal_value: Vec<f64>,
    pub internal_weight: Vec<f64>,
    pub internal_count: Vec<i32>,
    pub shrinkage: f64,
    pub is_linear: bool,
    pub leaf_const: Vec<f64>,
    pub leaf_coeff: Vec<Vec<f64>>,
    pub leaf_features: Vec<Vec<i32>>,
    /// Depth of the deepest leaf; computed lazily for SHAP (`-1` until then).
    max_depth: i32,
    // Training-time state (not serialized).
    pub leaf_parent: Vec<i32>,
    pub leaf_depth: Vec<i32>,
    pub split_feature_inner: Vec<i32>,
    pub threshold_in_bin: Vec<u32>,
}

/// Reusable TreeSHAP working memory (one per thread).
#[derive(Default)]
pub struct ShapScratch {
    path: Vec<PathElement>,
}

#[derive(Clone, Copy, Default)]
struct PathElement {
    feature_index: i32,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

#[inline]
pub fn is_zero(fval: f64) -> bool {
    (-K_ZERO_THRESHOLD..=K_ZERO_THRESHOLD).contains(&fval)
}

#[inline]
fn get_decision_type(decision_type: i8, mask: i8) -> bool {
    (decision_type & mask) > 0
}

#[inline]
fn get_missing_type(decision_type: i8) -> MissingType {
    MissingType::from_bits((decision_type >> 2) & 3)
}

impl Tree {
    /// Parse one tree block (the lines after `Tree=<i>`) from a model file.
    /// Returns the tree and the number of bytes consumed.
    ///
    /// upstream: src/io/tree.cpp `Tree::Tree(const char* str, size_t* used_len)`
    pub fn from_model_str(s: &str) -> Result<(Tree, usize)> {
        const MAX_NUM_LINE: usize = 22;
        let bytes = s.as_bytes();
        let mut kv: HashMap<&str, &str> = HashMap::new();
        let mut p = 0usize;
        let mut read_line = 0;
        while read_line < MAX_NUM_LINE && p < bytes.len() {
            if bytes[p] == b'\r' || bytes[p] == b'\n' {
                break;
            }
            let start = p;
            while p < bytes.len() && bytes[p] != b'\r' && bytes[p] != b'\n' {
                p += 1;
            }
            let line = &s[start..p];
            let (key, val) = line
                .split_once('=')
                .ok_or_else(|| Error::Parse(format!("tree line without '=': {line}")))?;
            kv.insert(key, val);
            read_line += 1;
            if p < bytes.len() && bytes[p] == b'\r' {
                p += 1;
            }
            if p < bytes.len() && bytes[p] == b'\n' {
                p += 1;
            }
        }
        let used = p;

        let req = |k: &str| {
            kv.get(k)
                .copied()
                .ok_or_else(|| Error::Parse(format!("tree model string should contain {k} field")))
        };
        let num_leaves: usize = parse_scalar(req("num_leaves")?)?;
        let num_cat: usize = parse_scalar(req("num_cat")?)?;
        let leaf_value: Vec<f64> = parse_array(req("leaf_value")?, num_leaves)?;
        let shrinkage = match kv.get("shrinkage") {
            Some(v) => atof_legacy(v)
                .ok_or_else(|| Error::Parse(format!("cannot parse shrinkage {v:?}")))?,
            None => 1.0,
        };
        let is_linear = match kv.get("is_linear") {
            Some(v) => parse_scalar::<i32>(v)? != 0,
            None => false,
        };
        let leaf_count = opt_array(&kv, "leaf_count", num_leaves)?;

        let mut tree = Tree {
            num_leaves,
            num_cat,
            leaf_value,
            leaf_count,
            shrinkage,
            is_linear,
            left_child: Vec::new(),
            right_child: Vec::new(),
            split_feature: Vec::new(),
            split_gain: Vec::new(),
            threshold: Vec::new(),
            decision_type: Vec::new(),
            cat_boundaries: Vec::new(),
            cat_threshold: Vec::new(),
            leaf_weight: Vec::new(),
            internal_value: Vec::new(),
            internal_weight: Vec::new(),
            internal_count: Vec::new(),
            leaf_const: Vec::new(),
            leaf_coeff: Vec::new(),
            leaf_features: Vec::new(),
            max_depth: -1,
            leaf_parent: Vec::new(),
            leaf_depth: Vec::new(),
            split_feature_inner: Vec::new(),
            threshold_in_bin: Vec::new(),
        };
        if num_leaves <= 1 && !is_linear {
            tree.max_depth = 0;
            return Ok((tree, used));
        }
        let n_int = num_leaves.saturating_sub(1);
        tree.left_child = parse_array(req("left_child")?, n_int)?;
        tree.right_child = parse_array(req("right_child")?, n_int)?;
        tree.split_feature = parse_array(req("split_feature")?, n_int)?;
        tree.threshold = parse_array(req("threshold")?, n_int)?;
        tree.split_gain = match kv.get("split_gain") {
            Some(v) => parse_array_fast_f64(v, n_int)?
                .into_iter()
                .map(|x| x as f32)
                .collect(),
            None => vec![0.0; n_int],
        };
        tree.internal_count = opt_array(&kv, "internal_count", n_int)?;
        tree.internal_value = opt_fast_f64(&kv, "internal_value", n_int)?;
        tree.internal_weight = opt_fast_f64(&kv, "internal_weight", n_int)?;
        tree.leaf_weight = opt_array(&kv, "leaf_weight", num_leaves)?;
        tree.decision_type = opt_array(&kv, "decision_type", n_int)?;

        if is_linear {
            tree.leaf_const = opt_array(&kv, "leaf_const", num_leaves)?;
            tree.leaf_coeff = vec![Vec::new(); num_leaves];
            tree.leaf_features = vec![Vec::new(); num_leaves];
            if let Some(nf) = kv.get("num_features") {
                let num_feat: Vec<usize> = parse_array(nf, num_leaves)?;
                let total: usize = num_feat.iter().sum();
                let feats: Option<Vec<i32>> = kv
                    .get("leaf_features")
                    .map(|v| parse_array(v, total))
                    .transpose()?;
                let coeffs: Option<Vec<f64>> = kv
                    .get("leaf_coeff")
                    .map(|v| parse_array(v, total))
                    .transpose()?;
                let mut off = 0;
                for (i, &n) in num_feat.iter().enumerate() {
                    if n > 0 {
                        if let Some(f) = &feats {
                            tree.leaf_features[i] = f[off..off + n].to_vec();
                        }
                        if let Some(c) = &coeffs {
                            tree.leaf_coeff[i] = c[off..off + n].to_vec();
                        }
                    }
                    off += n;
                }
            }
        }

        if num_cat > 0 {
            tree.cat_boundaries = parse_array(req("cat_boundaries")?, num_cat + 1)?;
            let n = *tree.cat_boundaries.last().unwrap_or(&0);
            let n =
                usize::try_from(n).map_err(|_| Error::Parse("negative cat_boundaries".into()))?;
            tree.cat_threshold = parse_array(req("cat_threshold")?, n)?;
        }
        tree.validate()?;
        tree.recompute_max_depth();
        Ok((tree, used))
    }

    /// Structural checks so prediction can index without bounds surprises.
    fn validate(&self) -> Result<()> {
        let n_int = self.num_leaves.saturating_sub(1);
        let check_child = |c: i32| {
            let ok = if c >= 0 {
                (c as usize) < n_int
            } else {
                ((!c) as usize) < self.num_leaves
            };
            if ok {
                Ok(())
            } else {
                Err(Error::Parse(format!("child index {c} out of range")))
            }
        };
        for i in 0..n_int {
            check_child(self.left_child[i])?;
            check_child(self.right_child[i])?;
            if self.split_feature[i] < 0 {
                return Err(Error::Parse("negative split_feature".into()));
            }
            if get_decision_type(self.decision_type[i], K_CATEGORICAL_MASK) {
                let ci = self.threshold[i] as i64;
                if ci < 0 || ci as usize >= self.num_cat {
                    return Err(Error::Parse(
                        "categorical threshold index out of range".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// upstream: `Tree::Tree(int max_leaves, bool track_branch_features, bool is_linear)`
    pub fn new_training(max_leaves: usize) -> Tree {
        let n_int = max_leaves.saturating_sub(1);
        Tree {
            num_leaves: 1,
            num_cat: 0,
            left_child: vec![0; n_int],
            right_child: vec![0; n_int],
            split_feature: vec![0; n_int],
            split_gain: vec![0.0; n_int],
            threshold: vec![0.0; n_int],
            decision_type: vec![0; n_int],
            cat_boundaries: vec![0],
            cat_threshold: Vec::new(),
            leaf_value: vec![0.0; max_leaves],
            leaf_weight: vec![0.0; max_leaves],
            leaf_count: vec![0; max_leaves],
            internal_value: vec![0.0; n_int],
            internal_weight: vec![0.0; n_int],
            internal_count: vec![0; n_int],
            shrinkage: 1.0,
            is_linear: false,
            leaf_const: Vec::new(),
            leaf_coeff: Vec::new(),
            leaf_features: Vec::new(),
            max_depth: -1,
            leaf_parent: {
                let mut v = vec![0; max_leaves];
                if let Some(first) = v.first_mut() {
                    *first = -1;
                }
                v
            },
            leaf_depth: vec![0; max_leaves],
            split_feature_inner: vec![0; n_int],
            threshold_in_bin: vec![0; n_int],
        }
    }

    /// upstream: `Tree::MaybeRoundToZero`
    #[inline]
    pub fn maybe_round_to_zero(v: f64) -> f64 {
        if is_zero(v) { 0.0 } else { v }
    }

    /// upstream: `Tree::SetLeafOutput`
    pub fn set_leaf_output(&mut self, leaf: usize, output: f64) {
        self.leaf_value[leaf] = Tree::maybe_round_to_zero(output);
    }

    /// upstream: private inline `Tree::Split` (shared by numerical and categorical splits)
    #[allow(clippy::too_many_arguments)]
    fn split_common(
        &mut self,
        leaf: usize,
        feature: i32,
        real_feature: i32,
        left_value: f64,
        right_value: f64,
        left_cnt: i32,
        right_cnt: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
    ) -> usize {
        let new_node = self.num_leaves - 1;
        let nl = self.num_leaves;
        let parent = self.leaf_parent[leaf];
        if parent >= 0 {
            let p = parent as usize;
            if self.left_child[p] == !(leaf as i32) {
                self.left_child[p] = new_node as i32;
            } else {
                self.right_child[p] = new_node as i32;
            }
        }
        self.split_feature_inner[new_node] = feature;
        self.split_feature[new_node] = real_feature;
        self.split_gain[new_node] = gain;
        self.left_child[new_node] = !(leaf as i32);
        self.right_child[new_node] = !(nl as i32);
        self.leaf_parent[leaf] = new_node as i32;
        self.leaf_parent[nl] = new_node as i32;
        self.internal_weight[new_node] = left_weight + right_weight;
        self.internal_value[new_node] = self.leaf_value[leaf];
        self.internal_count[new_node] = left_cnt + right_cnt;
        self.leaf_value[leaf] = if left_value.is_nan() { 0.0 } else { left_value };
        self.leaf_weight[leaf] = left_weight;
        self.leaf_count[leaf] = left_cnt;
        self.leaf_value[nl] = if right_value.is_nan() {
            0.0
        } else {
            right_value
        };
        self.leaf_weight[nl] = right_weight;
        self.leaf_count[nl] = right_cnt;
        self.leaf_depth[nl] = self.leaf_depth[leaf] + 1;
        self.leaf_depth[leaf] += 1;
        new_node
    }

    /// upstream: `Tree::Split` (numerical). Returns the new (right) leaf index.
    #[allow(clippy::too_many_arguments)]
    pub fn split_numerical(
        &mut self,
        leaf: usize,
        feature: i32,
        real_feature: i32,
        threshold_bin: u32,
        threshold_double: f64,
        left_value: f64,
        right_value: f64,
        left_cnt: i32,
        right_cnt: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: MissingType,
        default_left: bool,
    ) -> usize {
        let node = self.split_common(
            leaf,
            feature,
            real_feature,
            left_value,
            right_value,
            left_cnt,
            right_cnt,
            left_weight,
            right_weight,
            gain,
        );
        let mut dt: i8 = 0;
        if default_left {
            dt |= K_DEFAULT_LEFT_MASK;
        }
        dt = (dt & 3) | ((missing_type as i8) << 2);
        self.decision_type[node] = dt;
        self.threshold_in_bin[node] = threshold_bin;
        self.threshold[node] = threshold_double;
        self.num_leaves += 1;
        self.num_leaves - 1
    }

    /// upstream: `Tree::SplitCategorical`. `bitset` holds the categories that go left, as a
    /// bitset over category values (`Common::ConstructBitset`).
    #[allow(clippy::too_many_arguments)]
    pub fn split_categorical(
        &mut self,
        leaf: usize,
        feature: i32,
        real_feature: i32,
        bitset: &[u32],
        left_value: f64,
        right_value: f64,
        left_cnt: i32,
        right_cnt: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: MissingType,
    ) -> usize {
        let node = self.split_common(
            leaf,
            feature,
            real_feature,
            left_value,
            right_value,
            left_cnt,
            right_cnt,
            left_weight,
            right_weight,
            gain,
        );
        let dt: i8 = K_CATEGORICAL_MASK | ((missing_type as i8) << 2);
        self.decision_type[node] = dt;
        let cat_idx = self.num_cat;
        self.threshold_in_bin[node] = cat_idx as u32;
        self.threshold[node] = cat_idx as f64;
        self.num_cat += 1;
        let last = self.cat_boundaries.last().copied().unwrap_or(0);
        self.cat_boundaries.push(last + bitset.len() as i32);
        self.cat_threshold.extend_from_slice(bitset);
        self.num_leaves += 1;
        self.num_leaves - 1
    }

    /// upstream: `Tree::Shrinkage`
    pub fn apply_shrinkage(&mut self, rate: f64) {
        let n = self.num_leaves;
        for i in 0..n.saturating_sub(1) {
            self.leaf_value[i] = Tree::maybe_round_to_zero(self.leaf_value[i] * rate);
            self.internal_value[i] = Tree::maybe_round_to_zero(self.internal_value[i] * rate);
        }
        self.leaf_value[n - 1] = Tree::maybe_round_to_zero(self.leaf_value[n - 1] * rate);
        if self.is_linear {
            for i in 0..n {
                self.leaf_const[i] = Tree::maybe_round_to_zero(self.leaf_const[i] * rate);
                for c in &mut self.leaf_coeff[i] {
                    *c = Tree::maybe_round_to_zero(*c * rate);
                }
            }
        }
        self.shrinkage *= rate;
    }

    /// upstream: `Tree::AddBias`
    pub fn add_bias(&mut self, val: f64) {
        let n = self.num_leaves;
        for i in 0..n.saturating_sub(1) {
            self.leaf_value[i] = Tree::maybe_round_to_zero(self.leaf_value[i] + val);
            self.internal_value[i] = Tree::maybe_round_to_zero(self.internal_value[i] + val);
        }
        self.leaf_value[n - 1] = Tree::maybe_round_to_zero(self.leaf_value[n - 1] + val);
        if self.is_linear {
            for i in 0..n {
                self.leaf_const[i] = Tree::maybe_round_to_zero(self.leaf_const[i] + val);
            }
        }
        self.shrinkage = 1.0;
    }

    /// upstream: `Tree::AsConstantTree`
    pub fn as_constant_tree(&mut self, val: f64, count: i32) {
        self.num_leaves = 1;
        self.shrinkage = 1.0;
        self.leaf_value[0] = val;
        if self.is_linear {
            self.leaf_const[0] = val;
        }
        self.leaf_count[0] = count;
    }

    /// Drop unused capacity after training so the tree serializes like upstream.
    pub fn finish_training(&mut self) {
        let n = self.num_leaves;
        let ni = n.saturating_sub(1);
        self.left_child.truncate(ni);
        self.right_child.truncate(ni);
        self.split_feature.truncate(ni);
        self.split_gain.truncate(ni);
        self.threshold.truncate(ni);
        self.decision_type.truncate(ni);
        self.internal_value.truncate(ni);
        self.internal_weight.truncate(ni);
        self.internal_count.truncate(ni);
        self.split_feature_inner.truncate(ni);
        self.threshold_in_bin.truncate(ni);
        self.leaf_value.truncate(n);
        self.leaf_weight.truncate(n);
        self.leaf_count.truncate(n);
        self.leaf_parent.truncate(n);
        self.leaf_depth.truncate(n);
        if n <= 1 {
            // upstream single-leaf trees serialize without node arrays
            self.left_child.clear();
            self.right_child.clear();
        }
        self.recompute_max_depth();
    }

    /// upstream: `Tree::ToString`
    pub fn to_model_string(&self) -> String {
        use std::fmt::Write;
        let n_int = self.num_leaves.saturating_sub(1);
        let gains: Vec<String> = self.split_gain[..n_int.min(self.split_gain.len())]
            .iter()
            .map(|g| format_g(f64::from(*g), 6))
            .collect();
        let mut s = String::with_capacity(64 * self.num_leaves + 512);
        let _ = writeln!(s, "num_leaves={}", self.num_leaves);
        let _ = writeln!(s, "num_cat={}", self.num_cat);
        let _ = writeln!(
            s,
            "split_feature={}",
            join_display(head(&self.split_feature, n_int))
        );
        let _ = writeln!(s, "split_gain={}", gains.join(" "));
        let _ = writeln!(s, "threshold={}", join_g17(head(&self.threshold, n_int)));
        let _ = writeln!(
            s,
            "decision_type={}",
            join_display(head(&self.decision_type, n_int))
        );
        let _ = writeln!(
            s,
            "left_child={}",
            join_display(head(&self.left_child, n_int))
        );
        let _ = writeln!(
            s,
            "right_child={}",
            join_display(head(&self.right_child, n_int))
        );
        let _ = writeln!(s, "leaf_value={}", join_g17(&self.leaf_value));
        let _ = writeln!(s, "leaf_weight={}", join_g17(&self.leaf_weight));
        let _ = writeln!(s, "leaf_count={}", join_display(&self.leaf_count));
        let _ = writeln!(
            s,
            "internal_value={}",
            join_g(head(&self.internal_value, n_int))
        );
        let _ = writeln!(
            s,
            "internal_weight={}",
            join_g(head(&self.internal_weight, n_int))
        );
        let _ = writeln!(
            s,
            "internal_count={}",
            join_display(head(&self.internal_count, n_int))
        );
        if self.num_cat > 0 {
            let _ = writeln!(s, "cat_boundaries={}", join_display(&self.cat_boundaries));
            let _ = writeln!(s, "cat_threshold={}", join_display(&self.cat_threshold));
        }
        let _ = writeln!(s, "is_linear={}", i32::from(self.is_linear));
        if self.is_linear {
            let _ = writeln!(s, "leaf_const={}", join_g17(&self.leaf_const));
            let num_feat: Vec<usize> = self.leaf_coeff.iter().map(Vec::len).collect();
            let _ = writeln!(s, "num_features={}", join_display(&num_feat));
            s.push_str("leaf_features=");
            for (i, f) in self.leaf_features.iter().enumerate() {
                if num_feat[i] > 0 {
                    s.push_str(&join_display(f));
                    s.push(' ');
                }
                s.push(' ');
            }
            s.push('\n');
            s.push_str("leaf_coeff=");
            for (i, c) in self.leaf_coeff.iter().enumerate() {
                if num_feat[i] > 0 {
                    s.push_str(&join_g17(c));
                    s.push(' ');
                }
                s.push(' ');
            }
            s.push('\n');
        }
        let _ = writeln!(s, "shrinkage={}", format_g(self.shrinkage, 6));
        s.push('\n');
        s
    }

    /// upstream: `Tree::ToJSON` (stream precision 17, i.e. `%.17g`).
    pub fn to_json(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "\"num_leaves\":{},", self.num_leaves);
        let _ = writeln!(s, "\"num_cat\":{},", self.num_cat);
        let _ = writeln!(s, "\"shrinkage\":{},", g17(self.shrinkage));
        if self.num_leaves == 1 {
            s.push_str("\"tree_structure\":{");
            let _ = writeln!(s, "\"leaf_value\":{}, ", g17(self.leaf_value[0]));
            let count = self.leaf_count.first().copied().unwrap_or(0);
            if self.is_linear {
                let _ = writeln!(s, "\"leaf_count\":{count}, ");
                s.push_str(&self.linear_model_to_json(0));
            } else {
                let _ = write!(s, "\"leaf_count\":{count}");
            }
            s.push_str("}\n");
        } else {
            s.push_str("\"tree_structure\":");
            self.node_to_json(0, &mut s);
            s.push('\n');
        }
        s
    }

    fn linear_model_to_json(&self, index: usize) -> String {
        let mut s = format!("\"leaf_const\":{},\n", g17(self.leaf_const[index]));
        let feats = &self.leaf_features[index];
        if feats.is_empty() {
            s.push_str("\"leaf_features\":[],\n\"leaf_coeff\":[]\n");
        } else {
            let f: Vec<String> = feats.iter().map(ToString::to_string).collect();
            let c: Vec<String> = self.leaf_coeff[index].iter().map(|v| g17(*v)).collect();
            s.push_str(&format!("\"leaf_features\":[{}], \n", f.join(", ")));
            s.push_str(&format!("\"leaf_coeff\":[{}]\n", c.join(", ")));
        }
        s
    }

    /// upstream: `Tree::NodeToJSON`
    fn node_to_json(&self, node: i32, s: &mut String) {
        use std::fmt::Write;
        if node >= 0 {
            let i = node as usize;
            let dt = self.decision_type[i];
            s.push_str("{\n");
            let _ = writeln!(s, "\"split_index\":{i},");
            let _ = writeln!(s, "\"split_feature\":{},", self.split_feature[i]);
            let _ = writeln!(
                s,
                "\"split_gain\":{},",
                g17(f64::from(avoid_inf_f32(self.split_gain[i])))
            );
            if get_decision_type(dt, K_CATEGORICAL_MASK) {
                let cat_idx = self.threshold[i] as usize;
                let (lo, hi) = (
                    self.cat_boundaries[cat_idx] as usize,
                    self.cat_boundaries[cat_idx + 1] as usize,
                );
                let bits = &self.cat_threshold[lo..hi];
                let cats: Vec<String> = (0..(hi - lo) * 32)
                    .filter(|&c| find_in_bitset(bits, c as u32))
                    .map(|c| c.to_string())
                    .collect();
                let _ = writeln!(s, "\"threshold\":\"{}\",", cats.join("||"));
                s.push_str("\"decision_type\":\"==\",\n");
            } else {
                let _ = writeln!(s, "\"threshold\":{},", g17(avoid_inf(self.threshold[i])));
                s.push_str("\"decision_type\":\"<=\",\n");
            }
            let dl = get_decision_type(dt, K_DEFAULT_LEFT_MASK);
            let _ = writeln!(s, "\"default_left\":{dl},");
            let mt = match get_missing_type(dt) {
                MissingType::None => "None",
                MissingType::Zero => "Zero",
                MissingType::NaN => "NaN",
            };
            let _ = writeln!(s, "\"missing_type\":\"{mt}\",");
            let _ = writeln!(s, "\"internal_value\":{},", g17(self.internal_value[i]));
            let _ = writeln!(s, "\"internal_weight\":{},", g17(self.internal_weight[i]));
            let _ = writeln!(s, "\"internal_count\":{},", self.internal_count[i]);
            s.push_str("\"left_child\":");
            self.node_to_json(self.left_child[i], s);
            s.push_str(",\n\"right_child\":");
            self.node_to_json(self.right_child[i], s);
            s.push_str("\n}");
        } else {
            let i = (!node) as usize;
            s.push_str("{\n");
            let _ = writeln!(s, "\"leaf_index\":{i},");
            let _ = writeln!(s, "\"leaf_value\":{},", g17(self.leaf_value[i]));
            let _ = writeln!(
                s,
                "\"leaf_weight\":{},",
                g17(self.leaf_weight.get(i).copied().unwrap_or(0.0))
            );
            if self.is_linear {
                let _ = writeln!(s, "\"leaf_count\":{},", self.leaf_count[i]);
                s.push_str(&self.linear_model_to_json(i));
            } else {
                let _ = writeln!(s, "\"leaf_count\":{}", self.leaf_count[i]);
            }
            s.push('}');
        }
    }

    #[inline]
    pub fn max_feature_used(&self) -> i32 {
        let mut m = self.split_feature.iter().copied().max().unwrap_or(-1);
        for f in &self.leaf_features {
            m = m.max(f.iter().copied().max().unwrap_or(-1));
        }
        m
    }

    #[inline]
    fn numerical_decision(&self, mut fval: f64, node: usize) -> i32 {
        let dt = self.decision_type[node];
        let missing_type = get_missing_type(dt);
        if fval.is_nan() && missing_type != MissingType::NaN {
            fval = 0.0;
        }
        if (missing_type == MissingType::Zero && is_zero(fval))
            || (missing_type == MissingType::NaN && fval.is_nan())
        {
            return if get_decision_type(dt, K_DEFAULT_LEFT_MASK) {
                self.left_child[node]
            } else {
                self.right_child[node]
            };
        }
        if fval <= self.threshold[node] {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }

    #[inline]
    fn categorical_decision(&self, fval: f64, node: usize) -> i32 {
        if fval.is_nan() {
            return self.right_child[node];
        }
        // C++ static_cast<int>: truncation toward zero.
        let int_fval = fval as i32;
        if int_fval < 0 {
            return self.right_child[node];
        }
        let cat_idx = self.threshold[node] as usize;
        let lo = self.cat_boundaries[cat_idx] as usize;
        let hi = self.cat_boundaries[cat_idx + 1] as usize;
        if find_in_bitset(&self.cat_threshold[lo..hi], int_fval as u32) {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }

    #[inline]
    fn decision(&self, fval: f64, node: usize) -> i32 {
        if get_decision_type(self.decision_type[node], K_CATEGORICAL_MASK) {
            self.categorical_decision(fval, node)
        } else {
            self.numerical_decision(fval, node)
        }
    }

    /// Leaf reached by `features` (dense, indexed by original feature id).
    #[inline]
    pub fn get_leaf(&self, features: &[f64]) -> usize {
        let mut node: i32 = 0;
        if self.num_cat > 0 {
            while node >= 0 {
                let n = node as usize;
                node = self.decision(features[self.split_feature[n] as usize], n);
            }
        } else {
            while node >= 0 {
                let n = node as usize;
                node = self.numerical_decision(features[self.split_feature[n] as usize], n);
            }
        }
        (!node) as usize
    }

    #[inline]
    pub fn predict(&self, features: &[f64]) -> f64 {
        if self.is_linear {
            let leaf = if self.num_leaves > 1 {
                self.get_leaf(features)
            } else {
                0
            };
            let mut output = self.leaf_const[leaf];
            for (&f, &c) in self.leaf_features[leaf].iter().zip(&self.leaf_coeff[leaf]) {
                let v = features[f as usize];
                if v.is_nan() {
                    return self.leaf_value[leaf];
                }
                output = madd(c, v, output);
            }
            output
        } else if self.num_leaves > 1 {
            self.leaf_value[self.get_leaf(features)]
        } else {
            self.leaf_value[0]
        }
    }

    #[inline]
    pub fn predict_leaf_index(&self, features: &[f64]) -> usize {
        if self.num_leaves > 1 {
            self.get_leaf(features)
        } else {
            0
        }
    }

    #[inline]
    fn data_count(&self, node: i32) -> f64 {
        if node >= 0 {
            f64::from(self.internal_count[node as usize])
        } else {
            f64::from(self.leaf_count[(!node) as usize])
        }
    }

    /// upstream: `Tree::ExpectedValue`
    ///
    /// The upstream macOS/arm64 build vectorizes this loop: whole blocks of 16 leaves use
    /// an unfused multiply then a sequential add, and the scalar tail is a fused
    /// multiply-add. We reproduce that rounding so SHAP bias terms are bit-identical.
    pub fn expected_value(&self) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_value[0];
        }
        let total = f64::from(self.internal_count[0]);
        let n = self.num_leaves;
        let vec_end = if cfg!(all(target_arch = "aarch64", target_os = "macos")) && n >= 16 {
            n & !15
        } else {
            0
        };
        let mut v = 0.0;
        for i in 0..vec_end {
            v += (f64::from(self.leaf_count[i]) / total) * self.leaf_value[i];
        }
        for i in vec_end..n {
            v = madd(f64::from(self.leaf_count[i]) / total, self.leaf_value[i], v);
        }
        v
    }

    /// upstream: `Tree::RecomputeMaxDepth`
    pub fn recompute_max_depth(&mut self) {
        if self.num_leaves == 1 {
            self.max_depth = 0;
            return;
        }
        let mut max_depth = 0;
        let mut stack = vec![(0i32, 0i32)];
        while let Some((node, depth)) = stack.pop() {
            if node < 0 {
                max_depth = max_depth.max(depth);
            } else {
                stack.push((self.left_child[node as usize], depth + 1));
                stack.push((self.right_child[node as usize], depth + 1));
            }
        }
        self.max_depth = max_depth;
    }

    pub fn max_depth(&self) -> i32 {
        self.max_depth
    }

    /// Adds this tree's SHAP values to `output` (`num_features + 1` slots, bias last).
    ///
    /// upstream: `Tree::PredictContrib`
    pub fn predict_contrib(&self, features: &[f64], num_features: usize, output: &mut [f64]) {
        let mut scratch = ShapScratch::default();
        self.predict_contrib_with(features, num_features, output, &mut scratch);
    }

    /// Same as [`Tree::predict_contrib`], reusing `scratch` to avoid per-call allocation.
    pub fn predict_contrib_with(
        &self,
        features: &[f64],
        num_features: usize,
        output: &mut [f64],
        scratch: &mut ShapScratch,
    ) {
        output[num_features] += self.expected_value();
        if self.num_leaves > 1 {
            let max_path_len = (self.max_depth.max(0) as usize) + 1;
            let need = max_path_len * (max_path_len + 1) / 2;
            if scratch.path.len() < need {
                scratch.path.resize(need, PathElement::default());
            }
            self.tree_shap(features, output, 0, 0, &mut scratch.path, 0, 1.0, 1.0, -1);
        }
    }

    /// upstream: `Tree::TreeSHAP`. `base` is the offset of the parent's unique path in `path`.
    #[allow(clippy::too_many_arguments)]
    fn tree_shap(
        &self,
        features: &[f64],
        phi: &mut [f64],
        node: i32,
        mut unique_depth: usize,
        path: &mut [PathElement],
        parent_base: usize,
        parent_zero_fraction: f64,
        parent_one_fraction: f64,
        parent_feature_index: i32,
    ) {
        let base = parent_base + unique_depth;
        if unique_depth > 0 {
            path.copy_within(parent_base..parent_base + unique_depth, base);
        }
        let up = &mut path[base..];
        extend_path(
            up,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );

        if node < 0 {
            let leaf_value = self.leaf_value[(!node) as usize];
            for i in 1..=unique_depth {
                let w = unwound_path_sum(up, unique_depth, i);
                let el = up[i];
                let fi = el.feature_index as usize;
                phi[fi] = madd(
                    w * (el.one_fraction - el.zero_fraction),
                    leaf_value,
                    phi[fi],
                );
            }
        } else {
            let n = node as usize;
            let split_feature = self.split_feature[n];
            let hot_index = self.decision(features[split_feature as usize], n);
            let cold_index = if hot_index == self.left_child[n] {
                self.right_child[n]
            } else {
                self.left_child[n]
            };
            let w = self.data_count(node);
            let hot_zero_fraction = self.data_count(hot_index) / w;
            let cold_zero_fraction = self.data_count(cold_index) / w;
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            let mut path_index = 0;
            while path_index <= unique_depth {
                if up[path_index].feature_index == split_feature {
                    break;
                }
                path_index += 1;
            }
            if path_index != unique_depth + 1 {
                incoming_zero_fraction = up[path_index].zero_fraction;
                incoming_one_fraction = up[path_index].one_fraction;
                unwind_path(up, unique_depth, path_index);
                unique_depth -= 1;
            }
            self.tree_shap(
                features,
                phi,
                hot_index,
                unique_depth + 1,
                path,
                base,
                hot_zero_fraction * incoming_zero_fraction,
                incoming_one_fraction,
                split_feature,
            );
            self.tree_shap(
                features,
                phi,
                cold_index,
                unique_depth + 1,
                path,
                base,
                cold_zero_fraction * incoming_zero_fraction,
                0.0,
                split_feature,
            );
        }
    }
}

fn opt_array<T: ParseNum + Default + Clone>(
    kv: &HashMap<&str, &str>,
    key: &str,
    n: usize,
) -> Result<Vec<T>> {
    match kv.get(key) {
        Some(v) => parse_array(v, n),
        None => Ok(vec![T::default(); n]),
    }
}

fn g17(v: f64) -> String {
    format_g(v, 17)
}

/// upstream: `Common::AvoidInf(double)`
pub fn avoid_inf(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-1e300, 1e300)
    }
}

/// upstream: `Common::AvoidInf(float)`. Clamping at `1e38f` is equivalent to upstream's
/// comparisons against the double literal `1e38`.
pub fn avoid_inf_f32(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-1e38, 1e38)
    }
}

fn head<T>(v: &[T], n: usize) -> &[T] {
    &v[..n.min(v.len())]
}

fn opt_fast_f64(kv: &HashMap<&str, &str>, key: &str, n: usize) -> Result<Vec<f64>> {
    match kv.get(key) {
        Some(v) => parse_array_fast_f64(v, n),
        None => Ok(vec![0.0; n]),
    }
}

fn extend_path(
    up: &mut [PathElement],
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i32,
) {
    up[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    let d1 = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        up[i + 1].pweight += one_fraction * up[i].pweight * (i + 1) as f64 / d1;
        up[i].pweight = zero_fraction * up[i].pweight * (unique_depth - i) as f64 / d1;
    }
}

fn unwind_path(up: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = up[path_index].one_fraction;
    let zero_fraction = up[path_index].zero_fraction;
    let mut next_one_portion = up[unique_depth].pweight;
    let d1 = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = up[i].pweight;
            up[i].pweight = next_one_portion * d1 / ((i + 1) as f64 * one_fraction);
            next_one_portion = tmp - up[i].pweight * zero_fraction * (unique_depth - i) as f64 / d1;
        } else {
            up[i].pweight = (up[i].pweight * d1) / (zero_fraction * (unique_depth - i) as f64);
        }
    }
    for i in path_index..unique_depth {
        up[i].feature_index = up[i + 1].feature_index;
        up[i].zero_fraction = up[i + 1].zero_fraction;
        up[i].one_fraction = up[i + 1].one_fraction;
    }
}

fn unwound_path_sum(up: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = up[path_index].one_fraction;
    let zero_fraction = up[path_index].zero_fraction;
    let mut next_one_portion = up[unique_depth].pweight;
    let d1 = (unique_depth + 1) as f64;
    let mut total = 0.0;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * d1 / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion = madd(
                -(tmp * zero_fraction),
                (unique_depth - i) as f64 / d1,
                up[i].pweight,
            );
        } else {
            total += (up[i].pweight / zero_fraction) / ((unique_depth - i) as f64 / d1);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    // x0 <= 0.5 -> leaf0 (1.0); else x1 is NaN-default-left: <= 2.0 -> leaf1 (2.0) else leaf2 (3.0)
    const STUMP: &str = "num_leaves=3\nnum_cat=0\nsplit_feature=0 1\nsplit_gain=1 1\n\
threshold=0.5 2\ndecision_type=2 10\nleft_child=-1 -2\nright_child=1 -3\n\
leaf_value=1 2 3\nleaf_weight=1 1 2\nleaf_count=10 10 20\ninternal_value=0 0\n\
internal_weight=0 0\ninternal_count=40 30\nis_linear=0\nshrinkage=1\n\n";

    #[test]
    fn parses_and_reports_consumed_bytes() {
        let (t, used) = Tree::from_model_str(STUMP).unwrap();
        assert_eq!(t.num_leaves, 3);
        assert_eq!(&STUMP[used..], "\n");
        assert_eq!(t.decision_type, vec![2, 10]);
    }

    #[test]
    fn numerical_routing_and_nan_default_left() {
        let (t, _) = Tree::from_model_str(STUMP).unwrap();
        assert_eq!(t.predict(&[0.0, 0.0]), 1.0);
        assert_eq!(t.predict(&[1.0, 1.0]), 2.0);
        assert_eq!(t.predict(&[1.0, 5.0]), 3.0);
        // missing_type NaN (bits 2-3 = 2) + default_left -> leaf1
        assert_eq!(t.predict(&[1.0, f64::NAN]), 2.0);
        // feature 0 has missing_type None: NaN is treated as 0.0 -> left
        assert_eq!(t.predict(&[f64::NAN, 9.0]), 1.0);
    }

    #[test]
    fn missing_field_is_an_error() {
        let bad = STUMP.replace("left_child=-1 -2\n", "");
        assert!(Tree::from_model_str(&bad).is_err());
    }

    #[test]
    fn out_of_range_child_is_rejected() {
        let bad = STUMP.replace("left_child=-1 -2", "left_child=-1 -9");
        assert!(Tree::from_model_str(&bad).is_err());
    }

    #[test]
    fn expected_value_is_count_weighted_mean() {
        let (t, _) = Tree::from_model_str(STUMP).unwrap();
        assert_eq!(
            t.expected_value(),
            (10.0 * 1.0 + 10.0 * 2.0 + 20.0 * 3.0) / 40.0
        );
    }

    #[test]
    fn shap_values_sum_to_prediction() {
        let (t, _) = Tree::from_model_str(STUMP).unwrap();
        assert_eq!(t.max_depth(), 2);
        for x in [[0.0, 0.0], [1.0, 1.0], [1.0, 5.0], [1.0, f64::NAN]] {
            let mut out = [0.0; 3];
            t.predict_contrib(&x, 2, &mut out);
            let s: f64 = out.iter().sum();
            assert!((s - t.predict(&x)).abs() < 1e-12, "{x:?}: {out:?}");
        }
    }

    #[test]
    fn single_leaf_tree() {
        let s = "num_leaves=1\nnum_cat=0\nleaf_value=0.25\nis_linear=0\nshrinkage=1\n\n";
        let (t, _) = Tree::from_model_str(s).unwrap();
        assert_eq!(t.predict(&[]), 0.25);
        assert_eq!(t.predict_leaf_index(&[]), 0);
    }

    #[test]
    fn categorical_bitset_routing() {
        // categories {1, 33} go left
        let s = "num_leaves=2\nnum_cat=1\nsplit_feature=0\nsplit_gain=1\nthreshold=0\n\
decision_type=1\nleft_child=-1\nright_child=-2\nleaf_value=-1 1\nleaf_count=1 1\n\
internal_count=2\ncat_boundaries=0 2\ncat_threshold=2 2\nis_linear=0\nshrinkage=1\n\n";
        let (t, _) = Tree::from_model_str(s).unwrap();
        assert_eq!(t.predict(&[1.0]), -1.0);
        assert_eq!(t.predict(&[33.0]), -1.0);
        assert_eq!(t.predict(&[1.9]), -1.0); // truncates to 1
        assert_eq!(t.predict(&[2.0]), 1.0);
        assert_eq!(t.predict(&[-1.0]), 1.0);
        assert_eq!(t.predict(&[f64::NAN]), 1.0);
        assert_eq!(t.predict(&[1000.0]), 1.0);
    }
}
