//! Linear leaves (`linear_tree`): after a tree is grown, each leaf fits a regularized
//! least-squares model on the numerical features its branch splits on.
//! upstream: src/treelearner/linear_tree_learner.cpp (`CalculateLinear`, `GetLeafMap`)

use super::*;
use crate::io::bin::BinType;
use crate::utils::common::madd;

/// upstream: `kZeroThreshold`; coefficients with a smaller magnitude are dropped.
const K_ZERO_THRESHOLD: f64 = 1e-35f32 as f64;

/// Per-dataset facts the linear fit needs, computed once.
pub(super) struct LinearState {
    /// Real feature index -> raw column, for numerical features with raw values.
    numeric_column: Vec<Option<usize>>,
    /// Real feature index -> whether its raw values contain a NaN.
    contains_nan: Vec<bool>,
}

impl LinearState {
    /// Requires the raw values, which the dataset keeps when `linear_tree` is set.
    pub(super) fn new(ds: &Dataset, metas: &[FeatureMeta]) -> Self {
        let l = &ds.layout;
        let num_real = l.used_feature_map.len();
        let mut numeric_column = vec![None; num_real];
        let mut contains_nan = vec![false; num_real];
        for (real, slot) in numeric_column.iter_mut().enumerate() {
            let inner = l.used_feature_map[real];
            if inner < 0 || metas[inner as usize].bin_type == BinType::Categorical {
                continue;
            }
            // `numeric_feature_map` and `raw_data` are indexed by inner feature.
            let col = ds.numeric_feature_map[inner as usize];
            if col < 0 || ds.raw_data.is_empty() {
                continue;
            }
            *slot = Some(col as usize);
            contains_nan[real] = ds.raw_data[col as usize].iter().any(|v| v.is_nan());
        }
        LinearState {
            numeric_column,
            contains_nan,
        }
    }

    pub(super) fn raw_column(&self, real: i32) -> Option<usize> {
        usize::try_from(real)
            .ok()
            .and_then(|r| self.numeric_column.get(r).copied().flatten())
    }

    pub(super) fn contains_nan(&self, real: i32) -> bool {
        usize::try_from(real)
            .ok()
            .and_then(|r| self.contains_nan.get(r).copied())
            .unwrap_or(false)
    }
}

/// The real features each leaf's branch splits on, in split order (upstream:
/// `Tree::branch_features`).
fn branch_features(tree: &Tree) -> Vec<Vec<i32>> {
    let mut out = vec![Vec::new(); tree.num_leaves];
    if tree.num_leaves <= 1 {
        return out;
    }
    let mut stack: Vec<(i32, Vec<i32>)> = vec![(0, Vec::new())];
    while let Some((node, path)) = stack.pop() {
        if node < 0 {
            out[(!node) as usize] = path;
            continue;
        }
        let i = node as usize;
        let mut next = path;
        next.push(tree.split_feature[i]);
        stack.push((tree.left_child[i], next.clone()));
        stack.push((tree.right_child[i], next));
    }
    out
}

/// Solves `a x = b` for a dense `n x n` matrix (row-major) with complete pivoting, as
/// upstream's `fullPivLu`. `None` when the matrix is singular.
fn solve_complete_pivot(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let mut col_perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Largest remaining entry of the active block.
        let (mut bi, mut bj, mut best) = (k, k, 0.0f64);
        for i in k..n {
            for j in k..n {
                let v = a[i * n + j].abs();
                if v > best {
                    (bi, bj, best) = (i, j, v);
                }
            }
        }
        if best == 0.0 || best.is_nan() {
            return None;
        }
        if bi != k {
            for j in 0..n {
                a.swap(k * n + j, bi * n + j);
            }
            b.swap(k, bi);
        }
        if bj != k {
            for i in 0..n {
                a.swap(i * n + k, i * n + bj);
            }
            col_perm.swap(k, bj);
        }
        let pivot = a[k * n + k];
        for i in k + 1..n {
            let factor = a[i * n + k] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in k..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    // Back substitution in the permuted unknowns, then undo the column permutation.
    let mut y = vec![0.0f64; n];
    for k in (0..n).rev() {
        let mut s = b[k];
        for j in k + 1..n {
            s -= a[k * n + j] * y[j];
        }
        y[k] = s / a[k * n + k];
    }
    let mut x = vec![0.0f64; n];
    for (k, &orig) in col_perm.iter().enumerate() {
        x[orig] = y[k];
    }
    Some(x)
}

impl SerialTreeLearner {
    /// Replaces each leaf's output with the linear model fitted on its rows.
    /// upstream: `LinearTreeLearner::CalculateLinear` (single-threaded accumulation order).
    /// The first tree only records the constant outputs, as upstream does.
    pub(crate) fn fit_linear(
        &self,
        tree: &mut Tree,
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) {
        let Some(state) = self.linear.as_ref() else {
            return;
        };
        let num_leaves = tree.num_leaves;
        tree.is_linear = true;
        tree.leaf_const = vec![0.0; num_leaves];
        tree.leaf_coeff = vec![Vec::new(); num_leaves];
        tree.leaf_features = vec![Vec::new(); num_leaves];
        if is_first_tree {
            tree.leaf_const[..num_leaves].copy_from_slice(&tree.leaf_value[..num_leaves]);
            return;
        }
        // Any split feature with a NaN puts the fit on the NaN-aware counting path.
        let has_nan = tree.split_feature[..num_leaves.saturating_sub(1)]
            .iter()
            .any(|&f| state.contains_nan(f));
        let branch = branch_features(tree);
        let lambda = self.cfg.linear_lambda;

        for leaf in 0..num_leaves {
            // Numerical raw columns of this leaf's branch, sorted and unique.
            let mut real: Vec<i32> = branch[leaf].clone();
            real.sort_unstable();
            real.dedup();
            let feats: Vec<(i32, usize)> = real
                .iter()
                .filter_map(|&f| state.raw_column(f).map(|c| (f, c)))
                .collect();
            let nf = feats.len();
            let dim = nf + 1;
            let mut xthx = vec![0.0f64; dim * dim];
            let mut xtg = vec![0.0f64; dim];
            let mut curr = vec![0.0f64; dim];
            let mut nonzero = 0usize;
            for &row in self.partition.indices_on_leaf(leaf) {
                let row = row as usize;
                let mut nan_found = false;
                for (k, &(_, col)) in feats.iter().enumerate() {
                    let v = self.ds.raw_data[col][row];
                    if has_nan {
                        if v.is_nan() {
                            nan_found = true;
                            break;
                        }
                        nonzero += 1;
                    }
                    curr[k] = f64::from(v);
                }
                if nan_found {
                    continue;
                }
                curr[nf] = 1.0;
                let h = f64::from(hessians[row]);
                let g = f64::from(gradients[row]);
                for f1 in 0..dim {
                    let x1 = curr[f1];
                    xtg[f1] += x1 * g;
                    let x1h = x1 * h;
                    for f2 in f1..dim {
                        xthx[f1 * dim + f2] += x1h * curr[f2];
                    }
                }
            }
            let total = if has_nan {
                nonzero
            } else {
                self.partition.leaf_count(leaf)
            };
            if total < dim {
                tree.leaf_const[leaf] = tree.leaf_value[leaf];
                continue;
            }
            // Symmetric matrix from the upper triangle, with lambda on the feature diagonal
            // (not on the constant term).
            let mut a = vec![0.0f64; dim * dim];
            for f1 in 0..dim {
                for f2 in f1..dim {
                    let mut v = xthx[f1 * dim + f2];
                    if f1 == f2 && f1 < nf {
                        v += lambda;
                    }
                    a[f1 * dim + f2] = v;
                    a[f2 * dim + f1] = v;
                }
            }
            let Some(sol) = solve_complete_pivot(a, xtg, dim) else {
                tree.leaf_const[leaf] = tree.leaf_value[leaf];
                continue;
            };
            let coeffs: Vec<f64> = sol.iter().map(|c| -c).collect();
            let mut kept_features = Vec::new();
            let mut kept_coeffs = Vec::new();
            for (k, &(f, _)) in feats.iter().enumerate() {
                if coeffs[k].abs() > K_ZERO_THRESHOLD {
                    kept_features.push(f);
                    kept_coeffs.push(coeffs[k]);
                }
            }
            tree.leaf_features[leaf] = kept_features;
            tree.leaf_coeff[leaf] = kept_coeffs;
            tree.leaf_const[leaf] = coeffs[nf];
        }
    }

    /// The output of a linear leaf for one training row, read from the raw values. A NaN
    /// in any leaf feature falls back to the constant leaf value (upstream: `Tree::Predict`).
    pub(super) fn linear_leaf_output(&self, tree: &Tree, leaf: usize, row: usize) -> f64 {
        let Some(state) = self.linear.as_ref() else {
            return tree.leaf_value[leaf];
        };
        let mut out = tree.leaf_const[leaf];
        for (&f, &c) in tree.leaf_features[leaf].iter().zip(&tree.leaf_coeff[leaf]) {
            let Some(col) = state.raw_column(f) else {
                return tree.leaf_value[leaf];
            };
            let v = f64::from(self.ds.raw_data[col][row]);
            if v.is_nan() {
                return tree.leaf_value[leaf];
            }
            out = madd(c, v, out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn complete_pivot_solve_matches_a_known_system() {
        // [2 1; 1 3] x = [3; 5] has x = [0.8, 1.4].
        let x = solve_complete_pivot(vec![2.0, 1.0, 1.0, 3.0], vec![3.0, 5.0], 2).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12, "{x:?}");
        assert!((x[1] - 1.4).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn singular_systems_are_reported() {
        assert!(solve_complete_pivot(vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 2.0], 2).is_none());
    }

    #[test]
    fn branch_features_collect_the_splits_above_each_leaf() {
        let mut t = Tree::new_training(3);
        t.num_leaves = 3;
        // Root splits on feature 4; its left child (leaf 0) is then split on feature 2.
        t.split_feature = vec![4, 2];
        t.left_child = vec![1, !0];
        t.right_child = vec![!2, !1];
        let b = branch_features(&t);
        assert_eq!(b[0], vec![4, 2]);
        assert_eq!(b[1], vec![4, 2]);
        assert_eq!(b[2], vec![4]);
    }
}
