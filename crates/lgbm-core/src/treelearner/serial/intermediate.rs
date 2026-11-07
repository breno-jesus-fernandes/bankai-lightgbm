//! `monotone_constraints_method=intermediate`: when a monotone split is made, the output
//! bounds of the leaves on the other side are tightened to the new children's outputs, and
//! the leaves whose bounds moved are returned so their best splits get recomputed.
//!
//! upstream: src/treelearner/monotone_constraints.hpp (`IntermediateLeafConstraints`,
//! `GoUpToFindLeavesToUpdate`, `GoDownToFindLeavesToUpdate`, `ShouldKeepGoingLeftRight`).
//! The port keeps upstream's names and recursion so the two files can be read side by side.

use crate::tree::{K_CATEGORICAL_MASK, Tree};

use super::super::feature_histogram::FeatureMeta;
use super::super::split_info::{K_MIN_SCORE, SplitInfo};

/// Output bounds of one leaf: `(min, max)`, as upstream's `BasicConstraint`.
pub(super) type Bounds = (f64, f64);

/// upstream: `BasicConstraintEntry::UpdateMinAndReturnBoolIfChanged`
fn raise_min(b: &mut Bounds, new_min: f64) -> bool {
    if new_min > b.0 {
        b.0 = new_min;
        return true;
    }
    false
}

/// upstream: `BasicConstraintEntry::UpdateMaxAndReturnBoolIfChanged`
fn lower_max(b: &mut Bounds, new_max: f64) -> bool {
    if new_max < b.1 {
        b.1 = new_max;
        return true;
    }
    false
}

/// upstream: `BasicConstraintEntry::UpdateMin` (`std::max(new_min, min)`)
fn update_min(b: &mut Bounds, new_min: f64) {
    b.0 = if new_min < b.0 { b.0 } else { new_min };
}

/// upstream: `BasicConstraintEntry::UpdateMax` (`std::min(new_max, max)`)
fn update_max(b: &mut Bounds, new_max: f64) {
    b.1 = if b.1 < new_max { b.1 } else { new_max };
}

/// The split being applied, as the constraint update reads it.
pub(super) struct AppliedSplit {
    /// upstream: `is_numerical_split`
    pub numerical: bool,
    /// Inner index of the split feature (upstream: `inner_feature_index`).
    pub feature_inner: i32,
    /// Bin threshold of the split (upstream: `split_info.threshold`).
    pub threshold: u32,
    pub left_output: f64,
    pub right_output: f64,
    /// upstream: `best_split_info.monotone_type`
    pub monotone_type: i8,
}

/// Read-only view of the tree and split searches during one update.
struct Ctx<'a> {
    tree: &'a Tree,
    metas: &'a [FeatureMeta],
    best: &'a [SplitInfo],
    split: &'a AppliedSplit,
}

/// The splits on the path from the original leaf up to the current node (upstream: the three
/// `*_of_splits_going_up_from_original_leaf` vectors).
#[derive(Default)]
struct Path {
    features: Vec<i32>,
    thresholds: Vec<u32>,
    was_right_child: Vec<bool>,
}

/// Per-tree state of the intermediate method (upstream: `IntermediateLeafConstraints`).
pub(super) struct IntermediateConstraints {
    /// upstream: `leaf_is_in_monotone_subtree_`
    in_monotone_subtree: Vec<bool>,
    /// upstream: `node_parent_`, indexed by internal node (`new_leaf - 1`).
    node_parent: Vec<i32>,
}

impl IntermediateConstraints {
    /// upstream: `IntermediateLeafConstraints(config, num_leaves)`
    pub fn new(num_leaves: usize) -> Self {
        Self {
            in_monotone_subtree: vec![false; num_leaves],
            node_parent: vec![-1; num_leaves.saturating_sub(1)],
        }
    }

    /// upstream: `IntermediateLeafConstraints::Reset`
    pub fn reset(&mut self) {
        self.in_monotone_subtree.fill(false);
        self.node_parent.fill(-1);
    }

    /// upstream: `IntermediateLeafConstraints::BeforeSplit`. Called before the tree and the
    /// data partition are split, so `leaf_parent(leaf)` is still the parent of `leaf`.
    pub fn before_split(&mut self, tree: &Tree, leaf: usize, new_leaf: usize, monotone_type: i8) {
        if monotone_type != 0 || self.in_monotone_subtree[leaf] {
            self.in_monotone_subtree[leaf] = true;
            self.in_monotone_subtree[new_leaf] = true;
        }
        self.node_parent[new_leaf - 1] = tree.leaf_parent[leaf];
    }

    /// upstream: `IntermediateLeafConstraints::Update`. Called after the split is applied to
    /// the tree. Tightens the bounds in `bounds` and returns the leaves whose bounds moved.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        tree: &Tree,
        bounds: &mut [Bounds],
        metas: &[FeatureMeta],
        best: &[SplitInfo],
        leaf: usize,
        new_leaf: usize,
        split: &AppliedSplit,
    ) -> Vec<usize> {
        let mut leaves_to_update = Vec::new();
        if !self.in_monotone_subtree[leaf] {
            return leaves_to_update;
        }
        // upstream: UpdateConstraintsWithOutputs
        bounds[new_leaf] = bounds[leaf];
        if split.numerical {
            if split.monotone_type < 0 {
                update_min(&mut bounds[leaf], split.right_output);
                update_max(&mut bounds[new_leaf], split.left_output);
            } else if split.monotone_type > 0 {
                update_max(&mut bounds[leaf], split.right_output);
                update_min(&mut bounds[new_leaf], split.left_output);
            }
        }
        let ctx = Ctx {
            tree,
            metas,
            best,
            split,
        };
        let mut path = Path::default();
        self.go_up_to_find_leaves_to_update(
            &ctx,
            bounds,
            &mut leaves_to_update,
            tree.leaf_parent[new_leaf],
            &mut path,
        );
        leaves_to_update
    }

    /// upstream: `OppositeChildShouldBeUpdated`
    fn opposite_child_should_be_updated(
        path: &Path,
        is_split_numerical: bool,
        inner_feature: i32,
        is_in_right_child: bool,
    ) -> bool {
        // categorical splits are not handled by this optimisation, so the walk goes down
        // the other child to look for leaves to update.
        if !is_split_numerical {
            return false;
        }
        // only branches containing leaves contiguous to the original leaf need updating; for
        // the same feature, there is no use going down a second time on the same side.
        for i in 0..path.features.len() {
            if path.features[i] == inner_feature && path.was_right_child[i] == is_in_right_child {
                return false;
            }
        }
        true
    }

    /// upstream: `GoUpToFindLeavesToUpdate`
    fn go_up_to_find_leaves_to_update(
        &mut self,
        cx: &Ctx<'_>,
        bounds: &mut [Bounds],
        out: &mut Vec<usize>,
        node_idx: i32,
        path: &mut Path,
    ) {
        let parent_idx = self.node_parent[node_idx as usize];
        // stop at the root
        if parent_idx == -1 {
            return;
        }
        let p = parent_idx as usize;
        let tree = cx.tree;
        let inner_feature = tree.split_feature_inner[p];
        let monotone_type = cx.metas[inner_feature as usize].monotone_type;
        let is_in_right_child = tree.right_child[p] == node_idx;
        let is_split_numerical = is_numerical_split(tree, p);

        if Self::opposite_child_should_be_updated(
            path,
            is_split_numerical,
            inner_feature,
            is_in_right_child,
        ) {
            // no monotone constraint on a split: no relation between its children's values
            if monotone_type != 0 {
                let left_child_idx = tree.left_child[p];
                let right_child_idx = tree.right_child[p];
                let left_child_is_curr_idx = left_child_idx == node_idx;
                let opposite_child_idx = if left_child_is_curr_idx {
                    right_child_idx
                } else {
                    left_child_idx
                };
                let update_max_in_opposite_child = if monotone_type < 0 {
                    left_child_is_curr_idx
                } else {
                    !left_child_is_curr_idx
                };
                self.go_down_to_find_leaves_to_update(
                    cx,
                    bounds,
                    out,
                    opposite_child_idx,
                    path,
                    update_max_in_opposite_child,
                    true,
                    true,
                );
            }
            // the path to here is relevant going down, so record this split
            path.was_right_child.push(tree.right_child[p] == node_idx);
            path.thresholds.push(tree.threshold_in_bin[p]);
            path.features.push(tree.split_feature_inner[p]);
        }

        self.go_up_to_find_leaves_to_update(cx, bounds, out, parent_idx, path);
    }

    /// upstream: `GoDownToFindLeavesToUpdate`
    #[allow(clippy::too_many_arguments)]
    fn go_down_to_find_leaves_to_update(
        &mut self,
        cx: &Ctx<'_>,
        bounds: &mut [Bounds],
        out: &mut Vec<usize>,
        node_idx: i32,
        path: &Path,
        update_max_constraints: bool,
        use_left_leaf: bool,
        use_right_leaf: bool,
    ) {
        let split = cx.split;
        if node_idx < 0 {
            let leaf_idx = (!node_idx) as usize;
            // splits that are not to be used shall not be updated, including a leaf at max depth
            if cx.best[leaf_idx].gain == K_MIN_SCORE {
                return;
            }
            // a leaf contiguous with both new children may need to lie above both outputs
            // or below both; one contiguous with a single child only needs that child's output
            let (min_value, max_value) = if use_right_leaf && use_left_leaf {
                if split.right_output <= split.left_output {
                    (split.right_output, split.left_output)
                } else {
                    (split.left_output, split.right_output)
                }
            } else if use_right_leaf {
                (split.right_output, split.right_output)
            } else {
                (split.left_output, split.left_output)
            };
            let changed = if update_max_constraints {
                lower_max(&mut bounds[leaf_idx], min_value)
            } else {
                raise_min(&mut bounds[leaf_idx], max_value)
            };
            if changed {
                out.push(leaf_idx);
            }
            return;
        }

        // an internal node: keep going into the children that are contiguous with the leaf
        let node = node_idx as usize;
        let tree = cx.tree;
        let (keep_going_left, keep_going_right) = should_keep_going_left_right(tree, node, path);
        let inner_feature = tree.split_feature_inner[node];
        let threshold = tree.threshold_in_bin[node];
        let is_split_numerical = is_numerical_split(tree, node);
        let mut use_left_leaf_for_update_right = true;
        let mut use_right_leaf_for_update_left = true;
        // on the same feature, the threshold decides which child is contiguous with which leaf
        if is_split_numerical && inner_feature == split.feature_inner {
            if threshold >= split.threshold {
                use_left_leaf_for_update_right = false;
            }
            if threshold <= split.threshold {
                use_right_leaf_for_update_left = false;
            }
        }
        if keep_going_left {
            self.go_down_to_find_leaves_to_update(
                cx,
                bounds,
                out,
                tree.left_child[node],
                path,
                update_max_constraints,
                use_left_leaf,
                use_right_leaf_for_update_left && use_right_leaf,
            );
        }
        if keep_going_right {
            self.go_down_to_find_leaves_to_update(
                cx,
                bounds,
                out,
                tree.right_child[node],
                path,
                update_max_constraints,
                use_left_leaf_for_update_right && use_left_leaf,
                use_right_leaf,
            );
        }
    }
}

/// upstream: `Tree::IsNumericalSplit`
fn is_numerical_split(tree: &Tree, node: usize) -> bool {
    (tree.decision_type[node] & K_CATEGORICAL_MASK) == 0
}

/// upstream: `ShouldKeepGoingLeftRight`. Returns `(keep_going_left, keep_going_right)`.
fn should_keep_going_left_right(tree: &Tree, node: usize, path: &Path) -> (bool, bool) {
    let inner_feature = tree.split_feature_inner[node];
    let threshold = tree.threshold_in_bin[node];
    let mut keep_going_right = true;
    let mut keep_going_left = true;
    // left and right children are checked to see whether they are contiguous with the
    // original leaf; if so, the walk keeps going down them to update constraints
    if is_numerical_split(tree, node) {
        for i in 0..path.features.len() {
            if path.features[i] == inner_feature {
                if threshold >= path.thresholds[i] && !path.was_right_child[i] {
                    keep_going_right = false;
                    if !keep_going_left {
                        break;
                    }
                }
                if threshold <= path.thresholds[i] && path.was_right_child[i] {
                    keep_going_left = false;
                    if !keep_going_right {
                        break;
                    }
                }
            }
        }
    }
    (keep_going_left, keep_going_right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_updates_keep_the_tighter_side() {
        let mut b = (-1.0, 1.0);
        update_min(&mut b, 0.5);
        assert_eq!(b, (0.5, 1.0));
        update_min(&mut b, -3.0);
        assert_eq!(b.0, 0.5);
        update_max(&mut b, 0.25);
        assert_eq!(b.1, 0.25);
        update_max(&mut b, 9.0);
        assert_eq!(b.1, 0.25);
    }

    #[test]
    fn changed_flags_report_only_real_tightening() {
        let mut b = (0.0, 1.0);
        assert!(raise_min(&mut b, 0.5));
        assert!(!raise_min(&mut b, 0.25));
        assert!(lower_max(&mut b, 0.75));
        assert!(!lower_max(&mut b, 0.9));
        assert_eq!(b, (0.5, 0.75));
    }

    #[test]
    fn reset_clears_the_subtree_marks_and_parents() {
        let mut im = IntermediateConstraints::new(4);
        im.in_monotone_subtree[2] = true;
        im.node_parent[1] = 2;
        im.reset();
        assert!(!im.in_monotone_subtree[2]);
        assert_eq!(im.node_parent[1], -1);
    }
}
