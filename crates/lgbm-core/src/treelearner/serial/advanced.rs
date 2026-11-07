//! `monotone_constraints_method=advanced`: each leaf keeps, per feature, a list of
//! `(threshold, bound)` pairs, so a bound can differ from one threshold of the feature to the
//! next. Lists are rebuilt from the tree when a change made them stale.
//!
//! upstream: src/treelearner/monotone_constraints.hpp (`AdvancedLeafConstraints`,
//! `AdvancedConstraintEntry`, `AdvancedFeatureConstraints`). The walks share the intermediate
//! method's helpers (`intermediate.rs`), as upstream's class inherits them.

use crate::tree::Tree;
use crate::treelearner::monotone_constraints::{AdvancedFeature, FeatureMinOrMax, max_of, min_of};

use super::super::feature_histogram::FeatureMeta;
use super::intermediate::{
    IntermediateConstraints, LeafBounds, Path, is_numerical_split, should_keep_going_left_right,
};

/// Per leaf, per inner feature: the advanced constraint.
pub(super) type AdvancedStore = Vec<Vec<AdvancedFeature>>;

/// upstream: `AdvancedConstraintEntry(num_features)` for every leaf
pub(super) fn new_store(num_leaves: usize, num_features: usize) -> AdvancedStore {
    vec![vec![AdvancedFeature::default(); num_features]; num_leaves]
}

impl LeafBounds for AdvancedStore {
    fn copy_leaf(&mut self, from: usize, to: usize) {
        self[to] = self[from].clone();
    }

    fn raise_min(&mut self, leaf: usize, new_min: f64) -> bool {
        for f in &mut self[leaf] {
            f.update_min(new_min, true);
        }
        // upstream: "even if nothing changed, this could have been unconstrained so it needs
        // to be recomputed from the beginning"
        true
    }

    fn lower_max(&mut self, leaf: usize, new_max: f64) -> bool {
        for f in &mut self[leaf] {
            f.update_max(new_max, true);
        }
        true
    }

    fn set_min(&mut self, leaf: usize, new_min: f64) {
        for f in &mut self[leaf] {
            f.update_min(new_min, false);
        }
    }

    fn set_max(&mut self, leaf: usize, new_max: f64) {
        for f in &mut self[leaf] {
            f.update_max(new_max, false);
        }
    }
}

/// upstream: `AdvancedLeafConstraints::UpdateConstraints`. Applies `extremum` on
/// `[it_start, it_end)` to a list, keeping the list minimal: the previous constraint is put
/// back where the new one ends, and equal neighbours are merged.
pub(super) fn update_constraints(
    fc: &mut FeatureMinOrMax,
    extremum: f64,
    it_start: u32,
    it_end: u32,
    use_max_operator: bool,
    last_threshold: u32,
) {
    let pick = |a: f64, b: f64| {
        if use_max_operator {
            max_of(a, b)
        } else {
            min_of(a, b)
        }
    };
    let mut start_done = false;
    let mut end_done = false;
    // the previous constraint still applies where the new one ends, so it is remembered
    let mut previous_constraint = if use_max_operator {
        -f64::MAX
    } else {
        f64::MAX
    };
    let mut i: usize = 0;
    while i < fc.thresholds.len() {
        let current_constraint = fc.constraints[i];
        // easy case when the thresholds match
        if fc.thresholds[i] == it_start {
            fc.constraints[i] = pick(extremum, fc.constraints[i]);
            start_done = true;
        }
        if fc.thresholds[i] > it_start {
            // an existing constraint is updated if it lies inside the new range
            if fc.thresholds[i] < it_end {
                fc.constraints[i] = pick(extremum, fc.constraints[i]);
            }
            // a new threshold, and a new constraint, may need to be inserted
            if !start_done {
                start_done = true;
                if (use_max_operator && extremum > previous_constraint)
                    || (!use_max_operator && extremum < previous_constraint)
                {
                    fc.constraints.insert(i, extremum);
                    fc.thresholds.insert(i, it_start);
                    i += 1;
                }
            }
        }
        // easy case when the end thresholds match
        if fc.thresholds[i] == it_end {
            end_done = true;
            break;
        }
        // otherwise the previous constraint is added back where the current one ends
        if fc.thresholds[i] > it_end {
            if i != 0 && previous_constraint != fc.constraints[i - 1] {
                fc.constraints.insert(i, previous_constraint);
                fc.thresholds.insert(i, it_end);
            }
            end_done = true;
            break;
        }
        // two successive equal constraints: the second one is dropped
        if i != 0 && fc.constraints[i] == fc.constraints[i - 1] {
            fc.constraints.remove(i);
            fc.thresholds.remove(i);
            i -= 1;
        }
        previous_constraint = current_constraint;
        i += 1;
    }
    // the loop never reached `it_start`: the new constraint goes at the end
    if !start_done {
        let last = fc
            .constraints
            .last()
            .copied()
            .unwrap_or(previous_constraint);
        if (use_max_operator && extremum > last) || (!use_max_operator && extremum < last) {
            fc.constraints.push(extremum);
            fc.thresholds.push(it_start);
        } else {
            end_done = true;
        }
    }
    // the previous constraint is set back after `it_end`, unless `it_end` is the last threshold
    if !end_done && it_end != last_threshold {
        let last = fc
            .constraints
            .last()
            .copied()
            .unwrap_or(previous_constraint);
        if previous_constraint != last {
            fc.constraints.push(previous_constraint);
            fc.thresholds.push(it_end);
        }
    }
}

/// The tree and the feature metadata a constraint walk reads.
pub(super) struct WalkCtx<'a> {
    pub tree: &'a Tree,
    pub metas: &'a [FeatureMeta],
}

/// upstream: `LeftRightContainsRelevantInformation`. Whether each child of a node can hold a
/// value that constrains the leaf, `(left, right)`.
fn left_right_contains_relevant(
    metas: &[FeatureMeta],
    min_constraints_to_be_updated: bool,
    inner_feature: i32,
    split_feature_is_inner_feature: bool,
) -> (bool, bool) {
    if split_feature_is_inner_feature {
        return (true, true);
    }
    let monotone_type = metas[inner_feature as usize].monotone_type;
    if monotone_type == 0 {
        return (true, true);
    }
    if (monotone_type == -1 && min_constraints_to_be_updated)
        || (monotone_type == 1 && !min_constraints_to_be_updated)
    {
        (true, false)
    } else {
        // upstream: the same as the other pairing of the monotone direction and the side
        (false, true)
    }
}

impl IntermediateConstraints {
    /// upstream: `AdvancedConstraintEntry::RecomputeConstraintsIfNeeded`. When a change made
    /// the leaf's list of `feature` stale, rebuilds it from the tree, walking up from the leaf.
    /// `num_bin` is the feature's bin count (upstream: `it_end`).
    pub fn refresh_advanced(
        &self,
        ctx: &WalkCtx<'_>,
        store: &mut AdvancedStore,
        leaf: usize,
        feature: usize,
        num_bin: u32,
    ) {
        let entry = &mut store[leaf][feature];
        if !(entry.min_to_recompute || entry.max_to_recompute) {
            return;
        }
        // upstream: the min side is rebuilt when it is flagged, otherwise the max side
        let min_side = entry.min_to_recompute;
        let target = if min_side {
            &mut entry.min
        } else {
            &mut entry.max
        };
        target.reset(if min_side { -f64::MAX } else { f64::MAX });
        let mut path = Path::default();
        self.go_up_to_find_constraining_leaves(
            ctx,
            feature as i32,
            !(leaf as i32),
            &mut path,
            target,
            min_side,
            0,
            num_bin,
            num_bin,
        );
        entry.reset_updates();
    }

    /// upstream: `AdvancedLeafConstraints::GoUpToFindConstrainingLeaves`
    #[allow(clippy::too_many_arguments)]
    fn go_up_to_find_constraining_leaves(
        &self,
        ctx: &WalkCtx<'_>,
        feature_for_constraint: i32,
        node_idx: i32,
        path: &mut Path,
        feature_constraint: &mut FeatureMinOrMax,
        min_constraints_to_be_updated: bool,
        mut it_start: u32,
        mut it_end: u32,
        last_threshold: u32,
    ) {
        let tree = ctx.tree;
        let parent_idx = if node_idx < 0 {
            tree.leaf_parent[(!node_idx) as usize]
        } else {
            self.node_parent[node_idx as usize]
        };
        // if not at the root
        if parent_idx == -1 {
            return;
        }
        let p = parent_idx as usize;
        let inner_feature = tree.split_feature_inner[p];
        let monotone_type = ctx.metas[inner_feature as usize].monotone_type;
        let is_in_right_child = tree.right_child[p] == node_idx;
        let is_split_numerical = is_numerical_split(tree, p);
        let threshold = tree.threshold_in_bin[p];

        // by going up, the position of the original leaf is narrowed, so the thresholds
        // to visit going down are narrowed too
        if feature_for_constraint == inner_feature && is_split_numerical {
            if is_in_right_child {
                it_start = it_start.max(threshold);
            } else {
                it_end = it_end.min(threshold + 1);
            }
        }

        // this is only an optimisation: subtrees where no new constraining leaf can be found
        // are skipped
        if IntermediateConstraints::opposite_child_should_be_updated(
            path,
            is_split_numerical,
            inner_feature,
            is_in_right_child,
        ) {
            // if there is no monotone constraint on the split, there is no relation between
            // its children's values
            if monotone_type != 0 {
                let left_child_idx = tree.left_child[p];
                let right_child_idx = tree.right_child[p];
                let left_child_is_curr_idx = left_child_idx == node_idx;
                let update_min_in_curr_child_leaf = if monotone_type < 0 {
                    left_child_is_curr_idx
                } else {
                    !left_child_is_curr_idx
                };
                if update_min_in_curr_child_leaf == min_constraints_to_be_updated {
                    let opposite_child_idx = if left_child_is_curr_idx {
                        right_child_idx
                    } else {
                        left_child_idx
                    };
                    self.go_down_to_find_constraining_leaves(
                        ctx,
                        feature_for_constraint,
                        inner_feature,
                        opposite_child_idx,
                        min_constraints_to_be_updated,
                        it_start,
                        it_end,
                        path,
                        feature_constraint,
                        last_threshold,
                    );
                }
            }
            path.was_right_child.push(is_in_right_child);
            path.thresholds.push(threshold);
            path.features.push(inner_feature);
        }

        // since the current node is not the root, keep going up
        if parent_idx != 0 {
            self.go_up_to_find_constraining_leaves(
                ctx,
                feature_for_constraint,
                parent_idx,
                path,
                feature_constraint,
                min_constraints_to_be_updated,
                it_start,
                it_end,
                last_threshold,
            );
        }
    }

    /// upstream: `AdvancedLeafConstraints::GoDownToFindConstrainingLeaves`
    #[allow(clippy::too_many_arguments)]
    fn go_down_to_find_constraining_leaves(
        &self,
        ctx: &WalkCtx<'_>,
        feature_for_constraint: i32,
        root_monotone_feature: i32,
        node_idx: i32,
        min_constraints_to_be_updated: bool,
        it_start: u32,
        it_end: u32,
        path: &Path,
        feature_constraint: &mut FeatureMinOrMax,
        last_threshold: u32,
    ) {
        let tree = ctx.tree;
        // a leaf constrains the original leaf on the thresholds it covers
        if node_idx < 0 {
            let extremum = tree.leaf_value[(!node_idx) as usize];
            update_constraints(
                feature_constraint,
                extremum,
                it_start,
                it_end,
                min_constraints_to_be_updated,
                last_threshold,
            );
            return;
        }
        let node = node_idx as usize;
        // children contiguous with the original leaf are the only ones that can constrain it
        let (keep_going_left, keep_going_right) = should_keep_going_left_right(tree, node, path);
        let inner_feature = tree.split_feature_inner[node];
        let threshold = tree.threshold_in_bin[node];
        let split_feature_is_inner_feature = inner_feature == feature_for_constraint;
        let split_feature_is_monotone_feature = root_monotone_feature == feature_for_constraint;
        // make sure both children hold values that can help determine the constraints
        let (left_relevant, right_relevant) = left_right_contains_relevant(
            ctx.metas,
            min_constraints_to_be_updated,
            inner_feature,
            split_feature_is_inner_feature && !split_feature_is_monotone_feature,
        );
        // if both children are contiguous but one holds larger values, one side is enough
        if keep_going_left && (left_relevant || !keep_going_right) {
            let new_it_end = if split_feature_is_inner_feature {
                it_end.min(threshold + 1)
            } else {
                it_end
            };
            self.go_down_to_find_constraining_leaves(
                ctx,
                feature_for_constraint,
                root_monotone_feature,
                tree.left_child[node],
                min_constraints_to_be_updated,
                it_start,
                new_it_end,
                path,
                feature_constraint,
                last_threshold,
            );
        }
        if keep_going_right && (right_relevant || !keep_going_left) {
            let new_it_start = if split_feature_is_inner_feature {
                it_start.max(threshold + 1)
            } else {
                it_start
            };
            self.go_down_to_find_constraining_leaves(
                ctx,
                feature_for_constraint,
                root_monotone_feature,
                tree.right_child[node],
                min_constraints_to_be_updated,
                new_it_start,
                it_end,
                path,
                feature_constraint,
                last_threshold,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(constraints: &[f64], thresholds: &[u32]) -> FeatureMinOrMax {
        FeatureMinOrMax {
            constraints: constraints.to_vec(),
            thresholds: thresholds.to_vec(),
        }
    }

    #[test]
    fn a_range_constraint_splits_the_list_and_restores_the_previous_value() {
        // apply max-side 3 on [2, 5) to an unconstrained list, last threshold 10
        let mut fc = FeatureMinOrMax::new(f64::MAX);
        update_constraints(&mut fc, 3.0, 2, 5, false, 10);
        assert_eq!(fc, list(&[f64::MAX, 3.0, f64::MAX], &[0, 2, 5]));
    }

    #[test]
    fn a_min_constraint_on_the_whole_range_replaces_nothing_smaller() {
        let mut fc = FeatureMinOrMax::new(-f64::MAX);
        update_constraints(&mut fc, 1.5, 0, 10, true, 10);
        assert_eq!(fc, list(&[1.5], &[0]));
    }

    #[test]
    fn a_range_equal_to_its_neighbour_adds_no_entry() {
        let mut fc = list(&[1.0, 2.0], &[0, 4]);
        update_constraints(&mut fc, 2.0, 4, 6, true, 10);
        assert_eq!(fc, list(&[1.0, 2.0], &[0, 4]));
    }

    #[test]
    fn store_copies_and_flags_the_leaf() {
        let mut store = new_store(3, 2);
        store.set_min(0, 1.0);
        store.copy_leaf(0, 1);
        assert_eq!(store[1][0].min.constraints, vec![1.0]);
        assert!(store.raise_min(1, 2.0));
        assert!(store[1][1].min_to_recompute);
        assert!(store.lower_max(2, 0.0));
        assert!(store[2][0].max_to_recompute);
    }
}
