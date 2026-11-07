//! Per-leaf output bounds for the advanced monotone method: one lower and one upper bound per
//! threshold of a feature, and the cumulative view a threshold scan reads from them.
//!
//! upstream: src/treelearner/monotone_constraints.hpp (`FeatureMinOrMaxConstraints`,
//! `CumulativeFeatureConstraint`, `AdvancedFeatureConstraints`)

/// Upstream `std::max<double>(a, b)`: `b` when `a < b`, else `a`.
#[inline]
pub(crate) fn max_of(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// Upstream `std::min<double>(a, b)`: `b` when `b < a`, else `a`.
#[inline]
pub(crate) fn min_of(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// One side (min or max) of a leaf's constraint for one feature: constraint `i` holds on the
/// thresholds `[thresholds[i], thresholds[i + 1])`, and the last one on every threshold after it.
///
/// upstream: `FeatureMinOrMaxConstraints`
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMinOrMax {
    pub constraints: Vec<f64>,
    pub thresholds: Vec<u32>,
}

impl FeatureMinOrMax {
    /// upstream: `FeatureMinOrMaxConstraints(double extremum)`
    pub fn new(extremum: f64) -> Self {
        Self {
            constraints: vec![extremum],
            thresholds: vec![0],
        }
    }

    /// upstream: `Reset(double extremum)`
    pub fn reset(&mut self, extremum: f64) {
        self.constraints.clear();
        self.constraints.push(extremum);
        self.thresholds.clear();
        self.thresholds.push(0);
    }

    /// upstream: `Size()`
    pub fn size(&self) -> usize {
        self.thresholds.len()
    }

    /// upstream: `UpdateMin(double min)`: raises every constraint to at least `min`.
    pub fn update_min(&mut self, min: f64) {
        for c in &mut self.constraints {
            if min > *c {
                *c = min;
            }
        }
    }

    /// upstream: `UpdateMax(double max)`: lowers every constraint to at most `max`.
    pub fn update_max(&mut self, max: f64) {
        for c in &mut self.constraints {
            if max < *c {
                *c = max;
            }
        }
    }
}

/// The advanced constraint of one leaf for one feature.
///
/// upstream: `AdvancedFeatureConstraints`. The cumulative view is not stored here: the threshold
/// scan builds it from these lists (see [`Cumulative`]), which gives the same values.
#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedFeature {
    pub min: FeatureMinOrMax,
    pub max: FeatureMinOrMax,
    /// upstream: `min_constraints_to_be_recomputed`
    pub min_to_recompute: bool,
    /// upstream: `max_constraints_to_be_recomputed`
    pub max_to_recompute: bool,
}

impl Default for AdvancedFeature {
    fn default() -> Self {
        Self {
            min: FeatureMinOrMax::new(-f64::MAX),
            max: FeatureMinOrMax::new(f64::MAX),
            min_to_recompute: false,
            max_to_recompute: false,
        }
    }
}

impl AdvancedFeature {
    /// upstream: `Reset()`
    pub fn reset(&mut self) {
        self.min.reset(-f64::MAX);
        self.max.reset(f64::MAX);
    }

    /// upstream: `ConstraintDifferentDependingOnThreshold`
    pub fn varies_by_threshold(&self) -> bool {
        self.min.size() > 1 || self.max.size() > 1
    }

    /// upstream: `UpdateMax(double new_max, bool trigger_a_recompute)`
    pub fn update_max(&mut self, new_max: f64, trigger_a_recompute: bool) {
        if trigger_a_recompute {
            self.max_to_recompute = true;
        }
        self.max.update_max(new_max);
    }

    /// upstream: `UpdateMin(double new_min, bool trigger_a_recompute)`
    pub fn update_min(&mut self, new_min: f64, trigger_a_recompute: bool) {
        if trigger_a_recompute {
            self.min_to_recompute = true;
        }
        self.min.update_min(new_min);
    }

    /// upstream: `ResetUpdates()`
    pub fn reset_updates(&mut self) {
        self.min_to_recompute = false;
        self.max_to_recompute = false;
    }
}

/// The per-threshold view of one feature's constraint, as a threshold scan walks the thresholds.
/// Built fresh by each scan (upstream keeps it in the constraint object and calls
/// `InitCumulativeConstraints` at the start of every scan, so the result is the same).
///
/// upstream: `CumulativeFeatureConstraint`
#[derive(Debug)]
pub struct Cumulative {
    thresholds_min: Vec<u32>,
    thresholds_max: Vec<u32>,
    min_left_to_right: Vec<f64>,
    min_right_to_left: Vec<f64>,
    max_left_to_right: Vec<f64>,
    max_right_to_left: Vec<f64>,
    idx_min_left_to_right: usize,
    idx_min_right_to_left: usize,
    idx_max_left_to_right: usize,
    idx_max_right_to_left: usize,
}

/// upstream: `CumulativeFeatureConstraint::CumulativeExtremum`. Carries the running extremum
/// in the given direction, so each entry is the extremum of everything before (or after) it.
fn cumulative_extremum(extremum: fn(f64, f64) -> f64, left_to_right: bool, values: &mut [f64]) {
    let n = values.len();
    if n == 1 {
        return;
    }
    if left_to_right {
        for i in 0..n - 1 {
            values[i + 1] = extremum(values[i + 1], values[i]);
        }
    } else {
        for i in (1..n).rev() {
            values[i - 1] = extremum(values[i - 1], values[i]);
        }
    }
}

impl Cumulative {
    /// upstream: `CumulativeFeatureConstraint(min_constraints, max_constraints, REVERSE)`
    pub fn new(feature: &AdvancedFeature, reverse: bool) -> Self {
        let min = &feature.min;
        let max = &feature.max;
        let mut min_left_to_right = min.constraints.clone();
        let mut min_right_to_left = min.constraints.clone();
        let mut max_left_to_right = max.constraints.clone();
        let mut max_right_to_left = max.constraints.clone();
        cumulative_extremum(max_of, true, &mut min_left_to_right);
        cumulative_extremum(max_of, false, &mut min_right_to_left);
        cumulative_extremum(min_of, true, &mut max_left_to_right);
        cumulative_extremum(min_of, false, &mut max_right_to_left);
        let (idx_min, idx_max) = if reverse {
            (min.size() - 1, max.size() - 1)
        } else {
            (0, 0)
        };
        Self {
            thresholds_min: min.thresholds.clone(),
            thresholds_max: max.thresholds.clone(),
            min_left_to_right,
            min_right_to_left,
            max_left_to_right,
            max_right_to_left,
            idx_min_left_to_right: idx_min,
            idx_min_right_to_left: idx_min,
            idx_max_left_to_right: idx_max,
            idx_max_right_to_left: idx_max,
        }
    }

    /// upstream: `CumulativeFeatureConstraint::Update(int threshold)`. Moves the indices to the
    /// constraints that hold at `threshold`. The left side looks at `threshold - 1`, as the
    /// left child of a split at `threshold - 1` ends there.
    pub fn update(&mut self, threshold: i32) {
        let t = i64::from(threshold);
        // `while (thresholds[idx] > threshold - 1) --idx`, bounded at 0
        while self.idx_min_left_to_right > 0
            && i64::from(self.thresholds_min[self.idx_min_left_to_right]) > t - 1
        {
            self.idx_min_left_to_right -= 1;
        }
        while self.idx_min_right_to_left > 0
            && i64::from(self.thresholds_min[self.idx_min_right_to_left]) > t
        {
            self.idx_min_right_to_left -= 1;
        }
        while self.idx_max_left_to_right > 0
            && i64::from(self.thresholds_max[self.idx_max_left_to_right]) > t - 1
        {
            self.idx_max_left_to_right -= 1;
        }
        while self.idx_max_right_to_left > 0
            && i64::from(self.thresholds_max[self.idx_max_right_to_left]) > t
        {
            self.idx_max_right_to_left -= 1;
        }
    }

    /// Lower and upper bound of the left child at the current threshold.
    pub fn left(&self) -> (f64, f64) {
        (
            self.min_left_to_right[self.idx_min_left_to_right],
            self.max_left_to_right[self.idx_max_left_to_right],
        )
    }

    /// Lower and upper bound of the right child at the current threshold.
    pub fn right(&self) -> (f64, f64) {
        (
            self.min_right_to_left[self.idx_min_right_to_left],
            self.max_right_to_left[self.idx_max_right_to_left],
        )
    }
}
