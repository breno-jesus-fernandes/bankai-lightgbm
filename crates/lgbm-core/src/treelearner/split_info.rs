//! Best-split bookkeeping. upstream: src/treelearner/split_info.hpp

use std::cmp::Ordering;

/// upstream: `kMinScore` (`-inf` as score_t)
pub const K_MIN_SCORE: f64 = f64::NEG_INFINITY;

#[derive(Debug, Clone)]
pub struct SplitInfo {
    /// Real (original) feature index, `-1` if none.
    pub feature: i32,
    pub threshold: u32,
    pub left_count: i32,
    pub right_count: i32,
    pub num_cat_threshold: i32,
    pub left_output: f64,
    pub right_output: f64,
    pub gain: f64,
    pub left_sum_gradient: f64,
    pub left_sum_hessian: f64,
    pub right_sum_gradient: f64,
    pub right_sum_hessian: f64,
    pub cat_threshold: Vec<u32>,
    pub default_left: bool,
    pub monotone_type: i8,
}

impl Default for SplitInfo {
    fn default() -> Self {
        SplitInfo {
            feature: -1,
            threshold: 0,
            left_count: 0,
            right_count: 0,
            num_cat_threshold: 0,
            left_output: 0.0,
            right_output: 0.0,
            gain: K_MIN_SCORE,
            left_sum_gradient: 0.0,
            left_sum_hessian: 0.0,
            right_sum_gradient: 0.0,
            right_sum_hessian: 0.0,
            cat_threshold: Vec::new(),
            default_left: true,
            monotone_type: 0,
        }
    }
}

impl SplitInfo {
    /// upstream: `SplitInfo::Reset`
    pub fn reset(&mut self) {
        self.feature = -1;
        self.gain = K_MIN_SCORE;
    }

    /// upstream: `SplitInfo::operator>` — higher gain wins; ties go to the smaller
    /// feature index (`-1` counts as the largest).
    #[inline]
    pub fn better_than(&self, other: &SplitInfo) -> bool {
        // upstream compares `gain == NAN`, which is always false, so NaN is not remapped.
        let (a, b) = (self.gain, other.gain);
        if a != b {
            return a > b;
        }
        let fa = if self.feature == -1 {
            i32::MAX
        } else {
            self.feature
        };
        let fb = if other.feature == -1 {
            i32::MAX
        } else {
            other.feature
        };
        fa < fb
    }
}

/// upstream `ArrayArgs<SplitInfo>::ArgMax` (first maximum under `operator>`).
pub fn argmax_split(splits: &[SplitInfo]) -> usize {
    let mut best = 0;
    for i in 1..splits.len() {
        if splits[i].better_than(&splits[best]) {
            best = i;
        }
    }
    best
}

impl PartialEq for SplitInfo {
    fn eq(&self, other: &Self) -> bool {
        !self.better_than(other) && !other.better_than(self)
    }
}

impl PartialOrd for SplitInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.better_than(other) {
            Some(Ordering::Greater)
        } else if other.better_than(self) {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Equal)
        }
    }
}
