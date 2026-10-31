//! Row-to-leaf partition. upstream: src/treelearner/data_partition.hpp,
//! src/io/dense_bin.hpp (`SplitInner`), include/LightGBM/feature_group.h (`Split`)

use crate::io::bin::{BinMapper, BinType};
use crate::io::bin_data::{BinData, DenseBin};
use crate::io::dataset::{Dataset, GroupData};
use crate::meta::MissingType;
use crate::utils::common::find_in_bitset;

/// upstream: `DataPartition`. Indices of each leaf are contiguous in `indices`.
#[derive(Debug, Clone)]
pub struct DataPartition {
    num_data: usize,
    leaf_begin: Vec<usize>,
    leaf_count: Vec<usize>,
    indices: Vec<i32>,
    left_buf: Vec<i32>,
    right_buf: Vec<i32>,
}

impl DataPartition {
    pub fn new(num_data: usize, num_leaves: usize) -> Self {
        DataPartition {
            num_data,
            leaf_begin: vec![0; num_leaves],
            leaf_count: vec![0; num_leaves],
            indices: (0..num_data as i32).collect(),
            left_buf: Vec::with_capacity(num_data),
            right_buf: Vec::with_capacity(num_data),
        }
    }

    /// upstream: `DataPartition::Init` (no bagging)
    pub fn init(&mut self) {
        self.leaf_begin.fill(0);
        self.leaf_count.fill(0);
        self.leaf_count[0] = self.num_data;
        for (i, v) in self.indices.iter_mut().enumerate() {
            *v = i as i32;
        }
    }

    /// upstream: `DataPartition::SetUsedDataIndices` + `Init`: the root leaf holds `rows`
    /// (bagged rows, ascending) instead of all rows.
    pub fn init_with(&mut self, rows: &[i32]) {
        self.leaf_begin.fill(0);
        self.leaf_count.fill(0);
        self.leaf_count[0] = rows.len();
        self.indices[..rows.len()].copy_from_slice(rows);
    }

    pub fn leaf_count(&self, leaf: usize) -> usize {
        self.leaf_count[leaf]
    }

    pub fn indices_on_leaf(&self, leaf: usize) -> &[i32] {
        let b = self.leaf_begin[leaf];
        &self.indices[b..b + self.leaf_count[leaf]]
    }

    /// upstream: `DataPartition::Split` — stable partition of `leaf` into
    /// `leaf` (<= threshold) and `right_leaf` (> threshold).
    pub fn split(&mut self, leaf: usize, rule: &SplitRule<'_>, right_leaf: usize) {
        let begin = self.leaf_begin[leaf];
        let cnt = self.leaf_count[leaf];
        self.left_buf.clear();
        self.right_buf.clear();
        let idx = &self.indices[begin..begin + cnt];
        // Monomorphize the routing loop per bin width (upstream: templated DenseBin).
        match rule.bin {
            DenseBin::U4(d) => partition(
                idx,
                |i| u32::from((d[i >> 1] >> ((i & 1) << 2)) & 0xf),
                rule,
                &mut self.left_buf,
                &mut self.right_buf,
            ),
            DenseBin::U8(d) => partition(
                idx,
                |i| u32::from(d[i]),
                rule,
                &mut self.left_buf,
                &mut self.right_buf,
            ),
            DenseBin::U16(d) => partition(
                idx,
                |i| u32::from(d[i]),
                rule,
                &mut self.left_buf,
                &mut self.right_buf,
            ),
            DenseBin::U32(d) => {
                partition(idx, |i| d[i], rule, &mut self.left_buf, &mut self.right_buf)
            }
        }
        let left_cnt = self.left_buf.len();
        self.indices[begin..begin + left_cnt].copy_from_slice(&self.left_buf);
        self.indices[begin + left_cnt..begin + cnt].copy_from_slice(&self.right_buf);
        self.leaf_count[leaf] = left_cnt;
        self.leaf_begin[right_leaf] = begin + left_cnt;
        self.leaf_count[right_leaf] = cnt - left_cnt;
    }
}

#[inline(always)]
fn partition(
    idx: &[i32],
    get: impl Fn(usize) -> u32,
    rule: &SplitRule<'_>,
    left: &mut Vec<i32>,
    right: &mut Vec<i32>,
) {
    for &i in idx {
        if matches!(rule.route_bin(get(i as usize)), Route::Left) {
            left.push(i);
        } else {
            right.push(i);
        }
    }
}

/// Where each class of row goes, precomputed from upstream's `SplitInner` template flags.
#[derive(Debug, Clone, Copy)]
enum Route {
    Left,
    Right,
}

/// A numerical split rule over a dense bin column (upstream `DenseBin::SplitInner`).
pub struct SplitRule<'a> {
    bin: &'a DenseBin,
    use_min_bin: bool,
    minb: u32,
    maxb: u32,
    th: u32,
    t_zero_bin: u32,
    miss_is_zero: bool,
    miss_is_na: bool,
    mfb_is_zero: bool,
    mfb_is_na: bool,
    default_route: Route,
    missing_default_route: Route,
    /// `min_bin < max_bin` (otherwise the single-bin branch is used)
    multi_bin: bool,
    max_bin_route: Route,
    /// Categorical: bitset over bins
    cat_bitset: Option<(&'a [u32], u32, u32)>,
}

impl<'a> SplitRule<'a> {
    /// upstream: `FeatureGroup::Split` + `DenseBin::Split` for numerical features.
    pub fn numerical(
        ds: &'a Dataset,
        inner_feature: usize,
        threshold: u32,
        default_left: bool,
    ) -> SplitRule<'a> {
        let l = &ds.layout;
        let g = &l.groups[l.feature2group[inner_feature] as usize];
        let sub = l.feature2subfeature[inner_feature] as usize;
        let m: &BinMapper = &g.bin_mappers[sub];
        let default_bin = m.default_bin;
        let most_freq_bin = m.most_freq_bin;
        let (bin, min_bin, max_bin, use_min_bin) = match &ds.data
            [l.feature2group[inner_feature] as usize]
        {
            GroupData::Single(BinData::Dense(d)) => {
                let min_bin = g.bin_offsets[sub];
                let max_bin = g.bin_offsets[sub + 1] - 1;
                if g.bin_mappers.len() == 1 {
                    (d, 1, max_bin, false)
                } else {
                    (d, min_bin, max_bin, true)
                }
            }
            GroupData::Multi(cols) => match &cols[sub] {
                BinData::Dense(d) => {
                    let addi = u32::from(most_freq_bin != 0);
                    (d, 1, m.num_bin as u32 - 1 + addi, false)
                }
                BinData::Sparse(_) => unreachable!("sparse bins are densified before training"),
            },
            GroupData::Single(BinData::Sparse(_)) => unreachable!("single-value groups are dense"),
        };
        let missing_type = m.missing_type;
        let (miss_is_zero, miss_is_na, mfb_is_zero, mfb_is_na) = match missing_type {
            MissingType::None => (false, false, false, false),
            MissingType::Zero => (true, false, default_bin == most_freq_bin, false),
            MissingType::NaN => {
                let cond = if use_min_bin {
                    max_bin == most_freq_bin + min_bin && most_freq_bin > 0
                } else {
                    max_bin == most_freq_bin + 1 && most_freq_bin > 0
                };
                (false, true, false, cond)
            }
        };
        let mut th = threshold.wrapping_add(min_bin);
        let mut t_zero_bin = min_bin.wrapping_add(default_bin);
        if most_freq_bin == 0 {
            th = th.wrapping_sub(1);
            t_zero_bin = t_zero_bin.wrapping_sub(1);
        }
        let default_route = if most_freq_bin <= threshold {
            Route::Left
        } else {
            Route::Right
        };
        let missing_default_route = if (miss_is_zero || miss_is_na) && default_left {
            Route::Left
        } else {
            Route::Right
        };
        let max_bin_route = if max_bin <= th {
            Route::Left
        } else {
            Route::Right
        };
        SplitRule {
            bin,
            use_min_bin,
            minb: min_bin,
            maxb: max_bin,
            th,
            t_zero_bin,
            miss_is_zero,
            miss_is_na,
            mfb_is_zero,
            mfb_is_na,
            default_route,
            missing_default_route,
            multi_bin: min_bin < max_bin,
            max_bin_route,
            cat_bitset: None,
        }
    }

    #[inline(always)]
    fn route_bin(&self, bin: u32) -> Route {
        if let Some((bits, min_bin, offset)) = self.cat_bitset {
            if (self.use_min_bin && (bin < min_bin || bin > self.maxb))
                || (!self.use_min_bin && bin == 0)
            {
                return self.default_route;
            }
            return if find_in_bitset(bits, bin - min_bin + offset) {
                Route::Left
            } else {
                Route::Right
            };
        }
        if self.multi_bin {
            if (self.miss_is_zero && !self.mfb_is_zero && bin == self.t_zero_bin)
                || (self.miss_is_na && !self.mfb_is_na && bin == self.maxb)
            {
                self.missing_default_route
            } else if (self.use_min_bin && (bin < self.minb || bin > self.maxb))
                || (!self.use_min_bin && bin == 0)
            {
                if (self.miss_is_na && self.mfb_is_na) || (self.miss_is_zero && self.mfb_is_zero) {
                    self.missing_default_route
                } else {
                    self.default_route
                }
            } else if bin > self.th {
                Route::Right
            } else {
                Route::Left
            }
        } else if self.miss_is_zero && !self.mfb_is_zero && bin == self.t_zero_bin {
            self.missing_default_route
        } else if bin != self.maxb {
            if (self.miss_is_na && self.mfb_is_na) || (self.miss_is_zero && self.mfb_is_zero) {
                self.missing_default_route
            } else {
                self.default_route
            }
        } else if self.miss_is_na && !self.mfb_is_na {
            self.missing_default_route
        } else {
            self.max_bin_route
        }
    }

    #[inline]
    pub fn goes_left(&self, idx: usize) -> bool {
        matches!(self.route_bin(self.bin.get(idx)), Route::Left)
    }

    /// Only numerical bins are handled by this rule constructor.
    pub fn is_numerical(m: &BinMapper) -> bool {
        m.bin_type == BinType::Numerical
    }
}
