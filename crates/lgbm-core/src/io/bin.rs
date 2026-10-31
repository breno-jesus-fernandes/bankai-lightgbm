//! Feature value -> bin mapping. upstream: include/LightGBM/bin.h, src/io/bin.cpp

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::meta::{K_ZERO_THRESHOLD, MissingType};

/// upstream: `kSparseThreshold`
pub const K_SPARSE_THRESHOLD: f64 = 0.7;

/// upstream: `BinType`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BinType {
    #[default]
    Numerical = 0,
    Categorical = 1,
}

/// Maps raw feature values to bin indices. upstream: `BinMapper`
#[derive(Debug, Clone, Default)]
pub struct BinMapper {
    pub num_bin: i32,
    pub missing_type: MissingType,
    pub is_trivial: bool,
    pub sparse_rate: f64,
    pub bin_type: BinType,
    pub min_val: f64,
    pub max_val: f64,
    pub default_bin: u32,
    pub most_freq_bin: u32,
    /// Numerical features: inclusive upper bound of each bin (last is +inf or NaN).
    pub bin_upper_bound: Vec<f64>,
    /// Categorical features: category of each bin (bin 0 is the NaN/other bin, `-1`).
    pub bin_2_categorical: Vec<i32>,
    pub categorical_2_bin: HashMap<i32, u32>,
}

/// upstream: `Common::GetDoubleUpperBound` (`nextafter(a, +inf)`).
#[inline]
fn next_up(a: f64) -> f64 {
    a.next_up()
}

/// upstream: `Common::CheckDoubleEqualOrdered(a, b)`: `b <= nextafter(a, +inf)`.
#[inline]
fn check_double_equal_ordered(a: f64, b: f64) -> bool {
    b <= next_up(a)
}

/// upstream: `GreedyFindBin`
pub(crate) fn greedy_find_bin(
    distinct_values: &[f64],
    counts: &[i32],
    mut max_bin: i32,
    total_cnt: usize,
    min_data_in_bin: i32,
) -> Vec<f64> {
    let num_distinct_values = distinct_values.len();
    let mut bin_upper_bound = Vec::new();
    debug_assert!(max_bin > 0);
    if num_distinct_values as i32 <= max_bin {
        let mut cur_cnt_inbin = 0;
        for i in 0..num_distinct_values.saturating_sub(1) {
            cur_cnt_inbin += counts[i];
            if cur_cnt_inbin >= min_data_in_bin {
                let val = next_up((distinct_values[i] + distinct_values[i + 1]) / 2.0);
                if bin_upper_bound
                    .last()
                    .is_none_or(|&b| !check_double_equal_ordered(b, val))
                {
                    bin_upper_bound.push(val);
                    cur_cnt_inbin = 0;
                }
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    } else {
        if min_data_in_bin > 0 {
            max_bin = max_bin.min((total_cnt / min_data_in_bin as usize) as i32);
            max_bin = max_bin.max(1);
        }
        let mut mean_bin_size = total_cnt as f64 / f64::from(max_bin);
        let mut rest_bin_cnt = max_bin;
        let mut rest_sample_cnt = total_cnt as i32;
        let mut is_big_count_value = vec![false; num_distinct_values];
        for i in 0..num_distinct_values {
            if f64::from(counts[i]) >= mean_bin_size {
                is_big_count_value[i] = true;
                rest_bin_cnt -= 1;
                rest_sample_cnt -= counts[i];
            }
        }
        mean_bin_size = f64::from(rest_sample_cnt) / f64::from(rest_bin_cnt);
        let mb = max_bin as usize;
        let mut upper_bounds = vec![f64::INFINITY; mb];
        let mut lower_bounds = vec![f64::INFINITY; mb];
        let mut bin_cnt = 0usize;
        lower_bounds[bin_cnt] = distinct_values[0];
        let mut cur_cnt_inbin = 0;
        // `mean_bin_size * 0.5f`: the float literal promotes to double.
        for i in 0..num_distinct_values - 1 {
            if !is_big_count_value[i] {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];
            if is_big_count_value[i]
                || f64::from(cur_cnt_inbin) >= mean_bin_size
                || (is_big_count_value[i + 1]
                    && f64::from(cur_cnt_inbin) >= 1.0f64.max(mean_bin_size * 0.5))
            {
                upper_bounds[bin_cnt] = distinct_values[i];
                bin_cnt += 1;
                lower_bounds[bin_cnt] = distinct_values[i + 1];
                if bin_cnt as i32 >= max_bin - 1 {
                    break;
                }
                cur_cnt_inbin = 0;
                if !is_big_count_value[i] {
                    rest_bin_cnt -= 1;
                    mean_bin_size = f64::from(rest_sample_cnt) / f64::from(rest_bin_cnt);
                }
            }
        }
        bin_cnt += 1;
        for i in 0..bin_cnt - 1 {
            let val = next_up((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
            if bin_upper_bound
                .last()
                .is_none_or(|&b| !check_double_equal_ordered(b, val))
            {
                bin_upper_bound.push(val);
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    }
    bin_upper_bound
}

/// upstream: `FindBinWithPredefinedBin` (forced bin bounds)
fn find_bin_with_predefined_bin(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Vec<f64> {
    let n = distinct_values.len();
    let mut bin_upper_bound: Vec<f64> = Vec::new();
    let left_cnt = distinct_values
        .iter()
        .position(|&v| v > -K_ZERO_THRESHOLD)
        .unwrap_or(n);
    let right_start = distinct_values[left_cnt..]
        .iter()
        .position(|&v| v > K_ZERO_THRESHOLD)
        .map(|p| p + left_cnt);
    if max_bin == 2 {
        bin_upper_bound.push(if left_cnt == 0 {
            K_ZERO_THRESHOLD
        } else {
            -K_ZERO_THRESHOLD
        });
    } else if max_bin >= 3 {
        if left_cnt > 0 {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
        if right_start.is_some() {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        }
    }
    bin_upper_bound.push(f64::INFINITY);
    let max_to_insert = max_bin - bin_upper_bound.len() as i32;
    let mut num_inserted = 0;
    for &b in forced_upper_bounds {
        if num_inserted >= max_to_insert {
            break;
        }
        if b.abs() > K_ZERO_THRESHOLD {
            bin_upper_bound.push(b);
            num_inserted += 1;
        }
    }
    stable_sort_f64(&mut bin_upper_bound);

    let free_bins = max_bin - bin_upper_bound.len() as i32;
    let mut bounds_to_add: Vec<f64> = Vec::new();
    let mut value_ind = 0usize;
    for i in 0..bin_upper_bound.len() {
        let mut cnt_in_bin = 0i32;
        let mut distinct_cnt_in_bin = 0usize;
        let bin_start = value_ind;
        while value_ind < n && distinct_values[value_ind] < bin_upper_bound[i] {
            cnt_in_bin += counts[value_ind];
            distinct_cnt_in_bin += 1;
            value_ind += 1;
        }
        let bins_remaining = max_bin - bin_upper_bound.len() as i32 - bounds_to_add.len() as i32;
        // std::lround: round half away from zero
        let mut num_sub_bins =
            (f64::from(cnt_in_bin) * f64::from(free_bins) / total_sample_cnt as f64).round() as i32;
        num_sub_bins = num_sub_bins.min(bins_remaining) + 1;
        if i == bin_upper_bound.len() - 1 {
            num_sub_bins = bins_remaining + 1;
        }
        if distinct_cnt_in_bin == 0 {
            continue;
        }
        let new_bounds = greedy_find_bin(
            &distinct_values[bin_start..bin_start + distinct_cnt_in_bin],
            &counts[bin_start..bin_start + distinct_cnt_in_bin],
            num_sub_bins,
            cnt_in_bin as usize,
            min_data_in_bin,
        );
        bounds_to_add.extend_from_slice(&new_bounds[..new_bounds.len() - 1]);
    }
    bin_upper_bound.extend(bounds_to_add);
    stable_sort_f64(&mut bin_upper_bound);
    bin_upper_bound
}

fn stable_sort_f64(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// upstream: `FindBinWithZeroAsOneBin` (with optional forced bounds)
fn find_bin_with_zero_as_one_bin(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Vec<f64> {
    if !forced_upper_bounds.is_empty() {
        return find_bin_with_predefined_bin(
            distinct_values,
            counts,
            max_bin,
            total_sample_cnt,
            min_data_in_bin,
            forced_upper_bounds,
        );
    }
    let n = distinct_values.len();
    let mut bin_upper_bound = Vec::new();
    let (mut left_cnt_data, mut cnt_zero, mut right_cnt_data) = (0i32, 0i32, 0i32);
    for i in 0..n {
        if distinct_values[i] <= -K_ZERO_THRESHOLD {
            left_cnt_data += counts[i];
        } else if distinct_values[i] > K_ZERO_THRESHOLD {
            right_cnt_data += counts[i];
        } else {
            cnt_zero += counts[i];
        }
    }
    let left_cnt = distinct_values
        .iter()
        .position(|&v| v > -K_ZERO_THRESHOLD)
        .unwrap_or(n);
    if left_cnt > 0 && max_bin > 1 {
        let denom = total_sample_cnt as f64 - f64::from(cnt_zero);
        let mut left_max_bin = (f64::from(left_cnt_data) / denom * f64::from(max_bin - 1)) as i32;
        left_max_bin = left_max_bin.max(1);
        bin_upper_bound = greedy_find_bin(
            &distinct_values[..left_cnt],
            &counts[..left_cnt],
            left_max_bin,
            left_cnt_data as usize,
            min_data_in_bin,
        );
        if let Some(last) = bin_upper_bound.last_mut() {
            *last = -K_ZERO_THRESHOLD;
        }
    }
    let right_start = distinct_values[left_cnt..]
        .iter()
        .position(|&v| v > K_ZERO_THRESHOLD)
        .map(|p| p + left_cnt);
    let right_max_bin = max_bin - 1 - bin_upper_bound.len() as i32;
    match right_start {
        Some(rs) if right_max_bin > 0 => {
            let right = greedy_find_bin(
                &distinct_values[rs..],
                &counts[rs..],
                right_max_bin,
                right_cnt_data as usize,
                min_data_in_bin,
            );
            bin_upper_bound.push(K_ZERO_THRESHOLD);
            bin_upper_bound.extend(right);
        }
        _ => bin_upper_bound.push(f64::INFINITY),
    }
    bin_upper_bound
}

/// upstream: `NeedFilter`
fn need_filter(cnt_in_bin: &[i32], total_cnt: i32, filter_cnt: i32, bin_type: BinType) -> bool {
    if bin_type == BinType::Numerical {
        let mut sum_left = 0;
        for &c in &cnt_in_bin[..cnt_in_bin.len() - 1] {
            sum_left += c;
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    } else if cnt_in_bin.len() <= 2 {
        for &c in &cnt_in_bin[..cnt_in_bin.len() - 1] {
            if c >= filter_cnt && total_cnt - c >= filter_cnt {
                return false;
            }
        }
    } else {
        return false;
    }
    true
}

/// Arguments of upstream `BinMapper::FindBin` besides the sample values.
#[derive(Debug, Clone, Copy)]
pub struct FindBinParams<'a> {
    pub total_sample_cnt: usize,
    pub max_bin: i32,
    pub min_data_in_bin: i32,
    pub min_split_data: i32,
    pub pre_filter: bool,
    pub bin_type: BinType,
    pub use_missing: bool,
    pub zero_as_missing: bool,
    pub forced_upper_bounds: &'a [f64],
}

impl BinMapper {
    /// upstream: `BinMapper::FindBin`. `values` holds the sampled non-zero values
    /// (`|v| > kZeroThreshold` or NaN); zeros are implied by `total_sample_cnt`.
    ///
    /// Like upstream, `values` is modified in place: non-NaN values are compacted to the
    /// front and sorted; entries past them keep stale contents. Upstream's feature bundling
    /// later reads this mutated buffer, so the exact layout matters.
    pub fn find_bin(values: &mut [f64], p: &FindBinParams<'_>) -> BinMapper {
        let mut m = BinMapper::default();
        let num_sample_values = values.len();
        let mut non_na_cnt = 0usize;
        for i in 0..num_sample_values {
            if !values[i].is_nan() {
                values[non_na_cnt] = values[i];
                non_na_cnt += 1;
            }
        }
        let mut na_cnt = 0usize;
        m.missing_type = if !p.use_missing {
            MissingType::None
        } else if p.zero_as_missing {
            MissingType::Zero
        } else if non_na_cnt == num_sample_values {
            MissingType::None
        } else {
            na_cnt = num_sample_values - non_na_cnt;
            MissingType::NaN
        };
        let num_sample_values = non_na_cnt;
        m.bin_type = p.bin_type;
        m.default_bin = 0;
        let zero_cnt = (p.total_sample_cnt - num_sample_values - na_cnt) as i32;

        // std::stable_sort with operator< over the non-NaN prefix
        values[..num_sample_values]
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut distinct_values: Vec<f64> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();
        if num_sample_values == 0 || (values[0] > 0.0 && zero_cnt > 0) {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }
        if num_sample_values > 0 {
            distinct_values.push(values[0]);
            counts.push(1);
        }
        for i in 1..num_sample_values {
            if !check_double_equal_ordered(values[i - 1], values[i]) {
                if values[i - 1] < 0.0 && values[i] > 0.0 {
                    distinct_values.push(0.0);
                    counts.push(zero_cnt);
                }
                distinct_values.push(values[i]);
                counts.push(1);
            } else {
                // use the large value
                if let Some(last) = distinct_values.last_mut() {
                    *last = values[i];
                }
                if let Some(c) = counts.last_mut() {
                    *c += 1;
                }
            }
        }
        if num_sample_values > 0 && values[num_sample_values - 1] < 0.0 && zero_cnt > 0 {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }
        m.min_val = distinct_values[0];
        m.max_val = distinct_values[distinct_values.len() - 1];
        let mut cnt_in_bin: Vec<i32> = Vec::new();

        if p.bin_type == BinType::Numerical {
            match m.missing_type {
                MissingType::Zero => {
                    m.bin_upper_bound = find_bin_with_zero_as_one_bin(
                        &distinct_values,
                        &counts,
                        p.max_bin,
                        p.total_sample_cnt,
                        p.min_data_in_bin,
                        p.forced_upper_bounds,
                    );
                    if m.bin_upper_bound.len() == 2 {
                        m.missing_type = MissingType::None;
                    }
                }
                MissingType::None => {
                    m.bin_upper_bound = find_bin_with_zero_as_one_bin(
                        &distinct_values,
                        &counts,
                        p.max_bin,
                        p.total_sample_cnt,
                        p.min_data_in_bin,
                        p.forced_upper_bounds,
                    );
                }
                MissingType::NaN => {
                    m.bin_upper_bound = find_bin_with_zero_as_one_bin(
                        &distinct_values,
                        &counts,
                        p.max_bin - 1,
                        p.total_sample_cnt - na_cnt,
                        p.min_data_in_bin,
                        p.forced_upper_bounds,
                    );
                    // upstream writes `push_back(NaN)`, where `NaN` resolves to the enum
                    // `MissingType::NaN` (= 2), so the stored bound is 2.0. It is never
                    // compared (ValueToBin stops before the NaN bin) but it is serialized.
                    m.bin_upper_bound.push(f64::from(MissingType::NaN as i32));
                }
            }
            m.num_bin = m.bin_upper_bound.len() as i32;
            cnt_in_bin = vec![0; m.num_bin as usize];
            let mut i_bin = 0usize;
            for (i, &dv) in distinct_values.iter().enumerate() {
                while dv > m.bin_upper_bound[i_bin] && (i_bin as i32) < m.num_bin - 1 {
                    i_bin += 1;
                }
                cnt_in_bin[i_bin] += counts[i];
            }
            if m.missing_type == MissingType::NaN {
                let last = cnt_in_bin.len() - 1;
                cnt_in_bin[last] = na_cnt as i32;
            }
        } else {
            let mut distinct_values_int: Vec<i32> = Vec::new();
            let mut counts_int: Vec<i32> = Vec::new();
            for (i, &dv) in distinct_values.iter().enumerate() {
                let val = dv as i32; // static_cast<int>: truncation
                if val < 0 {
                    na_cnt += counts[i] as usize;
                    log::warn!(
                        "Met negative value in categorical features, will convert it to NaN"
                    );
                } else if distinct_values_int.last() != Some(&val) {
                    distinct_values_int.push(val);
                    counts_int.push(counts[i]);
                } else if let Some(c) = counts_int.last_mut() {
                    *c += counts[i];
                }
            }
            let rest_cnt = p.total_sample_cnt as i64 - na_cnt as i64;
            if rest_cnt > 0 {
                // upstream SortForPair(counts, values, 0, is_reverse=true): stable, descending by count
                let mut pairs: Vec<(i32, i32)> = counts_int
                    .iter()
                    .copied()
                    .zip(distinct_values_int.iter().copied())
                    .collect();
                pairs.sort_by_key(|p| std::cmp::Reverse(p.0));
                counts_int = pairs.iter().map(|p| p.0).collect();
                distinct_values_int = pairs.iter().map(|p| p.1).collect();
                // RoundInt((total - na) * 0.99f): size_t * float is computed in *float*,
                // then RoundInt adds 0.5f in double.
                let prod = (p.total_sample_cnt - na_cnt) as f32 * 0.99f32;
                let cut_cnt = (f64::from(prod) + 0.5) as i32;
                let mut cur_cat_idx = 0usize;
                m.categorical_2_bin.clear();
                m.bin_2_categorical.clear();
                let mut used_cnt = 0i32;
                let mut distinct_cnt = distinct_values_int.len() as i32;
                if na_cnt > 0 {
                    distinct_cnt += 1;
                }
                let max_bin = distinct_cnt.min(p.max_bin);
                m.bin_2_categorical.push(-1);
                m.categorical_2_bin.insert(-1, 0);
                cnt_in_bin.push(0);
                m.num_bin = 1;
                while cur_cat_idx < distinct_values_int.len()
                    && (used_cnt < cut_cnt || m.num_bin < max_bin)
                {
                    if counts_int[cur_cat_idx] < p.min_data_in_bin && cur_cat_idx > 1 {
                        break;
                    }
                    m.bin_2_categorical.push(distinct_values_int[cur_cat_idx]);
                    m.categorical_2_bin
                        .insert(distinct_values_int[cur_cat_idx], m.num_bin as u32);
                    used_cnt += counts_int[cur_cat_idx];
                    cnt_in_bin.push(counts_int[cur_cat_idx]);
                    m.num_bin += 1;
                    cur_cat_idx += 1;
                }
                m.missing_type = if cur_cat_idx == distinct_values_int.len() && na_cnt == 0 {
                    MissingType::None
                } else {
                    MissingType::NaN
                };
                cnt_in_bin[0] = (p.total_sample_cnt as i64 - i64::from(used_cnt)) as i32;
            }
        }

        m.is_trivial = m.num_bin <= 1;
        if !m.is_trivial
            && p.pre_filter
            && need_filter(
                &cnt_in_bin,
                p.total_sample_cnt as i32,
                p.min_split_data,
                p.bin_type,
            )
        {
            m.is_trivial = true;
        }
        if !m.is_trivial {
            m.default_bin = m.value_to_bin(0.0);
            m.most_freq_bin = argmax_first(&cnt_in_bin) as u32;
            let max_sparse_rate =
                f64::from(cnt_in_bin[m.most_freq_bin as usize]) / p.total_sample_cnt as f64;
            if m.most_freq_bin != m.default_bin && max_sparse_rate < K_SPARSE_THRESHOLD {
                m.most_freq_bin = m.default_bin;
            }
            m.sparse_rate =
                f64::from(cnt_in_bin[m.most_freq_bin as usize]) / p.total_sample_cnt as f64;
        } else {
            m.sparse_rate = 1.0;
        }
        m
    }

    /// upstream: `BinMapper::ValueToBin`
    #[inline]
    pub fn value_to_bin(&self, mut value: f64) -> u32 {
        if value.is_nan() {
            if self.bin_type == BinType::Categorical {
                return 0;
            } else if self.missing_type == MissingType::NaN {
                return (self.num_bin - 1) as u32;
            }
            value = 0.0;
        }
        if self.bin_type == BinType::Numerical {
            let mut l = 0i32;
            let mut r = self.num_bin - 1;
            if self.missing_type == MissingType::NaN {
                r -= 1;
            }
            while l < r {
                let mid = (r + l - 1) / 2;
                if value <= self.bin_upper_bound[mid as usize] {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            l as u32
        } else {
            let int_value = value as i32;
            if int_value < 0 {
                return 0;
            }
            self.categorical_2_bin.get(&int_value).copied().unwrap_or(0)
        }
    }

    /// upstream: `BinMapper::BinToValue`
    pub fn bin_to_value(&self, bin: u32) -> f64 {
        if self.bin_type == BinType::Numerical {
            self.bin_upper_bound[bin as usize]
        } else {
            f64::from(self.bin_2_categorical[bin as usize])
        }
    }

    /// Field-by-field equality with bit-exact float comparison (NaN == NaN).
    pub fn bitwise_eq(&self, other: &BinMapper) -> bool {
        let feq = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        self.num_bin == other.num_bin
            && self.missing_type == other.missing_type
            && self.is_trivial == other.is_trivial
            && feq(self.sparse_rate, other.sparse_rate)
            && self.bin_type == other.bin_type
            && feq(self.min_val, other.min_val)
            && feq(self.max_val, other.max_val)
            && self.default_bin == other.default_bin
            && self.most_freq_bin == other.most_freq_bin
            && self.bin_upper_bound.len() == other.bin_upper_bound.len()
            && self
                .bin_upper_bound
                .iter()
                .zip(&other.bin_upper_bound)
                .all(|(a, b)| feq(*a, *b))
            && self.bin_2_categorical == other.bin_2_categorical
    }

    /// upstream: `BinMapper::SizesInByte`
    pub fn sizes_in_byte(&self) -> usize {
        8 + 8
            + 8
            + 8
            + 8
            + 8
            + 8
            + 8
            + 8
            + if self.bin_type == BinType::Numerical {
                8 * self.num_bin as usize
            } else {
                4 * self.num_bin as usize
            }
    }

    /// upstream: `BinMapper::CopyFrom` / `BinMapper(const void* memory)`
    pub fn from_bytes(buf: &[u8]) -> Result<(BinMapper, usize)> {
        let mut r = super::binary::Reader::new(buf);
        let num_bin = r.aligned_i32()?;
        let missing_type = MissingType::from_bits(r.aligned_i32()? as i8);
        let is_trivial = r.aligned_bool()?;
        let sparse_rate = r.f64()?;
        let bin_type = match r.aligned_i32()? {
            0 => BinType::Numerical,
            1 => BinType::Categorical,
            t => return Err(Error::Parse(format!("unknown bin type {t}"))),
        };
        let min_val = r.f64()?;
        let max_val = r.f64()?;
        let default_bin = r.aligned_i32()? as u32;
        let most_freq_bin = r.aligned_i32()? as u32;
        let n = usize::try_from(num_bin).map_err(|_| Error::Parse("negative num_bin".into()))?;
        let mut m = BinMapper {
            num_bin,
            missing_type,
            is_trivial,
            sparse_rate,
            bin_type,
            min_val,
            max_val,
            default_bin,
            most_freq_bin,
            ..BinMapper::default()
        };
        if bin_type == BinType::Numerical {
            m.bin_upper_bound = (0..n).map(|_| r.f64()).collect::<Result<_>>()?;
        } else {
            m.bin_2_categorical = (0..n).map(|_| r.i32()).collect::<Result<_>>()?;
            for (i, &c) in m.bin_2_categorical.iter().enumerate() {
                m.categorical_2_bin.insert(c, i as u32);
            }
        }
        Ok((m, r.pos()))
    }
}

/// upstream `ArrayArgs::ArgMax`: index of the first maximum.
fn argmax_first(v: &[i32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(max_bin: i32) -> FindBinParams<'static> {
        FindBinParams {
            total_sample_cnt: 0,
            max_bin,
            min_data_in_bin: 3,
            min_split_data: 0,
            pre_filter: false,
            bin_type: BinType::Numerical,
            use_missing: true,
            zero_as_missing: false,
            forced_upper_bounds: &[],
        }
    }

    #[test]
    fn small_numeric_feature_gets_midpoint_bounds() {
        let mut v: Vec<f64> = (1..=9).map(|i| f64::from(i % 3 + 1)).collect(); // 1,2,3 x3
        let p = FindBinParams {
            total_sample_cnt: 9,
            ..params(255)
        };
        let m = BinMapper::find_bin(&mut v, &p);
        assert_eq!(m.num_bin, 4); // [<=0 (zero bin)], (1.5], (2.5], inf  -> zero has no data
        assert!(!m.is_trivial);
        assert_eq!(m.value_to_bin(1.0), m.value_to_bin(1.4));
        assert_ne!(m.value_to_bin(1.0), m.value_to_bin(2.0));
        assert_eq!(m.value_to_bin(1e9), (m.num_bin - 1) as u32);
    }

    #[test]
    fn nan_gets_last_bin() {
        let mut v = vec![1.0, 2.0, f64::NAN, 3.0, f64::NAN, 1.0, 2.0, 3.0];
        let p = FindBinParams {
            total_sample_cnt: 8,
            min_data_in_bin: 1,
            ..params(255)
        };
        let m = BinMapper::find_bin(&mut v, &p);
        assert_eq!(m.missing_type, MissingType::NaN);
        assert_eq!(m.bin_upper_bound.last(), Some(&2.0)); // upstream quirk, see find_bin
        assert_eq!(m.value_to_bin(f64::NAN), (m.num_bin - 1) as u32);
    }

    #[test]
    fn constant_feature_is_trivial_only_with_pre_filter() {
        let p = FindBinParams {
            total_sample_cnt: 10,
            ..params(255)
        };
        assert!(!BinMapper::find_bin(&mut [5.0; 10], &p).is_trivial);
        let p = FindBinParams {
            pre_filter: true,
            min_split_data: 1,
            ..p
        };
        assert!(BinMapper::find_bin(&mut [5.0; 10], &p).is_trivial);
        // all-zero column: a single bin, trivial even without the filter
        let p = FindBinParams {
            pre_filter: false,
            ..p
        };
        assert!(BinMapper::find_bin(&mut [], &p).is_trivial);
    }

    #[test]
    fn greedy_respects_max_bin() {
        let dv: Vec<f64> = (0..1000).map(f64::from).collect();
        let c = vec![1; 1000];
        let b = greedy_find_bin(&dv, &c, 10, 1000, 3);
        assert!(b.len() <= 10);
        assert_eq!(*b.last().unwrap_or(&0.0), f64::INFINITY);
    }
}
