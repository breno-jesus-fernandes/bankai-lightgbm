//! Dataset layout: used features, Exclusive Feature Bundling and feature groups.
//! upstream: src/io/dataset.cpp (`FindGroups`, `FastFeatureBundling`, `Dataset::Construct`),
//! include/LightGBM/feature_group.h

use rayon::prelude::*;

use super::bin::{BinMapper, BinType, K_SPARSE_THRESHOLD};
use super::bin_data::{BinData, DenseBin, SparseBin};
use super::binary::{BINARY_FILE_TOKEN, aligned};
use super::sample::SampleSet;
use crate::config::Config;
use crate::error::{Error, Result};
use crate::utils::random::Random;

/// upstream: `MultiValBin::multi_val_bin_sparse_threshold` (`0.25f` widened to double).
const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25f32 as f64;

/// A feature group's layout (upstream `FeatureGroup` without bin data).
#[derive(Debug, Clone)]
pub struct FeatureGroupLayout {
    /// Original (real) feature indices in this group, in sub-feature order.
    pub features: Vec<i32>,
    pub bin_mappers: Vec<BinMapper>,
    pub is_multi_val: bool,
    pub is_dense_multi_val: bool,
    pub is_sparse: bool,
    pub num_total_bin: u32,
    pub bin_offsets: Vec<u32>,
}

impl FeatureGroupLayout {
    /// upstream: `FeatureGroup(int num_feature, int8_t is_multi_val, ..., group_id)` +
    /// `CreateBinData(num_data, is_multi_val, force_dense=true, force_sparse=false)`.
    fn new(
        features: Vec<i32>,
        bin_mappers: Vec<BinMapper>,
        is_multi_val: bool,
        group_id: usize,
    ) -> Self {
        let n = bin_mappers.len();
        let mut sum_sparse_rate = 0.0f64;
        for m in &bin_mappers {
            sum_sparse_rate += m.sparse_rate;
        }
        sum_sparse_rate /= n as f64;
        let mut offset = 1u32;
        let mut is_dense_multi_val = false;
        if sum_sparse_rate < MULTI_VAL_BIN_SPARSE_THRESHOLD && is_multi_val {
            offset = 0;
            is_dense_multi_val = true;
        }
        let mut num_total_bin = offset;
        if group_id == 0 && n > 0 && is_dense_multi_val && bin_mappers[0].most_freq_bin > 0 {
            num_total_bin = 1;
        }
        let mut bin_offsets = vec![num_total_bin];
        for m in &bin_mappers {
            let mut num_bin = m.num_bin as u32;
            if m.most_freq_bin == 0 {
                num_bin -= offset;
            }
            num_total_bin += num_bin;
            bin_offsets.push(num_total_bin);
        }
        // CreateBinData(force_dense = true): single-value groups start dense.
        FeatureGroupLayout {
            features,
            bin_mappers,
            is_multi_val,
            is_dense_multi_val,
            is_sparse: false,
            num_total_bin,
            bin_offsets,
        }
    }
}

/// upstream `Dataset` structural fields (`SerializeHeader`), minus bin data.
#[derive(Debug, Clone)]
pub struct DatasetLayout {
    pub num_data: usize,
    pub num_total_features: usize,
    pub num_features: usize,
    pub used_feature_map: Vec<i32>,
    pub real_feature_idx: Vec<i32>,
    pub feature2group: Vec<i32>,
    pub feature2subfeature: Vec<i32>,
    pub group_bin_boundaries: Vec<u64>,
    pub group_feature_start: Vec<i32>,
    pub group_feature_cnt: Vec<i32>,
    pub groups: Vec<FeatureGroupLayout>,
}

fn get_conflict_count(mark: &[bool], indices: &[i32], max_cnt: i32) -> i32 {
    let mut ret = 0;
    for &i in indices {
        if mark[i as usize] {
            ret += 1;
        }
        if ret > max_cnt {
            return -1;
        }
    }
    ret
}

fn mark_used(mark: &mut [bool], indices: &[i32]) {
    for &i in indices {
        mark[i as usize] = true;
    }
}

/// upstream: `FixSampleIndices` — sample positions whose bin differs from the most
/// frequent bin, when that is not the zero (default) bin.
fn fix_sample_indices(
    m: &BinMapper,
    num_total_samples: i32,
    indices: &[i32],
    values: &[f64],
) -> Vec<i32> {
    let mut ret = Vec::new();
    if m.default_bin == m.most_freq_bin {
        return ret;
    }
    let (mut i, mut j) = (0i32, 0usize);
    let n = indices.len();
    while i < num_total_samples {
        if j < n && indices[j] < i {
            j += 1;
        } else if j < n && indices[j] == i {
            if m.value_to_bin(values[j]) != m.most_freq_bin {
                ret.push(i);
            }
            i += 1;
        } else {
            ret.push(i);
            i += 1;
        }
    }
    ret
}

/// upstream: `FindGroups`
#[allow(clippy::too_many_arguments)]
fn find_groups(
    mappers: &[Option<BinMapper>],
    find_order: &[i32],
    sample_indices: &[&[i32]],
    num_per_col: &[i32],
    num_sample_col: usize,
    total_sample_cnt: i32,
    num_data: i32,
    is_use_gpu: bool,
    is_sparse: bool,
) -> (Vec<Vec<i32>>, Vec<bool>) {
    const MAX_SEARCH_GROUP: i32 = 100;
    const MAX_BIN_PER_GROUP: i32 = 256;
    let single_val_max_conflict_cnt = total_sample_cnt / 10000;
    let mut rand = Random::new(num_data);
    let mut features_in_group: Vec<Vec<i32>> = Vec::new();
    let mut conflict_marks: Vec<Vec<bool>> = Vec::new();
    let mut group_used_row_cnt: Vec<i32> = Vec::new();
    let mut group_total_data_cnt: Vec<i32> = Vec::new();
    let mut group_num_bin: Vec<i32> = Vec::new();
    let mapper = |f: i32| mappers[f as usize].as_ref();

    for &fidx in find_order {
        let Some(m) = mapper(fidx) else { continue };
        let is_filtered = fidx as usize >= num_sample_col;
        let cur_non_zero_cnt = if is_filtered {
            0
        } else {
            num_per_col[fidx as usize]
        };
        let mfb_zero = i32::from(m.most_freq_bin == 0);
        let mut available_groups = Vec::new();
        for gid in 0..features_in_group.len() {
            let cur_num_bin = group_num_bin[gid] + m.num_bin - mfb_zero;
            if group_total_data_cnt[gid] + cur_non_zero_cnt
                <= total_sample_cnt + single_val_max_conflict_cnt
                && (!is_use_gpu || cur_num_bin <= MAX_BIN_PER_GROUP)
            {
                available_groups.push(gid);
            }
        }
        let mut search_groups = Vec::new();
        if let Some(&last_gid) = available_groups.last() {
            let last = available_groups.len() as i32 - 1;
            let idx = rand.sample(last, last.min(MAX_SEARCH_GROUP - 1));
            search_groups.push(last_gid);
            for i in idx {
                search_groups.push(available_groups[i as usize]);
            }
        }
        let mut best_gid: Option<usize> = None;
        let mut best_conflict_cnt = -1;
        for &gid in &search_groups {
            let rest_max_cnt =
                single_val_max_conflict_cnt - group_total_data_cnt[gid] + group_used_row_cnt[gid];
            let cnt = if is_filtered {
                0
            } else {
                get_conflict_count(
                    &conflict_marks[gid],
                    sample_indices[fidx as usize],
                    rest_max_cnt,
                )
            };
            if cnt >= 0 && cnt <= rest_max_cnt && cnt <= cur_non_zero_cnt / 2 {
                best_gid = Some(gid);
                best_conflict_cnt = cnt;
                break;
            }
        }
        match best_gid {
            Some(g) => {
                features_in_group[g].push(fidx);
                group_total_data_cnt[g] += cur_non_zero_cnt;
                group_used_row_cnt[g] += cur_non_zero_cnt - best_conflict_cnt;
                if !is_filtered {
                    mark_used(&mut conflict_marks[g], sample_indices[fidx as usize]);
                }
                // upstream uses GetDefaultBin() here (and GetMostFreqBin() elsewhere).
                group_num_bin[g] += m.num_bin - i32::from(m.default_bin == 0);
            }
            None => {
                features_in_group.push(vec![fidx]);
                let mut marks = vec![false; total_sample_cnt as usize];
                if !is_filtered {
                    mark_used(&mut marks, sample_indices[fidx as usize]);
                }
                conflict_marks.push(marks);
                group_total_data_cnt.push(cur_non_zero_cnt);
                group_used_row_cnt.push(cur_non_zero_cnt);
                group_num_bin.push(1 + m.num_bin - mfb_zero);
            }
        }
    }
    if !is_sparse {
        let n = features_in_group.len();
        return (features_in_group, vec![false; n]);
    }
    let mut second_round_features = Vec::new();
    let mut features_in_group2 = Vec::new();
    const DENSE_THRESHOLD: f64 = 0.4;
    for (gid, feats) in features_in_group.into_iter().enumerate() {
        let dense_rate = f64::from(group_used_row_cnt[gid]) / f64::from(total_sample_cnt);
        if dense_rate >= DENSE_THRESHOLD {
            features_in_group2.push(feats);
        } else {
            second_round_features.extend(feats);
        }
    }
    let mut multi_val = vec![false; features_in_group2.len()];
    let mut groups = features_in_group2;
    if !second_round_features.is_empty() {
        let mut marks = vec![false; total_sample_cnt as usize];
        let mut is_multi_val = is_use_gpu;
        let mut conflict_cnt = 0i32;
        let mut group = Vec::new();
        for fidx in second_round_features {
            group.push(fidx);
            if !is_multi_val {
                let rest_max_cnt = single_val_max_conflict_cnt - conflict_cnt;
                let cnt = get_conflict_count(&marks, sample_indices[fidx as usize], rest_max_cnt);
                conflict_cnt += cnt;
                if cnt < 0 || conflict_cnt > single_val_max_conflict_cnt {
                    is_multi_val = true;
                    continue;
                }
                mark_used(&mut marks, sample_indices[fidx as usize]);
            }
        }
        groups.push(group);
        multi_val.push(is_multi_val);
    }
    (groups, multi_val)
}

/// upstream: `FastFeatureBundling`
fn fast_feature_bundling(
    mappers: &[Option<BinMapper>],
    sample: &SampleSet,
    used_features: &[i32],
    is_use_gpu: bool,
    is_sparse: bool,
) -> (Vec<Vec<i32>>, Vec<bool>) {
    let num_sample_col = sample.values.len();
    let total_sample_cnt = sample.sample_cnt as i32;
    let num_data = sample.num_data as i32;
    let feature_non_zero_cnt: Vec<usize> = used_features
        .iter()
        .map(|&f| {
            if (f as usize) < num_sample_col {
                sample.indices[f as usize].len()
            } else {
                0
            }
        })
        .collect();
    let mut sorted_idx: Vec<usize> = (0..used_features.len()).collect();
    sorted_idx.sort_by(|&a, &b| feature_non_zero_cnt[b].cmp(&feature_non_zero_cnt[a])); // stable
    let feature_order_by_cnt: Vec<i32> = sorted_idx.iter().map(|&i| used_features[i]).collect();

    let mut fixed: Vec<Option<Vec<i32>>> = vec![None; num_sample_col];
    let mut tmp_num_per_col = vec![0i32; num_sample_col];
    for &f in used_features {
        let f = f as usize;
        if f >= num_sample_col {
            continue;
        }
        let Some(m) = mappers[f].as_ref() else {
            continue;
        };
        let ret = fix_sample_indices(m, total_sample_cnt, &sample.indices[f], &sample.values[f]);
        if ret.is_empty() {
            tmp_num_per_col[f] = sample.indices[f].len() as i32;
        } else {
            tmp_num_per_col[f] = ret.len() as i32;
            fixed[f] = Some(ret);
        }
    }
    let idx_refs: Vec<&[i32]> = (0..num_sample_col)
        .map(|f| fixed[f].as_deref().unwrap_or(&sample.indices[f]))
        .collect();
    let (mut groups, mut multi) = find_groups(
        mappers,
        used_features,
        &idx_refs,
        &tmp_num_per_col,
        num_sample_col,
        total_sample_cnt,
        num_data,
        is_use_gpu,
        is_sparse,
    );
    let (groups2, multi2) = find_groups(
        mappers,
        &feature_order_by_cnt,
        &idx_refs,
        &tmp_num_per_col,
        num_sample_col,
        total_sample_cnt,
        num_data,
        is_use_gpu,
        is_sparse,
    );
    if groups.len() > groups2.len() {
        groups = groups2;
        multi = multi2;
    }
    let num_group = groups.len() as i32;
    let mut rand = Random::new(num_data);
    for i in 0..num_group - 1 {
        let j = rand.next_short(i + 1, num_group);
        groups.swap(i as usize, j as usize);
        multi.swap(i as usize, j as usize);
    }
    (groups, multi)
}

impl DatasetLayout {
    /// Sample, bin and bundle a row-major dense matrix (upstream `LGBM_DatasetCreateFromMat`).
    pub fn construct_dense(
        data: &[f64],
        nrow: usize,
        ncol: usize,
        cfg: &Config,
        categorical: &[i32],
    ) -> Result<DatasetLayout> {
        let mut sample = SampleSet::from_dense(data, nrow, ncol, cfg)?;
        let mappers = sample.find_bins(cfg, categorical)?;
        Ok(DatasetLayout::construct(mappers, &sample, cfg))
    }

    /// upstream: `Dataset::Construct`
    pub fn construct(
        mut mappers: Vec<Option<BinMapper>>,
        sample: &SampleSet,
        cfg: &Config,
    ) -> DatasetLayout {
        let num_total_features = mappers.len();
        let used_features: Vec<i32> = (0..num_total_features)
            .filter(|&i| mappers[i].as_ref().is_some_and(|m| !m.is_trivial))
            .map(|i| i as i32)
            .collect();
        if used_features.is_empty() {
            log::warn!(
                "There are no meaningful features which satisfy the provided configuration. \
                 Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing \
                 Dataset might resolve this warning."
            );
        }
        let is_sparse = cfg.is_enable_sparse && cfg.device_type != "cuda";
        let (features_in_group, group_is_multi_val) =
            if cfg.enable_bundle && !used_features.is_empty() {
                let is_use_gpu = cfg.device_type == "gpu" || cfg.device_type == "cuda";
                fast_feature_bundling(&mappers, sample, &used_features, is_use_gpu, is_sparse)
            } else {
                let g: Vec<Vec<i32>> = used_features.iter().map(|&f| vec![f]).collect();
                let n = g.len();
                (g, vec![false; n])
            };

        let num_features: usize = features_in_group.iter().map(Vec::len).sum();
        let mut used_feature_map = vec![-1i32; num_total_features];
        let mut real_feature_idx = vec![0i32; num_features];
        let mut feature2group = vec![0i32; num_features];
        let mut feature2subfeature = vec![0i32; num_features];
        let mut group_bin_boundaries = vec![0u64];
        let mut group_feature_start = Vec::with_capacity(features_in_group.len());
        let mut group_feature_cnt = Vec::with_capacity(features_in_group.len());
        let mut groups = Vec::with_capacity(features_in_group.len());
        let mut cur_fidx = 0usize;
        let mut num_total_bin = 0u64;
        for (gi, feats) in features_in_group.into_iter().enumerate() {
            group_feature_start.push(cur_fidx as i32);
            group_feature_cnt.push(feats.len() as i32);
            let mut ms = Vec::with_capacity(feats.len());
            for (j, &real) in feats.iter().enumerate() {
                used_feature_map[real as usize] = cur_fidx as i32;
                real_feature_idx[cur_fidx] = real;
                feature2group[cur_fidx] = gi as i32;
                feature2subfeature[cur_fidx] = j as i32;
                ms.push(mappers[real as usize].take().unwrap_or_default());
                cur_fidx += 1;
            }
            let g = FeatureGroupLayout::new(feats, ms, group_is_multi_val[gi], gi);
            num_total_bin += u64::from(g.num_total_bin);
            group_bin_boundaries.push(num_total_bin);
            groups.push(g);
        }
        DatasetLayout {
            num_data: sample.num_data,
            num_total_features,
            num_features,
            used_feature_map,
            real_feature_idx,
            feature2group,
            feature2subfeature,
            group_bin_boundaries,
            group_feature_start,
            group_feature_cnt,
            groups,
        }
    }
}

/// Bin data of one feature group. upstream: `FeatureGroup::bin_data_` / `multi_bin_data_`
#[derive(Debug, Clone, PartialEq)]
pub enum GroupData {
    Single(BinData),
    /// One column per sub-feature.
    Multi(Vec<BinData>),
}

/// Labels, weights and query boundaries. upstream: `Metadata`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    pub label: Vec<f32>,
    pub weights: Vec<f32>,
    pub query_boundaries: Vec<i32>,
    pub init_score: Vec<f64>,
}

/// A binned training dataset. upstream: `Dataset`
#[derive(Debug, Clone)]
pub struct Dataset {
    pub layout: DatasetLayout,
    pub data: Vec<GroupData>,
    pub metadata: Metadata,
    pub feature_names: Vec<String>,
    pub label_idx: i32,
    pub max_bin: i32,
    pub bin_construct_sample_cnt: i32,
    pub min_data_in_bin: i32,
    pub use_missing: bool,
    pub zero_as_missing: bool,
    pub max_bin_by_feature: Vec<i32>,
    pub forced_bin_bounds: Vec<Vec<f64>>,
    /// Raw float32 values per numeric used feature (linear trees), column-major.
    pub raw_data: Vec<Vec<f32>>,
    pub numeric_feature_map: Vec<i32>,
}

impl FeatureGroupLayout {
    /// Fill this group's bins from a row-major dense matrix (upstream `PushOneRow` over all
    /// rows + `FinishLoad`). Within a bundle, later columns overwrite earlier ones.
    fn fill_dense(&self, data: &[f64], nrow: usize, ncol: usize) -> GroupData {
        if self.is_multi_val {
            let cols = self
                .features
                .iter()
                .zip(&self.bin_mappers)
                .map(|(&real, m)| {
                    let addi = u32::from(m.most_freq_bin != 0);
                    let num_bin = m.num_bin as u32 + addi;
                    let bin_of = |r: usize| -> Option<u32> {
                        let mut bin = m.value_to_bin(data[r * ncol + real as usize]);
                        if bin == m.most_freq_bin {
                            return None;
                        }
                        if m.most_freq_bin == 0 {
                            bin -= 1;
                        }
                        Some(bin + 1)
                    };
                    if m.sparse_rate >= K_SPARSE_THRESHOLD {
                        let pairs: Vec<(usize, u32)> = (0..nrow)
                            .filter_map(|r| bin_of(r).map(|b| (r, b)))
                            .collect();
                        BinData::Sparse(SparseBin::from_sorted_pairs(nrow, num_bin, &pairs))
                    } else {
                        let mut d = DenseBin::new(nrow, num_bin);
                        for r in 0..nrow {
                            if let Some(b) = bin_of(r) {
                                d.set(r, b);
                            }
                        }
                        BinData::Dense(d)
                    }
                })
                .collect();
            return GroupData::Multi(cols);
        }
        let mut d = DenseBin::new(nrow, self.num_total_bin);
        // push order is by original column index
        let mut order: Vec<usize> = (0..self.features.len()).collect();
        order.sort_by_key(|&i| self.features[i]);
        for r in 0..nrow {
            let row = &data[r * ncol..(r + 1) * ncol];
            for &sub in &order {
                let m = &self.bin_mappers[sub];
                let mut bin = m.value_to_bin(row[self.features[sub] as usize]);
                if bin == m.most_freq_bin {
                    continue;
                }
                if m.most_freq_bin == 0 {
                    bin -= 1;
                }
                d.set(r, bin + self.bin_offsets[sub]);
            }
        }
        GroupData::Single(BinData::Dense(d))
    }

    /// upstream: `FeatureGroup::SizesInByte` + `SerializeToBinary`
    fn write(&self, data: &GroupData, out: &mut Vec<u8>) {
        let mut body = Vec::new();
        for flag in [self.is_multi_val, self.is_dense_multi_val, self.is_sparse] {
            body.push(u8::from(flag));
            body.extend_from_slice(&[0u8; 7]);
        }
        body.extend_from_slice(&(self.bin_mappers.len() as i32).to_le_bytes());
        body.extend_from_slice(&[0u8; 4]);
        for m in &self.bin_mappers {
            m.write(&mut body);
        }
        match data {
            GroupData::Single(b) => b.write(&mut body),
            GroupData::Multi(cols) => cols.iter().for_each(|b| b.write(&mut body)),
        }
        out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        out.extend_from_slice(&body);
    }
}

fn write_aligned(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
    out.resize(out.len() + aligned(bytes.len()) - bytes.len(), 0);
}

fn write_aligned_i32s(out: &mut Vec<u8>, vals: &[i32]) {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_aligned(out, &bytes);
}

impl Dataset {
    /// upstream: `Metadata::SetQuery`: `sizes` are the rows of each query, in order
    /// (an empty slice clears the query information).
    pub fn set_group(&mut self, sizes: &[i32]) -> Result<()> {
        if sizes.is_empty() {
            self.metadata.query_boundaries.clear();
            return Ok(());
        }
        // Negative sizes are accepted like upstream; only ranking objectives reject them.
        let mut boundaries = Vec::with_capacity(sizes.len() + 1);
        boundaries.push(0i32);
        let mut total = 0i64;
        for &s in sizes {
            total += i64::from(s);
            let last = boundaries[boundaries.len() - 1];
            boundaries.push(
                last.checked_add(s).ok_or_else(|| {
                    Error::InvalidParameter("Sum of query counts overflows".into())
                })?,
            );
        }
        if total != self.num_data() as i64 {
            return Err(Error::InvalidParameter(
                "Sum of query counts is not same with #data".into(),
            ));
        }
        self.metadata.query_boundaries = boundaries;
        Ok(())
    }

    /// upstream: `Metadata::SetInitScore` (class-major; an empty slice clears the scores).
    /// Non-finite values are clamped like upstream's `Common::AvoidInf`.
    pub fn set_init_score(&mut self, scores: &[f64]) -> Result<()> {
        if scores.is_empty() {
            self.metadata.init_score.clear();
            return Ok(());
        }
        let num_data = self.num_data();
        if num_data == 0 || !scores.len().is_multiple_of(num_data) {
            return Err(Error::InvalidParameter(
                "Initial score size doesn't match data size".into(),
            ));
        }
        self.metadata.init_score = scores.iter().map(|&s| crate::tree::avoid_inf(s)).collect();
        Ok(())
    }

    /// upstream: `LGBM_DatasetSetField("label")` (labels are stored as `f32`).
    pub fn set_label(&mut self, label: &[f64]) -> Result<()> {
        if label.len() != self.num_data() {
            return Err(Error::InvalidParameter(format!(
                "Length of label ({}) is not same with #data ({})",
                label.len(),
                self.num_data()
            )));
        }
        self.metadata.label = label.iter().map(|&v| v as f32).collect();
        Ok(())
    }

    /// upstream: `Metadata::SetWeights` (an empty slice clears the weights).
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<()> {
        if weights.is_empty() {
            self.metadata.weights.clear();
            return Ok(());
        }
        if weights.len() != self.num_data() {
            return Err(Error::InvalidParameter(
                "Length of weights differs from the length of #data".into(),
            ));
        }
        self.metadata.weights = weights
            .iter()
            .map(|&w| crate::tree::avoid_inf_f32(w))
            .collect();
        Ok(())
    }

    /// Build a binned dataset from a row-major dense matrix and labels
    /// (upstream `LGBM_DatasetCreateFromMat` + `LGBM_DatasetSetField("label")`).
    pub fn from_dense(
        data: &[f64],
        nrow: usize,
        ncol: usize,
        label: &[f64],
        cfg: &Config,
        categorical: &[i32],
    ) -> Result<Dataset> {
        if label.len() != nrow {
            return Err(Error::InvalidParameter(format!(
                "Length of label ({}) is not same with #data ({nrow})",
                label.len()
            )));
        }
        let layout = DatasetLayout::construct_dense(data, nrow, ncol, cfg, categorical)?;
        let group_data: Vec<GroupData> = layout
            .groups
            .par_iter()
            .map(|g| g.fill_dense(data, nrow, ncol))
            .collect();
        let mut numeric_feature_map = vec![-1i32; layout.num_features];
        let mut num_numeric = 0;
        for (f, slot) in numeric_feature_map.iter_mut().enumerate() {
            let g = &layout.groups[layout.feature2group[f] as usize];
            if g.bin_mappers[layout.feature2subfeature[f] as usize].bin_type == BinType::Numerical {
                *slot = num_numeric;
                num_numeric += 1;
            }
        }
        let raw_data = if cfg.linear_tree {
            (0..layout.num_features)
                .filter(|&f| numeric_feature_map[f] >= 0)
                .map(|f| {
                    let real = layout.real_feature_idx[f] as usize;
                    (0..nrow).map(|r| data[r * ncol + real] as f32).collect()
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Dataset {
            feature_names: (0..ncol).map(|i| format!("Column_{i}")).collect(),
            label_idx: 0,
            max_bin: cfg.max_bin,
            bin_construct_sample_cnt: cfg.bin_construct_sample_cnt,
            min_data_in_bin: cfg.min_data_in_bin,
            use_missing: cfg.use_missing,
            zero_as_missing: cfg.zero_as_missing,
            max_bin_by_feature: cfg.max_bin_by_feature.clone(),
            forced_bin_bounds: vec![Vec::new(); ncol],
            metadata: Metadata {
                label: label.iter().map(|&v| v as f32).collect(),
                ..Metadata::default()
            },
            data: group_data,
            layout,
            raw_data,
            numeric_feature_map,
        })
    }

    /// Bin new rows using the reference layout and bin mappers.
    /// upstream: src/io/dataset_loader.cpp (v4.7.0) LoadFromMat with reference.
    pub fn from_dense_reference(
        data: &[f64],
        nrow: usize,
        ncol: usize,
        label: &[f64],
        reference: &Dataset,
    ) -> Result<Dataset> {
        if ncol != reference.layout.num_total_features
            || data.len() != nrow.saturating_mul(ncol)
            || label.len() != nrow
        {
            return Err(Error::InvalidParameter(
                "reference dataset dimensions do not match input".into(),
            ));
        }
        let mut ds = reference.clone();
        ds.layout.num_data = nrow;
        ds.data = ds
            .layout
            .groups
            .par_iter()
            .map(|g| g.fill_dense(data, nrow, ncol))
            .collect();
        ds.metadata = Metadata {
            label: label.iter().map(|&v| v as f32).collect(),
            ..Metadata::default()
        };
        ds.raw_data = if reference.raw_data.is_empty() {
            Vec::new()
        } else {
            (0..ds.layout.num_features)
                .filter(|&f| ds.numeric_feature_map[f] >= 0)
                .map(|f| {
                    let real = ds.layout.real_feature_idx[f] as usize;
                    (0..nrow).map(|r| data[r * ncol + real] as f32).collect()
                })
                .collect()
        };
        Ok(ds)
    }

    pub fn num_data(&self) -> usize {
        self.layout.num_data
    }

    /// upstream: `Dataset::GetSubset` — select rows while reusing the source bin mappers.
    pub fn subset(&self, rows: &[usize]) -> Result<Dataset> {
        if rows.is_empty() {
            return Err(Error::InvalidParameter(
                "Dataset subset must contain at least one row".into(),
            ));
        }
        if rows.iter().any(|&row| row >= self.num_data()) {
            return Err(Error::InvalidParameter(
                "Dataset subset row index is out of range".into(),
            ));
        }
        fn subset_bin(source: &BinData, rows: &[usize], num_bins: u32) -> BinData {
            match source {
                BinData::Dense(source) => {
                    let mut subset = DenseBin::new(rows.len(), num_bins);
                    for (dst, &src) in rows.iter().enumerate() {
                        subset.set(dst, source.get(src));
                    }
                    BinData::Dense(subset)
                }
                BinData::Sparse(source) => {
                    let dense = source.to_dense();
                    let pairs: Vec<(usize, u32)> = rows
                        .iter()
                        .enumerate()
                        .filter_map(|(dst, &src)| {
                            let value = dense[src];
                            (value != 0).then_some((dst, value))
                        })
                        .collect();
                    BinData::Sparse(SparseBin::from_sorted_pairs(rows.len(), num_bins, &pairs))
                }
            }
        }

        let data = self
            .data
            .iter()
            .zip(&self.layout.groups)
            .map(|(group_data, group_layout)| match group_data {
                GroupData::Single(bin) => {
                    GroupData::Single(subset_bin(bin, rows, group_layout.num_total_bin))
                }
                GroupData::Multi(bins) => GroupData::Multi(
                    bins.iter()
                        .zip(&group_layout.bin_mappers)
                        .map(|(bin, mapper)| {
                            let num_bins =
                                mapper.num_bin as u32 + u32::from(mapper.most_freq_bin != 0);
                            subset_bin(bin, rows, num_bins)
                        })
                        .collect(),
                ),
            })
            .collect();

        let mut layout = self.layout.clone();
        layout.num_data = rows.len();
        let label = rows.iter().map(|&row| self.metadata.label[row]).collect();
        let weights = if self.metadata.weights.is_empty() {
            Vec::new()
        } else {
            rows.iter().map(|&row| self.metadata.weights[row]).collect()
        };
        // Class-major: each class block is selected with the same rows.
        let init_score = if self.metadata.init_score.is_empty() {
            Vec::new()
        } else {
            let num_data = self.num_data();
            let num_classes = self.metadata.init_score.len() / num_data;
            (0..num_classes)
                .flat_map(|class| {
                    let block = &self.metadata.init_score[class * num_data..(class + 1) * num_data];
                    rows.iter().map(|&row| block[row]).collect::<Vec<_>>()
                })
                .collect()
        };
        let query_boundaries = if self.metadata.query_boundaries.is_empty() {
            Vec::new()
        } else {
            let mut boundaries = vec![0];
            let mut current_query = None;
            let mut current_size = 0i32;
            for &row in rows {
                let query = self
                    .metadata
                    .query_boundaries
                    .partition_point(|&boundary| boundary <= row as i32)
                    .saturating_sub(1);
                if current_query.is_some_and(|previous| previous != query) {
                    let previous = boundaries.last().copied().unwrap_or(0);
                    boundaries.push(previous + current_size);
                    current_size = 0;
                }
                current_query = Some(query);
                current_size += 1;
            }
            if current_size > 0 {
                let previous = boundaries.last().copied().unwrap_or(0);
                boundaries.push(previous + current_size);
            }
            boundaries
        };
        let raw_data = self
            .raw_data
            .iter()
            .map(|column| rows.iter().map(|&row| column[row]).collect())
            .collect();

        Ok(Dataset {
            layout,
            data,
            metadata: Metadata {
                label,
                weights,
                query_boundaries,
                init_score,
            },
            feature_names: self.feature_names.clone(),
            label_idx: self.label_idx,
            max_bin: self.max_bin,
            bin_construct_sample_cnt: self.bin_construct_sample_cnt,
            min_data_in_bin: self.min_data_in_bin,
            use_missing: self.use_missing,
            zero_as_missing: self.zero_as_missing,
            max_bin_by_feature: self.max_bin_by_feature.clone(),
            forced_bin_bounds: self.forced_bin_bounds.clone(),
            raw_data,
            numeric_feature_map: self.numeric_feature_map.clone(),
        })
    }

    /// upstream: `Dataset::SaveBinaryFile` — byte-identical to upstream's output.
    pub fn to_binary_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_aligned(&mut out, BINARY_FILE_TOKEN);
        // header
        let l = &self.layout;
        let mut h = Vec::new();
        for v in [
            l.num_data as i32,
            l.num_features as i32,
            l.num_total_features as i32,
            self.label_idx,
            self.max_bin,
            self.bin_construct_sample_cnt,
            self.min_data_in_bin,
        ] {
            write_aligned_i32s(&mut h, &[v]);
        }
        for b in [
            self.use_missing,
            self.zero_as_missing,
            !self.raw_data.is_empty(),
        ] {
            write_aligned(&mut h, &[u8::from(b)]);
        }
        write_aligned_i32s(&mut h, &l.used_feature_map);
        write_aligned_i32s(&mut h, &[l.groups.len() as i32]);
        write_aligned_i32s(&mut h, &l.real_feature_idx);
        write_aligned_i32s(&mut h, &l.feature2group);
        write_aligned_i32s(&mut h, &l.feature2subfeature);
        for b in &l.group_bin_boundaries {
            h.extend_from_slice(&b.to_le_bytes());
        }
        write_aligned_i32s(&mut h, &l.group_feature_start);
        write_aligned_i32s(&mut h, &l.group_feature_cnt);
        if self.max_bin_by_feature.is_empty() {
            write_aligned_i32s(&mut h, &vec![-1; l.num_total_features]);
        } else {
            write_aligned_i32s(&mut h, &self.max_bin_by_feature);
        }
        for name in &self.feature_names {
            write_aligned_i32s(&mut h, &[name.len() as i32]);
            write_aligned(&mut h, name.as_bytes());
        }
        for bounds in &self.forced_bin_bounds {
            write_aligned_i32s(&mut h, &[bounds.len() as i32]);
            for b in bounds {
                h.extend_from_slice(&b.to_le_bytes());
            }
        }
        out.extend_from_slice(&(h.len() as u64).to_le_bytes());
        out.extend_from_slice(&h);

        // metadata
        let md = &self.metadata;
        let mut m = Vec::new();
        let num_queries = md.query_boundaries.len().saturating_sub(1);
        for v in [
            l.num_data as i32,
            md.weights.len() as i32,
            num_queries as i32,
        ] {
            write_aligned_i32s(&mut m, &[v]);
        }
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        write_aligned(&mut m, &f32s(&md.label));
        if !md.weights.is_empty() {
            write_aligned(&mut m, &f32s(&md.weights));
        }
        if !md.query_boundaries.is_empty() {
            write_aligned_i32s(&mut m, &md.query_boundaries);
        }
        out.extend_from_slice(&(m.len() as u64).to_le_bytes());
        out.extend_from_slice(&m);

        for (g, d) in l.groups.iter().zip(&self.data) {
            g.write(d, &mut out);
        }
        // raw data, row-major over numeric features
        if !self.raw_data.is_empty() {
            for r in 0..l.num_data {
                for col in &self.raw_data {
                    out.extend_from_slice(&col[r].to_le_bytes());
                }
            }
        }
        out
    }

    pub fn save_binary(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_binary_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod subset_tests {
    use super::*;
    use std::collections::HashMap;

    fn bin_values(bin: &BinData) -> Vec<u32> {
        match bin {
            BinData::Dense(dense) => (0..dense.num_rows()).map(|row| dense.get(row)).collect(),
            BinData::Sparse(sparse) => sparse.to_dense(),
        }
    }

    #[test]
    fn subset_keeps_source_bins_and_selects_metadata_rows() -> crate::Result<()> {
        let values = [
            0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0,
        ];
        let labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let cfg = Config::from_params(&HashMap::new())?;
        let mut dataset = Dataset::from_dense(&values, 6, 2, &labels, &cfg, &[])?;
        dataset.set_weights(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        dataset.set_group(&[2, 2, 2])?;

        let rows = [0, 2, 3, 5];
        let subset = dataset.subset(&rows)?;
        assert_eq!(subset.num_data(), rows.len());
        assert_eq!(subset.metadata.label, vec![0.0, 2.0, 3.0, 5.0]);
        assert_eq!(subset.metadata.weights, vec![1.0, 3.0, 4.0, 6.0]);
        assert_eq!(subset.metadata.query_boundaries, vec![0, 1, 3, 4]);

        for (source_group, subset_group) in dataset.data.iter().zip(&subset.data) {
            let (source_bins, subset_bins): (Vec<&BinData>, Vec<&BinData>) =
                match (source_group, subset_group) {
                    (GroupData::Single(source), GroupData::Single(subset)) => {
                        (vec![source], vec![subset])
                    }
                    (GroupData::Multi(source), GroupData::Multi(subset)) => {
                        (source.iter().collect(), subset.iter().collect())
                    }
                    _ => {
                        return Err(Error::InvalidParameter(
                            "subset changed feature group layout".into(),
                        ));
                    }
                };
            for (source, selected) in source_bins.iter().zip(subset_bins) {
                let source = bin_values(source);
                let selected = bin_values(selected);
                for (dst, &src) in rows.iter().enumerate() {
                    assert_eq!(selected[dst], source[src]);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn negative_query_sizes_are_stored_like_upstream() -> crate::Result<()> {
        // upstream: src/io/metadata.cpp (v4.7.0) SetQueriesFromIterator only checks that
        // the sizes sum to #data, so `[-1, 13]` for 12 rows is accepted as-is.
        let values: Vec<f64> = (0..24).map(f64::from).collect();
        let labels: Vec<f64> = (0..12).map(f64::from).collect();
        let cfg = Config::from_params(&HashMap::new())?;
        let mut dataset = Dataset::from_dense(&values, 12, 2, &labels, &cfg, &[])?;
        dataset.set_group(&[-1, 13])?;
        assert_eq!(dataset.metadata.query_boundaries, vec![0, -1, 12]);
        Ok(())
    }

    fn dataset_with_rows(rows: usize) -> crate::Result<Dataset> {
        let values: Vec<f64> = (0..rows * 2).map(|v| v as f64).collect();
        let labels: Vec<f64> = (0..rows).map(|v| v as f64).collect();
        let cfg = Config::from_params(&HashMap::new())?;
        Dataset::from_dense(&values, rows, 2, &labels, &cfg, &[])
    }

    #[test]
    fn init_score_size_must_be_a_multiple_of_num_data() -> crate::Result<()> {
        // upstream: src/io/metadata.cpp (v4.7.0) SetInitScoresFromIterator
        let mut dataset = dataset_with_rows(12)?;
        let result = dataset.set_init_score(&[1.0; 11]);
        assert!(matches!(
            result,
            Err(Error::InvalidParameter(ref message))
                if message == "Initial score size doesn't match data size"
        ));
        assert!(dataset.metadata.init_score.is_empty());
        Ok(())
    }

    #[test]
    fn init_score_is_stored_class_major_with_avoid_inf() -> crate::Result<()> {
        // upstream: Common::AvoidInf(double): NaN -> 0, +-inf -> +-1e300
        let mut dataset = dataset_with_rows(3)?;
        let scores = [f64::NAN, f64::INFINITY, -f64::INFINITY, 0.5, 1.5, 2.5];
        dataset.set_init_score(&scores)?;
        assert_eq!(
            dataset.metadata.init_score,
            vec![0.0, 1e300, -1e300, 0.5, 1.5, 2.5]
        );
        dataset.set_init_score(&[])?;
        assert!(dataset.metadata.init_score.is_empty());
        Ok(())
    }

    #[test]
    fn subset_selects_init_score_rows_for_every_class() -> crate::Result<()> {
        // upstream: Metadata::Init(fullset, used_indices) copies init_score per class.
        let mut dataset = dataset_with_rows(4)?;
        // Two classes, class-major: class 0 = [0,1,2,3], class 1 = [10,11,12,13].
        dataset.set_init_score(&[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])?;
        let subset = dataset.subset(&[2, 0])?;
        assert_eq!(subset.metadata.init_score, vec![2.0, 0.0, 12.0, 10.0]);
        Ok(())
    }
}
