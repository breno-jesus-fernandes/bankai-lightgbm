//! Row sampling and bin-mapper construction for in-memory data.
//! upstream: src/c_api.cpp (`CreateSampleIndices`, `LGBM_DatasetCreateFromMats`),
//! src/io/dataset_loader.cpp (`ConstructFromSampleData`)

use rayon::prelude::*;

use super::bin::{BinMapper, BinType, FindBinParams};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::meta::K_ZERO_THRESHOLD;
use crate::utils::random::Random;

/// upstream: `CreateSampleIndices` — sorted row indices used to build bin mappers.
pub fn create_sample_indices(total_nrow: i32, cfg: &Config) -> Vec<i32> {
    let sample_cnt = total_nrow.min(cfg.bin_construct_sample_cnt);
    Random::new(cfg.data_random_seed).sample(total_nrow, sample_cnt)
}

/// Per-column sampled non-zero values and their positions within the sample
/// (upstream `sample_values` / `sample_idx`).
#[derive(Debug, Clone)]
pub struct SampleSet {
    pub num_data: usize,
    pub sample_cnt: usize,
    pub values: Vec<Vec<f64>>,
    pub indices: Vec<Vec<i32>>,
}

impl SampleSet {
    /// Sample a row-major dense matrix (upstream keeps `|v| > kZeroThreshold` or NaN).
    pub fn from_dense(data: &[f64], nrow: usize, ncol: usize, cfg: &Config) -> Result<SampleSet> {
        if data.len() != nrow * ncol {
            return Err(Error::InvalidParameter(
                "data size does not match nrow x ncol".into(),
            ));
        }
        let total =
            i32::try_from(nrow).map_err(|_| Error::InvalidParameter("too many rows".into()))?;
        let rows = create_sample_indices(total, cfg);
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); ncol];
        let mut indices: Vec<Vec<i32>> = vec![Vec::new(); ncol];
        for (i, &idx) in rows.iter().enumerate() {
            let row = &data[idx as usize * ncol..(idx as usize + 1) * ncol];
            for (k, &v) in row.iter().enumerate() {
                if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                    values[k].push(v);
                    indices[k].push(i as i32);
                }
            }
        }
        Ok(SampleSet {
            num_data: nrow,
            sample_cnt: rows.len(),
            values,
            indices,
        })
    }

    /// upstream: the bin-finding loop of `ConstructFromSampleData`. Mutates `values`
    /// exactly as upstream does (see [`BinMapper::find_bin`]).
    pub fn find_bins(
        &mut self,
        cfg: &Config,
        categorical: &[i32],
    ) -> Result<Vec<Option<BinMapper>>> {
        let ncol = self.values.len();
        if !cfg.max_bin_by_feature.is_empty() {
            if cfg.max_bin_by_feature.len() != ncol {
                return Err(Error::InvalidParameter(
                    "max_bin_by_feature size must equal number of features".into(),
                ));
            }
            if cfg.max_bin_by_feature.iter().copied().min().unwrap_or(2) <= 1 {
                return Err(Error::InvalidParameter(
                    "max_bin_by_feature values must be > 1".into(),
                ));
            }
        }
        let sample_cnt = self.sample_cnt;
        // upstream: filter_cnt = (int)((double)(min_data_in_leaf * total_sample_size) / num_data)
        let filter_cnt =
            ((f64::from(cfg.min_data_in_leaf) * sample_cnt as f64) / self.num_data as f64) as i32;
        self.values
            .par_iter_mut()
            .enumerate()
            .map(|(i, values)| {
                let is_cat = categorical.contains(&(i as i32));
                if is_cat && cfg.monotone_constraints.get(i).is_some_and(|&c| c != 0) {
                    return Err(Error::InvalidParameter(
                        "The output cannot be monotone with respect to categorical features".into(),
                    ));
                }
                let max_bin = if cfg.max_bin_by_feature.is_empty() {
                    cfg.max_bin
                } else {
                    cfg.max_bin_by_feature[i]
                };
                let p = FindBinParams {
                    total_sample_cnt: sample_cnt,
                    max_bin,
                    min_data_in_bin: cfg.min_data_in_bin,
                    min_split_data: filter_cnt,
                    pre_filter: cfg.feature_pre_filter,
                    bin_type: if is_cat {
                        BinType::Categorical
                    } else {
                        BinType::Numerical
                    },
                    use_missing: cfg.use_missing,
                    zero_as_missing: cfg.zero_as_missing,
                    forced_upper_bounds: &[],
                };
                Ok(Some(BinMapper::find_bin(values, &p)))
            })
            .collect()
    }
}

/// Build one `BinMapper` per column from a row-major dense matrix, exactly as upstream's
/// `LGBM_DatasetCreateFromMat` + `ConstructFromSampleData` do. `None` marks ignored columns.
pub fn construct_bin_mappers_dense(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    cfg: &Config,
    categorical: &[i32],
) -> Result<Vec<Option<BinMapper>>> {
    SampleSet::from_dense(data, nrow, ncol, cfg)?.find_bins(cfg, categorical)
}
