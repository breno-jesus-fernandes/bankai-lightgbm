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

/// Build one `BinMapper` per column from a row-major dense matrix, exactly as upstream's
/// `LGBM_DatasetCreateFromMat` + `ConstructFromSampleData` do. `None` marks ignored columns.
pub fn construct_bin_mappers_dense(
    data: &[f64],
    nrow: usize,
    ncol: usize,
    cfg: &Config,
    categorical: &[i32],
) -> Result<Vec<Option<BinMapper>>> {
    if data.len() != nrow * ncol {
        return Err(Error::InvalidParameter(
            "data size does not match nrow x ncol".into(),
        ));
    }
    let total = i32::try_from(nrow).map_err(|_| Error::InvalidParameter("too many rows".into()))?;
    let sample_indices = create_sample_indices(total, cfg);
    let sample_cnt = sample_indices.len();
    // Collect per-column sampled non-zero values (upstream keeps |v| > kZeroThreshold or NaN).
    let mut sample_values: Vec<Vec<f64>> = vec![Vec::new(); ncol];
    for &idx in &sample_indices {
        let row = &data[idx as usize * ncol..(idx as usize + 1) * ncol];
        for (k, &v) in row.iter().enumerate() {
            if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                sample_values[k].push(v);
            }
        }
    }
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
    // upstream: filter_cnt = (int)((double)(min_data_in_leaf * total_sample_size) / num_data)
    let filter_cnt = ((f64::from(cfg.min_data_in_leaf) * sample_cnt as f64) / nrow as f64) as i32;
    let mappers = sample_values
        .into_par_iter()
        .enumerate()
        .map(|(i, mut values)| {
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
            Ok(Some(BinMapper::find_bin(&mut values, &p)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(mappers)
}
