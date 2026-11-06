//! Row sampling and bin-mapper construction for in-memory data.
//! upstream: src/c_api.cpp (`CreateSampleIndices`, `LGBM_DatasetCreateFromMats`),
//! src/io/dataset_loader.cpp (`ConstructFromSampleData`)

use rayon::prelude::*;

use super::bin::{BinMapper, BinType, FindBinParams};
use super::sparse::CsrMatrix;
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
        let rows: Vec<usize> = create_sample_indices(total, cfg)
            .into_iter()
            .map(|idx| idx as usize)
            .collect();
        Ok(Self::from_rows(data, nrow, ncol, &rows))
    }

    /// Sample a row-major dense matrix the way upstream's two-round loader does: a reservoir
    /// over the rows in file order, drawing `Random::NextInt(0, line + 1)` for every line past the
    /// first `bin_construct_sample_cnt` (upstream: `TextReader::SampleFromFile`). The sample keeps
    /// the reservoir's slot order.
    pub fn from_dense_reservoir(
        data: &[f64],
        nrow: usize,
        ncol: usize,
        cfg: &Config,
    ) -> Result<SampleSet> {
        if data.len() != nrow * ncol {
            return Err(Error::InvalidParameter(
                "data size does not match nrow x ncol".into(),
            ));
        }
        let sample_cnt = usize::try_from(cfg.bin_construct_sample_cnt).unwrap_or(0);
        let mut random = Random::new(cfg.data_random_seed);
        let mut slots: Vec<usize> = Vec::with_capacity(sample_cnt.min(nrow));
        for line in 0..nrow {
            if slots.len() < sample_cnt {
                slots.push(line);
            } else {
                let idx = random.next_int(0, (line + 1) as i32) as usize;
                if idx < sample_cnt {
                    slots[idx] = line;
                }
            }
        }
        Ok(Self::from_rows(data, nrow, ncol, &slots))
    }

    /// Gather the non-zero (or NaN) values of the given rows, in the given order.
    fn from_rows(data: &[f64], nrow: usize, ncol: usize, rows: &[usize]) -> SampleSet {
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); ncol];
        let mut indices: Vec<Vec<i32>> = vec![Vec::new(); ncol];
        for (i, &row_idx) in rows.iter().enumerate() {
            let row = &data[row_idx * ncol..(row_idx + 1) * ncol];
            for (k, &v) in row.iter().enumerate() {
                if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                    values[k].push(v);
                    indices[k].push(i as i32);
                }
            }
        }
        SampleSet {
            num_data: nrow,
            sample_cnt: rows.len(),
            values,
            indices,
        }
    }

    /// Sample a CSR matrix the way `LGBM_DatasetCreateFromCSR` does: each sampled row's
    /// stored pairs are visited in storage order and kept when `|v| > kZeroThreshold` or
    /// NaN (a duplicated column contributes one sample value per stored entry). Absent
    /// entries are zeros, which upstream never samples (dense rows drop them too).
    pub fn from_csr(m: &CsrMatrix<'_>, cfg: &Config) -> Result<SampleSet> {
        m.check_columns()?;
        let total =
            i32::try_from(m.nrow()).map_err(|_| Error::InvalidParameter("too many rows".into()))?;
        let rows = create_sample_indices(total, cfg);
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); m.ncol()];
        let mut indices: Vec<Vec<i32>> = vec![Vec::new(); m.ncol()];
        for (i, &idx) in rows.iter().enumerate() {
            let (cols, vals) = m.row(idx as usize);
            for (&c, &v) in cols.iter().zip(vals) {
                if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                    values[c as usize].push(v);
                    indices[c as usize].push(i as i32);
                }
            }
        }
        Ok(SampleSet {
            num_data: m.nrow(),
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
        // upstream: DatasetLoader::GetForcedBins (forcedbins_filename), per feature.
        let categorical_set: std::collections::HashSet<i32> = categorical.iter().copied().collect();
        let forced =
            super::forced_bins::get_forced_bins(&cfg.forcedbins_filename, ncol, &categorical_set)?;
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
                    forced_upper_bounds: &forced[i],
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

#[cfg(test)]
mod reservoir_tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn reservoir_sample_keeps_cnt_distinct_rows_of_the_file() -> Result<()> {
        let nrow = 100;
        // Column 0 holds the row index, so the sampled values name the sampled rows.
        let data: Vec<f64> = (0..nrow).map(|i| (i + 1) as f64).collect();
        let mut params = HashMap::new();
        params.insert("bin_construct_sample_cnt".to_string(), "10".to_string());
        let cfg = Config::from_params(&params)?;
        let sample = SampleSet::from_dense_reservoir(&data, nrow, 1, &cfg)?;
        assert_eq!(sample.sample_cnt, 10);
        let mut rows: Vec<i64> = sample.values[0].iter().map(|&v| v as i64).collect();
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(rows.len(), 10);
        assert!(rows.iter().all(|&r| (1..=100).contains(&r)));
        Ok(())
    }
}
