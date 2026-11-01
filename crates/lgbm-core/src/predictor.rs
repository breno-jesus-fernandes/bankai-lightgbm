//! Batch prediction over dense and sparse (CSR/CSC) rows. upstream: src/application/predictor.hpp,
//! src/boosting/gbdt_prediction.cpp, src/boosting/prediction_early_stop.cpp

use rayon::prelude::*;

use crate::boosting::Gbdt;
use crate::error::{Error, Result};
use crate::meta::K_ZERO_THRESHOLD;
use crate::tree::ShapScratch;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredictKind {
    /// Transformed output (probabilities for classification).
    #[default]
    Normal,
    RawScore,
    /// Leaf index per tree (`pred_leaf`).
    LeafIndex,
    /// SHAP feature contributions (`pred_contrib`).
    Contrib,
}

#[derive(Debug, Clone)]
pub struct PredictConfig {
    pub kind: PredictKind,
    pub start_iteration: i32,
    /// `<= 0` means all iterations.
    pub num_iteration: i32,
    pub pred_early_stop: bool,
    pub pred_early_stop_freq: i32,
    pub pred_early_stop_margin: f64,
    /// `0` uses rayon's global pool.
    pub num_threads: usize,
}

impl Default for PredictConfig {
    fn default() -> Self {
        PredictConfig {
            kind: PredictKind::Normal,
            start_iteration: 0,
            num_iteration: -1,
            pred_early_stop: false,
            pred_early_stop_freq: 10,
            pred_early_stop_margin: 10.0,
            num_threads: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EarlyStop {
    None,
    Binary { margin: f64, period: usize },
    Multiclass { margin: f64, period: usize },
}

impl EarlyStop {
    fn period(&self) -> usize {
        match *self {
            EarlyStop::None => usize::MAX,
            EarlyStop::Binary { period, .. } | EarlyStop::Multiclass { period, .. } => period,
        }
    }

    #[inline]
    fn should_stop(&self, pred: &[f64]) -> bool {
        match *self {
            EarlyStop::None => false,
            EarlyStop::Binary { margin, .. } => 2.0 * pred[0].abs() > margin,
            EarlyStop::Multiclass { margin, .. } => {
                let (mut a, mut b) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &v in pred {
                    if v > a {
                        b = a;
                        a = v;
                    } else if v > b {
                        b = v;
                    }
                }
                a - b > margin
            }
        }
    }
}

/// Prepared prediction over a loaded model.
pub struct Predictor<'a> {
    model: &'a Gbdt,
    kind: PredictKind,
    start: usize,
    num: usize,
    early_stop: EarlyStop,
    num_pred_one_row: usize,
    pool: Option<rayon::ThreadPool>,
}

impl<'a> Predictor<'a> {
    pub fn new(model: &'a Gbdt, cfg: &PredictConfig) -> Result<Self> {
        let (start, num) = model.iteration_range(cfg.start_iteration, cfg.num_iteration);
        let mut early_stop = EarlyStop::None;
        if cfg.pred_early_stop && !model.need_accurate_prediction() {
            if cfg.pred_early_stop_freq <= 0 || cfg.pred_early_stop_margin < 0.0 {
                return Err(Error::InvalidParameter(
                    "pred_early_stop_freq must be > 0 and pred_early_stop_margin >= 0".into(),
                ));
            }
            let (margin, period) = (
                cfg.pred_early_stop_margin,
                cfg.pred_early_stop_freq as usize,
            );
            early_stop = if model.num_class() == 1 {
                EarlyStop::Binary { margin, period }
            } else {
                if model.num_tree_per_iteration() < 2 {
                    return Err(Error::InvalidParameter(
                        "Multiclass early stopping needs predictions to be of length two or larger"
                            .into(),
                    ));
                }
                EarlyStop::Multiclass { margin, period }
            };
        }
        if cfg.kind == PredictKind::Contrib && model.is_linear() {
            return Err(Error::Unsupported(
                "Predicting SHAP feature contributions is not implemented for linear trees.".into(),
            ));
        }
        // upstream: GBDT::NumPredictOneRow
        let num_pred_one_row = match cfg.kind {
            PredictKind::LeafIndex => model.num_class() * num,
            PredictKind::Contrib => model.num_tree_per_iteration() * (model.num_features() + 1),
            _ => model.num_class(),
        };
        let pool = if cfg.num_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cfg.num_threads)
                    .build()
                    .map_err(|e| Error::InvalidParameter(format!("thread pool: {e}")))?,
            )
        } else {
            None
        };
        Ok(Predictor {
            model,
            kind: cfg.kind,
            start,
            num,
            early_stop,
            num_pred_one_row,
            pool,
        })
    }

    pub fn num_pred_one_row(&self) -> usize {
        self.num_pred_one_row
    }

    /// Predict a row-major dense matrix of `nrow x ncol` values.
    /// Returns `nrow * num_pred_one_row` values, row-major.
    pub fn predict_dense(&self, data: &[f64], nrow: usize, ncol: usize) -> Result<Vec<f64>> {
        if data.len() != nrow * ncol {
            return Err(Error::InvalidParameter(format!(
                "data has {} values, expected {nrow} x {ncol}",
                data.len()
            )));
        }
        let mut out = vec![0.0; nrow * self.num_pred_one_row];
        if nrow == 0 {
            return Ok(out);
        }
        let k = self.num_pred_one_row;
        let nf = self.model.num_features();
        let run = |out: &mut [f64]| {
            out.par_chunks_mut(k)
                .zip(data.par_chunks(ncol.max(1)))
                .for_each_init(
                    || (vec![0.0f64; nf], ShapScratch::default()),
                    |(buf, scratch), (o, row)| {
                        fill_buffer(buf, row);
                        self.predict_one_with(buf, o, scratch);
                    },
                );
        };
        match &self.pool {
            Some(p) => p.install(|| run(&mut out)),
            None => run(&mut out),
        }
        Ok(out)
    }

    /// Predict a CSR matrix (`indptr` of length `nrow + 1`, column `indices`, `values`).
    /// Returns `nrow * num_pred_one_row` values, row-major, like [`Self::predict_dense`].
    ///
    /// Each row is scattered into a zeroed per-thread buffer the way upstream does it
    /// (upstream: src/c_api.cpp (v4.7.0) RowFunctionFromCSR and
    /// src/application/predictor.hpp CopyToPredictBuffer): entries with
    /// `|x| <= kZeroThreshold` (not NaN) are skipped, columns `>= num_features` are ignored,
    /// and for a duplicated column the last stored entry wins. Upstream switches to a
    /// map-based path above 100000 features; it reads absent features as `0.0` too, so the
    /// buffer gives the same results without a per-row hash map.
    pub fn predict_csr(&self, indptr: &[i64], indices: &[i32], values: &[f64]) -> Result<Vec<f64>> {
        let rows = CsrRows::new(indptr, indices, values)?;
        let k = self.num_pred_one_row;
        let mut out = vec![0.0; rows.nrow() * k];
        let nf = self.model.num_features();
        let run = |out: &mut [f64]| {
            out.par_chunks_mut(k.max(1)).enumerate().for_each_init(
                || (vec![0.0f64; nf], ShapScratch::default()),
                |(buf, scratch), (r, o)| {
                    let (cols, vals) = rows.row(r);
                    scatter_row(buf, cols, vals);
                    self.predict_one_with(buf, o, scratch);
                    clear_row(buf, cols);
                },
            );
        };
        if k > 0 {
            self.install(|| run(&mut out));
        }
        Ok(out)
    }

    /// Predict a CSC matrix of `nrow` rows (`col_ptr`, row `indices`, `values`).
    /// Rows are gathered in column order, then predicted like [`Self::predict_csr`].
    pub fn predict_csc(
        &self,
        col_ptr: &[i64],
        indices: &[i32],
        values: &[f64],
        nrow: usize,
    ) -> Result<Vec<f64>> {
        let (indptr, cols, vals) = csc_to_csr(col_ptr, indices, values, nrow)?;
        self.predict_csr(&indptr, &cols, &vals)
    }

    /// The columns upstream stores in sparse SHAP output, per class (sorted, bias last).
    ///
    /// upstream: include/LightGBM/tree.h (v4.7.0) PredictContribByMap and
    /// src/io/tree.cpp TreeSHAPByMap. The bias column `num_features` is always inserted, and
    /// TreeSHAP visits every leaf of a tree with more than one leaf, inserting each split
    /// feature on the path even when its contribution is zero. So the stored set is the same
    /// for every row: every split feature of the in-range trees of that class, plus the bias.
    pub fn contrib_columns(&self) -> Vec<Vec<i32>> {
        let m = self.model;
        let ntpi = m.num_tree_per_iteration();
        let nf = m.num_features();
        let mut used = vec![vec![false; nf + 1]; ntpi];
        let trees = m.trees();
        for it in self.start..self.start + self.num {
            for (k, used) in used.iter_mut().enumerate() {
                let t = &trees[it * ntpi + k];
                if t.num_leaves > 1 {
                    for &f in &t.split_feature[..t.num_leaves - 1] {
                        if let Some(u) = usize::try_from(f).ok().and_then(|f| used.get_mut(f)) {
                            *u = true;
                        }
                    }
                }
            }
        }
        used.into_iter()
            .map(|mut used| {
                used[nf] = true;
                (0..=nf).filter(|&c| used[c]).map(|c| c as i32).collect()
            })
            .collect()
    }

    /// SHAP contributions of a CSR matrix as one CSR matrix per class, with upstream's stored
    /// pattern ([`Self::contrib_columns`]). Indices within a row are sorted; upstream leaves
    /// them in hash-map order, which is unspecified.
    pub fn predict_contrib_csr(
        &self,
        indptr: &[i64],
        indices: &[i32],
        values: &[f64],
    ) -> Result<Vec<SparseMatrix>> {
        if self.kind != PredictKind::Contrib {
            return Err(Error::InvalidParameter(
                "sparse output is only produced for pred_contrib".into(),
            ));
        }
        let nrow = CsrRows::new(indptr, indices, values)?.nrow();
        let dense = self.predict_csr(indptr, indices, values)?;
        let width = self.model.num_features() + 1;
        let k = self.num_pred_one_row;
        Ok(self
            .contrib_columns()
            .into_iter()
            .enumerate()
            .map(|(class, cols)| {
                let nnz = cols.len();
                let mut data = Vec::with_capacity(nrow * nnz);
                for r in 0..nrow {
                    let row = &dense[r * k + class * width..r * k + (class + 1) * width];
                    data.extend(cols.iter().map(|&c| row[c as usize]));
                }
                SparseMatrix {
                    indptr: (0..=nrow as i64).map(|r| r * nnz as i64).collect(),
                    indices: cols.iter().copied().cycle().take(nrow * nnz).collect(),
                    data,
                }
            })
            .collect())
    }

    /// SHAP contributions of a CSC matrix as one CSC matrix per class (`num_features + 1`
    /// columns). Within a column, rows are ascending, as upstream writes them
    /// (upstream: src/c_api.cpp (v4.7.0) PredictSparseCSC).
    pub fn predict_contrib_csc(
        &self,
        col_ptr: &[i64],
        indices: &[i32],
        values: &[f64],
        nrow: usize,
    ) -> Result<Vec<SparseMatrix>> {
        let (indptr, cols, vals) = csc_to_csr(col_ptr, indices, values, nrow)?;
        let csr = self.predict_contrib_csr(&indptr, &cols, &vals)?;
        let width = self.model.num_features() + 1;
        let row_ids: Vec<i32> = (0..nrow).map(|r| r as i32).collect();
        Ok(csr
            .into_iter()
            .zip(self.contrib_columns())
            .map(|(m, cols)| {
                let nnz = cols.len();
                let mut counts = vec![0i64; width];
                for &c in &cols {
                    counts[c as usize] = nrow as i64;
                }
                let mut col_ptr = Vec::with_capacity(width + 1);
                col_ptr.push(0i64);
                let mut acc = 0;
                for c in counts {
                    acc += c;
                    col_ptr.push(acc);
                }
                let mut data = Vec::with_capacity(nrow * nnz);
                for j in 0..nnz {
                    data.extend((0..nrow).map(|r| m.data[r * nnz + j]));
                }
                SparseMatrix {
                    indptr: col_ptr,
                    indices: row_ids.iter().copied().cycle().take(nrow * nnz).collect(),
                    data,
                }
            })
            .collect())
    }

    fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }

    /// Predict one row given a dense feature buffer of length `num_features`.
    pub fn predict_one(&self, features: &[f64], out: &mut [f64]) {
        self.predict_one_with(features, out, &mut ShapScratch::default());
    }

    #[inline]
    fn predict_one_with(&self, features: &[f64], out: &mut [f64], scratch: &mut ShapScratch) {
        let m = self.model;
        let ntpi = m.num_tree_per_iteration();
        let trees = m.trees();
        match self.kind {
            PredictKind::LeafIndex => {
                let first = self.start * ntpi;
                for (i, t) in trees[first..first + self.num * ntpi].iter().enumerate() {
                    out[i] = t.predict_leaf_index(features) as f64;
                }
            }
            PredictKind::Contrib => {
                let nf = m.num_features();
                out.fill(0.0);
                for it in self.start..self.start + self.num {
                    for k in 0..ntpi {
                        trees[it * ntpi + k].predict_contrib_with(
                            features,
                            nf,
                            &mut out[k * (nf + 1)..(k + 1) * (nf + 1)],
                            scratch,
                        );
                    }
                }
            }
            PredictKind::RawScore | PredictKind::Normal => {
                // upstream: GBDT::PredictRaw (output has num_tree_per_iteration slots)
                let raw = &mut out[..ntpi];
                raw.fill(0.0);
                let period = self.early_stop.period();
                let mut counter = 0usize;
                for it in self.start..self.start + self.num {
                    for k in 0..ntpi {
                        raw[k] += trees[it * ntpi + k].predict(features);
                    }
                    counter += 1;
                    if counter == period {
                        if self.early_stop.should_stop(raw) {
                            break;
                        }
                        counter = 0;
                    }
                }
                if self.kind == PredictKind::Normal {
                    if m.average_output() {
                        for v in raw.iter_mut() {
                            *v /= self.num as f64;
                        }
                    }
                    if let Some(obj) = m.objective() {
                        obj.convert_output(raw);
                    }
                }
            }
        }
    }
}

/// Copy a dense row into the feature buffer the way upstream's dense-matrix path does:
/// values with `|x| <= kZeroThreshold` (and not NaN) are dropped, i.e. read as `0.0`;
/// columns past `num_features` are ignored.
#[inline]
fn fill_buffer(buf: &mut [f64], row: &[f64]) {
    let n = buf.len().min(row.len());
    for (b, &v) in buf[..n].iter_mut().zip(row) {
        *b = if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
            v
        } else {
            0.0
        };
    }
    buf[n..].fill(0.0);
}

/// A compressed sparse matrix (CSR or CSC, by context) produced by sparse prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f64>,
}

/// Validated view over CSR arrays, so per-row slicing cannot go out of bounds.
struct CsrRows<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    values: &'a [f64],
}

impl<'a> CsrRows<'a> {
    fn new(indptr: &'a [i64], indices: &'a [i32], values: &'a [f64]) -> Result<Self> {
        if indptr.is_empty() {
            return Err(Error::InvalidParameter("indptr must not be empty".into()));
        }
        if indices.len() != values.len() {
            return Err(Error::InvalidParameter(format!(
                "indices has {} entries but data has {}",
                indices.len(),
                values.len()
            )));
        }
        let nnz = indices.len() as i64;
        if indptr[0] < 0 || indptr.windows(2).any(|w| w[1] < w[0]) || indptr[indptr.len() - 1] > nnz
        {
            return Err(Error::InvalidParameter(
                "indptr must be non-decreasing and within the number of stored values".into(),
            ));
        }
        if indices.iter().any(|&c| c < 0) {
            return Err(Error::InvalidParameter(
                "sparse indices must be non-negative".into(),
            ));
        }
        Ok(CsrRows {
            indptr,
            indices,
            values,
        })
    }

    fn nrow(&self) -> usize {
        self.indptr.len() - 1
    }

    #[inline]
    fn row(&self, r: usize) -> (&'a [i32], &'a [f64]) {
        // Validated in `new`: offsets are non-negative, ordered and within bounds.
        let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        (&self.indices[lo..hi], &self.values[lo..hi])
    }
}

/// Gather a CSC matrix into CSR arrays, keeping column order within each row
/// (upstream builds each row by walking the columns in order: src/c_api.cpp (v4.7.0)
/// IterateFunctionFromCSC / RowFunctionFromCSC).
fn csc_to_csr(
    col_ptr: &[i64],
    indices: &[i32],
    values: &[f64],
    nrow: usize,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f64>)> {
    let cols = CsrRows::new(col_ptr, indices, values)?;
    if indices.iter().any(|&r| r as usize >= nrow) {
        return Err(Error::InvalidParameter(format!(
            "CSC row index out of range for {nrow} rows"
        )));
    }
    let mut indptr = vec![0i64; nrow + 1];
    for c in 0..cols.nrow() {
        for &r in cols.row(c).0 {
            indptr[r as usize + 1] += 1;
        }
    }
    for r in 0..nrow {
        indptr[r + 1] += indptr[r];
    }
    let mut next: Vec<usize> = indptr[..nrow].iter().map(|&p| p as usize).collect();
    let nnz = indptr[nrow] as usize;
    let (mut out_cols, mut out_vals) = (vec![0i32; nnz], vec![0.0f64; nnz]);
    for c in 0..cols.nrow() {
        let (rows, vals) = cols.row(c);
        for (&r, &v) in rows.iter().zip(vals) {
            let at = &mut next[r as usize];
            out_cols[*at] = c as i32;
            out_vals[*at] = v;
            *at += 1;
        }
    }
    Ok((indptr, out_cols, out_vals))
}

/// Scatter one sparse row into a zeroed buffer like upstream's CSR row extraction followed
/// by `CopyToPredictBuffer`: tiny values are dropped before the copy (so they do not
/// overwrite an earlier duplicate), columns past the buffer are ignored, last entry wins.
#[inline]
fn scatter_row(buf: &mut [f64], cols: &[i32], vals: &[f64]) {
    for (&c, &v) in cols.iter().zip(vals) {
        if (v.abs() > K_ZERO_THRESHOLD || v.is_nan())
            && let Some(b) = buf.get_mut(c as usize)
        {
            *b = v;
        }
    }
}

/// Reset the entries [`scatter_row`] may have written (upstream: ClearPredictBuffer).
#[inline]
fn clear_row(buf: &mut [f64], cols: &[i32]) {
    if cols.len() > buf.len() / 2 {
        buf.fill(0.0);
    } else {
        for &c in cols {
            if let Some(b) = buf.get_mut(c as usize) {
                *b = 0.0;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn scatter_row_skips_tiny_values_ignores_extra_columns_and_last_wins() {
        let mut buf = vec![0.0; 3];
        scatter_row(
            &mut buf,
            &[0, 1, 1, 2, 2, 7],
            &[5.0, 1.0, 2.0, 4.0, 1e-40, 9.0],
        );
        assert_eq!(buf, vec![5.0, 2.0, 4.0]);
        clear_row(&mut buf, &[0, 1, 1, 2, 2, 7]);
        assert_eq!(buf, vec![0.0; 3]);
    }

    #[test]
    fn csc_to_csr_keeps_column_order_within_rows() {
        // 3x3: col0 rows {0,2}, col1 rows {1}, col2 rows {0}
        let (p, c, v) = csc_to_csr(&[0, 2, 3, 4], &[0, 2, 1, 0], &[1.0, 2.0, 3.0, 4.0], 3).unwrap();
        assert_eq!(p, vec![0, 2, 3, 4]);
        assert_eq!(c, vec![0, 2, 1, 0]);
        assert_eq!(v, vec![1.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn malformed_sparse_input_is_an_error_not_a_panic() {
        assert!(CsrRows::new(&[], &[], &[]).is_err());
        assert!(CsrRows::new(&[0, 2], &[0], &[1.0]).is_err());
        assert!(CsrRows::new(&[0, 1], &[0], &[1.0, 2.0]).is_err());
        assert!(CsrRows::new(&[1, 0], &[0], &[1.0]).is_err());
        assert!(CsrRows::new(&[0, 1], &[-1], &[1.0]).is_err());
        assert!(csc_to_csr(&[0, 1], &[5], &[1.0], 3).is_err());
    }

    #[test]
    fn fill_buffer_zeroes_tiny_values_and_pads() {
        let mut buf = vec![9.0; 4];
        fill_buffer(&mut buf, &[1e-40, -0.0, f64::NAN]);
        assert_eq!(buf[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(buf[1].to_bits(), 0.0f64.to_bits());
        assert!(buf[2].is_nan());
        assert_eq!(buf[3], 0.0);
    }

    #[test]
    fn multiclass_margin_uses_top_two() {
        let es = EarlyStop::Multiclass {
            margin: 1.0,
            period: 1,
        };
        assert!(es.should_stop(&[0.0, 3.0, 1.5]));
        assert!(!es.should_stop(&[0.0, 3.0, 2.5]));
        let es = EarlyStop::Binary {
            margin: 1.0,
            period: 1,
        };
        assert!(es.should_stop(&[-0.6]));
        assert!(!es.should_stop(&[0.4]));
    }
}
