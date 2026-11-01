//! Borrowed compressed sparse row (CSR) input, shared by dataset construction and
//! prediction. upstream: src/c_api.cpp (v4.7.0) `RowFunctionFromCSR`, `CSC_RowIterator`.

use crate::error::{Error, Result};

/// A validated CSR matrix borrowed from the caller: row `r` holds the pairs
/// `indices[indptr[r]..indptr[r + 1]]` / `values[..]`, in storage order (which may be
/// unsorted or contain duplicate columns, as upstream accepts).
#[derive(Debug, Clone, Copy)]
pub struct CsrMatrix<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    values: &'a [f64],
    ncol: usize,
}

impl<'a> CsrMatrix<'a> {
    /// Check that `indptr` is non-empty, non-decreasing and within `indices`, that
    /// `indices` and `values` have the same length, and that indices are non-negative.
    /// Column indices are *not* checked against `ncol` here: upstream ignores columns past
    /// the model/dataset width when pushing and predicting, and rejects them only while
    /// sampling (see [`CsrMatrix::check_columns`]).
    pub fn new(
        indptr: &'a [i64],
        indices: &'a [i32],
        values: &'a [f64],
        ncol: usize,
    ) -> Result<Self> {
        if indptr.is_empty() {
            return Err(Error::InvalidParameter("indptr must not be empty".into()));
        }
        if indices.len() != values.len() {
            return Err(Error::InvalidParameter(format!(
                "Length mismatch: {} vs {}",
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
        Ok(CsrMatrix {
            indptr,
            indices,
            values,
            ncol,
        })
    }

    /// Rows.
    pub fn nrow(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Columns (the width the caller declared).
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// Stored entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// The stored `(column, value)` pairs of row `r`, in storage order.
    #[inline]
    pub fn row(&self, r: usize) -> (&'a [i32], &'a [f64]) {
        // Validated in `new`: offsets are non-negative, ordered and within bounds.
        let (lo, hi) = (self.indptr[r] as usize, self.indptr[r + 1] as usize);
        (&self.indices[lo..hi], &self.values[lo..hi])
    }

    /// upstream's `CHECK_LT(inner_data.first, num_col)` while sampling CSR rows.
    pub fn check_columns(&self) -> Result<()> {
        match self.indices.iter().find(|&&c| c as usize >= self.ncol) {
            Some(&c) => Err(Error::InvalidParameter(format!(
                "Check failed: (inner_data.first) < (num_col): column index {c} >= {}",
                self.ncol
            ))),
            None => Ok(()),
        }
    }
}

/// Owned CSR arrays (`indptr`, `indices`, `values`).
pub type CsrArrays = (Vec<i64>, Vec<i32>, Vec<f64>);

/// Gather a CSC matrix of `nrow` rows into CSR arrays, keeping column order within each
/// row: upstream builds CSC rows by walking the columns in order
/// (src/c_api.cpp (v4.7.0) `IterateFunctionFromCSC` / `RowFunctionFromCSC`).
pub fn csc_to_csr(
    col_ptr: &[i64],
    indices: &[i32],
    values: &[f64],
    nrow: usize,
) -> Result<CsrArrays> {
    // A CSC matrix is the CSR form of its transpose: rows of `cols` are columns.
    let cols = CsrMatrix::new(col_ptr, indices, values, nrow)?;
    if indices.iter().any(|&r| r as usize >= nrow) {
        return Err(Error::InvalidParameter(format!(
            "CSC row index out of range for {nrow} rows"
        )));
    }
    let mut indptr = vec![0i64; nrow + 1];
    for &r in indices {
        indptr[r as usize + 1] += 1;
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

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
        assert!(CsrMatrix::new(&[], &[], &[], 1).is_err());
        assert!(CsrMatrix::new(&[0, 2], &[0], &[1.0], 1).is_err());
        assert!(CsrMatrix::new(&[0, 1], &[0], &[1.0, 2.0], 1).is_err());
        assert!(CsrMatrix::new(&[1, 0], &[0], &[1.0], 1).is_err());
        assert!(CsrMatrix::new(&[0, 1], &[-1], &[1.0], 1).is_err());
        assert!(csc_to_csr(&[0, 1], &[5], &[1.0], 3).is_err());
        let m = CsrMatrix::new(&[0, 1], &[4], &[1.0], 2).unwrap();
        assert!(m.check_columns().is_err());
    }
}
