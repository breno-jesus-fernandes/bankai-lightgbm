//! Row-major copy of the dense single-value bins, for the row-wise histogram layout.
//!
//! The column-wise storage keeps one array per feature, so a row-wise pass reads one byte from
//! each array per row. A row-major copy puts every feature's bin for a row next to each other,
//! so one row is one contiguous read. upstream: `MultiValDenseBin` (row-major dense bins).

use crate::io::bin_data::DenseBin;

/// Row-major `u8` bins of several dense columns: row `r` holds column `j` at `r * ncol + j`.
#[derive(Debug, Clone, PartialEq)]
pub struct RowMajorBins {
    ncol: usize,
    data: Vec<u8>,
}

impl RowMajorBins {
    /// Builds the copy from column-wise dense bins, or returns `None` when a column holds bins
    /// above 255 (`U16`/`U32`), which a `u8` copy cannot represent. The caller then keeps the
    /// column-wise kernel for those groups.
    pub fn from_columns(columns: &[&DenseBin], num_data: usize) -> Option<RowMajorBins> {
        if columns
            .iter()
            .any(|c| matches!(c, DenseBin::U16(_) | DenseBin::U32(_)))
        {
            return None;
        }
        let ncol = columns.len();
        let mut data = vec![0u8; num_data * ncol];
        for (j, col) in columns.iter().enumerate() {
            for r in 0..num_data {
                // Bins are below 256 by the check above, so the cast cannot truncate.
                data[r * ncol + j] = col.get(r) as u8;
            }
        }
        Some(RowMajorBins { ncol, data })
    }

    /// Number of columns (dense groups) in each row.
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    /// The bins of row `row`, one per column, in the order the columns were given.
    pub fn row(&self, row: usize) -> &[u8] {
        &self.data[row * self.ncol..(row + 1) * self.ncol]
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::RowMajorBins;
    use crate::io::bin_data::DenseBin;

    fn column(values: &[u32]) -> DenseBin {
        let mut bin = DenseBin::new(values.len(), 200);
        for (r, &v) in values.iter().enumerate() {
            bin.set(r, v);
        }
        bin
    }

    #[test]
    fn row_holds_each_column_bin_in_order() {
        let a = column(&[3, 0, 7, 255, 1]);
        let b = column(&[9, 9, 2, 0, 4]);
        let rm = RowMajorBins::from_columns(&[&a, &b], 5).expect("u8 columns convert");
        assert_eq!(rm.ncol(), 2);
        for r in 0..5 {
            assert_eq!(u32::from(rm.row(r)[0]), a.get(r), "column 0, row {r}");
            assert_eq!(u32::from(rm.row(r)[1]), b.get(r), "column 1, row {r}");
        }
    }

    #[test]
    fn columns_above_255_bins_are_not_converted() {
        let wide = DenseBin::U16(vec![0, 300, 1]);
        assert!(RowMajorBins::from_columns(&[&wide], 3).is_none());
    }

    #[test]
    fn works_for_nibble_packed_columns() {
        let mut nib = DenseBin::new(4, 10);
        for (r, v) in [1u32, 15, 0, 7].into_iter().enumerate() {
            nib.set(r, v);
        }
        let rm = RowMajorBins::from_columns(&[&nib], 4).expect("nibble columns convert");
        let got: Vec<u32> = (0..4).map(|r| u32::from(rm.row(r)[0])).collect();
        assert_eq!(got, vec![1, 15, 0, 7]);
    }
}
