//! Histogram construction kernels. upstream: src/io/dense_bin.hpp
//! (`ConstructHistogramInner`), specialized per bin width.
//!
//! Unsafe allow-listed module (AGENTS.md): the inner loops skip bounds checks, like
//! upstream. Soundness rests on [`validate_bins`], which the tree learner runs once.

use crate::io::bin_data::DenseBin;

/// Read access to a dense bin column, monomorphized per storage width.
trait Bins {
    /// # Safety
    /// `idx` must be a valid row of this column.
    unsafe fn bin_unchecked(&self, idx: usize) -> usize;
}

struct B4<'a>(&'a [u8]);
struct B8<'a>(&'a [u8]);
struct B16<'a>(&'a [u16]);
struct B32<'a>(&'a [u32]);

impl Bins for B4<'_> {
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { ((*self.0.get_unchecked(idx >> 1) >> ((idx & 1) << 2)) & 0xf) as usize }
    }
}
impl Bins for B8<'_> {
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}
impl Bins for B16<'_> {
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}
impl Bins for B32<'_> {
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}

/// Proof that a dense column's bins are all `< num_bin`. Only [`BinsCheck::new`] creates
/// one, after a full scan, so it can be kept beside an immutable dataset and reused.
#[derive(Clone, Copy, Debug)]
pub struct BinsCheck {
    num_rows: usize,
    num_bin: usize,
}

impl BinsCheck {
    /// `None` when some of the first `num_rows` bins is `>= num_bin`.
    pub fn new(bin: &DenseBin, num_rows: usize, num_bin: usize) -> Option<Self> {
        let ok =
            bin.num_rows() >= num_rows && (0..num_rows).all(|i| (bin.get(i) as usize) < num_bin);
        ok.then_some(BinsCheck { num_rows, num_bin })
    }
}

/// A dense bin column whose bins were all verified to be `< num_bin`.
#[derive(Clone, Copy)]
pub struct CheckedBins<'a> {
    bin: &'a DenseBin,
    num_rows: usize,
    num_bin: usize,
}

impl<'a> CheckedBins<'a> {
    /// Attach a previously computed [`BinsCheck`] to the same column.
    pub fn with(bin: &'a DenseBin, check: BinsCheck) -> Self {
        CheckedBins {
            bin,
            num_rows: check.num_rows,
            num_bin: check.num_bin,
        }
    }

    /// `None` when some of the first `num_rows` bins is `>= num_bin`.
    pub fn new(bin: &'a DenseBin, num_rows: usize, num_bin: usize) -> Option<Self> {
        let ok =
            bin.num_rows() >= num_rows && (0..num_rows).all(|i| (bin.get(i) as usize) < num_bin);
        ok.then_some(CheckedBins {
            bin,
            num_rows,
            num_bin,
        })
    }
}

/// Row indices verified to be `< num_rows`; build once per leaf and share across features.
#[derive(Clone, Copy)]
pub struct CheckedRows<'a> {
    indices: &'a [i32],
    num_rows: usize,
}

impl<'a> CheckedRows<'a> {
    /// `None` when some index is negative or `>= num_rows`.
    pub fn new(indices: &'a [i32], num_rows: usize) -> Option<Self> {
        indices
            .iter()
            .all(|&i| (i as u32 as usize) < num_rows)
            .then_some(CheckedRows { indices, num_rows })
    }
}

/// Accumulate `[grad, hess]` per bin into `out` (`2 * num_bin` doubles). `og`/`oh` are in
/// iteration order (ordered by `rows` when given). With `const_hess`, the hessian slot
/// counts rows. Panics if the slice sizes do not fit the column.
pub fn construct(
    bins: &CheckedBins<'_>,
    rows: Option<&CheckedRows<'_>>,
    og: &[f32],
    oh: &[f32],
    const_hess: bool,
    out: &mut [f64],
) {
    assert!(out.len() >= bins.num_bin * 2, "histogram buffer too small");
    let indices = match rows {
        Some(r) => {
            assert!(
                r.num_rows == bins.num_rows,
                "rows checked for another column"
            );
            assert!(
                r.indices.len() >= og.len(),
                "ordered gradients longer than rows"
            );
            Some(r.indices)
        }
        None => {
            assert!(og.len() <= bins.num_rows, "more gradients than rows");
            None
        }
    };
    assert!(
        const_hess || oh.len() >= og.len(),
        "hessians shorter than gradients"
    );
    match bins.bin {
        DenseBin::U4(d) => dispatch(&B4(d), indices, og, oh, const_hess, out),
        DenseBin::U8(d) => dispatch(&B8(d), indices, og, oh, const_hess, out),
        DenseBin::U16(d) => dispatch(&B16(d), indices, og, oh, const_hess, out),
        DenseBin::U32(d) => dispatch(&B32(d), indices, og, oh, const_hess, out),
    }
}

#[inline(always)]
fn dispatch<B: Bins>(
    b: &B,
    indices: Option<&[i32]>,
    og: &[f32],
    oh: &[f32],
    const_hess: bool,
    out: &mut [f64],
) {
    match (indices, const_hess) {
        (Some(ix), true) => inner::<B, true, false>(b, ix, og, oh, out),
        (Some(ix), false) => inner::<B, true, true>(b, ix, og, oh, out),
        (None, true) => inner::<B, false, false>(b, &[], og, oh, out),
        (None, false) => inner::<B, false, true>(b, &[], og, oh, out),
    }
}

#[inline(always)]
fn inner<B: Bins, const USE_INDICES: bool, const USE_HESSIAN: bool>(
    b: &B,
    indices: &[i32],
    og: &[f32],
    oh: &[f32],
    out: &mut [f64],
) {
    let n = og.len();
    let out_len = out.len();
    let out = out.as_mut_ptr();
    for i in 0..n {
        // SAFETY: `construct` checked that `indices`/`oh` hold at least `n = og.len()`
        // entries, that every row index (or `i < n` without indices) is `< num_rows <=`
        // the column length, and that `out` holds `2 * num_bin` doubles. `CheckedBins`
        // guarantees every stored bin is `< num_bin`, so `ti + 1 < out_len`.
        unsafe {
            let idx = if USE_INDICES {
                *indices.get_unchecked(i) as usize
            } else {
                i
            };
            let ti = b.bin_unchecked(idx) << 1;
            debug_assert!(ti + 1 < out_len);
            *out.add(ti) += f64::from(*og.get_unchecked(i));
            if USE_HESSIAN {
                *out.add(ti + 1) += f64::from(*oh.get_unchecked(i));
            } else {
                *out.add(ti + 1) += 1.0;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn column(num_bin: u32, n: usize) -> DenseBin {
        let mut bin = DenseBin::new(n, num_bin);
        for i in 0..n {
            bin.set(i, ((i * 7 + 3) as u32) % num_bin);
        }
        bin
    }

    fn naive(
        bin: &DenseBin,
        num_bin: usize,
        rows: Option<&[i32]>,
        og: &[f32],
        oh: &[f32],
        const_hess: bool,
    ) -> Vec<f64> {
        let mut out = vec![0.0; num_bin * 2];
        for (k, &g) in og.iter().enumerate() {
            let r = rows.map_or(k, |ix| ix[k] as usize);
            let b = bin.get(r) as usize;
            out[2 * b] += f64::from(g);
            out[2 * b + 1] += if const_hess { 1.0 } else { f64::from(oh[k]) };
        }
        out
    }

    #[test]
    fn matches_naive_for_every_width() {
        let n = 101; // odd, to cover the 4-bit padding row
        for num_bin in [5u32, 16, 17, 200, 300, 70_000] {
            let bin = column(num_bin, n);
            let checked = CheckedBins::new(&bin, n, num_bin as usize).unwrap();
            let rows: Vec<i32> = (0..n as i32).rev().step_by(3).collect();
            let checked_rows = CheckedRows::new(&rows, n).unwrap();
            let g: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 - 3.0).collect();
            let h: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.5).collect();
            for const_hess in [false, true] {
                let want = naive(&bin, num_bin as usize, None, &g, &h, const_hess);
                let mut got = vec![0.0; num_bin as usize * 2];
                construct(&checked, None, &g, &h, const_hess, &mut got);
                assert_eq!(got, want, "all rows, num_bin={num_bin}");

                let og: Vec<f32> = rows.iter().map(|&i| g[i as usize]).collect();
                let oh: Vec<f32> = rows.iter().map(|&i| h[i as usize]).collect();
                let want = naive(&bin, num_bin as usize, Some(&rows), &og, &oh, const_hess);
                let mut got = vec![0.0; num_bin as usize * 2];
                construct(
                    &checked,
                    Some(&checked_rows),
                    &og,
                    &oh,
                    const_hess,
                    &mut got,
                );
                assert_eq!(got, want, "subset, num_bin={num_bin}");
            }
        }
    }

    #[test]
    fn rejects_bins_outside_the_declared_range() {
        let bin = column(16, 40);
        assert!(CheckedBins::new(&bin, 40, 16).is_some());
        assert!(CheckedBins::new(&bin, 40, 15).is_none());
        assert!(
            CheckedBins::new(&bin, 80, 16).is_none(),
            "more rows than stored"
        );
    }

    #[test]
    fn rejects_row_indices_outside_the_column() {
        assert!(CheckedRows::new(&[0, 39], 40).is_some());
        assert!(CheckedRows::new(&[0, 40], 40).is_none());
        assert!(CheckedRows::new(&[-1], 40).is_none());
    }

    #[test]
    #[should_panic(expected = "histogram buffer too small")]
    fn panics_on_short_output() {
        let bin = column(16, 8);
        let checked = CheckedBins::new(&bin, 8, 16).unwrap();
        let mut out = vec![0.0; 10];
        construct(&checked, None, &[1.0; 8], &[1.0; 8], false, &mut out);
    }

    #[test]
    #[should_panic(expected = "rows checked for another column")]
    fn panics_on_rows_from_another_column() {
        let bin = column(16, 8);
        let checked = CheckedBins::new(&bin, 8, 16).unwrap();
        let rows = CheckedRows::new(&[0, 1], 4).unwrap();
        let mut out = vec![0.0; 32];
        construct(&checked, Some(&rows), &[1.0; 2], &[1.0; 2], false, &mut out);
    }
}
