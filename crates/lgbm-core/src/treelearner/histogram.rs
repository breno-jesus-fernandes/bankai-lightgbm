//! Histogram construction kernels. upstream: src/io/dense_bin.hpp
//! (`ConstructHistogramInner`), specialized per bin width.
//!
//! Unsafe allow-listed module (AGENTS.md): the inner loops skip bounds checks, like
//! upstream. Soundness rests on [`validate_bins`], which the tree learner runs once.

use crate::io::bin_data::DenseBin;

/// Read access to a dense bin column, monomorphized per storage width.
trait Bins {
    fn len(&self) -> usize;
    /// # Safety
    /// `idx` must be a valid row of this column.
    unsafe fn bin_unchecked(&self, idx: usize) -> usize;
}

struct B4<'a>(&'a [u8]);
struct B8<'a>(&'a [u8]);
struct B16<'a>(&'a [u16]);
struct B32<'a>(&'a [u32]);

impl Bins for B4<'_> {
    fn len(&self) -> usize {
        self.0.len() * 2
    }
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { ((*self.0.get_unchecked(idx >> 1) >> ((idx & 1) << 2)) & 0xf) as usize }
    }
}
impl Bins for B8<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}
impl Bins for B16<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}
impl Bins for B32<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    unsafe fn bin_unchecked(&self, idx: usize) -> usize {
        // SAFETY: the caller guarantees `idx` is a valid row.
        unsafe { *self.0.get_unchecked(idx) as usize }
    }
}

/// Accumulate `[grad, hess]` per bin into `out`. `og`/`oh` are in iteration order
/// (ordered by `indices` when given). With `const_hess`, the hessian slot counts rows.
pub fn construct(
    bin: &DenseBin,
    indices: Option<&[i32]>,
    og: &[f32],
    oh: &[f32],
    const_hess: bool,
    out: &mut [f64],
) {
    match bin {
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
    assert!(!USE_INDICES || indices.len() >= n);
    assert!(!USE_HESSIAN || oh.len() >= n);
    assert!(b.len() >= n || USE_INDICES);
    let out_len = out.len();
    let out = out.as_mut_ptr();
    for i in 0..n {
        // SAFETY: `i < n <= og.len()`, and `indices`/`oh` were checked to hold at least
        // `n` entries. Row indices come from the data partition (`< num_data`), and the
        // column holds `num_data` rows. Every stored bin is `< out_len / 2`, which
        // `validate_bins` checked when the learner was built.
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

/// Check that every bin in `bin` is `< num_bin` (precondition of the unchecked kernels).
pub fn validate_bins(bin: &DenseBin, num_rows: usize, num_bin: usize) -> bool {
    (0..num_rows).all(|i| (bin.get(i) as usize) < num_bin)
}
