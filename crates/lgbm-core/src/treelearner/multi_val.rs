//! Row-wise bins of the multi-value feature group, used to build its histograms during
//! col-wise training. upstream: src/io/multi_val_sparse_bin.hpp,
//! src/io/multi_val_dense_bin.hpp, src/io/dataset.cpp (`GetMultiBinFromSparseFeatures`,
//! `PushDataToMultiValBin`), src/io/train_share_states.cpp (`CalcBinOffsets`,
//! `MultiValBinWrapper`), all v4.7.0.

use crate::io::bin_data::BinData;
use crate::io::dataset::FeatureGroupLayout;
use crate::meta::K_ZERO_THRESHOLD;

/// upstream `kAlignedSize` (bytes of alignment used by `SIZE_ALIGNED`).
const K_ALIGNED_SIZE: usize = 32;

/// Histogram entries merged per parallel task (upstream splits bins into blocks of 512).
const MERGE_CHUNK: usize = 1024;

/// upstream `MultiValSparseBin` / `MultiValDenseBin` (one row per data point).
#[derive(Debug, Clone)]
pub enum MultiValBin {
    /// CSR of group-relative histogram bins: a row's non-most-frequent features, each
    /// stored as `sub_bin + offsets[j] - (most_freq_bin == 0)`, in feature order.
    Sparse {
        row_ptr: Vec<usize>,
        data: Vec<u32>,
        num_bin: usize,
        estimate_element_per_row: f64,
    },
    /// Row-major raw sub-feature bins (every feature, most frequent bin included); the
    /// histogram bin is `bin + offsets[j]`.
    Dense {
        num_feature: usize,
        data: Vec<u32>,
        offsets: Vec<u32>,
        num_bin: usize,
    },
}

impl MultiValBin {
    /// Build the group's row-wise bins from its per-feature columns. `cols[j]` stores `0`
    /// for the most frequent bin and otherwise `sub_bin + 1 - (most_freq_bin == 0)`
    /// (upstream `FeatureGroup::PushData` into `multi_bin_data_`).
    pub fn from_group(group: &FeatureGroupLayout, cols: &[BinData], num_data: usize) -> Self {
        // upstream: CalcBinOffsets (col-wise) produces exactly the group's bin offsets.
        let offsets: Vec<u32> = group.bin_offsets.clone();
        let num_bin = *offsets.last().unwrap_or(&0) as usize;
        let num_feature = group.bin_mappers.len();
        if group.is_dense_multi_val {
            let mut data = vec![0u32; num_data * num_feature];
            for (j, (col, m)) in cols.iter().zip(&group.bin_mappers).enumerate() {
                let mfb = m.most_freq_bin;
                let shift = u32::from(mfb == 0);
                let raw = |v: u32| if v == 0 { mfb } else { v - 1 + shift };
                match col {
                    BinData::Dense(d) => {
                        for r in 0..num_data {
                            data[r * num_feature + j] = raw(d.get(r));
                        }
                    }
                    BinData::Sparse(s) => {
                        for r in 0..num_data {
                            data[r * num_feature + j] = mfb;
                        }
                        s.for_each_nonzero(|r, v| data[r * num_feature + j] = raw(v));
                    }
                }
            }
            return MultiValBin::Dense {
                num_feature,
                data,
                offsets,
                num_bin,
            };
        }
        // Sparse: count entries per row, then fill rows column by column so each row's
        // entries follow feature order (upstream pushes features in order per row).
        let mut row_ptr = vec![0usize; num_data + 1];
        let each_nonzero = |col: &BinData, f: &mut dyn FnMut(usize, u32)| match col {
            BinData::Dense(d) => {
                for r in 0..num_data {
                    let v = d.get(r);
                    if v != 0 {
                        f(r, v);
                    }
                }
            }
            BinData::Sparse(s) => s.for_each_nonzero(&mut *f),
        };
        for col in cols {
            each_nonzero(col, &mut |r, _| row_ptr[r + 1] += 1);
        }
        for r in 0..num_data {
            row_ptr[r + 1] += row_ptr[r];
        }
        let mut next = row_ptr[..num_data].to_vec();
        let mut data = vec![0u32; row_ptr[num_data]];
        for (j, col) in cols.iter().enumerate() {
            let base = offsets[j] - 1;
            each_nonzero(col, &mut |r, v| {
                data[next[r]] = base + v;
                next[r] += 1;
            });
        }
        let sum_sparse_rate: f64 =
            group.bin_mappers.iter().map(|m| m.sparse_rate).sum::<f64>() / num_feature as f64;
        MultiValBin::Sparse {
            row_ptr,
            data,
            num_bin,
            // upstream: CreateMultiValSparseBin(..., (1.0 - sparse_rate) * num_feature)
            estimate_element_per_row: (1.0 - sum_sparse_rate) * num_feature as f64,
        }
    }

    /// Histogram bins of the group (upstream `MultiValBin::num_bin`).
    pub fn num_bin(&self) -> usize {
        match self {
            MultiValBin::Sparse { num_bin, .. } | MultiValBin::Dense { num_bin, .. } => *num_bin,
        }
    }

    /// upstream `num_element_per_row`.
    fn num_element_per_row(&self) -> f64 {
        match self {
            MultiValBin::Sparse {
                estimate_element_per_row,
                ..
            } => *estimate_element_per_row,
            MultiValBin::Dense { num_feature, .. } => *num_feature as f64,
        }
    }

    /// upstream `MultiValBinWrapper` `min_block_size_`.
    fn min_block_size(&self) -> usize {
        let per_row = self.num_element_per_row();
        let size = ((0.3f32 as f64 * self.num_bin() as f64 / (per_row + K_ZERO_THRESHOLD)) as i64
            + 1)
        .min(1024);
        size.max(32) as usize
    }

    /// Accumulate `(gradient, hessian)` of rows `start..end` into `out` (`2 * num_bin`
    /// doubles, gradient then hessian per bin), in row order.
    ///
    /// `rows`: the data indices (or `None` for rows `0..`). `grad_at(i, row)` and
    /// `hess_at(i, row)` give the derivatives of position `i`, which is `row` itself or
    /// its position in `rows` when the caller gathered them (upstream `ORDERED`).
    fn accumulate(
        &self,
        rows: Option<&[i32]>,
        start: usize,
        end: usize,
        grad_at: &(impl Fn(usize, usize) -> f32 + Sync),
        hess_at: &(impl Fn(usize, usize) -> f32 + Sync),
        out: &mut [f64],
    ) {
        for i in start..end {
            let row = rows.map_or(i, |r| r[i] as usize);
            let g = f64::from(grad_at(i, row));
            let h = f64::from(hess_at(i, row));
            match self {
                MultiValBin::Sparse { row_ptr, data, .. } => {
                    for &bin in &data[row_ptr[row]..row_ptr[row + 1]] {
                        let ti = (bin as usize) << 1;
                        out[ti] += g;
                        out[ti + 1] += h;
                    }
                }
                MultiValBin::Dense {
                    num_feature,
                    data,
                    offsets,
                    ..
                } => {
                    let base = row * num_feature;
                    for (j, &bin) in data[base..base + num_feature].iter().enumerate() {
                        let ti = ((bin + offsets[j]) as usize) << 1;
                        out[ti] += g;
                        out[ti + 1] += h;
                    }
                }
            }
        }
    }

    /// The group's histogram (`out`, zeroed here) over `num_rows` rows, split into
    /// upstream's data blocks for `num_threads` threads: block 0 is written in place and
    /// later blocks are added to it in block order (upstream
    /// `MultiValBinWrapper::ConstructHistograms` + `HistMerge`), so the result does not
    /// depend on scheduling. `scratch` holds the other blocks' histograms and is reused
    /// across calls (upstream `hist_buf_`).
    #[allow(clippy::too_many_arguments)]
    pub fn construct(
        &self,
        rows: Option<&[i32]>,
        num_rows: usize,
        num_threads: usize,
        pool: Option<&crate::utils::spin_pool::SpinPool>,
        grad_at: &(impl Fn(usize, usize) -> f32 + Sync),
        hess_at: &(impl Fn(usize, usize) -> f32 + Sync),
        out: &mut [f64],
        scratch: &mut Vec<f64>,
    ) {
        let num_bin = self.num_bin();
        let out = &mut out[..num_bin * 2];
        out.fill(0.0);
        let (n_block, block_size) = block_info(num_threads, num_rows, self.min_block_size());
        let pool = match pool {
            Some(p) if n_block > 1 => p,
            _ => {
                self.accumulate(rows, 0, num_rows, grad_at, hess_at, out);
                return;
            }
        };
        let len = num_bin * 2;
        // Each block zeroes its own buffer inside its task (upstream memsets per block).
        scratch.resize(len * (n_block - 1), 0.0);
        {
            let mut bufs: Vec<&mut [f64]> = Vec::with_capacity(n_block);
            bufs.push(&mut *out);
            bufs.extend(scratch.chunks_mut(len));
            pool.for_each_mut(&mut bufs, |b, buf| {
                if b == 0 {
                    self.accumulate(rows, 0, block_size.min(num_rows), grad_at, hess_at, buf);
                } else {
                    buf.fill(0.0);
                    let start = b * block_size;
                    let end = (start + block_size).min(num_rows);
                    if start < end {
                        self.accumulate(rows, start, end, grad_at, hess_at, buf);
                    }
                }
            });
        }
        // upstream HistMerge: parallel over bin ranges, each bin adding blocks 1.. in order.
        let scratch: &[f64] = &scratch[..];
        pool.for_each_chunk_mut(out, MERGE_CHUNK, |c, dst| {
            let lo = c * MERGE_CHUNK;
            for b in 0..n_block - 1 {
                let src = &scratch[b * len + lo..b * len + lo + dst.len()];
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += s;
                }
            }
        });
    }
}

/// upstream `Threading::BlockInfo(num_threads, cnt, min_cnt_per_block, ...)`.
fn block_info(num_threads: usize, cnt: usize, min_cnt_per_block: usize) -> (usize, usize) {
    let n_block = num_threads
        .max(1)
        .min(cnt.div_ceil(min_cnt_per_block.max(1)));
    if n_block > 1 {
        let size = cnt.div_ceil(n_block);
        (n_block, size.div_ceil(K_ALIGNED_SIZE) * K_ALIGNED_SIZE)
    } else {
        (n_block, cnt)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    use crate::io::bin::BinMapper;
    use crate::io::bin_data::{DenseBin, SparseBin};
    use crate::utils::random::Random;

    const NUM_BIN: [i32; 3] = [4, 3, 5];
    const MFB: [u32; 3] = [0, 2, 1];

    /// Raw per-feature bins (mostly the most frequent one) for `n` rows.
    fn raw_bins(n: usize) -> Vec<[u32; 3]> {
        let mut rng = Random::new(3);
        (0..n)
            .map(|_| {
                let mut row = [0u32; 3];
                for j in 0..3 {
                    row[j] = if rng.next_float() < 0.6 {
                        MFB[j]
                    } else {
                        rng.next_short(0, NUM_BIN[j]) as u32
                    };
                }
                row
            })
            .collect()
    }

    /// A multi-value group over `raw`, its columns stored like `FeatureGroup::PushData`
    /// (`0` = most frequent, else `raw + 1 - (mfb == 0)`); odd features use sparse columns.
    fn group(raw: &[[u32; 3]], dense_multi_val: bool) -> (FeatureGroupLayout, Vec<BinData>) {
        let n = raw.len();
        let bin_mappers: Vec<BinMapper> = (0..3)
            .map(|j| BinMapper {
                num_bin: NUM_BIN[j],
                most_freq_bin: MFB[j],
                sparse_rate: if dense_multi_val { 0.1 } else { 0.6 },
                ..BinMapper::default()
            })
            .collect();
        let offset = u32::from(!dense_multi_val);
        let mut bin_offsets = vec![offset];
        for m in &bin_mappers {
            let num_bin = m.num_bin as u32 - if m.most_freq_bin == 0 { offset } else { 0 };
            bin_offsets.push(bin_offsets.last().unwrap() + num_bin);
        }
        let cols = (0..3)
            .map(|j| {
                let stored = |r: usize| {
                    let b = raw[r][j];
                    if b == MFB[j] {
                        0
                    } else {
                        b + 1 - u32::from(MFB[j] == 0)
                    }
                };
                let num_bin = NUM_BIN[j] as u32 + u32::from(MFB[j] != 0);
                if j % 2 == 1 {
                    let pairs: Vec<(usize, u32)> = (0..n)
                        .filter(|&r| stored(r) != 0)
                        .map(|r| (r, stored(r)))
                        .collect();
                    BinData::Sparse(SparseBin::from_sorted_pairs(n, num_bin, &pairs))
                } else {
                    let mut d = DenseBin::new(n, num_bin);
                    for r in 0..n {
                        d.set(r, stored(r));
                    }
                    BinData::Dense(d)
                }
            })
            .collect();
        let layout = FeatureGroupLayout {
            features: vec![0, 1, 2],
            bin_mappers,
            is_multi_val: true,
            is_dense_multi_val: dense_multi_val,
            is_sparse: false,
            num_total_bin: *bin_offsets.last().unwrap(),
            bin_offsets,
        };
        (layout, cols)
    }

    /// Where raw bin `b` of feature `j` lands in the group histogram, if it is stored.
    fn slot(layout: &FeatureGroupLayout, j: usize, b: u32) -> Option<usize> {
        let base = layout.bin_offsets[j] as usize;
        if layout.is_dense_multi_val {
            Some(base + b as usize)
        } else if b == MFB[j] {
            None
        } else {
            Some(base + b as usize - usize::from(MFB[j] == 0))
        }
    }

    #[test]
    fn histograms_match_a_naive_per_feature_sum() {
        let raw = raw_bins(3000);
        let grads: Vec<f32> = (0..raw.len())
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let hess: Vec<f32> = (0..raw.len())
            .map(|i| 0.5 + (i % 13) as f32 / 9.0)
            .collect();
        let rows: Vec<i32> = (0..raw.len() as i32).filter(|r| r % 3 != 1).collect();
        for dense_multi_val in [false, true] {
            let (layout, cols) = group(&raw, dense_multi_val);
            let mv = MultiValBin::from_group(&layout, &cols, raw.len());
            let num_bin = layout.num_total_bin as usize;
            for subset in [None, Some(rows.as_slice())] {
                let n = subset.map_or(raw.len(), <[i32]>::len);
                let row_of = |i: usize| subset.map_or(i, |r| r[i] as usize);
                let mut want = vec![0.0f64; num_bin * 2];
                for i in 0..n {
                    let r = row_of(i);
                    for j in 0..3 {
                        if let Some(t) = slot(&layout, j, raw[r][j]) {
                            want[t * 2] += f64::from(grads[r]);
                            want[t * 2 + 1] += f64::from(hess[r]);
                        }
                    }
                }
                let grad_at = |_: usize, r: usize| grads[r];
                let hess_at = |_: usize, r: usize| hess[r];
                let mut got = vec![f64::NAN; num_bin * 2];
                mv.construct(
                    subset,
                    n,
                    1,
                    None,
                    &grad_at,
                    &hess_at,
                    &mut got,
                    &mut Vec::new(),
                );
                assert_eq!(got, want, "dense_multi_val={dense_multi_val} single block");
                let mut blocked = vec![f64::NAN; num_bin * 2];
                let pool = crate::utils::spin_pool::SpinPool::new(4).unwrap();
                mv.construct(
                    subset,
                    n,
                    4,
                    Some(&pool),
                    &grad_at,
                    &hess_at,
                    &mut blocked,
                    &mut Vec::new(),
                );
                for (b, w) in blocked.iter().zip(&want) {
                    assert!(
                        (b - w).abs() <= 1e-9 * w.abs().max(1.0),
                        "blocked {b} vs {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn block_info_matches_upstream() {
        assert_eq!(block_info(1, 10_000, 32), (1, 10_000));
        assert_eq!(block_info(4, 10_000, 32), (4, 2528));
        assert_eq!(block_info(8, 100, 32), (4, 32));
        assert_eq!(block_info(4, 0, 32), (0, 0));
    }
}
