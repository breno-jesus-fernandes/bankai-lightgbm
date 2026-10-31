//! Bin storage. upstream: src/io/dense_bin.hpp, src/io/sparse_bin.hpp

use super::binary::aligned;

/// upstream `kNumFastIndex`
const K_NUM_FAST_INDEX: usize = 64;

/// Dense storage: one bin per row. upstream: `DenseBin<VAL_T, IS_4BIT>`
#[derive(Debug, Clone, PartialEq)]
pub enum DenseBin {
    /// Two rows per byte (low nibble = even row). Used when `num_bin <= 16`.
    U4(Vec<u8>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl DenseBin {
    /// upstream: `Bin::CreateDenseBin`
    pub fn new(num_data: usize, num_bin: u32) -> DenseBin {
        if num_bin <= 16 {
            DenseBin::U4(vec![0; num_data.div_ceil(2)])
        } else if num_bin <= 256 {
            DenseBin::U8(vec![0; num_data])
        } else if num_bin <= 65536 {
            DenseBin::U16(vec![0; num_data])
        } else {
            DenseBin::U32(vec![0; num_data])
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> u32 {
        match self {
            DenseBin::U4(d) => u32::from((d[idx >> 1] >> ((idx & 1) << 2)) & 0xf),
            DenseBin::U8(d) => u32::from(d[idx]),
            DenseBin::U16(d) => u32::from(d[idx]),
            DenseBin::U32(d) => d[idx],
        }
    }

    /// Overwrite row `idx` (equivalent to upstream `Push` + `FinishLoad`).
    #[inline]
    pub fn set(&mut self, idx: usize, value: u32) {
        match self {
            DenseBin::U4(d) => {
                let shift = (idx & 1) << 2;
                let b = &mut d[idx >> 1];
                *b = (*b & !(0xf << shift)) | (((value as u8) & 0xf) << shift);
            }
            DenseBin::U8(d) => d[idx] = value as u8,
            DenseBin::U16(d) => d[idx] = value as u16,
            DenseBin::U32(d) => d[idx] = value,
        }
    }

    /// upstream: `DenseBin::SaveBinaryToFile` (aligned raw array)
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        match self {
            DenseBin::U4(d) | DenseBin::U8(d) => out.extend_from_slice(d),
            DenseBin::U16(d) => d
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            DenseBin::U32(d) => d
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        }
        pad(out, start);
    }
}

/// Values of a sparse bin. upstream: `SparseBin<VAL_T>::vals_`
#[derive(Debug, Clone, PartialEq)]
pub enum SparseVals {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// Delta-encoded sparse storage. upstream: `SparseBin<VAL_T>`
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBin {
    pub num_data: usize,
    /// Row deltas; an extra trailing 0 guards reads past the end.
    pub deltas: Vec<u8>,
    pub vals: SparseVals,
    /// upstream `fast_index_`: (val index, row index) every `1 << fast_index_shift` rows.
    pub fast_index: Vec<(usize, usize)>,
    pub fast_index_shift: u32,
}

impl SparseBin {
    /// upstream: `Bin::CreateSparseBin` + `Push`es + `FinishLoad` / `LoadFromPair`.
    /// `pairs` must be sorted by row.
    pub fn from_sorted_pairs(num_data: usize, num_bin: u32, pairs: &[(usize, u32)]) -> SparseBin {
        let mut deltas = Vec::with_capacity(pairs.len() + 1);
        let mut vals: Vec<u32> = Vec::with_capacity(pairs.len());
        let mut last_idx = 0usize;
        for (i, &(cur_idx, bin)) in pairs.iter().enumerate() {
            let mut cur_delta = cur_idx - last_idx;
            if i > 0 && cur_delta == 0 {
                continue; // upstream disallows multiple values in one row
            }
            while cur_delta >= 256 {
                deltas.push(255);
                vals.push(0);
                cur_delta -= 255;
            }
            deltas.push(cur_delta as u8);
            vals.push(bin);
            last_idx = cur_idx;
        }
        deltas.push(0);
        let vals = if num_bin <= 256 {
            SparseVals::U8(vals.iter().map(|&v| v as u8).collect())
        } else if num_bin <= 65536 {
            SparseVals::U16(vals.iter().map(|&v| v as u16).collect())
        } else {
            SparseVals::U32(vals)
        };
        let mut s = SparseBin {
            num_data,
            deltas,
            vals,
            fast_index: Vec::new(),
            fast_index_shift: 0,
        };
        s.build_fast_index();
        s
    }

    pub fn num_vals(&self) -> usize {
        self.deltas.len() - 1
    }

    #[inline]
    pub fn val(&self, i: usize) -> u32 {
        match &self.vals {
            SparseVals::U8(v) => u32::from(v[i]),
            SparseVals::U16(v) => u32::from(v[i]),
            SparseVals::U32(v) => v[i],
        }
    }

    /// upstream: `SparseBin::GetFastIndex`
    fn build_fast_index(&mut self) {
        self.fast_index.clear();
        let mod_size = self.num_data.div_ceil(K_NUM_FAST_INDEX);
        let mut pow2 = 1usize;
        self.fast_index_shift = 0;
        while pow2 < mod_size {
            pow2 <<= 1;
            self.fast_index_shift += 1;
        }
        let num_vals = self.num_vals();
        let mut i_delta = 0usize;
        let mut cur_pos = 0usize;
        let mut next_threshold = 0usize;
        // upstream NextNonzero: ++i_delta; cur_pos += deltas[i_delta]; i_delta < num_vals
        let mut first = true;
        loop {
            if first {
                first = false;
            } else {
                i_delta += 1;
            }
            cur_pos += self.deltas[i_delta] as usize;
            if i_delta >= num_vals {
                cur_pos = self.num_data;
                break;
            }
            while next_threshold <= cur_pos {
                self.fast_index.push((i_delta, cur_pos));
                next_threshold += pow2;
            }
        }
        while next_threshold < self.num_data {
            self.fast_index.push((num_vals.wrapping_sub(1), cur_pos));
            next_threshold += pow2;
        }
    }

    /// upstream: `SparseBin::SaveBinaryToFile`
    pub fn write(&self, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&(self.num_vals() as i32).to_le_bytes());
        pad(out, start);
        let start = out.len();
        out.extend_from_slice(&self.deltas);
        pad(out, start);
        let start = out.len();
        match &self.vals {
            SparseVals::U8(v) => out.extend_from_slice(v),
            SparseVals::U16(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            SparseVals::U32(v) => v
                .iter()
                .for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        }
        pad(out, start);
    }

    /// Decode to a dense vector of bins (0 where no value is stored).
    pub fn to_dense(&self) -> Vec<u32> {
        let mut out = vec![0u32; self.num_data];
        let mut pos = 0usize;
        for i in 0..self.num_vals() {
            pos += self.deltas[i] as usize;
            if pos < self.num_data {
                out[pos] = self.val(i);
            }
        }
        out
    }
}

/// A bin column: dense or sparse. upstream: `Bin`
#[derive(Debug, Clone, PartialEq)]
pub enum BinData {
    Dense(DenseBin),
    Sparse(SparseBin),
}

impl BinData {
    pub fn write(&self, out: &mut Vec<u8>) {
        match self {
            BinData::Dense(d) => d.write(out),
            BinData::Sparse(s) => s.write(out),
        }
    }
}

fn pad(out: &mut Vec<u8>, start: usize) {
    let n = out.len() - start;
    out.resize(start + aligned(n), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_packing() {
        let mut d = DenseBin::new(5, 10);
        for (i, v) in [3u32, 15, 0, 7, 9].iter().enumerate() {
            d.set(i, *v);
        }
        assert_eq!(
            (0..5).map(|i| d.get(i)).collect::<Vec<_>>(),
            vec![3, 15, 0, 7, 9]
        );
        d.set(1, 2);
        assert_eq!(d.get(0), 3);
        assert_eq!(d.get(1), 2);
    }

    #[test]
    fn sparse_round_trip_with_long_gaps() {
        let pairs = vec![(0usize, 5u32), (300, 7), (301, 1), (1000, 2)];
        let s = SparseBin::from_sorted_pairs(1200, 10, &pairs);
        let dense = s.to_dense();
        for &(i, v) in &pairs {
            assert_eq!(dense[i], v);
        }
        assert_eq!(dense.iter().filter(|&&v| v != 0).count(), 4);
        assert_eq!(*s.deltas.last().unwrap_or(&1), 0);
    }
}
