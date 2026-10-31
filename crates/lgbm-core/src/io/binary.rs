//! Reader for upstream's binary dataset file (`Dataset::SaveBinaryFile`).
//! upstream: src/io/dataset.cpp (SerializeHeader), src/io/dataset_loader.cpp
//! (LoadFromBinFile), include/LightGBM/feature_group.h, src/io/metadata.cpp

use std::path::Path;

use super::bin::BinMapper;
use crate::error::{Error, Result};

/// upstream: `Dataset::binary_file_token`
pub const BINARY_FILE_TOKEN: &[u8] = b"______LightGBM_Binary_File_Token______\n";

/// Little-endian cursor implementing upstream's `AlignedWrite` (8-byte padding) layout.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

pub(crate) fn aligned(n: usize) -> usize {
    n.div_ceil(8) * 8
}

impl<'a> Reader<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.buf.len());
        let end = end.ok_or_else(|| Error::Parse("binary dataset truncated".into()))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn skip_to_aligned(&mut self, start: usize) -> Result<()> {
        let pad = aligned(self.pos - start) - (self.pos - start);
        self.take(pad).map(|_| ())
    }
    pub(crate) fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub(crate) fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    pub(crate) fn f64(&mut self) -> Result<f64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(f64::from_le_bytes(a))
    }
    pub(crate) fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    pub(crate) fn aligned_i32(&mut self) -> Result<i32> {
        let v = self.i32()?;
        self.take(4)?;
        Ok(v)
    }
    pub(crate) fn aligned_bool(&mut self) -> Result<bool> {
        let b = self.take(8)?;
        Ok(b[0] != 0)
    }
    pub(crate) fn aligned_i32_vec(&mut self, n: usize) -> Result<Vec<i32>> {
        let start = self.pos;
        let v = (0..n).map(|_| self.i32()).collect::<Result<Vec<_>>>()?;
        self.skip_to_aligned(start)?;
        Ok(v)
    }
    pub(crate) fn aligned_f32_vec(&mut self, n: usize) -> Result<Vec<f32>> {
        let start = self.pos;
        let v = (0..n).map(|_| self.f32()).collect::<Result<Vec<_>>>()?;
        self.skip_to_aligned(start)?;
        Ok(v)
    }
    pub(crate) fn aligned_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.take(n)?;
        let pad = aligned(n) - n;
        self.take(pad)?;
        Ok(s)
    }
}

/// upstream `Dataset` header fields (`SerializeHeader`).
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetHeader {
    pub num_data: i32,
    pub num_features: i32,
    pub num_total_features: i32,
    pub label_idx: i32,
    pub max_bin: i32,
    pub bin_construct_sample_cnt: i32,
    pub min_data_in_bin: i32,
    pub use_missing: bool,
    pub zero_as_missing: bool,
    pub has_raw: bool,
    pub used_feature_map: Vec<i32>,
    pub num_groups: i32,
    pub real_feature_idx: Vec<i32>,
    pub feature2group: Vec<i32>,
    pub feature2subfeature: Vec<i32>,
    pub group_bin_boundaries: Vec<u64>,
    pub group_feature_start: Vec<i32>,
    pub group_feature_cnt: Vec<i32>,
    pub max_bin_by_feature: Vec<i32>,
    pub feature_names: Vec<String>,
    pub forced_bin_bounds: Vec<Vec<f64>>,
}

/// upstream `Metadata` as stored in the binary file (init scores are not saved).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BinaryMetadata {
    pub label: Vec<f32>,
    pub weights: Vec<f32>,
    pub query_boundaries: Vec<i32>,
}

/// One feature group's definition plus its raw serialized bin data.
#[derive(Debug, Clone)]
pub struct FeatureGroupDef {
    pub is_multi_val: bool,
    pub is_dense_multi_val: bool,
    pub is_sparse: bool,
    pub bin_mappers: Vec<BinMapper>,
    /// Serialized bin data (dense/sparse/multi-val), not decoded yet.
    pub data: Vec<u8>,
}

/// A parsed upstream binary dataset file.
#[derive(Debug, Clone)]
pub struct BinaryDataset {
    pub header: DatasetHeader,
    pub metadata: BinaryMetadata,
    pub groups: Vec<FeatureGroupDef>,
    /// Raw float32 values (row-major, numeric features only) when `has_raw` (linear trees).
    pub raw_data: Vec<f32>,
}

impl BinaryDataset {
    pub fn read_file(path: impl AsRef<Path>) -> Result<BinaryDataset> {
        BinaryDataset::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(buf: &[u8]) -> Result<BinaryDataset> {
        let mut r = Reader::new(buf);
        let token = r.aligned_bytes(BINARY_FILE_TOKEN.len())?;
        if token != BINARY_FILE_TOKEN {
            return Err(Error::Parse(
                "Input file is not LightGBM binary file".into(),
            ));
        }
        let header_size = r.u64()? as usize;
        let header_start = r.pos();
        let header = read_header(&mut r)?;
        if r.pos() - header_start != header_size {
            return Err(Error::Parse(format!(
                "binary header size mismatch: declared {header_size}, parsed {}",
                r.pos() - header_start
            )));
        }
        let meta_size = r.u64()? as usize;
        let meta_start = r.pos();
        let metadata = read_metadata(&mut r)?;
        if r.pos() - meta_start != meta_size {
            return Err(Error::Parse("binary metadata size mismatch".into()));
        }
        let mut groups = Vec::with_capacity(header.num_groups as usize);
        for _ in 0..header.num_groups {
            let size = r.u64()? as usize;
            let start = r.pos();
            let is_multi_val = r.aligned_bool()?;
            let is_dense_multi_val = r.aligned_bool()?;
            let is_sparse = r.aligned_bool()?;
            let num_feature = r.aligned_i32()?;
            let mut bin_mappers = Vec::with_capacity(num_feature.max(0) as usize);
            for _ in 0..num_feature {
                let (m, used) = BinMapper::from_bytes(&buf[r.pos()..])?;
                r.take(used)?;
                bin_mappers.push(m);
            }
            let consumed = r.pos() - start;
            let data = r
                .take(
                    size.checked_sub(consumed)
                        .ok_or_else(|| Error::Parse("feature group size".into()))?,
                )?
                .to_vec();
            groups.push(FeatureGroupDef {
                is_multi_val,
                is_dense_multi_val,
                is_sparse,
                bin_mappers,
                data,
            });
        }
        let raw_data = if header.has_raw {
            let rest = &buf[r.pos()..];
            rest.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };
        Ok(BinaryDataset {
            header,
            metadata,
            groups,
            raw_data,
        })
    }
}

fn read_header(r: &mut Reader<'_>) -> Result<DatasetHeader> {
    let num_data = r.aligned_i32()?;
    let num_features = r.aligned_i32()?;
    let num_total_features = r.aligned_i32()?;
    let label_idx = r.aligned_i32()?;
    let max_bin = r.aligned_i32()?;
    let bin_construct_sample_cnt = r.aligned_i32()?;
    let min_data_in_bin = r.aligned_i32()?;
    let use_missing = r.aligned_bool()?;
    let zero_as_missing = r.aligned_bool()?;
    let has_raw = r.aligned_bool()?;
    let ntf = usize::try_from(num_total_features)
        .map_err(|_| Error::Parse("num_total_features".into()))?;
    let nf = usize::try_from(num_features).map_err(|_| Error::Parse("num_features".into()))?;
    let used_feature_map = r.aligned_i32_vec(ntf)?;
    let num_groups = r.aligned_i32()?;
    let ng = usize::try_from(num_groups).map_err(|_| Error::Parse("num_groups".into()))?;
    let real_feature_idx = r.aligned_i32_vec(nf)?;
    let feature2group = r.aligned_i32_vec(nf)?;
    let feature2subfeature = r.aligned_i32_vec(nf)?;
    let group_bin_boundaries = (0..=ng).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
    let group_feature_start = r.aligned_i32_vec(ng)?;
    let group_feature_cnt = r.aligned_i32_vec(ng)?;
    let mut max_bin_by_feature = r.aligned_i32_vec(ntf)?;
    if max_bin_by_feature.iter().all(|&v| v == -1) {
        max_bin_by_feature.clear();
    }
    let mut feature_names = Vec::with_capacity(ntf);
    for _ in 0..ntf {
        let len = r.aligned_i32()? as usize;
        let bytes = r.aligned_bytes(len)?;
        feature_names.push(String::from_utf8_lossy(bytes).into_owned());
    }
    let mut forced_bin_bounds = Vec::with_capacity(ntf);
    for _ in 0..ntf {
        let n = r.aligned_i32()? as usize;
        forced_bin_bounds.push((0..n).map(|_| r.f64()).collect::<Result<Vec<_>>>()?);
    }
    Ok(DatasetHeader {
        num_data,
        num_features,
        num_total_features,
        label_idx,
        max_bin,
        bin_construct_sample_cnt,
        min_data_in_bin,
        use_missing,
        zero_as_missing,
        has_raw,
        used_feature_map,
        num_groups,
        real_feature_idx,
        feature2group,
        feature2subfeature,
        group_bin_boundaries,
        group_feature_start,
        group_feature_cnt,
        max_bin_by_feature,
        feature_names,
        forced_bin_bounds,
    })
}

fn read_metadata(r: &mut Reader<'_>) -> Result<BinaryMetadata> {
    let num_data = r.aligned_i32()? as usize;
    let num_weights = r.aligned_i32()? as usize;
    let num_queries = r.aligned_i32()? as usize;
    let label = r.aligned_f32_vec(num_data)?;
    let weights = if num_weights > 0 {
        r.aligned_f32_vec(num_weights)?
    } else {
        Vec::new()
    };
    let query_boundaries = if num_queries > 0 {
        r.aligned_i32_vec(num_queries + 1)?
    } else {
        Vec::new()
    };
    Ok(BinaryMetadata {
        label,
        weights,
        query_boundaries,
    })
}
