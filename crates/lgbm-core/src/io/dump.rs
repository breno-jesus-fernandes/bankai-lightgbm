//! `Dataset._dump_text`: the text dump of a constructed dataset's bins.
//!
//! upstream: src/io/dataset.cpp (v4.7.0) `Dataset::DumpTextFile`. The layout and the number
//! formats follow it exactly, so the file is byte-identical to upstream's for the same bins.

use std::fmt::Write;

use super::bin_data::BinData;
use super::dataset::{Dataset, GroupData};

/// The raw bin value of every row in one stored column (absent sparse rows read as 0).
fn raw_column(bin: &BinData, num_data: usize) -> Vec<u32> {
    match bin {
        BinData::Dense(d) => (0..num_data).map(|row| d.get(row)).collect(),
        BinData::Sparse(s) => s.to_dense(),
    }
}

/// The bin index of one used feature in every row.
///
/// upstream: include/LightGBM/feature_group.h `FeatureGroup::SubFeatureIterator` and
/// src/io/dense_bin.hpp `DenseBinIterator::Get`: a raw value inside the feature's bin range is
/// shifted to the feature's own bins, and any other value is the feature's most frequent bin.
/// A bundled (multi-value) column covers bins `[1, num_bin - 1 + addi]`; a single column covers
/// `[bin_offsets[sub], bin_offsets[sub + 1] - 1]`.
fn sub_feature_bins(ds: &Dataset, feature: usize) -> Vec<u32> {
    let layout = &ds.layout;
    let group_index = layout.feature2group[feature] as usize;
    let sub = layout.feature2subfeature[feature] as usize;
    let group = &layout.groups[group_index];
    let mapper = &group.bin_mappers[sub];
    let most_freq_bin = mapper.most_freq_bin;
    let offset = u32::from(most_freq_bin == 0);
    let num_data = layout.num_data;
    let (min_bin, max_bin, raw) = match &ds.data[group_index] {
        GroupData::Single(bin) => {
            let min_bin = group.bin_offsets[sub];
            let max_bin = group.bin_offsets[sub + 1] - 1;
            (min_bin, max_bin, raw_column(bin, num_data))
        }
        GroupData::Multi(columns) => {
            let addi = u32::from(most_freq_bin != 0);
            let max_bin = (mapper.num_bin - 1) as u32 + addi;
            (1, max_bin, raw_column(&columns[sub], num_data))
        }
    };
    raw.into_iter()
        .map(|value| {
            if value >= min_bin && value <= max_bin {
                value - min_bin + offset
            } else {
                most_freq_bin
            }
        })
        .collect()
}

/// `printf("%lf")`: six decimals, with C's spellings for non-finite values.
fn c_double(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{value:.6}")
    }
}

/// The dump of `ds`, as upstream's `Dataset::DumpTextFile` writes it (without a final newline).
pub fn dump_text(ds: &Dataset) -> String {
    let layout = &ds.layout;
    let num_total = layout.num_total_features;
    let mut out = String::new();
    // Writes to a String cannot fail.
    let _ = writeln!(out, "num_features: {}", layout.num_features);
    let _ = writeln!(out, "num_total_features: {num_total}");
    let _ = writeln!(out, "num_groups: {}", layout.groups.len());
    let _ = writeln!(out, "num_data: {}", layout.num_data);
    out.push_str("feature_names: ");
    for name in &ds.feature_names {
        let _ = write!(out, "{name}, ");
    }
    out.push_str("\nmax_bin_by_feature: ");
    for bins in &ds.max_bin_by_feature {
        let _ = write!(out, "{bins}, ");
    }
    out.push('\n');
    for name in &ds.feature_names {
        let _ = write!(out, "{name}, ");
    }
    out.push_str("\nforced_bins: ");
    for feature in 0..num_total {
        let _ = write!(out, "\nfeature {feature}: ");
        if let Some(bounds) = ds.forced_bin_bounds.get(feature) {
            for bound in bounds {
                let _ = write!(out, "{}, ", c_double(*bound));
            }
        }
    }
    // Per total feature: its bin column, or None for an unused feature (printed as NA).
    let columns: Vec<Option<Vec<u32>>> = (0..num_total)
        .map(|feature| {
            let inner = layout.used_feature_map[feature];
            (inner >= 0).then(|| sub_feature_bins(ds, inner as usize))
        })
        .collect();
    for row in 0..layout.num_data {
        out.push('\n');
        for column in &columns {
            match column {
                None => out.push_str("NA, "),
                Some(bins) => {
                    let _ = write!(out, "{}, ", bins[row]);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::c_double;

    #[test]
    fn c_double_matches_printf_lf() {
        assert_eq!(c_double(1.5), "1.500000");
        assert_eq!(c_double(-0.0000004), "-0.000000");
        assert_eq!(c_double(f64::NAN), "nan");
        assert_eq!(c_double(f64::INFINITY), "inf");
        assert_eq!(c_double(f64::NEG_INFINITY), "-inf");
    }
}
