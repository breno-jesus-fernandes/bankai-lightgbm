//! Text data parsers: CSV, TSV and LibSVM. upstream: src/io/parser.cpp, src/io/parser.hpp.

use crate::error::{Error, Result};
use crate::meta::K_ZERO_THRESHOLD;
use crate::utils::common::atof_legacy;

/// The text format of a data file (upstream `DataType`, without `INVALID`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Csv,
    Tsv,
    LibSvm,
}

/// One parsed line: the label and the `(feature index, value)` pairs (upstream
/// `ParseOneLine`). Feature indices already skip the label column.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRow {
    pub label: f64,
    pub features: Vec<(i32, f64)>,
}

fn is_trim_char(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// upstream: `Common::Trim`.
fn trim(s: &str) -> &str {
    s.trim_matches(is_trim_char)
}

fn count_of(line: &str, c: char) -> usize {
    line.chars().filter(|&x| x == c).count()
}

/// Detect the format from the first lines (upstream `GetDataType`). Lines are the data
/// lines after the header; only their delimiter counts matter.
pub fn detect_data_type(lines: &[&str]) -> Result<DataType> {
    let first = lines
        .first()
        .ok_or_else(|| Error::Parse("Data file should have at least one line.".into()))?;
    let (commas, tabs, colons) = (
        count_of(first, ','),
        count_of(first, '\t'),
        count_of(first, ':'),
    );
    let detected = if lines.len() == 1 {
        if colons > 0 {
            Some(DataType::LibSvm)
        } else if tabs > 0 {
            Some(DataType::Tsv)
        } else if commas > 0 {
            Some(DataType::Csv)
        } else {
            None
        }
    } else {
        let second = lines[1];
        let (commas2, tabs2, colons2) = (
            count_of(second, ','),
            count_of(second, '\t'),
            count_of(second, ':'),
        );
        let mut detected = if colons > 0 || colons2 > 0 {
            Some(DataType::LibSvm)
        } else if tabs == tabs2 && tabs > 0 {
            Some(DataType::Tsv)
        } else if commas == commas2 && commas > 0 {
            Some(DataType::Csv)
        } else {
            None
        };
        // Every later line must repeat the delimiter count of the first line.
        if let Some(ty @ (DataType::Tsv | DataType::Csv)) = detected {
            for line in &lines[2..] {
                let same = match ty {
                    DataType::Tsv => count_of(line, '\t') == tabs,
                    _ => count_of(line, ',') == commas,
                };
                if !same {
                    detected = None;
                    break;
                }
            }
        }
        detected
    };
    detected.ok_or_else(|| {
        Error::Parse(
            "Unknown format of training data. Only CSV, TSV, and LibSVM (zero-based) formatted text files are supported."
                .into(),
        )
    })
}

/// Number of columns of a CSV or TSV line (upstream `GetDataType` `num_col`). LibSVM files
/// have no column count from one line; their width comes from the largest index.
pub fn delimited_num_columns(ty: DataType, first_line: &str) -> usize {
    match ty {
        DataType::Csv => count_of(first_line, ',') + 1,
        DataType::Tsv => count_of(first_line, '\t') + 1,
        DataType::LibSvm => 0,
    }
}

/// upstream: `LibSVMParser` constructor check.
pub fn label_index_checked(ty: DataType, label_idx: i32) -> Result<()> {
    if ty == DataType::LibSvm && label_idx > 0 {
        return Err(Error::Parse(
            "Label should be the first column in a LibSVM file".into(),
        ));
    }
    Ok(())
}

/// The label column for a file with `num_features` features (upstream `GetLabelIdxFor*`):
/// `-1` when the first line already has exactly `num_features` columns, so there is no label.
pub fn label_index(ty: DataType, first_line: &str, num_features: i32, label_idx: i32) -> i32 {
    if num_features <= 0 {
        return label_idx;
    }
    let line = trim(first_line);
    match ty {
        DataType::LibSvm => {
            let space = line.find([' ', '\x0c', '\n', '\r', '\t', '\x0b']);
            let colon = line.find(':');
            match (space, colon) {
                (None, _) => label_idx,
                (Some(s), Some(c)) if s < c => label_idx,
                (Some(_), None) => label_idx,
                _ => -1,
            }
        }
        DataType::Csv | DataType::Tsv => {
            let sep = if ty == DataType::Csv { ',' } else { '\t' };
            if line.split(sep).count() as i64 == i64::from(num_features) {
                -1
            } else {
                label_idx
            }
        }
    }
}

/// Parse one float field with the legacy parser (upstream `Common::Atof`): an empty field
/// is NaN, `na`/`nan`/`null` are NaN, `inf`/`infinity` are huge, anything else must be a number.
pub fn parse_legacy(token: &str) -> Result<f64> {
    let t = token.trim_matches(' ');
    if t.is_empty() {
        return Ok(f64::NAN);
    }
    atof_legacy(t).ok_or_else(|| Error::Parse(format!("Unknown token {t} in data file")))
}

/// Parse one float field with the correctly rounded parser (upstream `Common::AtofPrecise`).
/// An empty field is an error here, as upstream's conversion fails on it.
fn parse_precise(token: &str) -> Result<f64> {
    let t = token.trim_matches(' ');
    t.parse::<f64>()
        .map_err(|_| Error::Parse(format!("no conversion to double for: {t}")))
}

fn parse_value(token: &str, precise: bool) -> Result<f64> {
    if precise {
        parse_precise(token)
    } else {
        parse_legacy(token)
    }
}

/// Parse one CSV, TSV or LibSVM line (upstream `CSVParser`, `TSVParser`, `LibSVMParser`
/// `ParseOneLine`). `precise` selects the float parser for CSV, TSV and the LibSVM label.
pub fn parse_line(ty: DataType, line: &str, label_idx: i32, precise: bool) -> Result<ParsedRow> {
    match ty {
        DataType::Csv | DataType::Tsv => parse_delimited(ty, line, label_idx, precise),
        DataType::LibSvm => parse_libsvm(line, label_idx, precise),
    }
}

fn parse_delimited(ty: DataType, line: &str, label_idx: i32, precise: bool) -> Result<ParsedRow> {
    let sep = if ty == DataType::Csv { ',' } else { '\t' };
    let mut row = ParsedRow {
        label: 0.0,
        features: Vec::new(),
    };
    if line.is_empty() {
        return Ok(row);
    }
    // A trailing delimiter ends the line; it does not start another field.
    let body = line.strip_suffix(sep).unwrap_or(line);
    let mut offset = 0;
    for (idx, token) in body.split(sep).enumerate() {
        let idx = idx as i32;
        let val = parse_value(token, precise)?;
        if idx == label_idx {
            row.label = val;
            offset = -1;
        } else if val.abs() > K_ZERO_THRESHOLD || val.is_nan() {
            row.features.push((idx + offset, val));
        }
    }
    Ok(row)
}

fn parse_libsvm(line: &str, label_idx: i32, precise: bool) -> Result<ParsedRow> {
    let mut row = ParsedRow {
        label: 0.0,
        features: Vec::new(),
    };
    let mut tokens = line
        .split([' ', '\t', '\x0b', '\x0c'])
        .filter(|t| !t.is_empty());
    if label_idx == 0 {
        let token = tokens
            .next()
            .ok_or_else(|| Error::Parse("Input format error when parsing as LibSVM".into()))?;
        row.label = parse_value(token, precise)?;
    }
    for token in tokens {
        let (idx, val) = token
            .split_once(':')
            .ok_or_else(|| Error::Parse("Input format error when parsing as LibSVM".into()))?;
        let idx: i32 = idx.parse().map_err(|_| {
            Error::Parse(format!(
                "Input format error when parsing as LibSVM: {token}"
            ))
        })?;
        // upstream: LibSVMParser always uses the legacy Common::Atof for feature values.
        row.features.push((idx, parse_legacy(val)?));
    }
    Ok(row)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn detects_csv_tsv_and_libsvm_from_the_first_lines() {
        assert_eq!(
            detect_data_type(&["1,2,3", "4,5,6"]).unwrap(),
            DataType::Csv
        );
        assert_eq!(
            detect_data_type(&["1\t2\t3", "4\t5\t6"]).unwrap(),
            DataType::Tsv
        );
        assert_eq!(
            detect_data_type(&["1 0:2 3:4", "0 1:5"]).unwrap(),
            DataType::LibSvm
        );
        assert_eq!(detect_data_type(&["1,2,3"]).unwrap(), DataType::Csv);
    }

    #[test]
    fn rejects_lines_whose_delimiter_counts_differ() {
        let err = detect_data_type(&["1,2,3", "4,5", "6,7,8"]).unwrap_err();
        assert!(matches!(err, crate::Error::Parse(_)));
        let err = detect_data_type(&["plain text"]).unwrap_err();
        assert!(matches!(err, crate::Error::Parse(_)));
    }

    #[test]
    fn label_column_is_the_first_one_unless_the_file_has_no_label() {
        // num_features == number of columns: the file has no label column (upstream
        // GetLabelIdxForCSV returns -1).
        assert_eq!(label_index(DataType::Csv, "1,2,3", 3, 0), -1);
        assert_eq!(label_index(DataType::Csv, "1,2,3", 2, 0), 0);
        assert_eq!(label_index(DataType::Csv, "1,2,3", 0, 0), 0);
        assert_eq!(label_index(DataType::Tsv, "1\t2", 2, 0), -1);
        assert_eq!(label_index(DataType::LibSvm, "1 0:2", 0, 0), 0);
        // A LibSVM line starts with its label when the first space comes before the first colon.
        assert_eq!(label_index(DataType::LibSvm, "1 0:2", 2, 0), 0);
        assert_eq!(label_index(DataType::LibSvm, "0:2 1:3", 2, 0), -1);
    }

    #[test]
    fn csv_row_drops_zeros_and_keeps_nan_with_shifted_feature_indices() {
        let row = parse_line(DataType::Csv, "1,0,2.5,nan", 0, false).unwrap();
        assert_eq!(row.label, 1.0);
        assert_eq!(row.features.len(), 2);
        assert_eq!(row.features[0].0, 1);
        assert_eq!(row.features[0].1, 2.5);
        assert_eq!(row.features[1].0, 2);
        assert!(row.features[1].1.is_nan());
    }

    #[test]
    fn empty_csv_field_is_nan_with_the_legacy_parser_and_an_error_with_the_precise_one() {
        let row = parse_line(DataType::Csv, "1,,3", 0, false).unwrap();
        assert!(row.features[0].1.is_nan());
        let err = parse_line(DataType::Csv, "1,,3", 0, true).unwrap_err();
        assert!(matches!(err, crate::Error::Parse(_)));
    }

    #[test]
    fn unknown_tokens_are_errors_and_na_tokens_are_nan() {
        let err = parse_line(DataType::Csv, "1,abc,3", 0, false).unwrap_err();
        assert!(matches!(err, crate::Error::Parse(_)));
        let row = parse_line(DataType::Tsv, "1\tNA\tnull", 0, false).unwrap();
        assert!(row.features.iter().all(|(_, v)| v.is_nan()));
    }

    #[test]
    fn tsv_row_matches_csv_semantics_with_tabs() {
        let row = parse_line(DataType::Tsv, "0\t0\t-1.25", 0, false).unwrap();
        assert_eq!(row.label, 0.0);
        assert_eq!(row.features, vec![(1, -1.25)]);
    }

    #[test]
    fn libsvm_row_reads_the_label_then_index_value_pairs() {
        let row = parse_line(DataType::LibSvm, "1 0:0.5  3:2", 0, false).unwrap();
        assert_eq!(row.label, 1.0);
        assert_eq!(row.features, vec![(0, 0.5), (3, 2.0)]);
        let unlabeled = parse_line(DataType::LibSvm, "2:4", -1, false).unwrap();
        assert_eq!(unlabeled.label, 0.0);
        assert_eq!(unlabeled.features, vec![(2, 4.0)]);
    }

    #[test]
    fn libsvm_label_must_be_the_first_column() {
        let err = label_index_checked(DataType::LibSvm, 1);
        assert!(matches!(err, Err(crate::Error::Parse(_))));
    }

    #[test]
    fn precise_parser_handles_long_decimals_like_a_correctly_rounded_parse() {
        let row = parse_line(
            DataType::Csv,
            "0,0.1000000000000000055511151231257827",
            0,
            true,
        )
        .unwrap();
        assert_eq!(row.features[0].1, 0.1);
    }
}
