//! Load a whole text data file into dense rows and metadata (upstream
//! `DatasetLoader::LoadFromFile` without the bin construction). upstream: src/io/dataset_loader.cpp.

use std::collections::{HashMap, HashSet};

use super::parser::{
    DataType, delimited_num_columns, detect_data_type, label_index, label_index_checked, parse_line,
};
use crate::error::{Error, Result};

/// The column options of a text file (upstream `header`, `label_column`, `weight_column`,
/// `group_column`, `ignore_column` and `precise_float_parser`).
#[derive(Debug, Clone)]
pub struct TextLoadOptions<'a> {
    pub header: bool,
    pub label_column: &'a str,
    pub weight_column: &'a str,
    pub group_column: &'a str,
    pub ignore_column: &'a str,
    pub precise_float_parser: bool,
    /// Feature count of the model when predicting from a file (upstream `num_features` of
    /// `CreateParser`), so the label column is detected from the column count. `0` when
    /// training, where column 0 is the label unless `label_column` says otherwise.
    pub num_features: usize,
}

/// A loaded file: dense features (row-major, `num_data x num_features`), labels, and the
/// weight and group (query size) columns when they were requested.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedText {
    pub num_data: usize,
    pub num_features: usize,
    /// Header names of the kept feature columns (empty without a header).
    pub feature_names: Vec<String>,
    pub features: Vec<f64>,
    pub label: Vec<f64>,
    pub weight: Option<Vec<f32>>,
    /// Group sizes: runs of equal consecutive query ids (upstream `CalculateQueryBoundaries`).
    pub group: Option<Vec<i32>>,
}

const NAME_PREFIX: &str = "name:";
/// upstream `GetNumColFromLIBSVMFile`: at most 2^13 lines, stopping 2^7 lines after the
/// last line that raised the maximum index.
const LIBSVM_MAX_LINES: usize = 1 << 13;
const LIBSVM_STOP_ROUND: usize = 1 << 7;
/// upstream `ReadKLineFromFile` reads 32 lines for format detection.
const DETECT_LINES: usize = 32;

fn is_trim(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// upstream `GetNumColFromLIBSVMFile`: the width of a LibSVM file is its largest feature index
/// plus one, looking only at the scanned lines.
fn libsvm_num_col(lines: &[&str]) -> Result<usize> {
    let mut max_col: i64 = 0;
    let mut max_line = 0usize;
    for (i, line) in lines.iter().take(LIBSVM_MAX_LINES).enumerate() {
        let last = line
            .trim_matches(is_trim)
            .rsplit([' ', '\x0c', '\t', '\x0b'])
            .next()
            .unwrap_or("");
        let idx = last
            .split(':')
            .next()
            .unwrap_or("")
            .parse::<i64>()
            .unwrap_or(0);
        if idx > max_col {
            max_col = idx;
            max_line = i;
        }
        if i - max_line >= LIBSVM_STOP_ROUND {
            break;
        }
    }
    if max_col <= 0 {
        return Err(Error::Parse(
            "LibSVM file has no feature index greater than zero".into(),
        ));
    }
    Ok(max_col as usize + 1)
}

/// The label column (upstream `SetHeader`): `name:<column>` looks up the header, a number is an
/// index. Without a label name the default is column 0.
fn resolve_label(label_column: &str, names: &[String]) -> Result<i32> {
    if label_column.is_empty() {
        return Ok(0);
    }
    if let Some(name) = label_column.strip_prefix(NAME_PREFIX) {
        return names
            .iter()
            .position(|n| n == name)
            .map(|i| i as i32)
            .ok_or_else(|| {
                Error::Parse(format!(
                    "Could not find label column {name} in data file or data file doesn't contain header"
                ))
            });
    }
    label_column
        .parse::<i32>()
        .map_err(|_| Error::Parse("label_column is not a number, if you want to use a column name, please add the prefix \"name:\" to the column name".into()))
}

/// A weight, group or ignore column: `name:<column>` is looked up among the post-label feature
/// names, a number is a post-label feature index (upstream `SetHeader`).
fn resolve_column(
    column: &str,
    what: &str,
    name2idx: &HashMap<&str, usize>,
) -> Result<Option<usize>> {
    if column.is_empty() {
        return Ok(None);
    }
    if let Some(name) = column.strip_prefix(NAME_PREFIX) {
        return name2idx.get(name).copied().map(Some).ok_or_else(|| {
            Error::Parse(format!("Could not find {what} column {name} in data file"))
        });
    }
    let idx = column.parse::<i32>().map_err(|_| {
        Error::Parse(format!(
            "{what}_column is not a number, if you want to use a column name, please add the prefix \"name:\" to the column name"
        ))
    })?;
    Ok(usize::try_from(idx).ok())
}

/// Load `content` (the text of a data file) with the column options of `opts`.
pub fn load_text(content: &str, opts: &TextLoadOptions<'_>) -> Result<LoadedText> {
    let all: Vec<&str> = content.lines().collect();
    let (header, body) = if opts.header && !all.is_empty() {
        (Some(all[0]), &all[1..])
    } else {
        (None, &all[..])
    };
    let data: Vec<&str> = body
        .iter()
        .map(|l| l.trim_matches(is_trim))
        .filter(|l| !l.is_empty())
        .collect();
    let first = *data
        .first()
        .ok_or_else(|| Error::Parse("Data file should have at least one line.".into()))?;
    let ty = detect_data_type(&data[..data.len().min(DETECT_LINES)])?;

    let names: Vec<String> = header
        .map(|h| h.split(['\t', ',']).map(str::to_owned).collect())
        .unwrap_or_default();
    let label_idx = resolve_label(opts.label_column, &names)?;
    let label_idx = if opts.num_features > 0 {
        label_index(ty, first, opts.num_features as i32, label_idx)
    } else {
        label_idx
    };
    label_index_checked(ty, label_idx)?;
    let mut feature_names = names;
    if !feature_names.is_empty() && label_idx >= 0 {
        let li = label_idx as usize;
        if li >= feature_names.len() {
            return Err(Error::Parse(format!(
                "label column {li} is outside the {} header columns",
                feature_names.len()
            )));
        }
        feature_names.remove(li);
    }
    let name2idx: HashMap<&str, usize> = feature_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let num_total = match ty {
        DataType::LibSvm => libsvm_num_col(&data)?,
        _ => delimited_num_columns(ty, first),
    };
    let num_all = match ty {
        DataType::LibSvm => num_total,
        _ => num_total.saturating_sub(usize::from(label_idx >= 0)),
    };

    let weight_idx = resolve_column(opts.weight_column, "weight", &name2idx)?;
    let group_idx = resolve_column(opts.group_column, "group/query", &name2idx)?;
    // Weight and group columns are never features (upstream adds them to ignore_features_).
    let mut ignore: HashSet<usize> = weight_idx.iter().chain(group_idx.iter()).copied().collect();
    if let Some(names_to_ignore) = opts.ignore_column.strip_prefix(NAME_PREFIX) {
        for name in names_to_ignore.split(',') {
            let idx = name2idx.get(name).copied().ok_or_else(|| {
                Error::Parse(format!("Could not find ignore column {name} in data file"))
            })?;
            ignore.insert(idx);
        }
    } else if !opts.ignore_column.is_empty() {
        for token in opts.ignore_column.split(',') {
            let idx = token.parse::<i32>().map_err(|_| {
                Error::Parse("ignore_column is not a number, if you want to use a column name, please add the prefix \"name:\" to the column name".into())
            })?;
            if let Ok(idx) = usize::try_from(idx) {
                ignore.insert(idx);
            }
        }
    }
    let keep: Vec<usize> = (0..num_all).filter(|k| !ignore.contains(k)).collect();

    let mut features = Vec::with_capacity(data.len() * keep.len());
    let mut label = Vec::with_capacity(data.len());
    let mut weight = Vec::with_capacity(data.len());
    let mut group = Vec::with_capacity(data.len());
    for line in &data {
        let row = parse_line(ty, line, label_idx, opts.precise_float_parser)?;
        let mut dense = vec![0.0_f64; num_all];
        let mut weight_value = 0.0;
        let mut group_value = 0.0;
        for &(idx, val) in &row.features {
            let Ok(idx) = usize::try_from(idx) else {
                continue;
            };
            if idx >= num_all {
                continue;
            }
            dense[idx] = val;
            if Some(idx) == weight_idx {
                weight_value = val;
            }
            if Some(idx) == group_idx {
                group_value = val;
            }
        }
        for &k in &keep {
            features.push(dense[k]);
        }
        label.push(row.label);
        weight.push(weight_value as f32);
        group.push(group_value as i32);
    }

    let kept_names = if feature_names.is_empty() {
        Vec::new()
    } else {
        keep.iter()
            .filter_map(|&k| feature_names.get(k).cloned())
            .collect()
    };
    Ok(LoadedText {
        num_data: data.len(),
        num_features: keep.len(),
        feature_names: kept_names,
        features,
        label,
        weight: weight_idx.map(|_| weight),
        group: group_idx.map(|_| group_sizes(&group)),
    })
}

/// Run lengths of equal consecutive query ids.
fn group_sizes(ids: &[i32]) -> Vec<i32> {
    let mut sizes: Vec<i32> = Vec::new();
    for (i, id) in ids.iter().enumerate() {
        if i == 0 || ids[i - 1] != *id {
            sizes.push(0);
        }
        if let Some(last) = sizes.last_mut() {
            *last += 1;
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn opts() -> TextLoadOptions<'static> {
        TextLoadOptions {
            header: false,
            label_column: "",
            weight_column: "",
            group_column: "",
            ignore_column: "",
            precise_float_parser: false,
            num_features: 0,
        }
    }

    #[test]
    fn loads_csv_with_default_label_column() {
        let loaded = load_text("1,2,3\n0,4,5\n", &opts()).unwrap();
        assert_eq!(loaded.num_data, 2);
        assert_eq!(loaded.num_features, 2);
        assert_eq!(loaded.label, vec![1.0, 0.0]);
        assert_eq!(loaded.features, vec![2.0, 3.0, 4.0, 5.0]);
        assert!(loaded.weight.is_none() && loaded.group.is_none());
    }

    #[test]
    fn header_names_and_label_by_name() {
        let mut o = opts();
        o.header = true;
        o.label_column = "name:target";
        let loaded = load_text("a,target,b\n1,2,3\n", &o).unwrap();
        assert_eq!(loaded.feature_names, vec!["a", "b"]);
        assert_eq!(loaded.label, vec![2.0]);
        assert_eq!(loaded.features, vec![1.0, 3.0]);
    }

    #[test]
    fn unknown_label_name_is_an_error() {
        let mut o = opts();
        o.header = true;
        o.label_column = "name:missing";
        assert!(matches!(
            load_text("a,b\n1,2\n", &o),
            Err(crate::Error::Parse(_))
        ));
    }

    #[test]
    fn weight_column_index_counts_feature_columns_after_the_label() {
        // Post-label features are [0.5, 7, 3]; index 1 is the weight.
        let mut o = opts();
        o.weight_column = "1";
        let loaded = load_text("1,0.5,7,3\n0,2,8,4\n", &o).unwrap();
        assert_eq!(loaded.weight, Some(vec![7.0, 8.0]));
        assert_eq!(loaded.features, vec![0.5, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn group_column_by_name_becomes_group_sizes() {
        let mut o = opts();
        o.header = true;
        o.label_column = "name:y";
        o.group_column = "name:qid";
        let loaded = load_text("y,qid,x\n1,10,0.1\n0,10,0.2\n1,11,0.3\n", &o).unwrap();
        assert_eq!(loaded.group, Some(vec![2, 1]));
        // The query column is not a feature: it is ignored like upstream's ignore_features_.
        assert_eq!(loaded.feature_names, vec!["x"]);
        assert_eq!(loaded.features, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn ignored_columns_are_dropped_from_features() {
        let mut o = opts();
        o.ignore_column = "0";
        let loaded = load_text("1,5,6\n", &o).unwrap();
        assert_eq!(loaded.num_features, 1);
        assert_eq!(loaded.features, vec![6.0]);
    }

    #[test]
    fn libsvm_is_densified_with_the_largest_index_as_the_width() {
        let loaded = load_text("1 0:2 2:4\n0 1:3\n", &opts()).unwrap();
        assert_eq!(loaded.num_features, 3);
        assert_eq!(loaded.label, vec![1.0, 0.0]);
        assert_eq!(loaded.features, vec![2.0, 0.0, 4.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn a_zero_weight_is_kept_as_zero() {
        let mut o = opts();
        o.weight_column = "0";
        let loaded = load_text("1,0,9\n", &o).unwrap();
        assert_eq!(loaded.weight, Some(vec![0.0]));
        assert_eq!(loaded.features, vec![9.0]);
    }

    #[test]
    fn predicting_detects_the_label_column_from_the_model_feature_count() {
        // upstream: Predictor reads a file with num_features = max_feature_idx + 1; the first
        // line has exactly that many columns, so there is no label column.
        let mut o = opts();
        o.num_features = 2;
        let loaded = load_text("1,2\n3,4\n", &o).unwrap();
        assert_eq!(loaded.features, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded.label, vec![0.0, 0.0]);
        // One more column than the model uses: the first column is the label.
        let loaded = load_text("9,1,2\n", &o).unwrap();
        assert_eq!(loaded.features, vec![1.0, 2.0]);
        assert_eq!(loaded.label, vec![9.0]);
    }

    #[test]
    fn empty_input_is_an_error() {
        assert!(matches!(
            load_text("", &opts()),
            Err(crate::Error::Parse(_))
        ));
    }
}
