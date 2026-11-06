//! Side files next to a text data file: `<data>.weight`, `<data>.query` and `<data>.init`.
//! upstream: src/io/metadata.cpp (`LoadWeights`, `LoadQueryBoundaries`, `LoadInitialScore`).

use super::parser::parse_legacy;
use crate::error::{Error, Result};

/// The lines of a side file: `\n`-separated, `\r` removed, and no extra row for a trailing
/// newline. Blank lines are kept, as upstream's `TextReader::ReadAllLines` keeps them.
pub fn side_file_lines(content: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = content
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect();
    if content.is_empty() || content.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// upstream: `Common::AvoidInf(double)`: NaN -> 0, values beyond +-1e300 are clamped.
fn avoid_inf(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-1e300, 1e300)
    }
}

/// upstream: `Common::AvoidInf(float)`, applied to the `label_t` (float) weights.
fn avoid_inf_f32(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(-1e38, 1e38)
    }
}

/// The weights of `<data>.weight`, one per line (upstream `Metadata::LoadWeights`).
pub fn parse_weights(content: &str) -> Result<Vec<f32>> {
    side_file_lines(content)
        .into_iter()
        .map(|line| {
            let value = parse_legacy(line)?;
            Ok(avoid_inf_f32(value as f32))
        })
        .collect()
}

/// The group sizes of `<data>.query`, one per line (upstream `Metadata::LoadQueryBoundaries`).
pub fn parse_query(content: &str) -> Result<Vec<i32>> {
    side_file_lines(content)
        .into_iter()
        .map(|line| {
            line.trim_matches(' ')
                .parse::<i32>()
                .map_err(|_| Error::Parse(format!("Invalid query size in query file: {line}")))
        })
        .collect()
}

/// The initial scores of `<data>.init` (upstream `Metadata::LoadInitialScore`): the number of
/// tab-separated columns is the number of classes; values are stored class-major, so class `k`
/// of row `i` is at `k * num_rows + i`. An empty file gives `(0, [])`.
pub fn parse_init_score(content: &str) -> Result<(usize, Vec<f64>)> {
    let lines = side_file_lines(content);
    let Some(first) = lines.first() else {
        return Ok((0, Vec::new()));
    };
    let num_class = first.split('\t').count();
    let num_rows = lines.len();
    let mut values = vec![0.0_f64; num_rows * num_class];
    for (i, line) in lines.iter().enumerate() {
        let tokens: Vec<&str> = line.split('\t').collect();
        if tokens.len() != num_class {
            return Err(Error::Parse(
                "Invalid initial score file. Redundant or insufficient columns".into(),
            ));
        }
        for (k, token) in tokens.iter().enumerate() {
            values[k * num_rows + i] = avoid_inf(parse_legacy(token)?);
        }
    }
    Ok((num_class, values))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn weights_are_one_float_per_line_with_nan_as_zero_and_a_32_bit_clamp() {
        assert_eq!(parse_weights("0.5\n2\n").unwrap(), vec![0.5, 2.0]);
        assert_eq!(
            parse_weights("nan\n1e300\n-1e300\n").unwrap(),
            vec![0.0, 1e38, -1e38]
        );
    }

    #[test]
    fn query_file_lists_group_sizes() {
        assert_eq!(parse_query("2\n3\n1\n").unwrap(), vec![2, 3, 1]);
    }

    #[test]
    fn init_scores_with_one_column_are_one_value_per_row() {
        let (num_class, values) = parse_init_score("0.5\n-1.25\n").unwrap();
        assert_eq!(num_class, 1);
        assert_eq!(values, vec![0.5, -1.25]);
    }

    #[test]
    fn init_scores_with_several_columns_are_stored_class_major() {
        let (num_class, values) = parse_init_score("1\t2\n3\t4\n5\t6\n").unwrap();
        assert_eq!(num_class, 2);
        // class 0 for all rows, then class 1 for all rows
        assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn init_score_rows_must_have_the_same_number_of_columns() {
        assert!(matches!(
            parse_init_score("1\t2\n3\n"),
            Err(crate::Error::Parse(_))
        ));
    }

    #[test]
    fn a_trailing_newline_does_not_add_a_row() {
        assert_eq!(side_file_lines("1\n2\n"), vec!["1", "2"]);
        assert_eq!(side_file_lines("1\r\n\n2"), vec!["1", "", "2"]);
    }
}
