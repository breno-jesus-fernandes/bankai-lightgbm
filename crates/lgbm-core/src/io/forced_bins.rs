//! Forced upper bin bounds from `forcedbins_filename`.
//! upstream: src/io/dataset_loader.cpp `DatasetLoader::GetForcedBins` (v4.7.0)

use crate::error::{Error, Result};

use std::collections::HashSet;

/// Per-feature forced upper bin bounds from the JSON file at `path` (`forcedbins_filename`).
/// The file is a JSON array of `{"feature": i, "bin_upper_bound": [..]}`. Categorical features
/// are skipped with a warning, and consecutive duplicate bounds are removed (as `std::unique`
/// does upstream). An empty or unreadable path gives no bounds, with a warning when unreadable.
pub fn get_forced_bins(
    path: &str,
    num_total_features: usize,
    categorical: &HashSet<i32>,
) -> Result<Vec<Vec<f64>>> {
    let mut bounds: Vec<Vec<f64>> = vec![Vec::new(); num_total_features];
    if path.is_empty() {
        return Ok(bounds);
    }
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(_) => {
            log::warn!("Could not open {path}. Will ignore.");
            return Ok(bounds);
        }
    };
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| Error::Parse(format!("forced bins file {path}: {e}")))?;
    let items = json
        .as_array()
        .ok_or_else(|| Error::Parse("forced bins file must be a JSON array".into()))?;
    for item in items {
        let feature = item.get("feature").and_then(|v| v.as_i64()).unwrap_or(0);
        let feature = usize::try_from(feature)
            .ok()
            .filter(|&f| f < num_total_features)
            .ok_or_else(|| {
                Error::InvalidParameter(format!(
                    "forced bins feature {feature} is outside the {num_total_features} features"
                ))
            })?;
        if categorical.contains(&(feature as i32)) {
            log::warn!(
                "Feature {feature} is categorical. Will ignore forced bins for this feature."
            );
            continue;
        }
        if let Some(values) = item.get("bin_upper_bound").and_then(|v| v.as_array()) {
            bounds[feature].extend(values.iter().map(|v| v.as_f64().unwrap_or(0.0)));
        }
    }
    for feature_bounds in &mut bounds {
        feature_bounds.dedup();
    }
    Ok(bounds)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::collections::HashSet;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("bankai_forced_{}_{name}", std::process::id()));
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn parses_bounds_per_feature_and_removes_adjacent_duplicates() -> Result<()> {
        let path = write_temp(
            "a.json",
            r#"[{"feature": 0, "bin_upper_bound": [0.3, 0.35, 0.35, 0.4]},
                {"feature": 2, "bin_upper_bound": [-0.1]}]"#,
        );
        let bounds = get_forced_bins(path.to_str().unwrap(), 3, &HashSet::new())?;
        assert_eq!(bounds, vec![vec![0.3, 0.35, 0.4], vec![], vec![-0.1]]);
        Ok(())
    }

    #[test]
    fn categorical_features_are_ignored() -> Result<()> {
        let path = write_temp("b.json", r#"[{"feature": 1, "bin_upper_bound": [0.5]}]"#);
        let categorical: HashSet<i32> = [1].into_iter().collect();
        let bounds = get_forced_bins(path.to_str().unwrap(), 2, &categorical)?;
        assert_eq!(bounds, vec![Vec::<f64>::new(), Vec::new()]);
        Ok(())
    }

    #[test]
    fn empty_path_and_missing_file_give_no_bounds() -> Result<()> {
        let none = get_forced_bins("", 2, &HashSet::new())?;
        assert_eq!(none, vec![Vec::<f64>::new(), Vec::new()]);
        let missing = get_forced_bins("/nonexistent/forced.json", 2, &HashSet::new())?;
        assert_eq!(missing, vec![Vec::<f64>::new(), Vec::new()]);
        Ok(())
    }

    #[test]
    fn feature_index_outside_the_data_is_an_error() {
        let path = write_temp("c.json", r#"[{"feature": 5, "bin_upper_bound": [0.5]}]"#);
        assert!(get_forced_bins(path.to_str().unwrap(), 2, &HashSet::new()).is_err());
    }
}
