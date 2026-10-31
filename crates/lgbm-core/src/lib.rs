//! Core of bankai-lightgbm: config, dataset, binning, tree learner, boosting,
//! objectives and metrics. Module layout mirrors upstream LightGBM's `src/`.

pub mod boosting;
pub mod config;
pub mod error;
pub mod meta;
pub mod objective;
pub mod predictor;
pub mod tree;
pub mod utils;

pub use error::{Error, Result};

/// Upstream LightGBM release this crate targets for parity.
pub const UPSTREAM_VERSION: &str = "v4.7.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_version_matches_repo_pin() {
        let pinned = include_str!("../../../UPSTREAM_VERSION").trim();
        assert_eq!(UPSTREAM_VERSION, pinned);
    }
}
