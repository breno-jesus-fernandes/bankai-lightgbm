//! Core of bankai-lightgbm: config, dataset, binning, tree learner, boosting,
//! objectives and metrics. Module layout mirrors upstream LightGBM's `src/`.

// Index-based loops are kept where they mirror upstream's C++ loops one-to-one: the
// side-by-side correspondence matters more than iterator style for a parity port.
#![allow(clippy::needless_range_loop)]

pub mod boosting;
pub mod config;
pub mod error;
pub mod io;
pub mod meta;
pub mod metric;
pub mod objective;
pub mod predictor;
pub mod tree;
pub mod treelearner;
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
