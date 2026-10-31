//! `lightgbm` command-line tool, config-file compatible with upstream.

fn main() -> std::process::ExitCode {
    eprintln!(
        "bankai-lightgbm CLI (targets LightGBM {}): not implemented yet",
        lgbm_core::UPSTREAM_VERSION
    );
    std::process::ExitCode::FAILURE
}
