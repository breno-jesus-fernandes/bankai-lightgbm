//! A model file with CRLF line endings (a Windows checkout or editor) loads exactly like the
//! LF original. Upstream reads model files in text mode, which normalizes the line endings;
//! `tree_sizes` are byte offsets, so a CRLF file must not shift them.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use lgbm_core::boosting::Gbdt;
use support::*;

#[test]
fn crlf_model_file_loads_like_the_lf_file() {
    let dir = std::env::temp_dir().join(format!("bankai-crlf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for case in cases() {
        let lf = read_model_text(case.join("model.txt"));
        let name = case.file_name().unwrap().to_string_lossy().into_owned();
        let crlf_path = dir.join(format!("{name}.txt"));
        std::fs::write(&crlf_path, lf.replace('\n', "\r\n")).unwrap();

        let expected = Gbdt::from_model_string(&lf)
            .unwrap()
            .save_model_to_string(0, -1, 0)
            .unwrap();
        let loaded =
            Gbdt::from_model_file(&crlf_path).unwrap_or_else(|e| panic!("{}: {e}", case.display()));
        let text = loaded.save_model_to_string(0, -1, 0).unwrap();
        assert_eq!(text, expected, "{}", case.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
