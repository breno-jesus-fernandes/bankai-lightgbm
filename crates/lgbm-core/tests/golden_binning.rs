//! P2 gate: bin mappers computed by bankai equal those in upstream's `save_binary` output.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use lgbm_core::config::Config;
use lgbm_core::io::binary::BinaryDataset;
use lgbm_core::io::sample::construct_bin_mappers_dense;
use support::*;

#[test]
fn upstream_binary_files_parse() {
    for case in dataset_cases() {
        let ds = BinaryDataset::read_file(case.join("dataset.bin"))
            .unwrap_or_else(|e| panic!("{}: {e}", case.display()));
        let (y, _) = load_npy(&case.join("y.npy"));
        assert_eq!(ds.header.num_data as usize, y.len(), "{}", case.display());
        // labels are stored as float32
        for (a, b) in ds.metadata.label.iter().zip(&y) {
            assert_eq!(*a, *b as f32);
        }
        assert_eq!(
            ds.header.num_features as usize,
            ds.groups.iter().map(|g| g.bin_mappers.len()).sum::<usize>()
        );
    }
}

#[test]
fn bin_mappers_match_upstream() {
    for case in dataset_cases() {
        let upstream = BinaryDataset::read_file(case.join("dataset.bin")).unwrap();
        let (params, cats) = dataset_params(&case);
        let cfg = Config::from_params(&params).unwrap();
        let (x, shape) = load_npy(&case.join("x.npy"));
        let ours = construct_bin_mappers_dense(&x, shape[0], shape[1], &cfg, &cats).unwrap();

        let h = &upstream.header;
        assert_eq!(ours.len(), h.num_total_features as usize);
        for (fidx, mapper) in ours.iter().enumerate() {
            let used = h.used_feature_map[fidx];
            let ours_used = mapper.as_ref().is_some_and(|m| !m.is_trivial);
            assert_eq!(
                ours_used,
                used >= 0,
                "{} feature {fidx}: used mismatch",
                case.display()
            );
            if used < 0 {
                continue;
            }
            let inner = used as usize;
            let g = h.feature2group[inner] as usize;
            let sub = h.feature2subfeature[inner] as usize;
            let want = &upstream.groups[g].bin_mappers[sub];
            let got = mapper.as_ref().unwrap();
            assert!(
                got.bitwise_eq(want),
                "{} feature {fidx}:\n ours {got:?}\n want {want:?}",
                case.display()
            );
        }
    }
}
