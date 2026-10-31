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

#[test]
fn feature_groups_match_upstream() {
    use lgbm_core::io::dataset::DatasetLayout;
    for case in dataset_cases() {
        let upstream = BinaryDataset::read_file(case.join("dataset.bin")).unwrap();
        let (params, cats) = dataset_params(&case);
        let cfg = Config::from_params(&params).unwrap();
        let (x, shape) = load_npy(&case.join("x.npy"));
        let ours = DatasetLayout::construct_dense(&x, shape[0], shape[1], &cfg, &cats).unwrap();
        let h = &upstream.header;
        let c = case.display();
        assert_eq!(
            ours.num_features as i32, h.num_features,
            "{c}: num_features"
        );
        assert_eq!(ours.groups.len() as i32, h.num_groups, "{c}: num_groups");
        assert_eq!(
            ours.used_feature_map, h.used_feature_map,
            "{c}: used_feature_map"
        );
        assert_eq!(
            ours.real_feature_idx, h.real_feature_idx,
            "{c}: real_feature_idx"
        );
        assert_eq!(ours.feature2group, h.feature2group, "{c}: feature2group");
        assert_eq!(
            ours.feature2subfeature, h.feature2subfeature,
            "{c}: feature2subfeature"
        );
        assert_eq!(
            ours.group_bin_boundaries, h.group_bin_boundaries,
            "{c}: group_bin_boundaries"
        );
        assert_eq!(
            ours.group_feature_start, h.group_feature_start,
            "{c}: group_feature_start"
        );
        assert_eq!(
            ours.group_feature_cnt, h.group_feature_cnt,
            "{c}: group_feature_cnt"
        );
        for (g, (og, ug)) in ours.groups.iter().zip(&upstream.groups).enumerate() {
            assert_eq!(
                og.is_multi_val, ug.is_multi_val,
                "{c}: group {g} is_multi_val"
            );
            assert_eq!(
                og.is_dense_multi_val, ug.is_dense_multi_val,
                "{c}: group {g} is_dense_multi_val"
            );
            assert_eq!(og.is_sparse, ug.is_sparse, "{c}: group {g} is_sparse");
            for (a, b) in og.bin_mappers.iter().zip(&ug.bin_mappers) {
                assert!(a.bitwise_eq(b), "{c}: group {g} mapper");
            }
        }
    }
}
