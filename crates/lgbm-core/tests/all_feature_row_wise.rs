//! Upstream CalcBinOffsets(false) / GetMultiBinFromAllFeatures, including bundled groups.
#![allow(clippy::unwrap_used, clippy::panic)]
use lgbm_core::io::bin::BinMapper;
use lgbm_core::io::bin_data::{BinData, DenseBin, SparseBin};
use lgbm_core::io::dataset::{FeatureGroupLayout, GroupData};
use lgbm_core::treelearner::multi_val::{AllFeatureBins, MultiValBin};

fn column(values: &[u32]) -> BinData {
    let mut d = DenseBin::new(values.len(), 8);
    for (r, &b) in values.iter().enumerate() {
        d.set(r, b);
    }
    BinData::Dense(d)
}

fn group(
    features: Vec<i32>,
    bins: &[(i32, u32)],
    offsets: Vec<u32>,
    multi: bool,
    sparse_rate: f64,
) -> FeatureGroupLayout {
    FeatureGroupLayout {
        features,
        bin_mappers: bins
            .iter()
            .map(|&(num_bin, most_freq_bin)| BinMapper {
                num_bin,
                most_freq_bin,
                sparse_rate,
                ..BinMapper::default()
            })
            .collect(),
        is_multi_val: multi,
        is_dense_multi_val: multi && sparse_rate < 0.25,
        is_sparse: false,
        num_total_bin: *offsets.last().unwrap(),
        bin_offsets: offsets,
    }
}

#[test]
fn all_feature_row_wise_offsets_and_rows_follow_upstream() {
    for sparse in [false, true] {
        let rate = if sparse { 0.8 } else { 0.1 };
        let groups = vec![
            group(vec![0, 1], &[(3, 0), (3, 0)], vec![1, 3, 6], false, rate),
            group(
                vec![2, 3],
                &[(3, 0), (4, 2)],
                if sparse { vec![1, 3, 7] } else { vec![0, 3, 7] },
                true,
                rate,
            ),
            group(vec![4], &[(3, 0)], vec![1, 3], false, rate),
        ];
        let data = vec![
            GroupData::Single(column(&[1, 0, 4, 5])),
            GroupData::Multi(vec![
                column(&[0, 2, 1, 0]),
                BinData::Sparse(SparseBin::from_sorted_pairs(
                    4,
                    5,
                    &[(1, 1), (2, 4), (3, 2)],
                )),
            ]),
            GroupData::Single(column(&[2, 1, 0, 2])),
        ];
        let all = AllFeatureBins::from_groups(&groups, &data, 4).unwrap();
        if sparse {
            assert_eq!(all.feature_hist_offsets, [1, 3, 6, 8, 12]);
            assert_eq!(all.bins.num_bin(), 14);
            match &all.bins {
                MultiValBin::Sparse { row_ptr, data, .. } => {
                    assert_eq!(row_ptr, &[0, 2, 5, 8, 11]);
                    assert_eq!(data, &[1, 13, 7, 8, 12, 4, 6, 11, 5, 9, 13]);
                }
                _ => panic!("expected sparse all-feature storage"),
            }
        } else {
            assert_eq!(all.feature_hist_offsets, [1, 3, 7, 9, 14]);
            assert_eq!(all.bins.num_bin(), 16);
            match &all.bins {
                MultiValBin::Dense {
                    num_feature,
                    offsets,
                    data,
                    ..
                } => {
                    assert_eq!(*num_feature, 4);
                    assert_eq!(offsets, &[0, 6, 9, 13]);
                    assert_eq!(data, &[1, 0, 2, 2, 0, 2, 0, 1, 4, 1, 3, 0, 5, 0, 1, 2]);
                }
                _ => panic!("expected dense all-feature storage"),
            }
        }
    }
}

#[test]
fn all_feature_row_wise_reserves_first_bin_for_nonzero_most_frequent() {
    let groups = vec![group(
        vec![0, 1],
        &[(4, 2), (3, 0)],
        vec![1, 5, 8],
        true,
        0.1,
    )];
    let data = vec![GroupData::Multi(vec![column(&[0, 1]), column(&[1, 0])])];
    let all = AllFeatureBins::from_groups(&groups, &data, 2).unwrap();
    assert_eq!(all.feature_hist_offsets, [1, 6]);
    assert_eq!(all.bins.num_bin(), 8);
    match all.bins {
        MultiValBin::Dense { offsets, data, .. } => {
            assert_eq!(offsets, [1, 5]);
            assert_eq!(data, [2, 1, 0, 0]);
        }
        _ => panic!("expected dense storage"),
    }
}

#[test]
fn all_feature_row_wise_selects_dense_and_sparse_storage_on_real_layouts() {
    use lgbm_core::config::Config;
    use lgbm_core::io::dataset::Dataset;
    use lgbm_core::utils::random::Random;
    let config = Config::from_str_params("max_bin=63 min_data_in_bin=1 num_threads=1").unwrap();
    for dense_features in [2, 9] {
        let mut rng = Random::new(61);
        let values: Vec<f64> = (0..4096 * 12)
            .map(|i| {
                let v = f64::from(rng.next_float()) * 2.0 - 1.0;
                if i % 12 >= dense_features && rng.next_float() < 0.8 {
                    0.0
                } else {
                    v
                }
            })
            .collect();
        let labels = vec![0.0; 4096];
        let ds = Dataset::from_dense(&values, 4096, 12, &labels, &config, &[]).unwrap();
        assert!(ds.layout.groups.iter().any(|g| g.is_multi_val));
        assert!(ds.layout.groups.iter().any(|g| !g.is_multi_val));
        let all = AllFeatureBins::from_groups(&ds.layout.groups, &ds.data, 4096).unwrap();
        assert_eq!(
            matches!(all.bins, MultiValBin::Dense { .. }),
            dense_features == 9
        );
    }
}

#[test]
fn row_wise_training_accepts_sparse_single_value_groups() {
    use lgbm_core::boosting::Gbdt;
    use lgbm_core::config::Config;
    use lgbm_core::io::dataset::Dataset;
    let config = Config::from_str_params("force_row_wise=true num_threads=1 num_iterations=5 min_data_in_leaf=2 min_data_in_bin=1 max_bin=15 deterministic=true verbosity=-1").unwrap();
    let values: Vec<f64> = (0..513)
        .map(|r| if r % 7 == 0 { (r % 17 + 1) as f64 } else { 0.0 })
        .collect();
    let labels: Vec<f64> = values
        .iter()
        .enumerate()
        .map(|(r, x)| x * 2.0 + (r % 3) as f64 * 0.1)
        .collect();
    let dense = Dataset::from_dense(&values, 513, 1, &labels, &config, &[]).unwrap();
    assert_eq!(dense.data.len(), 1);
    let GroupData::Single(BinData::Dense(col)) = &dense.data[0] else {
        panic!("expected single-value group");
    };
    let pairs: Vec<(usize, u32)> = (0..513)
        .filter_map(|r| {
            let b = col.get(r);
            (b != 0).then_some((r, b))
        })
        .collect();
    let mut sparse = dense.clone();
    sparse.layout.groups[0].is_sparse = true;
    sparse.data[0] = GroupData::Single(BinData::Sparse(SparseBin::from_sorted_pairs(
        513,
        dense.layout.groups[0].num_total_bin,
        &pairs,
    )));
    let want = Gbdt::train(&config, &dense)
        .unwrap()
        .save_model_to_string(0, -1, 0)
        .unwrap();
    let got = Gbdt::train(&config, &sparse)
        .unwrap()
        .save_model_to_string(0, -1, 0)
        .unwrap();
    assert_eq!(got, want);
}
