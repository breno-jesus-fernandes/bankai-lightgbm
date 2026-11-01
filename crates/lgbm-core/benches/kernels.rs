//! Micro-benchmarks for the training hot paths (AGENTS.md section 5).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lgbm_core::boosting::Gbdt;
use lgbm_core::config::Config;
use lgbm_core::io::bin::BinType;
use lgbm_core::io::bin_data::DenseBin;
use lgbm_core::io::dataset::Dataset;
use lgbm_core::meta::MissingType;
use lgbm_core::predictor::{PredictConfig, Predictor};
use lgbm_core::treelearner::data_partition::{DataPartition, SplitRule};
use lgbm_core::treelearner::feature_histogram::{
    FeatureMeta, SplitParams, find_best_threshold_numerical,
};
use lgbm_core::treelearner::histogram::{CheckedBins, CheckedRows, construct};
use lgbm_core::treelearner::split_info::SplitInfo;
use lgbm_core::utils::random::Random;
use std::collections::HashMap;
use std::sync::Arc;

const N: usize = 200_000;

fn column(num_bin: u32) -> DenseBin {
    let mut bin = DenseBin::new(N, num_bin);
    let mut x = 12345u32;
    for i in 0..N {
        x = x.wrapping_mul(214_013).wrapping_add(2_531_011);
        bin.set(i, (x >> 8) % num_bin);
    }
    bin
}

fn histogram(c: &mut Criterion) {
    let g: Vec<f32> = (0..N).map(|i| (i % 17) as f32 - 8.0).collect();
    let h = vec![1.0f32; N];
    let mut group = c.benchmark_group("histogram");
    for num_bin in [16u32, 255, 4096] {
        let bin = column(num_bin);
        let checked = CheckedBins::new(&bin, N, num_bin as usize).unwrap();
        let mut out = vec![0.0f64; num_bin as usize * 2];
        group.bench_function(BenchmarkId::new("all_rows", num_bin), |b| {
            b.iter(|| {
                construct(&checked, None, &g, &h, false, black_box(&mut out));
            });
        });
        // A leaf holding a third of the rows, in partition (ascending) order.
        let rows: Vec<i32> = (0..N as i32).step_by(3).collect();
        let checked_rows = CheckedRows::new(&rows, N).unwrap();
        let og: Vec<f32> = rows.iter().map(|&i| g[i as usize]).collect();
        let oh: Vec<f32> = rows.iter().map(|&i| h[i as usize]).collect();
        group.bench_function(BenchmarkId::new("leaf_subset", num_bin), |b| {
            b.iter(|| {
                construct(
                    &checked,
                    Some(&checked_rows),
                    &og,
                    &oh,
                    false,
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn partition(c: &mut Criterion) {
    let config = Config::from_str_params("max_bin=255 min_data_in_bin=1").unwrap();
    let values: Vec<f64> = (0..N).map(|i| (i % 255) as f64).collect();
    let labels = vec![0.0; N];
    let dataset = Dataset::from_dense(&values, N, 1, &labels, &config, &[]).unwrap();
    let rule = SplitRule::numerical(&dataset, 0, 126, true);
    let mut part = DataPartition::new(N, 2);
    c.bench_function("partition/numerical/200k", |b| {
        b.iter(|| {
            part.init();
            part.split(0, &rule, 1, None);
            black_box((part.leaf_count(0), part.leaf_count(1)));
        });
    });
}

fn split_finding(c: &mut Criterion) {
    let config = Config::from_str_params("num_leaves=31 min_data_in_leaf=20").unwrap();
    let params = SplitParams::from_config(&config);
    let mut meta = FeatureMeta {
        num_bin: 255,
        missing_type: MissingType::None,
        offset: 0,
        default_bin: 0,
        monotone_type: 0,
        penalty: 1.0,
        bin_type: BinType::Numerical,
        rand: Random::new(1),
    };
    let hist: Vec<f64> = (0..255)
        .flat_map(|bin| [f64::from(bin) - 127.0, 800.0])
        .collect();
    let mut output = SplitInfo::default();
    c.bench_function("split_finding/numerical/255_bins", |b| {
        b.iter(|| {
            find_best_threshold_numerical(
                &hist,
                &mut meta,
                &params,
                0.0,
                204_000.0,
                200_000,
                0.0,
                None,
                &mut output,
            );
            black_box(output.gain);
        });
    });
}

fn prediction(c: &mut Criterion) {
    const ROWS: usize = 4096;
    const FEATURES: usize = 16;
    let mut params = HashMap::from([
        ("objective".to_owned(), "regression".to_owned()),
        ("num_iterations".to_owned(), "20".to_owned()),
        ("num_leaves".to_owned(), "15".to_owned()),
        ("min_data_in_leaf".to_owned(), "5".to_owned()),
        ("min_data_in_bin".to_owned(), "1".to_owned()),
        ("max_bin".to_owned(), "63".to_owned()),
    ]);
    params.insert("num_threads".to_owned(), "1".to_owned());
    let config = Arc::new(Config::from_params(&params).unwrap());
    let data: Vec<f64> = (0..ROWS * FEATURES)
        .map(|i| ((i * 17 + i / FEATURES) % 101) as f64)
        .collect();
    let labels: Vec<f64> = (0..ROWS).map(|i| (i % 101) as f64).collect();
    let dataset =
        Arc::new(Dataset::from_dense(&data, ROWS, FEATURES, &labels, &config, &[]).unwrap());
    let model = Gbdt::train_shared(config, dataset).unwrap();
    let predictor = Predictor::new(&model, &PredictConfig::default()).unwrap();
    c.bench_function("prediction/dense/4096x16x20", |b| {
        b.iter(|| {
            black_box(
                predictor
                    .predict_dense(black_box(&data), ROWS, FEATURES)
                    .unwrap(),
            )
        });
    });
}

criterion_group!(benches, histogram, partition, split_finding, prediction);
criterion_main!(benches);
