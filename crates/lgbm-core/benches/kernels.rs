//! Micro-benchmarks for the training hot paths (AGENTS.md section 5).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use lgbm_core::io::bin_data::DenseBin;
use lgbm_core::treelearner::histogram::{CheckedBins, CheckedRows, construct};

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

criterion_group!(benches, histogram);
criterion_main!(benches);
