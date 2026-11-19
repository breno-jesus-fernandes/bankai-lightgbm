//! SPEC-CONC-6: dispatch and the scalar fallback preserve every accumulation bit.
#![allow(clippy::unwrap_used)]

use lgbm_core::io::bin_data::DenseBin;
use lgbm_core::treelearner::histogram::{CheckedBins, CheckedRows, construct, construct_scalar};

#[test]
fn simd_histogram_matches_scalar_and_ordered_oracle() {
    for num_bin in [16, 255, 4096, 70_000] {
        for n in [0, 1, 3, 17, 257] {
            let mut bins = DenseBin::new(n, num_bin);
            for row in 0..n {
                // Repeated destinations catch gather/scatter collision mistakes.
                bins.set(row, ((row / 7) % 3) as u32);
            }
            let checked = CheckedBins::new(&bins, n, num_bin as usize).unwrap();
            let rows: Vec<i32> = (0..n as i32).rev().step_by(2).collect();
            let checked_rows = CheckedRows::new(&rows, n).unwrap();
            for subset in [false, true] {
                let indices: Vec<usize> = if subset {
                    rows.iter().map(|&r| r as usize).collect()
                } else {
                    (0..n).collect()
                };
                // Cancellation and disparate magnitudes expose changed addition order.
                let pattern = [1e20_f32, 1.0, -1e20, -0.0, f32::MIN_POSITIVE, -3.25, 0.5];
                let g: Vec<f32> = indices.iter().map(|&r| pattern[r % 7]).collect();
                let h: Vec<f32> = indices.iter().map(|&r| (r % 11) as f32 * 0.125).collect();
                for constant in [false, true] {
                    // Offset the output by one double to exercise unaligned vector access.
                    let mut scalar = vec![-0.0; num_bin as usize * 2 + 2];
                    let mut simd = scalar.clone();
                    let mut oracle = scalar.clone();
                    for (i, &row) in indices.iter().enumerate() {
                        let slot = bins.get(row) as usize * 2 + 1;
                        oracle[slot] += f64::from(g[i]);
                        oracle[slot + 1] += if constant { 1.0 } else { f64::from(h[i]) };
                    }
                    let selected = subset.then_some(&checked_rows);
                    let hess = if constant { &[][..] } else { &h };
                    construct_scalar(&checked, selected, &g, hess, constant, &mut scalar[1..]);
                    construct(&checked, selected, &g, hess, constant, &mut simd[1..]);
                    for (label, actual) in [("scalar", scalar), ("dispatch", simd)] {
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            oracle.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            "{label}, bins={num_bin}, n={n}, subset={subset}, constant={constant}"
                        );
                    }
                }
            }
        }
    }
}
