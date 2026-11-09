# PROGRESS: dependency graph of remaining work

Nodes are topologically sorted: a node may start only when every dependency is `[x]`. Each node
lists its status, dependencies, the command that verifies it, and its benchmark target. Node
status is the only place progress is recorded. `ROADMAP.md` is the user-facing narrative and
is updated in the same PR.

Status legend:
- `[ ] Not Started`
- `[~] In Progress`
- `[x] Parity Verified` (all verify commands pass on the branch)
- `[!] Blocked` (three consecutive red attempts; reason and DECISIONS.md entry required)

Verification command shorthand (defined in AGENTS.md section 13):
- `GATES`: `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `maturin develop -r`, `pytest tests/parity tests/python_package_test`.
- `STAGE`: the stage golden diff (Phase C of the harness, see the plan). Until it exists, `STAGE` is
  `n/a` and the node's verify line says so.

## Tier 0: done and locked

Changing these nodes requires a DECISIONS.md entry and a parity test that fails first.

- [x] T0.1 Config, aliases, validation, `auto.rs` generation. Verify: `python scripts/gen_config.py --check`, `cargo test -p lgbm-core config`. Bench: n/a.
- [x] T0.2 `utils::random`, `utils::common`, `utils::array_args`. Verify: `cargo test -p lgbm-core utils`. Bench: n/a.
- [x] T0.3 Bin mappers, missing handling, dense/4-bit/sparse bins. Verify: `cargo test -p lgbm-core --test golden_binning`. Bench: n/a.
- [x] T0.4 Model text and JSON round-trip. Verify: `cargo test -p lgbm-core --test golden_model_text --test golden_dump_model`. Bench: n/a.
- [x] T0.5 Tree and predictor (normal, raw, leaf, contrib, early stop). Verify: `cargo test -p lgbm-core --test golden_predict`, `pytest tests/parity -k predict`. Bench: `benches/bench_predict.py`.
- [x] T0.6 Objectives and metrics. Verify: `cargo test -p lgbm-core --test golden_metrics`, `pytest tests/parity -k objective`. Bench: n/a.

## Tier 1: foundations for the remaining work

- [x] T1.1 (PR #86) Tie-break audit: read upstream `split_info.hpp` and `feature_histogram.hpp` (verify), record the exact gain comparison and tie rule in SPEC.md (SPEC-NUM-5, SPEC-NUM-7), add a parity test on constructed ties. Depends: T0.3. Verify: `pytest tests/parity -k tie`. Bench: n/a.
- [x] T1.2 (PR #87; a pin, not an implementation, see DECISIONS D-004) `deterministic=true` implemented per SPEC-NUM-2, with a 1-thread structure parity test. Depends: T1.1. Verify: `pytest tests/parity -k deterministic`, GATES. Bench: n/a.
- [x] T1.3 (PR #88; the 40 PyResult entry points are guarded, infallible getters are not) `catch_unwind` at the PyO3 boundary (SPEC-ERR-4): panics surface as `LightGBMError`, with a regression test that forces a panic. Depends: T0.5. Verify: `pytest tests/parity -k panic`, GATES. Bench: n/a.
- [x] T1.4 (PR #90 proptest, PR #91 fuzz targets; JSON model parsing has no entry point yet) proptest and fuzz scaffolding (AGENTS.md section 4): binning invariants, serialization round-trips, and fuzz targets for model text, JSON, CSV and LibSVM. Depends: T0.4. Verify: `cargo test -p lgbm-core --test proptest_binning`, `cargo +nightly fuzz build`. Bench: n/a.
- [x] T1.5 (PR #89) Miri coverage for `utils::spin_pool` (the current unsafe module without it). Depends: none. Verify: `scripts/miri.sh spin_pool`. Bench: n/a.

## Tier 2: histograms and the training hot path

- [x] T2.1 (PR #84, row-major copy in PR #92; column-wise stays the auto default, see DECISIONS D-006) Row-wise histogram layout for `force_row_wise` (dense single-value groups). Depends: T0.3. Verify: `pytest tests/parity/test_row_wise_training_parity.py`, GATES. Bench: `cargo bench -p lgbm-core` (kernels) plus `benches/bench_train_interleaved.py` for the forced row-wise path. Status: merged in PR #84 (kernel gate passed; the forced row-wise training benchmark is still outstanding). Auto-choice and the multi-value row-wise path are not done.
- [ ] T2.2 Row-wise layout auto-choice (upstream `force_col_wise`/`force_row_wise` selection rules). Depends: T2.1. Verify: `pytest tests/parity -k auto_layout`, GATES. Bench: `benches/bench_train_interleaved.py` (dense, both thread counts).
- [ ] T2.3 Row-wise for multi-value groups (upstream `GetMultiBinFromAllFeatures`, all groups). Depends: T2.1. Verify: `pytest tests/parity/test_multi_val_training_parity.py -k row_wise`, GATES. Bench: `benches/bench_train_interleaved.py` (sparse).
- [ ] T2.4 `spin_pool -> rayon` migration (SPEC-CONC-1, SPEC-CONC-5). Depends: T1.5, T2.1. Verify: `cargo test -p lgbm-core --test thread_invariance` for 1, 2 and 3 threads, GATES. Bench: interleaved kernels and `benches/bench_train_interleaved.py` at 2 threads; no slowdown above 3% against the spin_pool baseline.
- [ ] T2.5 Histogram pool and subtraction trick (`histogram_pool_size`). Depends: T2.1. Verify: `pytest tests/parity -k histogram_pool`, GATES. Bench: `benches/bench_train_interleaved.py`.
- [ ] T2.6 SIMD histogram kernels with scalar fallback, bit-exact to the scalar path (SPEC-CONC-6). Depends: T2.1. Verify: `cargo test -p lgbm-core simd`, miri for the histogram module. Bench: criterion kernels, no regression.
- [ ] T2.7 Quantized training (`use_quantized_grad`, `num_grad_quant_bins`). Depends: T2.1, T1.2. Verify: `pytest tests/parity -k quantized`, GATES. Bench: `benches/bench_train_interleaved.py`.

## Tier 3: tree learner gaps

- [ ] T3.1 Sparse data partition and thread scaling (current gap: sparse 1.14–1.41x upstream time). Depends: T2.4. Verify: GATES, `pytest tests/parity -k partition`. Bench: `benches/bench_train_interleaved.py` sparse, target ≤1.0x upstream.
- [ ] T3.2 Split search: sequential threshold phase scaling. Depends: T2.4. Verify: GATES, thread_invariance. Bench: criterion split_finding, `benches/bench_train_interleaved.py` dense 2 threads.
- [ ] T3.3 Forced splits for categorical features (currently `NotImplementedError`). Depends: T0.3. Verify: `pytest tests/parity -k forced_split`, GATES. Bench: n/a.
- [ ] T3.4 CEGB (`cegb_*`). Depends: T2.1. Verify: `pytest tests/parity -k cegb`, GATES. Bench: n/a.

## Tier 4: boosting gaps

- [ ] T4.1 Unbiased lambdarank for `rank_xendcg` with positions. Depends: T0.6. Verify: `pytest tests/parity -k position`, GATES. Bench: n/a.
- [ ] T4.2 `bagging_by_query` for `rank_xendcg`. Depends: T4.1. Verify: `pytest tests/parity -k bagging_by_query`, GATES. Bench: n/a.
- [ ] T4.3 Model merge (upstream `Booster.merge` semantics). Depends: T0.4. Verify: `pytest tests/parity -k merge`, GATES. Bench: n/a.
- [ ] T4.4 Monotone `advanced` with categorical features (currently refused; needs the upstream undefined-behaviour decision in DECISIONS.md first). Depends: T3.2. Verify: `pytest tests/parity -k monotone`, GATES. Bench: n/a.

## Tier 5: bindings and surface

- [ ] T5.1 Native layer: remove the remaining float64 copy for row-major dense and CSR input (SPEC-PY-1). Depends: T1.3. Verify: `pytest tests/parity -k zero_copy`, GATES. Bench: `benches/bench_dataset.py`, no slowdown.
- [ ] T5.2 Dataset surface: pyarrow input, `forced_bin_bounds` in the text dump. Depends: T0.3. Verify: `pytest tests/python_package_test -k dataset`, GATES. Bench: n/a.
- [ ] T5.3 `lgbm-capi`: `LGBM_*` symbols compatible with upstream `c_api.h` (SPEC-ERR-5). Depends: T1.3. Verify: a C test harness built against the upstream header, GATES. Bench: n/a.
- [ ] T5.4 `lgbm-cli`: `task=train|predict|convert_model|refit|save_binary`. Depends: T5.3. Verify: parity against the upstream CLI on the example configs, GATES. Bench: n/a.
- [ ] T5.5 `convert_model` (C++ codegen output). Depends: T5.4. Verify: `pytest tests/parity -k convert_model`, GATES. Bench: n/a.

## Tier 6: post-1.0

- [ ] T6.1 Network layer (socket + MPI), `tree_learner=data|feature|voting`. Depends: T2.4. Verify: `pytest tests/parity -k distributed`. Bench: n/a.
- [ ] T6.2 Dask integration. Depends: T6.1. Verify: `pytest tests/python_package_test -k dask`. Bench: n/a.
- [ ] T6.3 CUDA (`device_type=cuda`) behind the `cuda` feature. Depends: T2.4. Verify: `cargo test -p lgbm-core --features cuda`, GATES on the CUDA runner. Bench: n/a.
- [ ] T6.4 OpenCL / ROCm. Depends: T6.3. Verify: GATES on the device runner. Bench: n/a.
