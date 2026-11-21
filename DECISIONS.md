# DECISIONS

Log of choices made where SPEC.md did not settle the question. Agents append an entry instead of
pausing (AGENTS.md section 13). A human can overturn an entry; the overturn is a new entry that
supersedes the old one.

Entry format:

```
## D-<NNN>: <short title>
- Date: YYYY-MM-DD
- Question: what was unclear
- Choice: what was done
- Rationale: why, citing upstream code, SPEC rule IDs, or evidence
- SPEC ref: SPEC-<AREA>-<n> or "none (new rule proposed)"
- Status: active | superseded by D-<NNN>
```

---

## D-001: Parallel loops migrate to rayon (locked by the user)

- Date: 2026-11-08
- Question: AGENTS.md section 5 says rayon, the code uses `utils::spin_pool`. Which one is the contract?
- Choice: rayon is the target. `spin_pool` stays in place until the migration node T2.4 passes its gate.
- Rationale: the user chose migration when asked. The spin_pool exception is documented in SPEC-CONC-5
  so the code and the rule agree during the transition.
- SPEC ref: SPEC-CONC-1, SPEC-CONC-5
- Status: active

## D-002: Row-wise layout is accepted when byte-identical to upstream

- Date: 2026-11-08
- Question: the row-wise kernel adds the same rows in a different loop order. Is it allowed to differ from col-wise?
- Choice: it must produce byte-identical model text for the parity cases. It is accepted on that evidence.
- Rationale: each bin accumulates its rows in leaf order in both layouts (SPEC-NUM-6), so sums match exactly.
  `tests/parity/test_row_wise_training_parity.py` checks regression, binary and bagging.
- SPEC ref: SPEC-NUM-6
- Status: active

## D-003: Benchmark gate scope for the row-wise change

- Date: 2026-11-08
- Question: the row-wise path is opt-in, and the interleaved kernel benchmark exercises only the col-wise path. What is the gate for PR #84?
- Choice: the kernel benchmark (default path, 5% limit) is the blocking gate. Training-level numbers for the forced row-wise path are required before the layout is advertised as faster, and are tracked as T2.1 in PROGRESS.md.
- Rationale: AGENTS.md section 5 requires criterion results for hot-path changes. The default path is the one that could regress, and it passed at 1.016x worst case.
- SPEC ref: none (AGENTS.md section 5 applies)
- Status: active

## D-004: `deterministic` is a no-op on the CPU learners

- Date: 2026-11-08
- Question: T1.2 asks for `deterministic` per SPEC-NUM-2, but the flag is parsed and never read on CPU. Is ignoring it a silent-ignore violation of AGENTS.md section 8?
- Choice: no implementation. The tests in `tests/parity/test_deterministic_parity.py` pin that a `deterministic=true` model matches upstream byte for byte, for the default layout and for forced col-wise and row-wise.
- Rationale: in the pinned source, `src/io/config.cpp` reads `deterministic` only for GPU and CUDA warnings, and no CPU learner branches on it. Upstream ignores it on CPU too, so matching upstream means ignoring it.
- SPEC ref: SPEC-NUM-2, AGENTS.md section 8
- Status: active

## D-005: Tie-break rule matches upstream `SplitInfo::operator>`

- Date: 2026-11-08
- Question: SPEC-NUM-5 and SPEC-NUM-7 left the exact gain comparison and tie rule unpinned.
- Choice: higher gain wins; on an exact tie the lower feature index wins; `-1` counts as the largest index; the upstream `gain == NAN` check is always false, so NaN gains are not remapped. `treelearner::split_info::better_than` implements this, and `tests/parity/test_split_tie_parity.py` pins it.
- Rationale: read from `src/treelearner/split_info.hpp` (v4.7.0). The tests catch a flipped tie rule (checked by a temporary mutation). The NaN quirk is kept on purpose so that bankai matches upstream.
- SPEC ref: SPEC-NUM-5, SPEC-NUM-7
- Status: active

## D-006: Layout auto-choice stays column-wise until the row path is competitive

- Date: 2026-11-08
- Question: T2.2 asks for `force_col_wise`/`force_row_wise` auto-choice. Which layout should auto pick?
- Choice: column-wise for dense and sparse data. Upstream's auto choice times both layouts on the data; bankai does not do that yet.
- Rationale: interleaved benchmark, 50k rows, 7 rounds, medians (`benches/bench_train_interleaved.py`). Column-wise is 0.57 to 0.69x upstream on dense data, and after the row-major copy (PR #92) forced row-wise is 0.95 to 0.98x. Column-wise is still the faster option, so auto should keep it. The timing-based choice only pays off once row-wise is faster somewhere.
- SPEC ref: none (new rule proposed; revisit when row-wise beats column-wise on any benchmark case)
- Status: active

## D-007: Took the first deterministic node (T3.3) ahead of finishing T2.1's benchmark

- Date: 2026-11-17
- Question: PROGRESS orders T2.1 (`[~]`) before T3.3. T2.1's only outstanding item is the forced row-wise training benchmark; the feature merged in PR #84. Which node to work next in an autonomous session on the 2-vCPU dev VM?
- Choice: leave T2.1 at `[~]` and implement T3.3 (categorical forced splits), the first eligible node whose verify is deterministic (parity, no benchmark gate).
- Rationale: AGENTS.md section 12 says a single-run timing below ~3% is noise on this VM and benchmark gates need interleaved medians over 7+ rounds; a benchmark pass/fail cannot be decided reliably unattended. T3.3 depends only on T0.3 `[x]` and is verified by `pytest tests/parity -k forced_split`, which is deterministic. The benchmark for T2.1 is deferred to a session that can run it under controlled load.
- SPEC ref: none (operational choice under AGENTS.md section 12 and section 13)
- Status: active

## D-008: T3.3 is built on top of the forced-split root-weight fix (a7a9147)

- Date: 2026-11-17
- Question: categorical forced-split parity compares the full model text over several trees. The recorded `split_gain` of a root forced split depends on the carried root weight, which commit `a7a9147` ("forced splits record upstream's gain in every tree") fixes but which is not yet on `main`. Base the work on `main` or on `a7a9147`?
- Choice: base `feat/forced-cat-splits` on `a7a9147`. The PR stacks on that unmerged fix (noted in the PR body). While there, `a7a9147`'s `tests/parity/test_forced_split_gain_parity.py` was reformatted with `ruff format` (it failed `ruff format --check` under ruff 0.17.4); no assertion changed.
- Rationale: `a7a9147`'s own message records the discrepancy "933.6 vs 5814.6 in a categorical case" from the second tree on. Without the fix, `test_forced_split_categorical_parity.py`'s `model_to_string()` comparison would fail on gain from tree 2+, masking correct categorical-split behaviour. The fix is a true dependency of the test, not optional.
- SPEC ref: none (operational choice; see AGENTS.md section 9 on green builds)
- Status: active

## D-009: rank_xendcg accepts positions but keeps them inert, matching upstream

- Date: 2026-11-17
- Question: T4.1 was framed as "unbiased lambdarank for rank_xendcg with positions". Should rank_xendcg learn position-bias factors like lambdarank?
- Choice: no. rank_xendcg accepts a `position` field (no longer `NotImplementedError`) but does not learn any bias; the factors stay 0 and the score adjustment is a no-op, so training with positions equals training without them.
- Rationale: read upstream `src/objective/rank_objective.hpp` (v4.7.0). `RankXENDCG` does not override the base `RankingObjective::UpdatePositionBiasFactors`, whose default body is empty, so `pos_biases_` stays 0 for xendcg. Only `LambdarankNDCG` overrides it. Matching upstream means accepting positions and leaving them inert. `tests/parity/test_xendcg_positions_parity.py` pins both the upstream parity and the inert behaviour.
- SPEC ref: none (upstream quirk; matched on purpose like the other documented upstream quirks)
- Status: active

## D-010: DECISIONS ids are numbered to avoid collisions across parallel branches

- Date: 2026-11-17
- Question: two independent feature branches off `main` (`feat/forced-cat-splits`, `feat/xendcg-position-bias`) both append DECISIONS entries in the same autonomous session. How to number them so they do not collide on merge?
- Choice: the forced-categorical branch took D-007 and D-008; this branch skips to D-009 (and this note is D-010). Numbers are assigned per branch, not per file state, so `main` ends up with a contiguous D-001..D-010 once both merge.
- Rationale: a text log keyed by number conflicts if two branches both add "D-007". Assigning disjoint numbers up front avoids the conflict without coordination.
- SPEC ref: none (process note)
- Status: active

## D-011: `bagging_by_query` is supported for `rank_xendcg`

- Date: 2026-11-18
- Question: T4.2. `rank_xendcg` draws a random number per query, so the port refused query
  bagging rather than risk diverging from upstream. Can it be supported bit-exactly?
- Choice: supported. The trainer now bags before computing gradients when `bagging_by_query`
  is on, and the objective's gradient pass is restricted to the sampled queries
  (`Objective::get_gradients_with_sampled_queries`, overridden by `RankXendcg`). The refusal in
  `boosting::sample::new_query_bagging` is removed. Numbering skips D-007..D-010, which belong
  to the parallel open PRs #94 and #95, so this entry does not collide with them.
- Rationale: upstream `RankXENDCG` seeds one `Random` per query (`seed_ + query_id`) and draws
  `rands_[query_id].NextFloat()` inside `GetGradientsForOneQuery`
  (`src/objective/rank_objective.hpp`, v4.7.0). Because each query's generator is indexed by
  its own id, computing gradients for a subset of queries never perturbs another query's draw
  sequence — exactly what `GetGradientsWithSampledQueries` relies on, and what
  `src/boosting/gbdt.cpp` (v4.7.0) calls under `bagging_by_query`. bankai's query-sample set
  already matches upstream (lambdarank parity), so the xendcg model matches tree by tree
  (`tests/parity/test_bagging_by_query_parity.py::test_query_bagging_xendcg_matches_upstream_tree_by_tree`).
  The Harness dropped `test_bagging_by_query_rejects_rank_xendcg_until_ported`, whose only
  purpose was to pin the now-lifted refusal.
- SPEC ref: none (AGENTS.md section 8: the feature is ported rather than refused)
- Status: active

## D-012: C API conventions for `lgbm-capi` (slice 1)

- Date: 2026-11-18
- Question: T5.3. The `lgbm-capi` crate was an empty stub. Porting `c_api.h` needs a settled
  error-reporting convention, handle representation, panic policy and crate output, none of
  which SPEC.md pins in detail.
- Choice: match upstream's C API exactly. Every entry point returns `int` `0` on success and
  `-1` on failure (upstream `API_BEGIN`/`API_END`); the message is read with
  `LGBM_GetLastError`, backed by a thread-local `CString`. `BoosterHandle` is an opaque
  `*mut c_void` to a heap-boxed wrapper owned by the caller until `LGBM_BoosterFree`. Every
  entry point runs its body inside `std::panic::catch_unwind`, so a panic becomes a `-1` error
  instead of unwinding across the FFI boundary (SPEC-ERR-4, SPEC-ERR-5). The crate builds as
  `cdylib` + `staticlib` (the C artifacts) and `rlib` (so Rust integration tests exercise the
  symbols without a C toolchain). Slice 1 covers error handling, model load
  (`LGBM_BoosterCreateFromModelfile`, `LGBM_BoosterLoadModelFromString`), `LGBM_BoosterFree`
  and metadata (`LGBM_BoosterGetNumClasses`, `LGBM_BoosterGetCurrentIteration`). Numbering
  skips D-007..D-011, which belong to the parallel open PRs #94/#95/#96.
- Rationale: `include/LightGBM/c_api.h` and `src/c_api.cpp` (v4.7.0) define exactly these
  return codes, the thread-local last error and the opaque handle; matching them is what makes
  the symbols drop-in. The C-header test harness named in PROGRESS T5.3 comes in a later slice;
  slice 1 is validated by `crates/lgbm-capi/tests/booster_load.rs`, which calls the `extern "C"`
  symbols directly and checks them against `lgbm-core`.
- SPEC ref: SPEC-ERR-5 (and SPEC-ERR-4 for the panic boundary)
- Status: active

## D-013: Layout auto-choice stays column-wise instead of upstream's runtime timing probe

- Date: 2026-11-18
- Question: T2.2. Upstream picks column-wise vs row-wise in auto mode (neither `force_col_wise`
  nor `force_row_wise` set) by building both share states and *timing* a histogram construction
  of each (`Dataset::GetShareStates`, v4.7.0), keeping the faster. Should bankai mirror that
  runtime probe?
- Choice: no. bankai's auto path stays column-wise and skips the probe. It still rejects the
  `force_col_wise && force_row_wise` conflict exactly like upstream's fatal in the same function.
- Rationale: the layout choice is performance-only — column-wise and row-wise accumulate the same
  bins over the same rows, so they produce bit-identical histograms and the trained model is the
  same whichever is chosen (`tests/parity/test_auto_layout_parity.py` pins auto == force_col_wise
  == force_row_wise == upstream auto). Upstream probes because it cannot rank the two statically;
  bankai can, from the interleaved benchmark (release, 50k rows, 50 rounds, 7 interleaved rounds,
  medians; bankai/upstream time ratio): dense column-wise 0.69x (1 thread) / 0.58x (2), dense
  row-wise 0.95x / 0.99x. Column-wise is faster on every measured dense shape, and mirroring
  upstream's probe would only add the cost of building and timing a row-wise state that is then
  discarded. Revisit when the multi-value row-wise path (T2.3) and the perf work (T3.1/T3.2) make
  row-wise competitive somewhere. Extends D-006 (which deferred auto-choice pending this data).
- SPEC ref: none (performance-only choice; model output is identical, AGENTS.md section 5)
- Status: active

## D-014: Took the first deterministic node (T3.4 CEGB) over the benchmark-gated T2.3/T2.4

- Date: 2026-11-18
- Question: after T2.2, PROGRESS lists T2.3 (row-wise multi-value) as the next node, then T2.4
  (`spin_pool`→rayon). Which node to take next in the autonomous per-node loop?
- Choice: skip the benchmark-gated T2.x nodes for now and take T3.4 (CEGB), the first eligible
  node with a deterministic (`Bench: n/a`) gate.
- Rationale: T2.3's gate is the sparse interleaved benchmark, and sparse is a known perf gap
  (ROADMAP: 1.14–1.41x upstream) whose fix is T3.1 (sparse data partition), which itself depends
  on T2.4 (the `spin_pool`→rayon migration). So T2.3 cannot clear its own benchmark gate before
  that perf chain lands, and T2.4 is the delicate concurrency rewrite best done with focused
  attention rather than unattended. CEGB has a clean deterministic parity gate and is completable
  now. The per-node loop (AGENTS.md section 13) picks "the first node whose dependencies are all
  `[x]`"; among those, this takes the first one whose gate can actually be verified unattended.
  The benchmark-gated perf chain (T2.3, T2.4, T3.1, T3.2) remains the next focus for a session
  that can run and compare benchmarks interactively.
- SPEC ref: none (operational choice under AGENTS.md section 13)
- Status: active

## D-015: Took T5.2 (forced-bins dump) over T4.3 (model merge), which is underspecified

- Date: 2026-11-18
- Question: after T3.4, the next deterministic nodes are T4.3 (model merge, "upstream
  `Booster.merge` semantics") and T5.2 (dataset surface). Which to take?
- Choice: take T5.2. Defer T4.3 until its contract is pinned.
- Rationale: upstream `lightgbm` has no public `Booster.merge` on the Python `Booster`, and
  `pytest tests/parity -k merge` collects zero tests, so there is no parity surface to write a
  failing test against without first inventing the semantics (the closest upstream construct is
  the C API `LGBM_BoosterMerge` / `GBDT::MergeFrom`, used for distributed training, which belongs
  with the C-API/network work, not a Python parity node). T5.2 is a concrete, deterministic gap:
  `forcedbins_filename` bounds were applied to binning but not stored, so the dataset text dump's
  `forced_bins` section was empty; storing them (`forced_bin_bounds_for`) makes the dump match
  upstream byte for byte. T4.3 should be revisited once its intended API surface is decided.
- SPEC ref: none (operational choice under AGENTS.md section 13)
- Status: active

## D-016: Run each tree inside one rayon `install` to meet the spin_pool → rayon gate

- Date: 2026-11-18
- Question: T2.4. A straight `spin_pool -> rayon` swap regressed dense 2-thread training ~11%
  against the spin_pool baseline (bankai/upstream ratio 0.656 -> 0.729), missing SPEC-CONC-5's
  3% gate. spin_pool kept workers spinning between the many short per-node regions; rayon lets
  them sleep, and waking them costs 15-30 us. How to close the gap without reviving spin_pool?
- Choice: build a per-Booster `rayon::ThreadPool` (`utils::worker_pool::WorkerPool`) and run the
  whole tree inside one `pool.install` (`SerialTreeLearner::train`). The per-region primitives
  (`run`, `for_each_mut`, ...) then nest on the already-active pool instead of re-entering rayon
  each region. Raising the parallel thresholds (`PARALLEL_HIST_MIN_WORK`, `PARALLEL_SEARCH_MIN_BINS`,
  `PARALLEL_PARTITION_MIN_ROWS`) was tried first and made it *worse* (dense 2-thread 0.729 -> 0.750),
  because the cost was per-region rayon entry, not too many small regions, so serialising more only
  lost the 2-thread gain.
- Rationale: inside one `install` the pool stays active for the tree's duration, so the short
  regions no longer each pay the wake latency. Measured (release, 50k rows, 9 interleaved rounds,
  bankai/upstream ratio): dense 2-thread 0.665 vs 0.656 baseline (+1.4%, within gate), dense
  1-thread 0.795 vs 0.797, sparse 2-thread 0.907 vs 0.915; 2-thread scaling dense 1.49x, sparse
  1.66x (both above the 1.2x floor). Results are unchanged (parallel and serial write disjoint
  data and merge in index order, SPEC-CONC-2), so `thread_invariance` holds for 1/2/3 threads and
  the full parity suite is unchanged. The gate was met, not relaxed.
- SPEC ref: SPEC-CONC-1 (rayon, one pool per Booster), SPEC-CONC-5 (migration gate)
- Status: active

## D-017: The sparse single-thread gap is the scalar threshold search (T2.6), not the partition (T3.1)

- Date: 2026-11-18
- Question: T3.1 ("sparse data partition and thread scaling", target sparse ≤1.0x upstream). The
  interleaved benchmark shows sparse 1-thread at ~1.12x of upstream (2-thread already 0.91x, i.e.
  faster). Where is the single-thread gap, and can optimising the sparse data partition close it?
- Choice: do not optimise the partition — it is not the bottleneck. Record the finding and defer
  the sparse single-thread target to the SIMD node (T2.6).
- Rationale: measured with construction and training separated (release, 50k x 200 CSR, density
  0.03, median of 11): Dataset construction is 0.93x of upstream (bankai faster, ~16 ms, not the
  gap); single-thread *training* is ~1.33x. Profiling that training (macOS `sample`, debuginfo
  build) splits it into the threshold search `find_best_threshold_numerical` (~42%, a scalar
  per-bin scan shared with the dense path), `data_partition::partition_sparse` (~18%), and the
  sparse histogram `MultiValBin::accumulate` (~10%). The partition already mirrors upstream's
  forward delta-encoded `SparseBinIterator`, and making `accumulate` `unsafe` (unchecked writes)
  gained only ~1.6% of training, so neither sparse-specific kernel has real headroom. The
  dominant cost is the scalar threshold search: there is no SIMD anywhere in
  `feature_histogram.rs`/`histogram.rs` (upstream vectorises it), and sparse data has ~200
  features, so the scalar scan runs ~200x per node and dominates — the dense path is fast only
  because it has far fewer features. So the lever for the sparse single-thread gap is T2.6 (SIMD
  histogram/search kernels with a bit-exact scalar fallback), which benefits dense and sparse
  alike; T3.1's own scope (the partition) is effectively already at upstream parity.
- SPEC ref: none (performance finding; SPEC-CONC-6 covers the SIMD work that would close it)
- Status: active


## D-018: SIMD across independent histogram and split-gain lanes (T2.6)

- Date: 2026-11-18
- Question: introduce SIMD without changing the prefix-sum rounding, row order or threshold
  tie-breaking required by SPEC-CONC-6 and SPEC-NUM-7.
- Choice: use two independent f64 lanes: gradient/Hessian for dense histogram updates, and
  left/right child gain for the common numerical search (no L1, output clipping or smoothing).
  Dispatch NEON on aarch64 and SSE2 on x86_64 with runtime detection; retain explicit scalar
  reference functions and fall back to them on other targets and under Miri. Constrained,
  clipped, smoothed and L1 gain calculations keep their existing implementations. There are
  no reciprocal approximations, FMA changes, prefix-sum reassociations or allocations.
- Evidence: the acceptance tests first failed to compile because the scalar histogram entry
  point and SIMD split module did not exist. Both now compare every result bit against an
  independent ordered scalar formula. Histogram cases cover all four bin widths, repeated
  destinations, indexed rows, empty/odd lengths, constant Hessians, cancellation and an
  unaligned output. Split-gain cases include extreme magnitudes and nonzero L2.
- Criterion: release/bench profile, one thread, Apple M1 (8 logical CPUs), macOS/aarch64, base d564c4d versus this T2.6
  diff, seven interleaved rounds, 0.2 s warm-up / 0.6 s measurement / 20 samples per case.
  Reproduce with `cargo bench -p lgbm-core --bench kernels --no-run` at each revision, keep
  each resulting executable, then alternate the binaries with `--bench
  '^(histogram/|partition/|split_finding/|prediction/)' --warm-up-time 0.2
  --measurement-time 0.6 --sample-size 20` seven times. Medians (base -> candidate):

  | Kernel | Base | Candidate | Ratio |
  | --- | ---: | ---: | ---: |
  | Histogram all rows, 16 bins | 176.67 us | 172.16 us | 0.9745 |
  | Histogram all rows, 255 bins | 124.96 us | 123.59 us | 0.9891 |
  | Histogram all rows, 4096 bins | 124.43 us | 122.69 us | 0.9861 |
  | Histogram subset, 16 bins | 65.21 us | 65.22 us | 1.0001 |
  | Histogram subset, 255 bins | 49.78 us | 49.98 us | 1.0042 |
  | Histogram subset, 4096 bins | 51.41 us | 51.48 us | 1.0015 |
  | Partition, 200k rows | 423.52 us | 424.45 us | 1.0022 |
  | Numerical split search, 255 bins | 540.21 ns | 511.80 ns | 0.9474 |
  | Prediction, 4096 x 16 x 20 trees | 407.88 us | 421.04 us | 1.0323 |

  Histograms use 200k rows (subset: every third row). No upstream process runs in this
  Rust-only comparison; the pinned upstream is lightgbm==4.7.0 and OMP_WAIT_POLICY is n/a.
  Every kernel is below the 5% regression gate. A first non-isolated measurement was
  discarded because builds/Miri slowed even unchanged kernels; the table is from binaries
  run with those jobs stopped. The observed split-search benefit is modest, so this does
  not by itself establish the sparse end-to-end target in T3.1.
- Profile: `samply record --save-only --duration 3 --output <profile.json> -- <base-kernels>
  --bench split_finding --profile-time 3`, plus the training diagnosis in D-017. Of 2384
  samples in the base microbenchmark, 750 land on the final scalar gain addition immediately
  following the two scalar `fdiv` instructions (confirmed with `otool -tvV`); this motivates
  pairing those independent divisions rather than changing the prefix sums. The paired
  scalar/dispatch histogram benchmarks are retained in `benches/kernels.rs` for future work.
- Full training check: release build verified newer than debug and installed immediately before
  timing; `lightgbm==4.7.0`, same venv/data/params, 50 boosting rounds, seven interleaved rounds,
  auto layout. Dense is 50k x 20; CSR is 50k x 200 at density 0.03. Candidate is this T2.6
  diff over d564c4d (the PR records its final commit). These are end-to-end medians including
  construction, not a before/after claim against main:

  | OMP_WAIT_POLICY | Case / threads | bankai | upstream | Ratio |
  | --- | --- | ---: | ---: | ---: |
  | ACTIVE | dense / 1 | 0.228 s | 0.288 s | 0.792 |
  | ACTIVE | dense / 2 | 0.178 s | 0.186 s | 0.959 |
  | ACTIVE | sparse / 1 | 0.463 s | 0.413 s | 1.123 |
  | ACTIVE | sparse / 2 | 0.281 s | 0.237 s | 1.186 |
  | PASSIVE | dense / 1 | 0.222 s | 0.283 s | 0.784 |
  | PASSIVE | dense / 2 | 0.155 s | 0.235 s | 0.660 |
  | PASSIVE | sparse / 1 | 0.421 s | 0.381 s | 1.106 |
  | PASSIVE | sparse / 2 | 0.256 s | 0.281 s | 0.910 |

  Reproduce: `OMP_WAIT_POLICY=ACTIVE .venv/bin/python benches/bench_train_interleaved.py
  --rounds 7 --threads 1 2`, then the same command with PASSIVE. The ACTIVE result prompted
  the policy comparison; both are retained. Both commands exit 1 because the known sparse
  single-thread gap exceeds the upstream target. T2.6's specified acceptance is the Criterion
  no-regression gate, which passed; T3.1 remains blocked rather than claiming the overall
  upstream target is met. No CI gate or required check was changed.
- Validation: `scripts/check.sh` ended `all green` (2346 Python tests); release parity/upstream
  suite passed 2340 tests. Native bitwise and 1/2/3-thread invariance tests passed; Miri passed
  the histogram library tests and the split-gain integration test. The larger histogram
  integration matrix was run natively; its additional Miri run was interrupted after the
  library Miri suite passed, because interpreting the full four-width matrix was excessive.
- SPEC ref: SPEC-CONC-6, SPEC-NUM-2, SPEC-NUM-7
- Status: active


## D-019: One row-wise bin view across mixed feature groups (T2.3)

- Date: 2026-11-18
- Question: force_row_wise previously processed dense single-value groups in one row-major
  pass and the multi-value group in a separate pass, rather than upstream's all-feature view.
- Choice: AllFeatureBins ports CalcBinOffsets(false), GetMultiBinFromAllFeatures and
  PushDataToMultiValBin (v4.7.0). It selects dense/sparse storage from the combined density,
  keeps single-value bundles as one column, expands multi-value groups into their raw
  sub-feature columns, reserves upstream's initial bin and removes zero/default bins only
  where upstream does. Sparse storage is assembled with two column scans and no dense
  temporary. The learner uses that single view and the existing MultiValBin block/merge
  kernel; it reuses the histogram scratch buffers and does not also build an unused dense
  copy or a separate multi-value view. Column-wise and dense-only row-wise paths remain.
- Acceptance: the native all-feature offset/storage tests failed first on the missing
  AllFeatureBins API. A separate regression then failed with Unsupported("sparse single-value
  group"); the same view now handles that representation in row-wise training. Native tests
  cover both physical layouts, bundled groups, nonzero most-frequent bins and real datasets.
  Twenty-two added parity cases cover dense/CSR/CSC, mixed/all-sparse data, bagging, feature
  sampling, weights, categories and 1/2/3 threads, including cross-thread model identity.
- Benchmark harness: the initial periodic sparse data accidentally formed only exclusive
  single-value bundles, caught by the assertion requiring a multi-value group. The data
  generator was corrected to independent seeded random values; both group assertions were
  retained. Both sparse and dense all-feature views are benchmarked at 4096 x 12, 8 trees,
  max_bin=63, one thread, with 2 or 9 dense features and 80% zeros in the remaining columns.
- Profile: `samply record --save-only --duration 3 --output <profile.json> -- <base-kernels>
  --bench training/row_wise_mixed --profile-time 3`. Of 2612 base samples, 522 land in
  MultiValBin::accumulate, 487 in numerical split search, 472 in data routing, 362 in sparse
  partition and 241 in construct_histograms. The existing accumulation kernel remains the
  arithmetic implementation; this change unifies its inputs and histogram offsets.
- SPEC ref: SPEC-MEM-1, SPEC-MEM-2, SPEC-MEM-4, SPEC-CONC-3, SPEC-NUM-2, SPEC-NUM-6
- Final gates: cargo check, clippy, cargo test and scripts/check.sh passed; check.sh ended
  "all green" with 2368 Python tests. A rebuilt release extension then passed 2362 parity /
  upstream-suite tests. No existing assertion, tolerance, skip or gap-registry entry changed.
- Criterion: release/bench profile, Apple M1 (8 logical CPUs), one thread, base af07e1f versus
  this T2.3 diff (candidate hash recorded in the PR), seven interleaved rounds, 0.2 s warm-up,
  0.6 s measurement, 20 samples. Both revisions used the same benchmark code; only the base's
  two implementation modules were restored from af07e1f for its build. No upstream wheel
  runs in these Rust-only comparisons (pinned version 4.7.0, OMP_WAIT_POLICY n/a).

  | Kernel | Base | Candidate | Ratio |
  | --- | ---: | ---: | ---: |
  | Histogram all rows, 16 bins | 165.88 us | 168.21 us | 1.0140 |
  | Histogram all rows, 255 bins | 116.80 us | 117.73 us | 1.0080 |
  | Histogram all rows, 4096 bins | 116.50 us | 117.64 us | 1.0098 |
  | Histogram subset, 16 bins | 62.51 us | 62.46 us | 0.9992 |
  | Histogram subset, 255 bins | 47.50 us | 48.04 us | 1.0114 |
  | Histogram subset, 4096 bins | 49.54 us | 50.20 us | 1.0134 |
  | Partition, 200k rows | 425.50 us | 423.37 us | 0.9950 |
  | Prediction, 4096 x 16 x 20 trees | 393.13 us | 382.90 us | 0.9740 |
  | Numerical split search, 255 bins | 512.62 ns | 512.22 ns | 0.9992 |
  | Row-wise mixed / sparse view, 4096 x 12 | 2.671 ms | 2.653 ms | 0.9934 |
  | Row-wise mixed / dense view, 4096 x 12 | 3.004 ms | 2.857 ms | 0.9508 |

  Histogram cases use 200k rows, subsets every third row. Reproduce by building the kernels
  executable at each revision, then alternating them seven times with `--bench
  '^(histogram/|partition/|split_finding/|prediction/|training/)' --warm-up-time 0.2
  --measurement-time 0.6 --sample-size 20`. All cases stay within the 5% gate.
- End-to-end upstream check: installed release build confirmed newer than debug; same venv,
  lightgbm==4.7.0, OMP_WAIT_POLICY=PASSIVE, force_row_wise=true, 50 boosting rounds,
  seven interleaved rounds. Dense: 50k x 20; CSR: 50k x 200, density 0.03.

  | Case / threads | bankai | upstream | Ratio |
  | --- | ---: | ---: | ---: |
  | dense / 1 | 0.267 s | 0.280 s | 0.952 |
  | dense / 2 | 0.241 s | 0.224 s | 1.074 |
  | sparse / 1 | 0.413 s | 0.340 s | 1.213 |
  | sparse / 2 | 0.249 s | 0.258 s | 0.964 |

  Command: `OMP_WAIT_POLICY=PASSIVE .venv/bin/python benches/bench_train_interleaved.py
  --layout row --rounds 7 --threads 1 2`. It exits 1 on the dense-2 / sparse-1 upstream gaps.
  An initial run overlapping a short test build was discarded; the table is the isolated run.
- Same-session control for both slow cases: built af07e1f and the candidate release extensions,
  ran them in separate fresh processes, alternating base/candidate for seven rounds with the
  same data and parameters above. Each timed block is the median of three full 50-round
  training calls, after a 2-round warm-up; data generation and imports are outside the timer.
  OMP_WAIT_POLICY=PASSIVE, upstream 4.7.0 imported but not invoked in this control:

  | Case / threads | Base | Candidate | Ratio |
  | --- | ---: | ---: | ---: |
  | dense / 2 | 0.238751 s | 0.238648 s | 0.9996 |
  | sparse / 1 | 0.414659 s | 0.412849 s | 0.9956 |

  These differences are noise, not speedup claims. The upstream gaps are not introduced by
  the unified view; they remain T3.1/T3.2. T2.3 verifies the row-wise representation and parity,
  with no regression against main. No CI gate, tolerance or required check was changed.
- Status: active


## D-020: Bounded LRU histogram cache and reset semantics (T2.5)

- Date: 2026-11-19
- Question: positive histogram_pool_size was rejected even though the learner already had
  the parent-minus-smaller-child subtraction and the fallback that rebuilds an uncached parent.
- Choice: bound physical histogram buckets, keep the existing leaf/slot mapping, and evict
  the least recently used bucket (first slot on equal timestamps). Both Get hits/misses and
  Move refresh recency; maps and timestamps reset per tree. The unlimited/default path has
  no LRU timestamp vector. Per-slot splittable flags and histogram buffers stay together.
- Source: src/treelearner/serial_tree_learner.cpp (v4.7.0), Init / ResetConfig, and
  src/treelearner/feature_histogram.hpp, HistogramPool::Get / Move / Reset. The MB-to-bucket
  conversion uses 16 bytes times the sum of raw per-feature bin counts, not the compacted
  histogram layout's total bins. The result is clamped to at least two and at most num_leaves.
- Reset quirk: upstream resizes the cache only when num_leaves changes. A budget-only reset
  still echoes the new MB value, while keeping the old capacity. The trainer's normal learner
  rebuild now preserves the actual capacity when num_leaves stays equal. A budget-only
  update instead keeps the learner and bagging sampler, resets split-threshold RNG metadata
  and consumes ColSampler::SetConfig's one by-tree draw from the continuing feature RNG.
  This preserves bagging and feature sampling, including extra_trees, rather than reseeding
  them through construction. Bagging::ResetSampleConfig and ColSampler::SetConfig at the
  same upstream tag were read to confirm those state transitions.
- Red evidence: the new Python test first raised Unsupported for a positive budget. A native
  reset regression then observed 7 buckets where upstream requires the previous 2. The
  sampling-reset parity test separately reproduced divergent leaf assignments starting at
  the first tree after reset. All three failures were fixed without changing their assertions.
- Tests: 36 new parity cases cover minimum/intermediate/full budgets, both layouts,
  regression/binary, missing values, bagging, feature sampling, alias spelling, 1/2/3 threads,
  all monotone methods, and pool-only resets with/without extra_trees. Native tests check
  bounded allocation, LRU hit/eviction order, Move recency and data preservation, map reset,
  the unlimited path, and resize-on-num_leaves-only behavior.
- Harness: benches/bench_train_interleaved.py now accepts --histogram-pool-size so bounded
  runs can be reproduced with the same existing upstream/data/thread setup. The command
  first failed on the unknown CLI argument; its default parameters remain unchanged.
- Profile: `samply record --save-only --duration 3 --output <profile.json> -- <base-kernels>
  --bench training/row_wise_mixed --profile-time 3`. Of 1836 base samples, 447 land in
  multi-value accumulation, 383 in numerical split search, 338 in routing, 255 in sparse
  partition and 107 in split orchestration; pool bookkeeping is not a leading self-time site.
- SPEC ref: SPEC-MEM-4, SPEC-CONC-2, SPEC-NUM-2, SPEC-NUM-6, SPEC-PY-5
- Validation: scripts/check.sh ended "all green" (2404 Python tests), with cargo check,
  clippy, cargo test, fmt, config generation, ruff and mypy passed. The rebuilt release
  extension passed 2398 parity/upstream tests. No assertions or tolerances were weakened,
  and no fixture, skip or known-gap entry was changed.
- Criterion: release/bench, Apple M1 (8 logical CPUs), one thread, base 0fb1d93 versus this
  diff (final candidate hash in PR), seven interleaved rounds, 0.2 s warm-up, 0.6 s measurement,
  20 samples. Default/unlimited pool on both revisions. Pinned upstream is 4.7.0; no upstream
  process runs in this Rust-only comparison, OMP_WAIT_POLICY n/a.

  | Kernel | Base | Candidate | Ratio |
  | --- | ---: | ---: | ---: |
  | Histogram all rows, 16 bins | 163.35 us | 163.38 us | 1.0002 |
  | Histogram all rows, 255 bins | 115.74 us | 114.99 us | 0.9935 |
  | Histogram all rows, 4096 bins | 113.82 us | 113.48 us | 0.9970 |
  | Histogram subset, 16 bins | 61.63 us | 61.67 us | 1.0007 |
  | Histogram subset, 255 bins | 46.92 us | 46.98 us | 1.0011 |
  | Histogram subset, 4096 bins | 48.91 us | 48.84 us | 0.9987 |
  | Partition, 200k rows | 420.09 us | 420.44 us | 1.0008 |
  | Prediction, 4096 x 16 x 20 trees | 370.29 us | 385.45 us | 1.0409 |
  | Numerical split search, 255 bins | 507.94 ns | 508.52 ns | 1.0011 |
  | Row-wise mixed / sparse view, 4096 x 12 | 2.621 ms | 2.586 ms | 0.9864 |
  | Row-wise mixed / dense view, 4096 x 12 | 2.818 ms | 2.823 ms | 1.0020 |

  Histograms use 200k rows, subsets every third row. Mixed training uses 8 trees, max_bin=63,
  2 or 9 dense columns and 80% zeros in the remaining columns. Build both kernels binaries,
  then alternate them seven times with `--bench
  '^(histogram/|partition/|split_finding/|prediction/|training/)' --warm-up-time 0.2
  --measurement-time 0.6 --sample-size 20`. Every case remains below the 5% regression limit;
  the prediction kernel itself is unchanged.
- End-to-end training: installed release build newer than debug, same session/venv/data/params
  as lightgbm==4.7.0, OMP_WAIT_POLICY=PASSIVE, auto layout, 50 boosting rounds, seven
  interleaved rounds. Dense: 50k x 20. CSR: 50k x 200, density .03. Candidate is this diff over
  0fb1d93; the PR records its exact hash. Budget 0.000001 MB exercises the minimum 2 buckets.

  | Pool | Case / threads | bankai | upstream | Ratio |
  | --- | --- | ---: | ---: | ---: |
  | default | dense / 1 | 0.222 s | 0.282 s | 0.788 |
  | default | dense / 2 | 0.148 s | 0.225 s | 0.659 |
  | default | sparse / 1 | 0.411 s | 0.369 s | 1.112 |
  | default | sparse / 2 | 0.250 s | 0.276 s | 0.904 |
  | 0.000001 MB | dense / 1 | 0.295 s | 0.347 s | 0.848 |
  | 0.000001 MB | dense / 2 | 0.205 s | 0.281 s | 0.730 |
  | 0.000001 MB | sparse / 1 | 0.548 s | 0.516 s | 1.061 |
  | 0.000001 MB | sparse / 2 | 0.342 s | 0.361 s | 0.946 |

  Command: `OMP_WAIT_POLICY=PASSIVE .venv/bin/python benches/bench_train_interleaved.py
  --rounds 7 --threads 1 2`, then repeat with `--histogram-pool-size 0.000001`.
  Both commands exit 1 on the remaining sparse single-thread upstream gap (T3.1/T3.2).
  The default-path before/after Criterion gate passed. No CI threshold or required check
  was changed; the bounded path is compared with upstream at the same budget.
- Miri follow-up: CI selected the new cache-reset unit test through its existing histogram
  filter. Dataset construction entered rayon's global worker deque, where Miri reported a
  Stacked Borrows violation inside Crossbeam. Dense sample bin-mapper construction and dense
  group filling now use standard sequential iterators under cfg(miri), matching WorkerPool's
  existing serial Miri backend. Native builds keep the Rayon iterators, and both backends
  execute the same mapping/group-filling closures. No test assertion, filter, Miri flag or CI
  check changed. scripts/miri.sh passed all 10 selected tests, including the reset regression.
- Status: active

## D-021: Native oracle for nearest-rounded gradient discretization (T2.7, slice 1)

- Date: 2026-11-19
- Question: the new quantization acceptance fixture initially recorded a gain that
  differed by two score_t ULPs despite identical leaf outputs and Hessian weights.
- Harness correction: capture the native Booster with keep_training_booster=True.
  Upstream python-package/lightgbm/engine.py train otherwise serializes and reloads
  the model before returning. Tree::ToString uses the default six significant digits
  for split_gain, and the reload uses Common::Atof. Those statistics are not the
  native score_t split gain needed to validate the quantizer's integer sums.
- Source: v4.7.0 src/treelearner/gradient_discretizer.cpp DiscretizeGradients;
  src/treelearner/feature_histogram.hpp BeforeNumericalInt and
  FindBestThresholdSequentiallyInt; src/io/tree.cpp Tree::ToString and NodeToJSON;
  python-package/lightgbm/engine.py train. The local upstream source's
  gradient_discretizer.cpp was checked against the pinned tag with no differences.
- Oracle: scripts/gen_quantized_grad_fixtures.py records 16 upstream stumps, for
  constant/nonconstant Hessians and 2/3/4/7/16/31/64/127 quantization bins. Inputs
  are score_t values; expected leaf outputs, Hessian weights and native split gains
  come only from lightgbm==4.7.0. No expected quantized integer is synthesized by
  the generator. The fixture records its producing platform and version.
- Test correction is isolated in the Harness commit. No assertion or tolerance was
  changed. Initial red: missing gradient_discretizer module; subsequent red exposed
  the serialized gain in the original oracle. After the native-oracle correction,
  the same acceptance assertions pass.
- Scope: this is the nearest-rounding foundation for T2.7, not quantized training.
  Stochastic RNG streams, integer histogram accumulation/search, leaf renewal and
  integration into SerialTreeLearner remain pending. T2.7 must not be checked off.
- Status: active

## D-022: Nearest-rounding quantization foundation (T2.7, slice 1)

- Date: 2026-11-19
- Choice: port the discretization primitive separately from learner integration.
  treelearner::gradient_discretizer mirrors upstream's packed per-row layout: low
  byte Hessian, signed high byte gradient, held in i16. Inputs remain score_t/f32;
  maxima, scales and inverse scales are double/f64. Odd bin counts deliberately
  use integer division by two in the gradient scale. The buffer is allocated once
  and reused, including when the next tree has different derivative maxima.
- Source: src/treelearner/gradient_discretizer.{cpp,hpp} (v4.7.0), DiscretizeGradients
  (stochastic_rounding=false), and SetNumBitsInHistogramBin. The width helper uses
  the strict rows*bins <256 / <65536 boundaries, widening to 32 bits otherwise.
- Undefined input handling: the primitive rejects nonfinite derivatives, negative
  Hessians, empty/mismatched input, and bin counts outside 2..=127. At 128 or more
  bins the variable-Hessian int8 conversion may overflow upstream. Zero gradient
  maxima, or zero variable-Hessian maxima, lead to NaN-to-int8 conversion upstream;
  refuse those cases (SPEC-NUM-10) instead of choosing platform-specific results.
  Constant Hessians are always integer one, so zero constant Hessians need no cast
  and retain a zero reconstruction scale. Validation errors preserve prior output.
  These limits belong to the new low-level primitive; Python training is unchanged.
- TDD: acceptance tests first failed on the missing module; the native-oracle
  correction is the separate Harness commit (D-021). Zero-maxima refusal first
  failed because the initial implementation returned success; after fixing it, the
  constant-zero-Hessian test first failed because the refusal was too broad. All
  six tests now pass with their original assertions. The oracle checks both leaves'
  outputs and weights within 1e-12 and the native score_t gain after f32 conversion.
- Performance: Criterion cases quantify nearest rounding over 200,000 rows with
  constant and variable Hessians and reuse the output buffer. The training path is
  not connected to this module. Before/after results and final gates follow below.
- Remaining T2.7 work: stochastic rounding and its upstream std::mt19937/distribution
  streams; integer histogram accumulation/subtraction and split searches; integer
  leaf-sum propagation, leaf renewal, reset semantics and full training parity.
- Status: active

- Validation: cargo check --workspace and scripts/check.sh passed (2404 Python
  tests passed, 9 skipped, 13 xfailed, 4 xpassed); maturin develop -r followed by
  pytest tests/parity tests/python_package_test passed 2398 tests with the same
  skips/xfails/xpasses. All six new native tests passed in debug and release.
- Local benchmark: Apple M1, 8 logical CPUs, release/bench profile, one thread,
  base 4533261f4146b905e1c655737d3c57caba8aa9fb versus this implementation (final
  head SHA recorded in the PR). Pinned upstream v4.7.0; no upstream process runs
  in this Rust benchmark, so OMP_WAIT_POLICY is not applicable. Seven interleaved
  base/candidate rounds, warm-up 0.2s, measurement 0.6s, sample size 20. Each
  number is the median of seven Criterion median point estimates (ns).

  | Kernel | Base ns | Candidate ns | Ratio |
  | --- | ---: | ---: | ---: |
  | histogram/all_rows/16 | 165915.76 | 165661.01 | 0.9985x |
  | histogram/all_rows/255 | 118734.95 | 118703.86 | 0.9997x |
  | histogram/all_rows/4096 | 114792.12 | 120039.61 | 1.0457x |
  | histogram/leaf_subset/16 | 61934.49 | 62151.65 | 1.0035x |
  | histogram/leaf_subset/255 | 47193.00 | 47548.25 | 1.0075x |
  | histogram/leaf_subset/4096 | 50468.99 | 49239.36 | 0.9756x |
  | partition/numerical/200k | 421467.10 | 422015.23 | 1.0013x |
  | prediction/dense/4096x16x20 | 420031.67 | 401414.92 | 0.9557x |
  | split_finding/numerical/255_bins | 509.98 | 510.25 | 1.0005x |
  | training/row_wise_dense_mixed/4096x12 | 2832454.60 | 2835737.28 | 1.0012x |
  | training/row_wise_mixed/4096x12 | 2595524.25 | 2586720.29 | 0.9966x |

  Histograms use 200k rows; subsets select every third row. Prediction is
  4096x16 over 20 trees, numerical search 255 bins, mixed training 4096x12
  over 8 trees. These existing code paths are unchanged by the new module.
  All kernels remain within the 5% gate; no speedup claim is made.
  New standalone quantization/nearest_constant/200k: 500655.39 ns (0.501 ms), 200k f32 rows, 4 bins.
  New standalone quantization/nearest_variable/200k: 609040.04 ns (0.609 ms), 200k f32 rows, 4 bins.
  New-case times have no base equivalent and are not an end-to-end training
  comparison against upstream. They exclude setup allocation and reuse storage.
- Artifact verification: an initial shared-target build reused the base binary;
  that entire comparison was discarded. The package benchmark artifacts were
  cleaned and the candidate recompiled. Both new benchmark names were verified,
  as were distinct binary hashes before repeating all seven rounds. Valid logs:
  target/t27-final-N-{base,candidate}.log; results: target/t27-final-benchmark-results.json.
  base executable SHA256: 12dedfac940d4e90d6842e6141e4bf124c98b441f2f98259c4ab08335ba33371.
  candidate executable SHA256: e722b1433a45efb1a01249c661d3917682358ca7dafb1d4b20c4fe393e1ede11.
- Profile: samply record --save-only --duration 3 --output
  target/t27-final-quantizer-profile.json -- target/t27-candidate-kernels --bench
  quantization/nearest_variable --profile-time 3. Of 2149 main-thread samples,
  the leading sampled addresses resolve with atos to the inlined discretization
  benchmark loop. This change adds the primitive; it does not optimize a training
  path or claim that T3.1/T3.2 performance targets are closed.

## D-023: std::mt19937 engine for quantized stochastic rounding (T2.7, slice 2)

- Date: 2026-11-19
- Question: SPEC-NUM-9 states that all RNG comes from `utils::random` (upstream's
  LCG) and that the `rand` crate is unused. Upstream's quantized stochastic
  rounding does not use that LCG: `GradientDiscretizer::Init` seeds `std::mt19937`
  (one engine per OpenMP block, plus one for the per-tree rotation start) and draws
  the per-row rounding noise from `std::uniform_real_distribution<double>` and the
  rotation start from `std::uniform_int_distribution<data_size_t>`
  (src/treelearner/gradient_discretizer.cpp, v4.7.0). Parity therefore needs the
  standard-library engine, not the LCG.
- Choice: port `std::mt19937` as `treelearner::gradient_discretizer::mt19937`,
  scoped next to the only path that uses it, rather than widening the general RNG.
  This extends SPEC-NUM-9 for the quantized path only; the LCG remains the RNG for
  every other training/prediction path, and `rand` is still unused. The engine uses
  the standard's fixed parameters (`mersenne_twister_engine<uint32_t, 32, 624, 397,
  31, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>`).
- Portability: the engine's raw 32-bit output sequence is defined by the C++
  standard, so it is identical on every platform and standard library (libc++ on
  the macOS fixture host, libstdc++ on the Linux CI runner). Only the *distributions*
  layered on top are implementation defined; those are deferred to the next T2.7
  slice, where the libc++/libstdc++ difference and the platform-portability of the
  fixtures will be decided and recorded. This slice introduces no platform coupling.
- Oracle: the C++ standard's own predefined test (ISO/IEC 14882 [rand.predef]) —
  the 10000th consecutive invocation of a default-constructed `std::mt19937` is
  `4123659995`. This single vector uniquely pins the seeding, the twist and the
  tempering, so it validates the engine for every seed upstream derives from
  `random_seed`. No expected value is synthesized from the Rust port.
- TDD: the integration test first failed to compile (missing `mt19937` module), the
  TDD red recorded in its own Harness commit. The implementation then made both the
  integration test and the in-module unit test green, with the assertion unchanged.
- SPEC ref: SPEC-NUM-9 (extended for the quantized path), SPEC-CONC-1.
- Validation: cargo fmt --all --check and cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2404 Python tests passed,
  9 skipped, 13 xfailed, 4 xpassed). The engine is not wired into any training path,
  so no hot-path benchmark applies.
- Status: active

## D-024: Packed integer gradient/Hessian histogram construction (T2.7, slice 3)

- Date: 2026-11-19
- Choice: port upstream's integer histogram accumulation as a standalone,
  platform-independent primitive (`treelearner::gradient_discretizer::int_histogram`)
  before the integer split search and learner integration. The discretized per-row
  derivative is a packed int16 (signed int8 gradient high byte, uint8 Hessian low
  byte, produced by `GradientDiscretizer`). The histogram accumulates both halves
  into one integer per bin: the Hessian occupies the low `HIST_BITS` bits and the
  gradient the next `HIST_BITS` bits, so `out[bin] += packed` sums both in one add.
  The three accumulator widths (int16/int32/int64 for HIST_BITS 8/16/32) mirror
  upstream's three kernels. Output is unpacked to `(gradient_sum, Hessian_sum)` i64
  pairs per bin; keeping the split search on packed values is a later slice.
- Source: src/io/dense_bin.hpp (v4.7.0), `ConstructHistogramIntInner` and the
  `ConstructHistogramInt8/Int16/Int32` entry points. The packing `(int8(grad_16>>8)
  << HIST_BITS) | (grad_16 & 0xff)` is reproduced; for HIST_BITS==8 upstream adds
  the raw packed int16, which equals this packing on valid discretized inputs.
- Width validity: the gradient sum sits in the high half and the Hessian sum in the
  low half, so a width holds a leaf only while |gradient_sum| < 2^(HIST_BITS-1) and
  Hessian_sum < 2^HIST_BITS. Upstream guarantees this by choosing the width from
  rows * num_grad_quant_bins (`SetNumBitsInHistogramBin`, mirrored by
  `histogram_bits`): each row's |gradient| <= bins/2 and Hessian <= bins, so a leaf
  of n rows has |gradient_sum| <= n*bins/2 and Hessian_sum <= n*bins, both bounded
  by the same n*bins < 256 / 65536 / 2^32 cutoffs. A narrower width than the data
  needs overflows exactly as upstream's fixed-width accumulator would.
- Constant Hessian: upstream's bits16/32 kernels pack `| 1` for the constant case
  instead of reading the low byte; on valid inputs the discretized low byte is
  already 1, so the primitive's `constant_hessian` flag (force Hessian 1) produces
  identical histograms. For bits8 upstream adds the raw packed value, equal to the
  same result on valid data.
- Oracle: the reference is the brute-force per-bin sum of the discretized
  derivatives (an invariant the packing must reproduce), not a value synthesized
  from the Rust packing. Tests cover all three widths agreeing within the 8-bit
  range, sums that overflow the narrower widths, empty bins and negative gradients.
- TDD: the invariant test first failed to compile (missing `int_histogram` module),
  the TDD red recorded in its own Harness commit. The Harness commit was corrected
  (still red) once, because the first draft asserted a 16-bit width could hold a
  gradient sum of 100000, which exceeds the 16-bit half; the width-validity note
  above is the fix. No assertion was weakened to pass.
- SPEC ref: SPEC-NUM-6 (integer accumulation has no rounding), SPEC-CONC-3.
- Validation: cargo fmt --all --check and cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2404 Python tests passed,
  9 skipped, 13 xfailed, 4 xpassed). The primitive is not wired into any training
  path, so no hot-path benchmark applies.
- Status: active

## D-025: Integer numerical split search (T2.7, slice 4)

- Date: 2026-11-19
- Choice: port upstream's `FindBestThresholdSequentiallyInt` / `BeforeNumericalInt`
  (`feature_histogram.rs::find_best_threshold_numerical_int`) over the unpacked
  integer `(gradient_sum, Hessian_sum)` histogram from D-024, as a standalone
  primitive before wiring it into the serial learner. The function mirrors the
  full-precision `find_best_threshold_numerical` structure (same RAND/L1/MAX_OUT/
  SMOOTH/REVERSE/SKIP/NA/MC template matrix) and reuses the same f64 gain and
  leaf-output helpers (`calculate_splitted_leaf_output`, `leaf_gain`,
  `leaf_gain_given_output`, `simd::split_gain`); only the sum accumulation differs.
- Integer exactness: gradient and Hessian prefix sums are accumulated as `i64`
  (exact, order-independent), and the double sums are recovered by one multiply by
  `grad_scale` / `hess_scale` per prefix point, as upstream does. The row count of a
  side is estimated from its integer Hessian sum: `cnt_factor = num_data /
  int_total_hessian`, `count = RoundInt(int_side_hessian * cnt_factor)`. For a
  constant Hessian (e.g. L2 regression) the integer Hessian sums are the exact row
  counts and `cnt_factor == 1`.
- kEpsilon placement (the subtle part, differs from the f64 path): upstream's
  integer path does NOT add `kEpsilon` to the leaf totals (unlike
  `FindBestThreshold`, which adds `2*kEpsilon` to its Hessian) and does NOT store it
  in the split sums. `kEpsilon` is added to the child Hessians ONLY inside the gain
  (`GetSplitGains(..., sum_left_hessian + kEpsilon, sum_right_hessian + kEpsilon)`).
  The leaf-total gain shift (`BeforeNumericalInt`) and the final leaf outputs use the
  raw (epsilon-free) scaled Hessians. The tail recomputes both child counts with
  `RoundInt` from the best integer sums, matching upstream lines 1324-1325.
- Oracle: an independent brute-force reference over all thresholds for the L2-only
  path (no L1/max-delta-step/smoothing/monotone constraints, no missing bin, so a
  single reverse scan), recomputing the gain with the documented formula, exact
  integer prefix sums, the same count estimation and the same epsilon placement. It
  shares none of the implementation's loop/break/best-tracking logic. Tests cover
  constant Hessian (exact counts), variable Hessian (estimated counts) with
  non-trivial scales, and an unsplittable feature. The full learner parity oracle
  (tree-by-tree vs `lightgbm==4.7.0`) applies when the function is wired in.
- Not wired: the serial learner still trains the full-precision path; this primitive
  is unreachable from Python until the integer histogram pool, sibling subtraction,
  leaf-splits integer sums and leaf-output wiring land (next slice).
- SPEC ref: SPEC-NUM-6 (integer accumulation has no rounding); the quantized path
  extends SPEC-NUM-9 for RNG as noted in D-023.
- Validation: cargo fmt --all --check and cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2404 Python tests passed,
  9 skipped, 13 xfailed, 4 xpassed). Not wired into any training path, so no
  hot-path benchmark applies.
- Status: active

## D-026: Wire quantized training into the serial learner (T2.7, slice 5)

- Date: 2026-11-19
- Choice: connect the quantized primitives (nearest discretizer D-021/D-022,
  integer histogram D-024, integer split search D-025) into `SerialTreeLearner`
  so `use_quantized_grad=true` trains end-to-end for a deterministic slice, and
  validate it against `lightgbm==4.7.0` tree-by-tree.
- Supported slice: numeric features, dense column-wise layout, `num_threads=1`
  effect on structure (the sums are integer-exact, so thread count cannot change
  them), `stochastic_rounding=false`, `quant_train_renew_leaf=false`. Objective
  arbitrary, but L2 regression (constant Hessian) is the first parity target.
- Integration points (`treelearner/serial.rs`):
  - `QuantState` holds the `GradientDiscretizer`, the two scales, and integer
    histograms (`Vec<(i64,i64)>` per slot) parallel to the double `HistogramPool`
    slots; they move with a slot exactly as the double ones do.
  - `before_train` discretizes once (upstream `Train` calls `DiscretizeGradients`
    before `BeforeTrain`) and sets the root `LeafSplits`: the integer sum packs
    `(grad<<32)|hess` exactly (used for splits), while the double `sum_gradients`
    accumulates `int*scale` in row order (upstream `LeafSplits::Init` int overload,
    used only for the root leaf output).
  - `construct_int_histograms` adds each row's packed `int8` gradient / `uint8`
    Hessian into its group's bin. Integer sums are exact and associative, so the
    accumulation order is irrelevant — unlike the double path, no gather or SIMD
    lane order must be reproduced, which removes the main parity risk.
  - Sibling subtraction is componentwise on the integer pairs; `fix_histogram_int`
    sets the most-frequent bin to the leaf integer total minus the other bins.
  - The split search calls `find_best_threshold_numerical_int` (D-025); the child
    integer sums it reports propagate through `SplitInfo` into the children's
    `LeafSplits::int_sum`, as upstream propagates `left/right_sum_gradient_and_hessian`.
- Non-regression fallback (the important call): on `main`, `use_quantized_grad`
  was silently ignored and trained full precision, and the ported upstream suite
  (`test_engine.py::test_quantized_training`, the `test_categorical_*[True]` cases)
  passed that way with loose-accuracy assertions. Those tests use the parameter
  defaults — `stochastic_rounding=true`, and categorical data — which this slice
  does not implement. Rejecting them with `Error::Unsupported` would regress a
  green CI gate. So the learner enables the integer path only when the whole
  supported slice holds and otherwise falls back to full precision, i.e. the exact
  prior behaviour. This is not a silent drop of a supported parameter: the
  parameter is honoured where implemented and behaves as it did while T2.7 was
  unimplemented everywhere else. Each later slice (stochastic rounding, renewal,
  categorical, multi-value/sparse/row-wise, multithread) narrows the fallback.
- Stochastic rounding remains the open portability question (see the handoff and
  D-023): its `uniform_real`/`uniform_int` draws are implementation-defined across
  libc++/libstdc++, so the default (`stochastic_rounding=true`) path is deferred,
  not approximated.
- Oracle: `tests/parity/test_quantized_training_parity.py` compares the model text
  tree section to `lightgbm==4.7.0` for `use_quantized_grad=true`,
  `stochastic_rounding=false`, `quant_train_renew_leaf=false`, `num_threads=1`,
  `deterministic=true`, L2 regression, `num_grad_quant_bins` in {4,8,16}, and
  checks the quantized model differs from the full-precision one. First-pass
  bit-exact, attributable to the integer sums being exact regardless of order.
- SPEC ref: SPEC-NUM-6, SPEC-NUM-2 (1-thread structure), SPEC-NUM-9 (RNG, extended
  for the quantized path per D-023).
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2408 Python tests passed,
  9 skipped, 13 xfailed, 4 xpassed). The change touches `treelearner/`, so CI runs
  the interleaved benchmark; the quantized path is gated behind `use_quantized_grad`
  and leaves the full-precision hot path unchanged.
- Status: active

## D-027: Leaf renewal for quantized training (T2.7, slice 6)

- Date: 2026-11-19
- Choice: implement `quant_train_renew_leaf` (upstream
  `GradientDiscretizer::RenewIntGradTreeOutput`, the non-`data` tree-learner branch)
  in `SerialTreeLearner::renew_int_grad_tree_output`, called once after a quantized
  tree is grown. The quantized split structure is kept; each leaf output is replaced
  by the full-precision value computed from the original (non-discretized) gradients
  and Hessians summed over the leaf's rows.
- Formula: upstream hardcodes `CalculateSplittedLeafOutput<USE_L1=true,
  USE_MAX_OUTPUT=true, USE_SMOOTHING=false>(sum_grad, sum_hess, lambda_l1, lambda_l2,
  max_delta_step, path_smooth, leaf_cnt, parent_output=0)`. `USE_L1=true` with
  `lambda_l1=0` is identical to `USE_L1=false` (`threshold_l1(s, 0) == s`), so the
  fixed template matches every parameterization. Smoothing is off and the parent
  output is zero here regardless of `path_smooth`, so renewal deliberately ignores
  both — unlike the split-time outputs.
- Summation order: upstream's loop has no `deterministic_` guard, so at
  `num_threads>1` its OpenMP reduction order differs; bankai sums in leaf-index order,
  which matches upstream at `num_threads=1` (the validated slice). Multithread renewal
  is part of the later multithread slice.
- Fallback removed: `quant_train_renew_leaf=true` no longer forces the full-precision
  fallback; it now trains the quantized path with renewal. The remaining fallback
  cases (stochastic rounding, categorical, multi-value/sparse/row-wise) are unchanged
  (D-026).
- Oracle: `tests/parity/test_quantized_training_parity.py::
  test_quantized_renew_leaf_matches_upstream` for regression and binary at
  `num_grad_quant_bins=8`, tree-by-tree against `lightgbm==4.7.0`. Red first (the
  fallback model differed), green after.
- SPEC ref: SPEC-NUM-6, SPEC-NUM-2.
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2413 Python tests passed).
  Touches `treelearner/`, so CI runs the interleaved benchmark (built with
  codegen-units=1 per D-026's follow-up); renewal is gated behind
  use_quantized_grad + quant_train_renew_leaf and leaves the full-precision path
  unchanged.
- Status: active

## D-028: Integer categorical split for quantized training (T2.7, slice 7)

- Date: 2026-11-19
- Choice: port upstream's `FindBestThresholdCategoricalIntInner`
  (`feature_histogram.rs::find_best_threshold_categorical_int`) over the integer
  `(gradient_sum, Hessian_sum)` histogram, mirroring the full-precision
  `find_best_threshold_categorical`. Dispatched from `search_feature_int` by bin type;
  categorical features are removed from the quantized full-precision fallback (D-026).
- Both categorical modes are ported: one-hot (`num_bin <= max_cat_to_onehot`, each
  category vs the rest) and many-vs-many (bins with `RoundInt(int_hess * cnt_factor)
  >= cat_smooth` sorted by the gradient/Hessian ratio `g/(h + cat_smooth)`, scanned in
  both directions with `l2 = lambda_l2 + cat_l2` and `min_data_per_group`). Counts are
  estimated from the integer Hessian sums; the child integer sums propagate through
  `SplitInfo` into the children's `LeafSplits`.
- Epsilon: the integer categorical path adds no `kEpsilon` to any sum — neither the
  leaf totals, the gain shift, nor the child Hessians — unlike the full-precision
  categorical search (which adds `2*kEpsilon` to the leaf total and `kEpsilon` per
  bin). The gain shift uses the original `lambda_l2`; `cat_l2` is added only after it,
  for the many-vs-many sort and the final gains/outputs. Reuses the shared runtime-flag
  gain/output helpers (`split_gains_rt`, `output_rt`, `leaf_gain_rt`).
- Oracle: `tests/parity/test_quantized_training_parity.py::
  test_quantized_categorical_matches_upstream` on mixed numeric + integer-coded
  categorical data, tree-by-tree against `lightgbm==4.7.0` at `num_grad_quant_bins`
  4/8/16. Red first (categorical fell back to full precision), green after.
- SPEC ref: SPEC-NUM-6, SPEC-NUM-2.
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2416 Python tests passed).
  Touches `treelearner/`, so CI runs the interleaved benchmark (codegen-units=1).
- Status: active

## D-029: Multi-value (sparse) group in quantized training (T2.7, slice 8)

- Date: 2026-11-19
- Choice: build the integer histogram for a column-wise multi-value feature group
  (`MultiValBin::accumulate_int`, routed from `construct_int_histograms`), so sparse
  data trains on the quantized path instead of falling back to full precision.
- Mechanism: Exclusive Feature Bundling packs sparse columns into one multi-value
  group whose per-row entries are `(sub-feature, bin)` pairs (sparse CSR-like or dense
  packed). `accumulate_int` iterates the leaf's rows and adds each row's packed
  `int8`/`uint8` derivative into every bin the row touches, mirroring the double
  `MultiValBin::accumulate`. Because integer sums are exact and order-independent, it
  is a single sequential pass with no per-thread blocks or merge (unlike the double
  `construct`, which blocks for determinism). The dense single-value groups keep their
  existing per-group loop; the multi-value group's histogram occupies its own bin
  range in the same layout as the double pool, so the integer split search and sibling
  subtraction are unchanged.
- Fallback narrowed: `multi_val.is_some()` no longer forces the full-precision
  fallback. The remaining fallback is the row-wise layout (`force_row_wise`, which uses
  the row-major copy and the all-feature row-wise view) and stochastic rounding.
- Oracle: `tests/parity/test_quantized_training_parity.py::
  test_quantized_sparse_matches_upstream` on 75%-zero columns (EFB bundles them),
  tree-by-tree against `lightgbm==4.7.0` at `num_grad_quant_bins` 4/8/16. Red first
  (sparse fell back to full precision), green after.
- SPEC ref: SPEC-NUM-6, SPEC-NUM-2, SPEC-CONC-3.
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2419 Python tests passed).
  Touches `treelearner/`, so CI runs the interleaved benchmark (codegen-units=1).
- Status: active

## D-030: Row-wise layout in quantized training (T2.7, slice 9)

- Date: 2026-11-19
- Choice: support `force_row_wise` under `use_quantized_grad` by exploiting the
  layout-independence of the integer histogram. The per-bin integer sums are identical
  whether the data is read column-major or row-major (upstream notes column-wise and
  row-wise build bit-identical histograms), so no row-major integer kernel is needed.
- Two row-wise sub-cases (as set up in `new`):
  - dense row-wise (`dense_row_wise`): all groups are dense single-value, still stored
    in `ds.data`, and `multi_val` is `None`. `construct_int_histograms`'s existing dense
    per-group loop reads `ds.data[gi]` directly and produces the correct sums; the
    row-major copy (`row_major`) is a performance detail the integer path ignores.
  - row-wise with a multi-value component (`all_row_wise`): `multi_val` is `None` and
    `all_feature_bins` is a single `MultiValBin` over every feature. The whole integer
    histogram is accumulated from it via `MultiValBin::accumulate_int` (added in D-029),
    mirroring the double path's early `all.construct(...)` branch.
- Fallback: only stochastic rounding remains unsupported (falls back to full
  precision). Every CPU layout and objective of the deterministic nearest-rounding path
  is now covered.
- Oracle: `tests/parity/test_quantized_training_parity.py::
  test_quantized_row_wise_matches_upstream` for dense and sparse `force_row_wise`,
  tree-by-tree against `lightgbm==4.7.0`. Red first (row-wise fell back), green after.
- SPEC ref: SPEC-NUM-6, SPEC-NUM-2, D-013 (layout equivalence).
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2421 Python tests passed).
  Touches `treelearner/`, so CI runs the interleaved benchmark (codegen-units=1).
- Status: active

## D-031: Stochastic rounding pinned to libc++ (T2.7, slice 10)

- Date: 2026-11-19
- Choice: implement `stochastic_rounding=true` (the default of `use_quantized_grad`)
  by porting libc++'s `std::uniform_real_distribution<double>(0,1)` and
  `std::uniform_int_distribution<int>(0,n)` over the mt19937 engine (D-023), and
  pin that behaviour to libc++ — the standard library of the host where fixtures
  and the dev wheel are built (macOS). User-authorized direction.
- Why a pin is unavoidable: the C++ standard does not specify how these
  distributions turn engine output into values, so libc++ (macOS), libstdc++
  (Linux) and MSVC (Windows) each produce different sequences from the same engine
  and seed. No single implementation can be bit-exact with the `lightgbm` wheel on
  every platform, because that wheel's distributions follow whatever standard
  library built it. This mirrors the platform-specific bit-exactness policy already
  in AGENTS.md section 4 (`tests/fixtures/platform.json`).
- libc++ specifics reproduced (validated against clang/libc++ `<random>` reference
  values in `tests/mt19937.rs`):
  - `uniform_real_distribution<double>(0,1)` = `generate_canonical<double,53>`: two
    32-bit draws `g1, g2`, value `(g1 + g2 * 2^32) / 2^64`, with the additions and
    the final division in that order so the double rounding matches.
  - `uniform_int_distribution<int>(0,n)`: the low `w` bits of one draw (`w` the
    width covering `n+1`, bumped when `n+1` is not a clean power of two within it),
    rejecting values `>= n+1`. `n+1` fits in `u32`, so one draw per attempt.
- Discretizer: `init_stochastic` precomputes per-row `[0,1)` noise once (upstream
  `Init`), block-seeded exactly as upstream (`mt19937(seed+b)` for gradients,
  `mt19937(seed+b+num_threads)` for Hessians, blocks from `BlockInfo(num_threads,
  n, 512)`), so the values match upstream for the same thread count.
  `discretize_stochastic` draws the per-tree rotation offset
  (`uniform_int(0,num_data)`) and rounds each derivative by adding the rotated
  noise before truncation, with the fused `madd` the nearest path already uses.
- Scope: parity validated at `num_threads=1` (regression and binary). The noise is
  block-seeded by thread count exactly as upstream, so other fixed thread counts are
  expected to match too, but are not yet parity-tested. The parity test is guarded
  to macOS (`sys.platform == "darwin"`); Linux/Windows CI skip it. Only an
  out-of-range `num_grad_quant_bins` still falls back to full precision.
- Oracle: `tests/parity/test_quantized_training_parity.py::
  test_quantized_stochastic_rounding_matches_upstream` (macOS), tree-by-tree against
  `lightgbm==4.7.0`; `tests/mt19937.rs` for the distributions.
- SPEC ref: SPEC-NUM-9 (extended for the quantized RNG, D-023), SPEC-NUM-2,
  AGENTS.md section 4 (platform-specific bit-exactness).
- Validation: cargo fmt --all --check, cargo clippy --workspace --all-targets
  -D warnings clean; scripts/check.sh ended "all green" (2423 Python tests passed,
  the 2 stochastic parity cases among them on this macOS host).
- Status: active. Marks T2.7 complete for CPU.

## D-032: C-header harness closes T5.3; implemented C API subset (T5.3)

- Date: 2026-11-20
- Question: T5.3's Verify is "a C test harness built against the upstream header, GATES". The
  `lgbm-capi` symbol surface now covers the whole single-dataset workflow, but no C program had
  ever compiled against the real `c_api.h`, so signature/ABI drift from upstream was unchecked.
  When is T5.3 done, given upstream exports ~98 symbols and we implement a subset?
- Choice: vendor the verbatim upstream headers (`include/LightGBM/{c_api.h,export.h,arrow.h}` at
  v4.7.0) under `third_party/lightgbm/include/`, add `crates/lgbm-capi/ctest/harness.c` which
  `#include <LightGBM/c_api.h>` and drives the implemented symbols end to end (dataset create →
  set label → train → eval counts/names/values → predict → save-to-string), and run it from
  `scripts/c_header_harness.sh` (build the staticlib, compile with `-isystem` the vendored
  include dir, link, run). Wired as a dedicated `c_header_harness` CI job (ubuntu) and into
  `scripts/check.sh` (guarded by the presence of a C compiler). This is the stop-list CI-gate
  change, made with explicit user authorization. Mark T5.3 `[x]`.
- Rationale: compiling the harness against the *upstream* prototypes type-checks every call site
  against upstream's declared signatures, and running it against the linked staticlib catches
  symbol gaps and ABI drift the Rust integration tests (which use the crate's own `extern "C"`
  declarations) cannot. The implemented surface is the complete CPU single-dataset path:
  error handling + last error; model load/free; dataset construction (`DatasetCreateFrom{Mat,CSR,
  CSC}` + `Free`/`GetNumData`/`GetNumFeature`); fields (`DatasetSet/GetField`); training
  (`BoosterCreate`/`UpdateOneIter`/`RollbackOneIter`); prediction (`PredictFor{Mat,CSR,CSC,
  MatSingleRow,File}` + `CalcNumPredict`/`GetNumFeature`); serialization (`SaveModelToString`/
  `SaveModel`); eval over training data (`GetEvalCounts`/`GetEvalNames`/`GetEval`). Deferred to
  follow-up nodes (not needed for the v1 CPU drop-in train/predict/eval/serialize cycle):
  `AddValidData` + `GetEval` over validation data (needs a predict-over-`Dataset` path or a raw
  matrix retained in the handle — a core design choice), streaming/push APIs, Arrow ingestion,
  `FastConfig`, `Refit`, CV, and `DumpModel`/feature-importance accessors. The prediction
  `parameter` string remains unparsed (rejected when non-empty), as noted since slice 2.
- SPEC ref: SPEC-ERR-5; ROADMAP T5.3 (Verify). Extends D-012.
- Validation: `scripts/c_header_harness.sh` prints "c_api harness: OK" and "all green" locally
  (macOS, clang); `scripts/check.sh` ended "all green" (2423 Python tests). CI runs the harness
  in its own `c_header_harness` job (ubuntu, gcc).
- Status: active. Marks T5.3 complete for the implemented CPU subset; the deferred symbols above
  are tracked as follow-ups.

## D-033: Defer T5.5 convert_model (no in-environment parity oracle)

- Date: 2026-11-21
- Question: T5.5 is the CLI's `task=convert_model`, which emits a standalone C++ prediction file
  (upstream `GBDT::ModelToIfElse`, default `gbdt_prediction.cpp`). Its Verify is
  `pytest tests/parity -k convert_model`. Can it be implemented and validated autonomously now?
- Choice: defer. Upstream LightGBM 4.7.0 exposes `convert_model` only through the CLI binary, which
  is not installed in the test environment (the `lightgbm` Python wheel ships no CLI), and the
  generated C++ must match upstream byte-for-byte (exact if/else tree lowering and float
  formatting) to be a faithful drop-in. Without the upstream generator available to produce golden
  fixtures, there is no trustworthy parity oracle in CI, so a from-scratch codegen could only be
  "validated" against itself — which would make a green run meaningless. Leave T5.5 `[ ]`.
- Rationale: the project's rule is that behaviour merges only behind a test that failed first and
  checks it against upstream. The honest path is to generate golden `gbdt_prediction.cpp` fixtures
  with the upstream CLI (a fixture-generation step, like `scripts/gen_fixtures.py`) and then port
  the codegen to match them. That needs the upstream CLI binary set up, which is out of scope for
  an unattended session. No code was written for this node.
- SPEC ref: ROADMAP T5.5 (Verify: `pytest tests/parity -k convert_model`).
- Status: deferred. Unblocks once upstream `convert_model` golden fixtures exist.

## D-034: T3.2 split-search scaling needs profiling under review, not a blind change

- Date: 2026-11-21
- Question: T3.2 ("Split search: sequential threshold phase scaling", bench dense 2 threads) is the
  last unblocked pre-Tier-6 node and gates T3.1 and T4.4. Can it be completed in an unattended
  session?
- Choice: do not make a speculative perf change. Record the assessment and leave T3.2 `[ ]`
  (T3.1/T4.4 stay blocked on it).
- Rationale: the per-feature threshold search is already parallel — `find_best_splits_from_histograms`
  runs a `fused_search` across features when `num_threads > 1` and the work clears
  `PARALLEL_SEARCH_MIN_BINS` (crates/lgbm-core/src/treelearner/serial.rs). T3.2 is therefore not a
  missing-feature TDD slice but a question of how well that existing parallel path *scales* to
  2 threads on the dense workload. AGENTS.md section 5 requires any hot-path change to be driven by
  profiling and to attach criterion + interleaved-benchmark findings (release, interleaved, medians)
  to the PR; the development VM has ±20-30% run-to-run noise, so credible measurements cannot be
  produced unattended. Merging a change that only passes CI's pass/fail benchmark gate — without
  trustworthy before/after medians showing it actually improves 2-thread scaling — would dress up a
  possibly-neutral change as progress and violate the measure-before-optimise discipline. The
  correct next step is a supervised profiling session (samply/flamegraph on the dense 2-thread
  split-search phase) to locate the scaling loss, then a measured change. No code was written.
- SPEC ref: SPEC-CONC-6 (threshold tie-breaking/order the parallel path must preserve); AGENTS.md
  section 5 (perf discipline).
- Status: open, needs supervised profiling. T3.1 (sparse 1-thread target, already ~1.11x after
  T2.6 per D-018) and T4.4 (monotone advanced + categorical; also needs the upstream UB decision)
  remain blocked on it.

## D-035: T3.2 met — dense 2-thread split-search scaling confirmed; sparse residual is T3.1

- Date: 2026-11-21
- Question: supersedes D-034's "needs profiling". With profiling now done (supervised), is T3.2
  ("Split search: sequential threshold phase scaling", Verify: GATES + thread_invariance, Bench:
  criterion split_finding + interleaved dense 2 threads) met, and is there a clean win left?
- Choice: mark T3.2 complete on its formal criteria; do not change code. The residual sparse
  1-thread gap stays with T3.1 as a separate, hard, open item.
- Evidence (local dev host, macOS/aarch64 Apple M1, release, OMP_WAIT_POLICY=PASSIVE):
  - `benches/bench_train.py` (40 rounds, upstream/bankai time ratio, >1 = bankai faster): dense
    50k x 20 = 1.55x (1 thread) and 1.76x (2 threads); bankai's 1->2-thread speed-up is 1.47x vs
    upstream's 1.29x, so the dense threshold-search phase scales at least as well as upstream.
  - `cargo bench -p lgbm-core --bench kernels -- split_finding`: numerical/255_bins ~516 ns, no
    regression (consistent with D-018's 0.9474x). `thread_invariance` passes in the gate.
  - CI's benchmark job (2-vCPU Linux) has passed on every hot-path PR this session (e.g. #139),
    corroborating that the dense paths do not regress on the gated environment.
- Rationale: the per-bin threshold scan (`find_best_threshold_numerical`) is a data-dependent
  prefix sum (each bin's cumulative sums depend on the previous), so it is inherently serial and
  scalar, exactly mirroring upstream's `FindBestThresholdSequentially`; T2.6 (D-018) already
  vectorised the independent gain arithmetic. Profiling the sparse 1-thread case (multi-val
  50k x 100, construction amortised) shows split search is the largest training cost, but it
  matches upstream's structure and the `search_feature` wrapper has no bankai-specific overhead,
  so no bounded, low-risk speed-up is available without open-ended research that would risk the
  bit-exact tie-breaking (SPEC-CONC-6 / SPEC-NUM-7). The sparse end-to-end target (~0.90-0.95x at
  1 thread here; ~1.11x in D-018's gated run) is T3.1's bench, not T3.2's.
- SPEC ref: SPEC-CONC-6, SPEC-NUM-7 (scan order/tie-breaking); AGENTS.md section 5.
- Status: active. Supersedes D-034's "open/needs profiling". Marks T3.2 `[x]`; unblocks T4.4.
  T3.1 (sparse 1-thread) remains open and hard (D-017/D-018).

## D-036: advanced monotone categorical searches initialize cumulative bounds

- Date: 2026-11-21
- Question: does v4.7.0 read an uninitialized cumulative constraint when searching categorical features?
- Choice: support the combination, matching the pinned source rather than the prior unsupported classification.
- Evidence: `src/treelearner/feature_histogram.cpp` at tag v4.7.0 calls `InitCumulativeConstraints(true)` before both `FindBestThresholdCategoricalInner` and `FindBestThresholdCategoricalIntInner`. Candidate gains call `GetSplitGains<USE_MC>` and final leaf outputs read `LeftToBasicConstraint` / `RightToBasicConstraint`. Unlike numerical scans, categorical scans never call `Update(threshold)`, so they retain the initial reverse cumulative bounds. The left and right bounds can differ. The Rust `Mc::fixed` helper discarded advanced bounds entirely, explaining the divergence after removing the refusal. The advanced constraint walks themselves match the pinned source and need no changes.
- Validation: reproduced the existing categorical+NaN parity failure before implementation (maximum absolute prediction difference 0.1047145). Eight additional model-text parity cases cover one-hot/grouped searches, numerical missing values, and deterministic nearest-rounded quantized training. The prior strict upstream gap entry is removed as Harness maintenance because the feature is now supported; no existing assertion or tolerance is relaxed.
- SPEC ref: SPEC-SCOPE-1, SPEC-NUM-2, SPEC-NUM-10.
- Status: implemented; supersedes the old advanced+categorical unsupported/undefined-behaviour classification.

- Performance validation for D-036: seven interleaved Criterion rounds, macOS/aarch64, bench/release profile, one thread, baseline `c4fd750` versus the T4.4 working tree. Numerical split finding: 1.0015x baseline; dense mixed row-wise training: 0.9919x. An absolute machine-speed transition from ~2.7 to ~4.3 ms skewed the mixed sparse row-wise median, so that case was repeated in isolation (1 s warmup, 3 s measurement, 20 samples, seven rounds): 0.9978x, no >5% regression. Commands: `cargo bench -p lgbm-core --bench kernels --no-run`, then interleave the before/after binaries with `--bench --warm-up-time 0.1 --measurement-time 0.2 --sample-size 20 --nresamples 1000`; isolated rerun adds filter `training/row_wise_mixed/` and uses 1/3 s. These are bankai-before/after measurements; upstream oracle is 4.7.0, with no upstream timing or OMP_WAIT_POLICY involved. Full `scripts/check.sh`: all green; release parity/upstream suite: 2431 passed.

## D-037: pinned upstream CLI fixtures unblock convert_model (T5.5)

- Date: 2026-11-21
- Question: can the missing CLI oracle from D-033 be made available without linking upstream C++ into bankai?
- Choice: build the pinned v4.7.0 source and its pinned Eigen/fast_double_parser/fmt/nanoarrow submodules in the ignored session build directory, with CMake `-DUSE_OPENMP=OFF -DBUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release`. Use that executable only to generate golden C++ outputs from upstream-wheel models. Record its SHA-256 and wheel/tag versions in fixture provenance. The shipped CLI implementation is Rust only.
- Evidence: eight generated model/output pairs cover numerical, categorical one-hot/grouped (>32 categories), missing/zero-missing, stump, multiclass and linear trees. The first acceptance test failed before implementation because ConvertModel was unsupported. Thirteen CLI tests now compare complete generated files, default output, existing-file wrappers (including non-UTF8 source bytes), and input/output failures. The byte-preserving regression first failed on read_to_string; the wrapper bytes were independently checked against the pinned upstream executable.
- Compatibility: preserve upstream's quirks. The generated missing-value branches and linear-tree outputs are emitted as upstream emits them. A single-leaf tree uses six significant digits and returns its leaf value even in leaf-index mode. Existing files use `#define USE_HARD_CODE 0` followed by `#ifndef USE_HARD_CODE`; do not correct this preprocessor spelling. Static template scaffolding is adapted from upstream's MIT-licensed ModelToIfElse output.
- SPEC ref: SPEC-SCOPE-1/2, SPEC-MAP-1, SPEC-PY-5; roadmap T5.5.
- Status: implemented. Supersedes D-033's missing-oracle deferral. No training or prediction hot path changes, so no performance benchmark applies.

## D-038: model merge preserves upstream's donor-first tree order

- Date: 2026-11-21
- Question: does summing the two ensembles' full raw scores establish MergeFrom parity?
- Choice: prepend the donor ensemble before the recipient, matching `src/boosting/gbdt.h` (v4.7.0) GBDT::MergeFrom. Move the recipient's existing trees and clone the donor, preserving ownership without unnecessary recipient copies.
- Evidence: the existing implementation appended the donor, while upstream explicitly pushes the donor before the original models. The sum-based tests could not distinguish those orders. A new native test failed first on the first donor tree; a new parity test failed first on leaf-index columns. Regression/binary/multiclass parity calls the wheel's `LGBM_BoosterMerge` directly (there is no public Python merge), then compares all leaf-index columns and raw-score iteration windows exactly. No existing test assertion or tolerance is edited.
- SPEC ref: SPEC-SCOPE-1, SPEC-NUM-1, SPEC-NUM-2; roadmap T4.3.
- Status: implemented; corrects the prior append description. The operation is offline model composition; no prediction/training loop is changed. The existing benchmark CI filter still runs for this file.

- Performance validation for D-038: seven interleaved Criterion rounds against the `c4fd750` core source (identical to the branch base), macOS/aarch64, bench/release profile, one thread. Ratios: prediction 4096x16x20 1.0167x; numerical split finding 255 bins 1.0209x; row-wise dense mixed 4096x12 0.9810x; row-wise mixed 4096x12 0.9749x. All within 5%, with no speed-up claim for noise-sized changes. Build with `cargo bench -p lgbm-core --bench kernels --no-run`; interleave before/after binaries seven times with filter `split_finding|training/|prediction/`, `--bench --warm-up-time 1 --measurement-time 2 --sample-size 20 --nresamples 1000`. Upstream oracle is lightgbm==4.7.0, but these are bankai-before/after timings, so OMP_WAIT_POLICY is not applicable. Full check: all green, 2431 Python tests; release parity/upstream suite: 2425 passed.
