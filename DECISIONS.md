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
