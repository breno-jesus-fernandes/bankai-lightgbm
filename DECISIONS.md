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
