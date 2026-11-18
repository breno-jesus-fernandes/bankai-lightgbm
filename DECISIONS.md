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
