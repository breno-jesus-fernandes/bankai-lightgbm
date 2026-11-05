# ROADMAP

Target: feature parity with LightGBM **v4.7.0** (see `UPSTREAM_VERSION`). Every phase ends only when its
**parity gate** and **bench gate** pass. Follow the TDD rules in `AGENTS.md`.

## Implementation order

Phase numbers group features; they do not define implementation order. The work now runs in
three steps, in this order. Each step ends only when its gate passes. Remaining items (below
the steps) start after step 3.

### Step 1: Python surface (P6)
Goal: `import bankai_lightgbm as lgb` works for the usual sklearn-style workflow.
- [ ] sklearn estimators: `LGBMModel`, `LGBMClassifier`, `LGBMRegressor`, `LGBMRanker`
      (names, signatures, defaults, `fit`/`predict`/`predict_proba`, `LightGBMError` behavior as upstream)
      Ported and parity-tested (`tests/parity/test_sklearn_parity.py`): regression, binary and multiclass classification,
      ranking, `eval_set` with early stopping, `feature_importances_`, pandas feature names (also in the model text).
      Still missing: `LightGBMError` cases beyond the input-length check, and `set_params`/`get_params` edge cases.
- [x] Plotting helpers: `plot_importance`, `plot_split_value_histogram`, `plot_metric`, `plot_tree`, `create_tree_digraph`
- [ ] Port upstream `tests/python_package_test` and run it green against `bankai_lightgbm`
      Done: all ported files in `tests/python_package_test` (`test_plotting`, `test_sklearn`, `test_basic`,
      `test_engine`, `test_callback`, `test_utilities`, `test_consistency`, `test_pandas`, `test_arrow`, `test_polars`,
      `test_dual`), run unmodified except imports. Upstream cases that still fail are strict `xfail`s listed in
      `tests/known_upstream_gaps.tsv` with the owning ROADMAP item (51 node ids; a fix turns them into
      failing XPASS, so the line is removed with the fix). `test_dask.py` belongs to P7 (distributed).
      Arrow/polars ports pass in full (`Dataset._dump_text` is byte-identical to upstream). `get_data` on pyarrow/polars subsets now keeps the table type. Done: all-ones weights become `None`, as upstream; arrow/polars column names feed categorical features; pyarrow/polars predict densely.
      Open work from the list: the "not triaged" entries (RF value mismatch, pandas dtypes, early-stopping
      naming, categorical handle).
- Gap order: (1) the smaller Python-surface gaps ("not triaged" P6 entries, pandas dtypes, Dataset and Booster surface, early stopping, callbacks, zero-copy input, log callback); (2) the large features last: linear_tree, refit, cv/CVBooster, forced splits, monotone beyond basic, CEGB, continued training, RF, bagging_by_query, position bias.
- [x] `Dataset.feature_num_bin` (index or name), matching upstream's `LGBM_DatasetGetFeatureNumBin`
- Gate: the ported upstream suite passes for everything the step names; unsupported parts raise `NotImplementedError`.

### Step 2: Text input and file prediction (P2/P1)
Goal: load CSV/TSV/LibSVM files and predict from a file path, as upstream does.
- [ ] Parsers: CSV/TSV/LibSVM, header, `label_column`, `weight_column`, `group_column`, `ignore_column`, `two_round`
- [ ] Predict from file (`data` as path)
- Gate: parity tests against `lightgbm==4.7.0` on the same files (model text and predictions match as the parity tolerances in `AGENTS.md`).

### Step 3: Speed on sparse and dense data (P3/P2)
Goal: training with `num_threads` 1 and 2 is at least as fast as upstream on both sparse and dense data.
Start with profiling evidence, then change one thing at a time with interleaved benchmarks.
- [ ] Row-wise training layout (`force_row_wise`; upstream `GetMultiBinFromAllFeatures`) and layout auto-choice
- [ ] Multi-value histogram: per-call fixed cost and block merge
- [ ] Data partition: sparse path and thread scaling
- [ ] Split search: sequential threshold phase scaling
- Gate: interleaved benchmark against `lightgbm==4.7.0` (dense and sparse, 1 and 2 threads, medians over at least 7 rounds)
  shows bankai at or below upstream within the 3% noise threshold; models stay identical to the previous commit.

### After step 3
The remaining features follow in the order of the sections below: P1–P5 items not yet done,
then P7 (CLI and C API). P8 (distributed and GPU) starts only after CPU parity and the gates above pass.

For each feature, first run a failing upstream parity/regression test, implement it, update
this roadmap and the changelog, and run `scripts/check.sh` before committing and pushing.
Hot-path changes also require profile evidence and before/after Criterion results. Keep a
checkbox open until every behavior it names and its applicable performance gates pass.

## P0: Foundations
- [x] Cargo workspace + crate skeleton (`lgbm-core`, `lgbm-io`, `lgbm-cli`, `lgbm-capi`, `lgbm-python`)
- [x] maturin/PyO3 package `bankai_lightgbm` (Python >= 3.11)
- [x] CI: fmt, clippy, test, Python 3.11–3.13 on ubuntu/macos/windows
- [x] Fixture generator `scripts/gen_fixtures.py` (upstream 4.7.0)
- [x] Parity harness helpers (`tests/parity/conftest.py`, `crates/lgbm-core/tests/support`), platform-aware bit-exactness
- [x] Python benchmark vs upstream (`benches/bench_predict.py`, non-zero exit if slower)
- [x] Criterion micro-benchmarks and CI regression gate (`crates/lgbm-core/benches/kernels.rs`: histogram construction, partition, split finding, prediction; CI compares shared results to the base revision and fails above 5%)
- [x] `cargo miri` CI job for the unsafe-allowlisted histogram module (`scripts/miri.sh`)
- [x] Parameter table code generator from upstream `config.h` (`scripts/gen_config.py`)

## P1: Config + model I/O + prediction
- [x] All params + aliases + validation (`Config::from_params`, `str2map`, conflicts); parameters block round-trips byte-identically
- [x] Model text parser and writer (`gbdt_model_text.cpp`), byte-identical round-trip
- [x] `Tree` (numerical, categorical, and linear leaves), `dump_model` JSON
- [x] Predict: normal, `raw_score`, `pred_leaf`, `pred_contrib` (TreeSHAP), `start_iteration`/`num_iteration`
- [x] Prediction early stopping (`pred_early_stop*`)
- [x] Python `Booster`: load/predict/save/dump/importance/pickle (dense numpy, lists, pandas)
- [x] Sparse (CSR/CSC) prediction input, sparse `pred_contrib` output (bit-exact vs upstream: per-row buffer like `CopyToPredictBuffer`, duplicates last-wins, other scipy formats converted with upstream's warning; float32 input gives float32 SHAP output; sparse path matches dense-path speed, see P3 bench note)
- [ ] Predict from file (`data` as path) — needs P2 parsers
- **Gate:** every golden model predicts bit-exactly ✅ (24 fixtures; 1.25–1.33x faster predict, 1.05x SHAP)
- Bench note: on linux/x86_64 (2-vCPU VM) dense predict measured 0.90–0.99x of upstream (both libraries' sparse/dense ratios match); profile the tree traversal on x86 before claiming the gate there.

## P2: Dataset + binning
- [x] `BinMapper`: numerical/categorical, `max_bin`, `max_bin_by_feature`, `min_data_in_bin`, `bin_construct_sample_cnt`
- [x] Missing handling: `use_missing`, `zero_as_missing`, NaN/zero bins
- [x] Dense, 4-bit dense, and sparse bins; multi-val (per-feature) bins
- [x] `MultiValBin` for col-wise training: the multi-value group's histograms come from row-wise sparse/dense multi-value bins with upstream's offsets, kernels, data blocks and merge order (byte-identical model text vs upstream across objectives, sampling, constraints, categorical, multiclass, ranking, CSR/CSC/dense input, and with 2 threads; `tests/parity/test_multi_val_training_parity.py`). Sparse multi-value columns are partitioned with a forward `SparseBin` cursor, without densifying
- [ ] Row-wise training layout (`force_row_wise`; upstream `GetMultiBinFromAllFeatures`)
- [x] Exclusive Feature Bundling (`enable_bundle`), sampling identical to upstream
- [ ] Metadata: label, weight, and group/query implemented; weight/query length errors and nested subsets covered; position metadata done (set/get, remapped int32 ids, subsets; ranking position bias pending P4); init_score done (class-major storage, training, validation metrics, subsets)
- [x] Sparse CSR/CSC input (`LGBM_DatasetCreateFromCSR`/`FromCSC`): sampling, push order (stored pairs then `FinishOneRow` zero pushes), reference datasets and linear-tree raw values, O(nnz) without densifying; `save_binary` byte-identical to upstream incl. bundles, unsorted indices and multi-value groups. Known gap: a duplicated column within a row that lands in a *sparse* multi-value column (upstream keeps whichever entry its unstable `std::sort` leaves first, which is stdlib-dependent; bankai keeps the first). Construction speed (`benches/bench_dataset.py`, linux/x86_64 2 vCPU): CSR 0.8–1.05x, CSC 0.75–4x upstream; dense construction measured 0.32–0.82x (pre-existing, profile `value_to_bin`/`find_groups`)
- [ ] Parsers: CSV/TSV/LibSVM, header, `label_column`, `weight_column`, `group_column`, `ignore_column`, `two_round`
- [x] Binary dataset file (`save_binary`) byte-identical to upstream (20 fixtures); reader for upstream files
- [ ] Load a `Dataset` from an upstream `.bin` file (decode bin data)
- [ ] `forcedbins_filename`, `linear_tree` raw-data retention, Arrow ingest
- **Gate:** identical bin upper bounds and feature groups ✅

## P3: Serial tree learner + GBDT
- [ ] Histogram construction: col-wise and row-wise, `force_col_wise`/`force_row_wise` auto-choice, SIMD
- [ ] Histogram pool + subtraction trick, `histogram_pool_size`
- [x] Leaf-wise growth, `num_leaves`, `max_depth`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`
- [x] Regularization: `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_delta_step`, `path_smooth`
- [x] Categorical splits (`max_cat_threshold`, `cat_l2`, `cat_smooth`, `max_cat_to_onehot`, `min_data_per_group`; byte-identical, one-hot and sorted search, NaN categories, bagging; monotone + categorical still unsupported)
- [ ] `feature_fraction`, `feature_fraction_bynode`, `bagging_fraction`/`bagging_freq`, pos/neg bagging (done, byte-identical; `extra_trees` byte-identical across seeds; `bagging_by_query` still missing)
- [ ] Monotone constraints: basic (done, byte-identical, with `monotone_penalty`); intermediate and advanced still missing
- [x] `interaction_constraints` (byte-identical; by-tree and by-node sampling)
- [ ] `forcedsplits_filename`, CEGB (`cegb_*`)
- [ ] `linear_tree` (`linear_lambda`); training currently refuses `linear_tree=true` with an explicit error
- [ ] Quantized training (`use_quantized_grad`, `num_grad_quant_bins`, ...)
- [ ] `refit`, `deterministic`
- **Gate:** identical trees on fixed seeds; training as fast as upstream
- Bench note (`benches/bench_train.py`, linux/x86_64 2 vCPU, training only, 50 rounds): 1 thread at parity (dense 1.04x, multi-value group 0.95–0.97x); no thread scaling yet (0.47–0.68x at 2 threads): split finding, sparse partition and single-group histograms run serially, while upstream parallelizes them. Benchmark a release build (`maturin develop -r`); `scripts/check.sh` installs a debug build.
- Thread scaling (open): upstream's OpenMP threads spin between regions, rayon's sleep. Rayon fork/join after serial work cost 15-30 us here, which cancelled the gain of the parallel split search and partition (reverted). `utils::spin_pool` (allowlisted module, safe helpers for the learner) runs search, partition and dense histograms on spinning workers, and the sparse partition seeks each block's cursor through the fast index. Interleaved medians, 50 rounds, 2-vCPU VM: dense 2-thread scaling 1.18x (upstream 1.46x), multi-value 1.21x (upstream 1.68x); one thread at 0.85-1.0x of upstream. Open: the multi-value histogram and merge still scale poorly, and per-thread throughput drops when both threads run (the dense histogram kernel alone scales 1.49x).

## P4: Objectives + metrics
- [x] Regression: `l2`, `l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie` (byte-identical training, weighted and unweighted)
- [x] `binary`, `multiclass`, `multiclassova`, `cross_entropy`, `cross_entropy_lambda` (weighted and unweighted)
- [x] Ranking: `lambdarank`, `rank_xendcg` (byte-identical training, weighted and unweighted)
- [ ] Ranking: unbiased lambdarank position bias (`position` field, `lambdarank_position_bias_regularization`)
- [x] Custom evaluation (`feval` in `train`/`cv`) and custom objectives (`fobj`/callable aliases): native derivatives, regression/multiclass and CV parity, invalid dimensions/lengths, raw-score evaluation, GBDT/DART/GOSS switching, upstream-compatible RF rejection.
- [x] All metrics: l1/l2/rmse/quantile/huber/fair/poisson/mape/gamma/gamma_deviance/tweedie/r2, ndcg@k, map@k, auc, average_precision, binary_logloss/error, auc_mu, multi_logloss/error (`multi_error_top_k`), cross_entropy, cross_entropy_lambda, kullback_leibler (bit-exact vs upstream on 24 fixtures; wiring into training/early stopping is P5)

## P5: Boosting variants + training loop
- [x] DART (`drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `uniform_drop`; byte-identical, 4 fixtures; the drop replays recorded splits on all rows; custom objectives see the score after the iteration's drop, as `DART::GetTrainingScore`. Upstream never initializes `is_update_score_cur_iter_`, so with a custom objective its first drop draw is undefined behaviour; bankai uses `false` and reproduces upstream's majority outcome, and parity tests use RNG-independent drop settings)
- [x] RF (`boosting=rf`, byte-identical; known gap: a first tree with no split is scored on out-of-bag rows only, matching upstream's non-subset path, the subset path is not modelled)
- [x] GOSS (`data_sample_strategy=goss`, byte-identical; both the subset and dense regimes)
- [x] Early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`; validation metrics and callback parity)
- [ ] Continued training: `init_model` ✅ for gbdt/multiclass (file, Booster, validation and callbacks; DART rejected for now); `init_score` ✅ (byte-identical training, multiclass, validation metrics, subsets; RF rejects it like upstream)
- [ ] Model merge; `shuffle_models` ✅ (upstream order with Random(17)); `set_leaf_output` ✅ (edits the model and continued training's trees); `rollback_one_iter` ✅ (gbdt, bagging and GOSS byte-identical scores; rejected for RF/DART and init_model trees)

## P6: Python package (drop-in)
- [ ] `Dataset` (done: dense numpy and pandas incl. categoricals, weight, group, categorical_feature, subsets with source bin mapper reuse; scipy.sparse CSR/CSC (other formats converted like upstream) ✅, `save_binary` ✅ (never overwrites, like upstream); pending: pyarrow, files, complete subset compatibility (unsorted/duplicate indices, weighted query metadata, and empty/out-of-range index errors covered), `get_weight`/`get_group`/`get_init_score`/`set_init_score` ✅; `set_label`/`set_weight`/`set_group` ✅ (also on constructed datasets); remaining `set_field`/`get_field`, `set_position` and other setters; dense shared reference binning implemented)
- [ ] `Booster` (training works: `lgb.train` gives byte-identical model text; Rust `GbdtTrainer` and Python `Booster.update()` support incremental training; `cv` supports seeded/custom folds, metric aggregation, custom evaluation, callbacks, and `CVBooster`; pending: `cv` preprocessing/continued training and the remaining Booster methods)
- [ ] Callbacks: `early_stopping`, `log_evaluation`, `record_evaluation` ✅; `reset_parameter` supports learning-rate schedules; pending resets for other parameters
- [ ] sklearn API: `LGBMModel`, `LGBMClassifier`, `LGBMRegressor`, `LGBMRanker`
- [ ] Plotting: `plot_importance`, `plot_split_value_histogram`, `plot_metric`, `plot_tree`, `create_tree_digraph`
- [ ] Port upstream `tests/python_package_test` and run it green

## P7: CLI + C API
- [ ] `lightgbm` CLI: `task=train|predict|convert_model|refit|save_binary`, config files
- [ ] `convert_model` (C++ codegen output)
- [ ] `lgbm-capi`: `LGBM_*` symbols compatible with `c_api.h`

## P8 (post-1.0): Distributed + GPU
- [ ] Network layer: socket + MPI; `tree_learner=data|feature|voting`
- [ ] Dask integration
- [ ] CUDA (`device_type=cuda`) via `cudarc`, behind the `cuda` feature
- [ ] OpenCL (`device_type=gpu`) / ROCm
