# ROADMAP

Target: feature parity with LightGBM **v4.7.0** (see `UPSTREAM_VERSION`). Every phase ends only when its
**parity gate** and **bench gate** pass. Follow the TDD rules in `AGENTS.md`.

## P0: Foundations
- [x] Cargo workspace + crate skeleton (`lgbm-core`, `lgbm-io`, `lgbm-cli`, `lgbm-capi`, `lgbm-python`)
- [x] maturin/PyO3 package `bankai_lightgbm` (Python >= 3.11)
- [x] CI: fmt, clippy, test, Python 3.11–3.13 on ubuntu/macos/windows
- [x] Fixture generator `scripts/gen_fixtures.py` (upstream 4.7.0)
- [x] Parity harness helpers (`tests/parity/conftest.py`, `crates/lgbm-core/tests/support`), platform-aware bit-exactness
- [x] Python benchmark vs upstream (`benches/bench_predict.py`, non-zero exit if slower)
- [ ] criterion micro-benchmarks + regression gate in CI
- [ ] `cargo miri` job for the unsafe-allowlisted modules
- [ ] Parameter table code generator from upstream `config.h`

## P1: Config + model I/O + prediction
- [ ] All params + aliases + validation (`Config::from_map`, `config_auto` equivalent)
- [x] Model text parser and writer (`gbdt_model_text.cpp`), byte-identical round-trip
- [x] `Tree` (numerical, categorical, and linear leaves), `dump_model` JSON
- [x] Predict: normal, `raw_score`, `pred_leaf`, `pred_contrib` (TreeSHAP), `start_iteration`/`num_iteration`
- [x] Prediction early stopping (`pred_early_stop*`)
- [x] Python `Booster`: load/predict/save/dump/importance/pickle (dense numpy, lists, pandas)
- [ ] Sparse (CSR/CSC) prediction input, sparse `pred_contrib` output
- [ ] Predict from file (`data` as path) — needs P2 parsers
- **Gate:** every golden model predicts bit-exactly ✅ (24 fixtures; 1.25–1.33x faster predict, 1.05x SHAP)

## P2: Dataset + binning
- [ ] `BinMapper`: numerical/categorical, `max_bin`, `max_bin_by_feature`, `min_data_in_bin`, `bin_construct_sample_cnt`
- [ ] Missing handling: `use_missing`, `zero_as_missing`, NaN/zero bins
- [ ] Dense, 4-bit dense, and sparse bins; multi-val dense/sparse bins
- [ ] Exclusive Feature Bundling (`enable_bundle`, `max_conflict_rate`)
- [ ] Metadata: label, weight, group/query, init_score, position
- [ ] Parsers: CSV/TSV/LibSVM, header, `label_column`, `weight_column`, `group_column`, `ignore_column`, `two_round`
- [ ] Binary dataset file (`save_binary`) compatible with upstream
- [ ] `forcedbins_filename`, `linear_tree` raw-data retention, Arrow ingest
- **Gate:** identical bin upper bounds and feature groups

## P3: Serial tree learner + GBDT
- [ ] Histogram construction: col-wise and row-wise, `force_col_wise`/`force_row_wise` auto-choice, SIMD
- [ ] Histogram pool + subtraction trick, `histogram_pool_size`
- [ ] Leaf-wise growth, `num_leaves`, `max_depth`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`
- [ ] Regularization: `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_delta_step`, `path_smooth`
- [ ] Categorical splits (`max_cat_threshold`, `cat_l2`, `cat_smooth`, `max_cat_to_onehot`, `min_data_per_group`)
- [ ] `feature_fraction`, `feature_fraction_bynode`, `bagging_fraction`/`bagging_freq`, pos/neg bagging, `extra_trees`
- [ ] Monotone constraints: basic, intermediate, advanced; `monotone_penalty`
- [ ] `interaction_constraints`, `forcedsplits_filename`, CEGB (`cegb_*`)
- [ ] `linear_tree` (`linear_lambda`)
- [ ] Quantized training (`use_quantized_grad`, `num_grad_quant_bins`, ...)
- [ ] `refit`, `deterministic`
- **Gate:** identical trees on fixed seeds; training as fast as upstream

## P4: Objectives + metrics
- [ ] Regression: `l2`, `l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`
- [ ] `binary`, `multiclass`, `multiclassova`, `cross_entropy`, `cross_entropy_lambda`
- [ ] Ranking: `lambdarank` (incl. position bias), `rank_xendcg`
- [ ] Custom objective (fobj) path
- [ ] All metrics: l1/l2/rmse/quantile/huber/fair/poisson/mape/gamma/gamma_deviance/tweedie, ndcg@k, map@k, auc, average_precision, binary_logloss/error, auc_mu, multi_logloss/error (`multi_error_top_k`), cross_entropy, cross_entropy_lambda, kullback_leibler

## P5: Boosting variants + training loop
- [ ] DART (`drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `uniform_drop`)
- [ ] RF, GOSS (`data_sample_strategy=goss`)
- [ ] Early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`)
- [ ] Continued training (`init_model`), `init_score`
- [ ] Model merge/shuffle, `set_leaf_output`, `rollback_one_iter`

## P6: Python package (drop-in)
- [ ] `Dataset` (numpy, pandas incl. categoricals, scipy.sparse, pyarrow, files, references, subsets)
- [ ] `Booster` (full method surface), `train`, `cv`
- [ ] Callbacks: `early_stopping`, `log_evaluation`, `record_evaluation`, `reset_parameter`
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
