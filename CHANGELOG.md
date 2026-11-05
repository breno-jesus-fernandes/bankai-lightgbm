# Changelog

## Unreleased

- `Dataset.subset` of a Dataset built from `Sequence` objects reads the selected rows from the sources, instead of indexing the list of sources.

- `Booster.valid_sets` holds the validation Datasets and `name_valid_sets` their names, as upstream does. `Booster.eval` on a Dataset that is not registered adds it as a validation set, and a Booster evaluates before its first iteration (the native training handle is created on demand).

- `CVBooster` can be pickled (`pickle`, `joblib`, `cloudpickle`) and loaded with `CVBooster(model_file=...)`, as upstream does. A pickled Booster drops its training Dataset, which holds native handles, so a `cv(return_cvbooster=True)` result can be saved.

- `Booster.trees_to_dataframe` returns one row per node of the fitted trees, with upstream's columns and node names (`<tree>-S<split>`, `<tree>-L<leaf>`), built from `dump_model()`.

- A Booster loaded from a model file or string types each `parameters:` value by upstream's `Config::ParameterTypes` (new `param_types()` binding): `bool` values come back as `True`/`False`, `int` and `double` as numbers, and vectors as lists.

- Changing a constructed Dataset's binning parameters (`max_bin`, `min_data_in_bin`, `categorical_feature`, `feature_pre_filter`, ...) through `train`/`cv` raises upstream's "Cannot change ... after constructed Dataset handle." error, and reducing `min_data_in_leaf` with `feature_pre_filter=true` is rejected. The check is native (`lgbm-core` `check_dataset_reset_config`, port of upstream `CheckDatasetResetConfig`) and runs when the Dataset's `free_raw_data=True` (upstream frees the raw data then); bankai keeps raw data, so the condition stands in for upstream's freed state.

- Before construction, `Dataset.get_label`, `get_weight`, `get_init_score`, `get_group` and `get_position` return the values as provided, and `get_field` raises `Cannot get <field> before construct Dataset`, as upstream does. After construction, label, weight, init score and group are read back from the native Dataset (NaN becomes 0, infinities are clamped, group becomes int32 sizes), so the Python attributes match what training uses.

- A validation Dataset built with `reference=` takes the reference's parameters when it is constructed, as upstream does. A warning "Overriding the parameters from Reference Dataset." is emitted when they differ, except when only the categorical-feature settings differ.

- `Booster.eval_train` returns the training data's metrics as `(dataset_name, metric_name, value, is_higher_better)` tuples, as upstream does. It raises `LightGBMError` when the Booster has no training Dataset (for example after `lgb.train` frees it).

- `Booster(model_file=...)` and `Booster(model_str=...)` take their parameters from the model's `parameters:` block, skipping empty values and nested lists typed as upstream does. An unknown name is dropped with upstream's warning, and passing `params` with a model warns that it is ignored.

- A Dataset records the categorical feature indices it resolved (including pandas category columns) in `params["categorical_column"]`, as upstream does.

- Pandas input with a column of an unsupported dtype (string, timezone-aware datetime, period or interval) raises upstream's `ValueError` naming the bad columns, after categorical columns are converted to codes.

- `Dataset` accepts a list of 2-D NumPy matrices and concatenates them by rows, as upstream does; the result is identical to one matrix holding the same rows.

- `train` resolves `early_stopping_round` from the global parameters (and its aliases) before the Booster is built, drops it when it is `None`, and rejects a non-integer value with upstream's `TypeError`. A single string passed as `valid_names` names the one validation set.

- `Dataset.init_score` reads back the value from the constructed Dataset, so a pandas DataFrame passed as init score is returned as a NumPy array, as upstream does.

- Validation sets are binned with the training set's bin mappers, as upstream does: `train` calls `Dataset.set_reference` on each validation set (new method). Before, each validation set was binned from its own data, which repeated the native "no meaningful features" warning and gave it different bins from the training set.

- Training no longer prints the native "no meaningful features" warning twice for a run with a validation set built without a reference.

- Training logs the same Info lines as upstream for a binary objective and at tree-learner start: class counts, `Total Bins`, the training data-point count, and `[binary:BoostFromScore]` with the average and init score.

- `log_evaluation` prints its `[i]` lines by default and sends them to a logger registered with `register_logger`, as upstream does. Before, bankai wrote them to the stdlib logger, so by default nothing was printed.

- `train` runs every requested round even after an iteration finds no splits, evaluating the validation sets on each round, as upstream does. Before, training stopped at the first such round and the validation history was shorter.

- `cv` no longer leaves its parameters on the Dataset after it returns, so a later `cv` with other metrics reports only its own metrics, as upstream does.

- `Booster.eval` accepts `feval` (one callable or a list), adding the custom metrics to the built-in ones as upstream does. `eval_valid` shares the same code.

- A parameter given to `train` is no longer kept on the training Dataset after the call, so a later call on the same Dataset does not inherit it (for example GOSS followed by bagging). Upstream restores the Dataset's parameters the same way.

- `train` and `cv` raise upstream's `TypeError` messages when `train_set` is not a Dataset, and when an item of `valid_sets` is not a Dataset (naming the item index and its type).

- `categorical_feature` given through a parameter or its alias (for example `categorical_column`) is applied when the Dataset is built, as upstream does. Before, the alias was ignored and the column was split as numeric.

- `Dataset.save_binary` reports a failed write as upstream does (`Cannot write binary data to <file>, wrote N of M bytes`) instead of the raw OS error.

- Label and weight inputs follow upstream's `_list_to_1d_numpy`: pandas Series with non-numeric dtypes raise `ValueError` naming the bad fields, and nested lists raise `TypeError`.

- `Dataset.feature_num_bin` returns the bins of a feature given by index or name (0 for unused features), as upstream does.

- Config warnings reach Python: the native library logs through `PyLogger`, so `[LightGBM] [Warning] ...` lines print as upstream does. `Booster(params, train_set=...)` validates its parameters when constructed, so the `max_depth` warning (raised for `max_depth >= 5` with `num_leaves` omitted) and the alias conflict warnings match upstream's text and timing.

- `Dataset.construct` sets `feature_name` to the resolved names, and `set_feature_name` marks them as user-set and checks their length, as upstream does.

- `Dataset._dump_text` writes the bins of every feature in every row, byte-identical to upstream's `Dataset::DumpTextFile` (`tests/parity/test_dataset_dump_parity.py`). Upstream cases that compare dumps (pandas dtypes, row- and column-major construction) now pass.

- `Dataset` accepts `position`, with `set_position`, `get_position` and `get_field("position")` returning dense int32 ids in first-seen order, as upstream does. Subsets keep their parent's ids. Training a ranking objective with positions raises `NotImplementedError` until the position bias is ported (P4).

- `Dataset.subset` of a pyarrow Table or polars DataFrame keeps its type, so `get_data()` returns the table, as upstream does.

- pyarrow tables and polars frames predict densely, as upstream does, and their column names are used for `categorical_feature` names and feature names.

- A weight whose values are all one is stored as `None`, as upstream does (`Dataset(..., weight=...)` and `set_weight`).

- Upstream's `test_arrow.py`, `test_polars.py` and `test_dual.py` are ported to `tests/python_package_test`. `pyarrow` and `polars` join the `dev` extra, and every arrow and polars case passes.

- The leaf gain is computed with a fused multiply-add on aarch64, as upstream's arm64 build does (`madd`). On macOS arm64, multi-value training with `path_smooth` chose a different default direction for a missing value than upstream, because the two directions' gains differ only in the last bits; models now match upstream there (`test_multi_value_training_matches_upstream`, previously an xfail on Darwin). Linux and Windows output is unchanged.

- Upstream's `tests/python_package_test` is ported in full (`test_basic`, `test_engine`, `test_callback`, `test_utilities`, `test_consistency`, `test_pandas`, plus the earlier sklearn and plotting files). Upstream cases that still fail are strict `xfail`s listed in `tests/known_upstream_gaps.tsv` with their ROADMAP item. Behavior fixed along the way: `EarlyStopException` is exported; `early_stopping_round` set to `None` no longer errors; `predict(pred_leaf=True)` with `start_iteration` past the last tree returns an empty `(n, 0)` array instead of panicking; sparse `pred_contrib` is predicted in row blocks, so very large CSR inputs no longer exhaust memory; `Dataset` gains `create_valid`, `get_data`, `get_feature_name`, `get_params`, `set_field`, `get_field`; `register_logger` and `EvalResult`/`Sequence` are exported; `reset_parameter` accepts keyword arguments.

- Upstream's `test_sklearn.py` and `test_plotting.py` are ported to `tests/python_package_test` and pass. Behavior fixed along the way, as upstream: set-valued `metric` parameters are passed (they were dropped); `categorical_feature` accepts names from an explicit `feature_name`; pandas `auto` categoricals skip ordered columns; `label`/`weight` column vectors are flattened with upstream's warning; unknown parameters print `[LightGBM] [Warning] Unknown parameter` when `verbosity >= 0`; `validate_features` reports upstream's messages; `fit()` no longer leaves `valid_sets` on the Booster; `Booster._get_loaded_param`; `lgb.dask` exposes only the import-error types (Dask estimators are not implemented, P7).

- Plotting helpers `plot_importance`, `plot_split_value_histogram`, `plot_metric`, `plot_tree` and `create_tree_digraph` are exported, ported from upstream `python-package/lightgbm/plotting.py` (MIT, see `THIRD_PARTY_NOTICES.md`). Their output (bar and line data, labels, graphviz source) is checked against `lightgbm==4.7.0`. Also: `Booster.get_split_value_histogram`.

- `Dataset` built from a pandas DataFrame (or with `feature_name`) names the model's features after the columns, as upstream does; the names appear in the model text.
- sklearn estimators `LGBMModel`, `LGBMRegressor`, `LGBMClassifier` and `LGBMRanker` are exported. They are ported from upstream `python-package/lightgbm/sklearn.py` (MIT, see `THIRD_PARTY_NOTICES.md`) with the same signatures and defaults; predictions match `lightgbm==4.7.0`. `narwhals` is a new dependency. Also: `Booster.free_dataset`, `record_evaluation` accepts an empty evaluation list (no validation sets) as upstream does.

- Split search with `num_threads > 1` fixes and derives each feature's histogram in the same task as its threshold search, so phase 1 runs in parallel and the histogram is not handed between cores. Models are unchanged; on the 2-vCPU benchmark machine training is about 1.35x faster on sparse data and 1.1x on dense data at two threads.

- Training with `num_threads > 1` runs its parallel regions (split search, data partition, dense histograms, multi-value histograms) on a persistent pool of spinning workers instead of rayon, which cuts the per-region overhead on small trees. Models are unchanged; the speedup on two threads is modest on the 2-vCPU benchmark machine (see ROADMAP P3).

- Train on data whose features bundle into a multi-value group (typical once most values are zero, dense or sparse input): histograms are built from upstream's row-wise multi-value bins with the same offsets, block split and merge order, so models match upstream byte for byte (also with several threads). Previously training raised "not supported yet: training with multi-value (sparse) feature groups".

- Labels are stored like upstream's `Metadata::SetLabel` (float32 through `AvoidInf`: NaN becomes 0, infinities are clamped to ±1e38); NaN labels used to poison the gradients so that no tree could split. `Dataset.get_label()`/`get_weight()` return the stored (sanitized) values once the Dataset is constructed, as upstream does.

- `train()` now returns the reparsed model unless `keep_training_booster=True`, like upstream (`model_from_string(model_to_string())`): the model text matches upstream's byte for byte in every case (e.g. a one-leaf tree's empty `leaf_weight`), and, as upstream, such a booster cannot continue training.

- Nested list parameters are passed like upstream (`interaction_constraints=[[0, 1], [2]]` becomes `[0,1],[2]`); they were flattened into a single group, which silently disabled interaction constraints given as Python lists.

- Custom objectives (`fobj`) now receive upstream's internal training score instead of a fresh prediction on the training rows: the `init_score` is included, and with DART the trees dropped for that iteration are excluded (previously the drop happened after `fobj` ran, so DART with a custom objective trained differently from upstream).

- `Dataset` accepts `scipy.sparse` CSR and CSC matrices (other sparse formats are converted with `csr_matrix`, as upstream does), for training, validation sets with `reference`, `cv` and custom objectives/metrics. Binning reproduces upstream's `LGBM_DatasetCreateFromCSR`/`FromCSC` exactly, without densifying the matrix. Training on data whose features bundle into multi-value groups is not supported yet (see ROADMAP P2 `MultiValBin`).

- Add `Dataset.save_binary(filename)`, writing upstream's binary format byte for byte; like upstream it never overwrites an existing file (it warns and keeps the file).

- Training with `linear_tree=true` (or `linear_trees`) now raises an error instead of silently training constant-leaf trees; linear-tree models still load and predict.

- `early_stopping` now follows upstream exactly: metrics are checked in order and the first one that stops improving decides, `best_score` holds that iteration's full evaluation list, the best iteration is also reported when training ends without stopping, `min_delta` accepts a list, `stopping_rounds <= 0` disables it (non-integers raise `TypeError`), and DART or training-set-only evaluation disable it with upstream's warnings. `train` sets `best_iteration = 0` and names a training set passed in `valid_sets` `"training"` (evaluated first), like upstream.

- `train` and `cv` resolve the number of boosting rounds like upstream: `num_iterations` or any of its aliases in `params` (e.g. `n_estimators`) take precedence over `num_boost_round`, conflicting aliases warn, and the chosen value is recorded in the model's parameters on every path (previously training with `valid_sets` or callbacks recorded the default `100`). Invalid counts raise upstream's message.

- `Booster.predict` accepts `scipy.sparse` CSR and CSC matrices natively (normal, raw score, leaf index and SHAP), matching upstream bit for bit without densifying the whole matrix; other sparse formats are converted with upstream's "Converting data to scipy sparse matrix." warning. `pred_contrib` on sparse input returns a sparse matrix of the same format (a list per class for multiclass) with upstream's stored entries and the input's float dtype.

- Cover `extra_trees` with upstream parity tests across seeds (byte-identical model text and predictions).

- Add `Dataset.set_label()`, `set_weight()` and `set_group()`, which also update an already constructed Dataset in place.

- Add `Dataset.get_weight()`, `get_group()`, `get_init_score()` and `set_init_score()` with upstream's array shapes and dtypes; multiclass init scores come back as `(num_data, num_class)`.

- Add `Booster.shuffle_models(start_iteration, end_iteration)` with upstream's iteration order, keeping continued training consistent with the shuffled trees.

- Add `Booster.set_leaf_output(tree, leaf, value)`, which edits the model and the trees continued training uses, like upstream's `SetLeafValue`.

- Add `Booster.rollback_one_iter()`: removes the last iteration's trees and their training-score contribution like upstream, so training continues identically.

- Support `init_model` continued training for gbdt (file path or Booster, including multiclass and validation callbacks), matching upstream trees and predictions. DART with `init_model` is rejected explicitly; non-Booster, non-path objects raise `TypeError` where upstream silently ignores them.

- Support `init_score` in Dataset (class-major, 2-D Fortran-order flattening like upstream): training starts from the scores without the automatic boost-from-average, validation metrics include them, subsets select them per class, and random forest rejects them as upstream does.

- Accept negative query sizes as upstream does (non-ranking objectives train normally); ranking objectives now raise a clear error instead of aborting.

- Match upstream Python errors for invalid weight lengths and query-count sums; cover nested subsets with zero-sized query groups.

- Match upstream errors for empty, negative, and out-of-range Dataset subset selections before converting indices to native unsigned values.

- Sort Dataset subset indices like upstream, preserving label, weight, and query-group alignment for unsorted and duplicate selections.

- Construct dense reference-backed datasets using the source bin mappers and feature groups, including weighted and categorical training.

- Recognize callable objectives through all upstream objective aliases in training and CV. Complete multiclass, DART/GOSS switching, RF rejection, and raw-score custom evaluation acceptance coverage.

- Support switching trained regression/binary boosters to custom objectives while retaining trees and scores; reject updates without derivatives after switching.

- Validate custom gradient/Hessian dimensions and lengths in Python with upstream-compatible errors before native updates; rejected inputs preserve trained model state.

- Apply CV training parameters before constructing dataset bins; verify custom multiclass objectives and CV fold predictions against upstream.

- Add external gradient/Hessian updates and callable regression objectives for GBDT/DART training; reject custom RF objectives as upstream does. Broader custom-objective compatibility remains in progress.

- Add custom evaluation functions (`feval`) to training and cross-validation, including multiple metrics returned by one function.
- Add seeded cross-validation with custom folds, train/validation metric aggregation, callbacks, early stopping, and optional serializable `CVBooster` results.
- Add `Dataset.subset()` with source bin mapper reuse for fold construction.
- Add `reset_parameter` schedules for learning rate by list or per-iteration callable.
- Add `log_evaluation` and `record_evaluation` callbacks for validation metrics.
- Add validation metric evaluation and the `early_stopping` callback, including best-score and best-iteration tracking.
- Add Python `Booster.update()` for one-at-a-time training with built-in objectives.
- Expose incremental Rust training through `boosting::GbdtTrainer`, with per-iteration updates and model snapshots. Add `Gbdt::train_shared` to train without copying shared dataset storage; the Python training binding now uses this path.
