# Changelog

## Unreleased

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
