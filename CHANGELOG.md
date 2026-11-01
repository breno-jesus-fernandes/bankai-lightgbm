# Changelog

## Unreleased

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
