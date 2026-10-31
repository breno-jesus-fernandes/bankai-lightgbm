# Changelog

## Unreleased

- Add validation metric evaluation and the `early_stopping` callback, including best-score and best-iteration tracking.
- Add Python `Booster.update()` for one-at-a-time training with built-in objectives.
- Expose incremental Rust training through `boosting::GbdtTrainer`, with per-iteration updates and model snapshots. Add `Gbdt::train_shared` to train without copying shared dataset storage; the Python training binding now uses this path.
