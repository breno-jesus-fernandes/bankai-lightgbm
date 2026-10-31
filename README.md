# bankai-lightgbm

[LightGBM](https://github.com/lightgbm-org/LightGBM) rewritten in Rust, with a drop-in Python API.

- **Parity target:** LightGBM `v4.7.0` (see [`UPSTREAM_VERSION`](UPSTREAM_VERSION))
- **Performance:** equal to or faster than upstream, enforced by benchmarks in CI
- **Model files:** text and JSON formats work in both directions with upstream
- **Python:** `>= 3.11`. Use `import bankai_lightgbm as lgb`.

Status: **P0, foundations.** See [`ROADMAP.md`](ROADMAP.md). Contributors and AI agents must follow [`AGENTS.md`](AGENTS.md).

## Development

```bash
uv venv -p 3.11 && source .venv/bin/activate
uv pip install maturin pytest numpy scipy ruff mypy "lightgbm==4.7.0"
maturin develop
cargo test --workspace
pytest
```

## Layout

| Path | Purpose |
|---|---|
| `crates/lgbm-core` | config, dataset, binning, tree learner, boosting, objectives, metrics |
| `crates/lgbm-io` | CSV/TSV/LibSVM/Arrow parsers, model text/JSON I/O |
| `crates/lgbm-cli` | `lightgbm` CLI (upstream config-file compatible) |
| `crates/lgbm-capi` | `c_api.h`-compatible FFI |
| `crates/lgbm-python` + `python/bankai_lightgbm` | PyO3 bindings + Python API |
| `tests/fixtures` | golden outputs from upstream (`scripts/gen_fixtures.py`) |
| `tests/parity` | parity tests vs upstream |
