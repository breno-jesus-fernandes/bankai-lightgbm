# Vendored upstream files

Copied verbatim from [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) at the tag in
`../../UPSTREAM_VERSION` (MIT license, see `LICENSE`).

| File | Upstream path | Used by |
|---|---|---|
| `config.h` | `include/LightGBM/config.h` | `scripts/gen_config.py` → `crates/lgbm-core/src/config/auto.rs` |

Refresh these files when you bump `UPSTREAM_VERSION`, then run `python scripts/gen_config.py`.
