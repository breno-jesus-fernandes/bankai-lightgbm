# Vendored upstream files

Copied verbatim from [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) at the tag in
`../../UPSTREAM_VERSION` (MIT license, see `LICENSE`).

| File | Upstream path | Used by |
|---|---|---|
| `config.h` | `include/LightGBM/config.h` | `scripts/gen_config.py` → `crates/lgbm-core/src/config/auto.rs` |
| `include/LightGBM/c_api.h` | `include/LightGBM/c_api.h` | `scripts/c_header_harness.sh` (compiles `crates/lgbm-capi/ctest/harness.c` against it) |
| `include/LightGBM/export.h` | `include/LightGBM/export.h` | included by `c_api.h` |
| `include/LightGBM/arrow.h` | `include/LightGBM/arrow.h` | included by `c_api.h` |

The `include/LightGBM/` layout mirrors upstream so the header's own `#include <LightGBM/...>` lines
resolve with `-isystem third_party/lightgbm/include`.

Refresh these files when you bump `UPSTREAM_VERSION`, then run `python scripts/gen_config.py`.
