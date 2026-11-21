# Upstream C++ code generation fixtures

The `.txt` models were trained by `lightgbm==4.7.0`; the `.cpp` files were emitted by
upstream's `task=convert_model`, independently of bankai. `provenance.json` records the
wheel/tag versions and executable SHA-256.

The fixture CLI was compiled from upstream tag `v4.7.0` (commit `8f7036f03`) with its
pinned submodules, using CMake `-DUSE_OPENMP=OFF -DBUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release`
and `cmake --build <build> --target lightgbm -j2`. It is used only for generation and is
never linked into bankai. Regenerate into an empty destination with:

```sh
python scripts/gen_convert_model_fixtures.py --upstream-cli /path/to/pinned/lightgbm
```

The generator refuses to overwrite existing C++ goldens, because upstream wraps existing
files instead of replacing their contents. `--cases` selects cases when adding fixtures.

The expected output preserves upstream quirks, including the missing-value branch spelling,
linear-tree behavior, six-digit stump values even for leaf-index functions, and the existing-file
`#define USE_HARD_CODE 0` / `#ifndef USE_HARD_CODE` wrapper. Generated C++ is intended for
upstream's hard-coded prediction build and includes upstream headers; it is not a standalone
program.
