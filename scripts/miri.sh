#!/usr/bin/env bash
# Run the unsafe-allowlisted modules under Miri (needs `rustup +nightly component add miri`).
# Slow (~1 min), so it is not part of scripts/check.sh; run it when touching `unsafe`.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo +nightly miri test -q -p lgbm-core --lib histogram
echo "miri.sh: clean"
