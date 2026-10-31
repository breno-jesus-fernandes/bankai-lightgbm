#!/usr/bin/env bash
# Full local gate: run before every commit (see AGENTS.md).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all --check
python3 scripts/gen_config.py --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  ruff format --check python scripts tests benches
  ruff check python scripts tests benches
  mypy
  maturin develop -q
  pytest -q
fi
echo "check.sh: all green"
