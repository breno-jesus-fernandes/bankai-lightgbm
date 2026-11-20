#!/usr/bin/env bash
# C-header harness for lgbm-capi (ROADMAP T5.3 "Verify"): compile a C program against the verbatim
# upstream header (third_party/lightgbm/include/LightGBM/c_api.h, v4.7.0), link the lgbm-capi static
# library, and run it. Fails if the build, link, or the end-to-end run fails.
set -euo pipefail

cd "$(dirname "$0")/.."
root="$(pwd)"

cargo build -p lgbm-capi --release
lib="$root/target/release/liblgbm_capi.a"
if [ ! -f "$lib" ]; then
  echo "c_header_harness: static library not found at $lib" >&2
  exit 1
fi

cc="${CC:-cc}"
inc="$root/third_party/lightgbm/include"
src="$root/crates/lgbm-capi/ctest/harness.c"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
bin="$workdir/harness"

# Rust static libraries need the platform's base runtime libraries at link time.
extra_libs=()
case "$(uname -s)" in
  Linux) extra_libs=(-lpthread -ldl -lm) ;;
  Darwin) extra_libs=(-lm -framework CoreFoundation -framework Security) ;;
esac

# The header include dir is passed with -isystem so warnings from the verbatim upstream header
# (e.g. -Wstatic-in-inline on its own inline helpers, which we must not edit) are suppressed, while
# -Wall -Wextra still check the harness itself.
echo "c_header_harness: $cc -std=c11 -isystem $inc $src $lib ${extra_libs[*]}"
"$cc" -std=c11 -Wall -Wextra -isystem "$inc" "$src" "$lib" "${extra_libs[@]}" -o "$bin"
"$bin"
echo "c_header_harness: all green"
