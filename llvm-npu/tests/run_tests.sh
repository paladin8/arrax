#!/usr/bin/env bash
# Run LLVM tests for the Xnpu vendor extension.
# Requires llc built with the Xnpu patches (see ../README.md).
#
# Usage: ./run_tests.sh [path/to/llc]
#
# If llc is not on PATH, pass it as the first argument.

set -euo pipefail

LLC="${1:-llc}"

if ! command -v "$LLC" &>/dev/null; then
    echo "ERROR: llc not found. Build LLVM with Xnpu patches first."
    echo "Usage: $0 [path/to/llc]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0

run_test() {
    local ll_file="$1"
    local expected_mnemonic="$2"
    local test_name
    test_name="$(basename "$ll_file" .ll):$expected_mnemonic"

    if "$LLC" -mtriple=riscv32 -mattr=+f,+xnpu "$ll_file" -o - 2>/dev/null \
        | grep -q "$expected_mnemonic"; then
        echo "  PASS  $test_name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $test_name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Xnpu LLVM Backend Tests ==="
echo ""

echo "--- Vector ops ---"
for mnemonic in npu.fvadd npu.fvsub npu.fvexp npu.fvrelu npu.fvgelu \
                npu.fvmul npu.fvdiv npu.fvsub.sc npu.fvmac; do
    run_test "$SCRIPT_DIR/test_vector_ops.ll" "$mnemonic"
done

echo ""
echo "--- Reductions ---"
for mnemonic in npu.fvreduce npu.fvmax; do
    run_test "$SCRIPT_DIR/test_reductions.ll" "$mnemonic"
done

echo ""
echo "--- Scalar ops ---"
for mnemonic in npu.fmacc npu.frsqrt npu.frstacc npu.frelu npu.fgelu; do
    run_test "$SCRIPT_DIR/test_scalar_ops.ll" "$mnemonic"
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
