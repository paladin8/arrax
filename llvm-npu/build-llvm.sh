#!/usr/bin/env bash
# Build LLVM with Xnpu vendor extension patches.
#
# Usage:
#   ./build-llvm.sh [llvm-project-dir]
#
# If llvm-project-dir is not provided, clones LLVM into ./llvm-project.
# Requires: GNU sed, cmake, ninja, gcc/g++ (or clang), ~30 GB disk, ~4 GB RAM.
#
# The script is idempotent — re-running skips completed steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_DIR="${1:-$SCRIPT_DIR/llvm-project}"
BUILD_DIR="$LLVM_DIR/build"
JOBS="${JOBS:-2}"  # default to -j2 to avoid OOM on 8 GB machines

# GNU sed is required (BSD sed on macOS has different -i and append syntax).
if ! sed --version 2>/dev/null | grep -q GNU; then
    echo "ERROR: GNU sed required. On macOS: brew install gnu-sed"
    exit 1
fi

echo "=== Xnpu LLVM Build ==="
echo "LLVM source: $LLVM_DIR"
echo "Build dir:   $BUILD_DIR"
echo "Parallelism: -j$JOBS (set JOBS=N to override)"
echo ""

# --- Step 1: Clone LLVM if needed ---

if [ ! -d "$LLVM_DIR/llvm" ]; then
    echo "--- Cloning LLVM (shallow) ---"
    git clone --depth 1 https://github.com/llvm/llvm-project.git "$LLVM_DIR"
else
    echo "--- LLVM source found, skipping clone ---"
fi

echo "LLVM commit: $(git -C "$LLVM_DIR" rev-parse HEAD)"

# --- Step 2: Copy extension files ---

echo "--- Copying Xnpu extension files ---"
cp "$SCRIPT_DIR/RISCVInstrInfoXnpu.td"  "$LLVM_DIR/llvm/lib/Target/RISCV/"
cp "$SCRIPT_DIR/IntrinsicsRISCVXnpu.td" "$LLVM_DIR/llvm/include/llvm/IR/"

# --- Step 3: Patch existing files (idempotent — grep before adding) ---

echo "--- Patching LLVM source files ---"

# Helper: patch a file if not already patched, verify the patch took effect.
# Usage: patch_file <file> <check_pattern> <sed_command> <label>
patch_file() {
    local file="$1" check="$2" sed_cmd="$3" label="$4"
    if grep -q "$check" "$file"; then
        echo "  $label already patched"
        return
    fi
    sed -i "$sed_cmd" "$file"
    if ! grep -q "$check" "$file"; then
        echo "ERROR: Failed to patch $label — sed anchor line not found."
        echo "  This may mean the LLVM version changed. Check the file manually."
        exit 1
    fi
    echo "  Patched $label"
}

# RISCVInstrInfo.td — add include
patch_file \
    "$LLVM_DIR/llvm/lib/Target/RISCV/RISCVInstrInfo.td" \
    'RISCVInstrInfoXnpu' \
    '/include "RISCVInstrInfoXAIF.td"/a include "RISCVInstrInfoXnpu.td"' \
    "RISCVInstrInfo.td"

# IntrinsicsRISCV.td — add include
patch_file \
    "$LLVM_DIR/llvm/include/llvm/IR/IntrinsicsRISCV.td" \
    'IntrinsicsRISCVXnpu' \
    '/include "llvm\/IR\/IntrinsicsRISCVXMIPS.td"/a include "llvm\/IR\/IntrinsicsRISCVXnpu.td"' \
    "IntrinsicsRISCV.td"

# RISCVFeatures.td — add feature definition
FEATURES="$LLVM_DIR/llvm/lib/Target/RISCV/RISCVFeatures.td"
if ! grep -q 'FeatureVendorXnpu' "$FEATURES"; then
    sed -i '/^def FeatureVendorXVentanaCondOps/i \
// NPU Floating-Point Coprocessor\
\
def FeatureVendorXnpu\
    : RISCVExtension<0, 1, "NPU Floating-Point Coprocessor">;\
def HasVendorXnpu : Predicate<"Subtarget->hasVendorXnpu()">,\
                    AssemblerPredicate<(all_of FeatureVendorXnpu),\
                        "'"'"'Xnpu'"'"' (NPU Floating-Point Coprocessor)">;\
' "$FEATURES"
    if ! grep -q 'FeatureVendorXnpu' "$FEATURES"; then
        echo "ERROR: Failed to patch RISCVFeatures.td — anchor line not found."
        exit 1
    fi
    echo "  Patched RISCVFeatures.td"
else
    echo "  RISCVFeatures.td already patched"
fi

# --- Step 4: Configure ---

if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "--- Configuring (RISC-V target only) ---"
    cmake -S "$LLVM_DIR/llvm" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="RISCV" \
        -DLLVM_ENABLE_PROJECTS=""
else
    echo "--- Build already configured, skipping cmake ---"
fi

# --- Step 5: Build llc ---

echo "--- Building llc (-j$JOBS) ---"
ninja -C "$BUILD_DIR" -j"$JOBS" llc

echo ""
echo "=== Build complete ==="
echo "llc binary: $BUILD_DIR/bin/llc"
echo ""

# --- Step 6: Verify ---

echo "--- Running Xnpu tests ---"
"$SCRIPT_DIR/tests/run_tests.sh" "$BUILD_DIR/bin/llc"
