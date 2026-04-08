# Xnpu: LLVM RISC-V Vendor Extension for NPU Floating-Point Coprocessor

RISC-V vendor extension adding 16 floating-point NPU instructions to the
LLVM RISC-V backend. All instructions use R-type encoding with opcode
0x2B (CUSTOM_1).

## Quick start

The `build-llvm.sh` script automates cloning, patching, and building:

```bash
./build-llvm.sh                     # clones LLVM into ./llvm-project/
./build-llvm.sh /path/to/llvm       # uses existing LLVM checkout
JOBS=4 ./build-llvm.sh              # more parallelism (needs more RAM)
```

Requires GNU sed, cmake, ninja, and a C++ compiler. The build produces
a single `llc` binary (~10 min for RISC-V-only, ~4 GB RAM at -j2).

## Files

| File                      | Destination in LLVM tree                         |
|---------------------------|--------------------------------------------------|
| `RISCVInstrInfoXnpu.td`  | `llvm/lib/Target/RISCV/RISCVInstrInfoXnpu.td`   |
| `IntrinsicsRISCVXnpu.td` | `llvm/include/llvm/IR/IntrinsicsRISCVXnpu.td`   |
| `build-llvm.sh`           | Automated build script (runs all steps below)    |

## Manual setup

If you prefer to patch manually instead of using `build-llvm.sh`:

### 1. Clone LLVM

```bash
git clone --depth 1 https://github.com/llvm/llvm-project.git
cd llvm-project
```

### 2. Copy extension files

```bash
cp /path/to/llvm-npu/RISCVInstrInfoXnpu.td   llvm/lib/Target/RISCV/
cp /path/to/llvm-npu/IntrinsicsRISCVXnpu.td   llvm/include/llvm/IR/
```

### 3. Add includes to existing files

**`llvm/lib/Target/RISCV/RISCVInstrInfo.td`** — add near other vendor includes:
```tablegen
include "RISCVInstrInfoXnpu.td"
```

**`llvm/include/llvm/IR/IntrinsicsRISCV.td`** — add at the end:
```tablegen
include "IntrinsicsRISCVXnpu.td"
```

### 4. Add feature definition

**`llvm/lib/Target/RISCV/RISCVFeatures.td`** — add with other vendor features:
```tablegen
def FeatureVendorXnpu
    : RISCVExtension<0, 1, "NPU Floating-Point Coprocessor">;
def HasVendorXnpu : Predicate<"Subtarget->hasVendorXnpu()">,
                    AssemblerPredicate<(all_of FeatureVendorXnpu),
                        "'Xnpu' (NPU Floating-Point Coprocessor)">;
```

`RISCVExtension` auto-generates the subtarget boolean — no manual
`RISCVSubtarget.h` edit needed.

### 5. Build LLVM

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="RISCV" \
  -DLLVM_ENABLE_PROJECTS=""

ninja -C build -j2 llc
```

### 6. Verify

```bash
build/bin/llc -march=riscv32 -mattr=+f,+xnpu \
  tests/test_vector_ops.ll -o - | grep npu
```

## Usage

Compile LLVM IR to RISC-V assembly with NPU instructions:

```bash
llc -march=riscv32 -mattr=+f,+xnpu -o output.S input.ll
```

## Instruction set

All 16 FP instructions, opcode 0x2B (CUSTOM_1), R-type encoding:

| Mnemonic       | funct3 | funct7 | rd          | rs1       | rs2       | Operation                          |
|----------------|--------|--------|-------------|-----------|-----------|------------------------------------|
| npu.fmacc      | 0      | 0x00   | f0 (ign)    | FPR(a)    | FPR(b)    | facc += a * b                      |
| npu.fvmac      | 0      | 0x01   | GPR(n)      | GPR(lhs)  | GPR(rhs)  | facc += dot(lhs[0..n], rhs[0..n])  |
| npu.fvexp      | 0      | 0x02   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = exp(src[i])               |
| npu.frsqrt     | 0      | 0x03   | FPR(res)    | FPR(src)  | f0 (ign)  | res = 1/sqrt(src)                  |
| npu.fvmul      | 0      | 0x04   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = src[i] * facc             |
| npu.fvreduce   | 0      | 0x05   | FPR(res)    | GPR(src)  | GPR(n)    | res = sum(src[0..n])               |
| npu.fvmax      | 0      | 0x06   | FPR(res)    | GPR(src)  | GPR(n)    | res = max(src[0..n])               |
| npu.fvadd      | 0      | 0x07   | GPR(n)      | GPR(src1) | GPR(dst)  | dst[i] += src1[i]                  |
| npu.fvsub      | 0      | 0x08   | GPR(n)      | GPR(src1) | GPR(dst)  | dst[i] = src1[i] - dst[i]          |
| npu.fvrelu     | 0      | 0x09   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = max(src[i], 0)            |
| npu.fvgelu     | 0      | 0x0A   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = gelu(src[i])              |
| npu.fvdiv      | 0      | 0x0B   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = src[i] / facc             |
| npu.fvsub.sc   | 0      | 0x0C   | GPR(n)      | GPR(src)  | GPR(dst)  | dst[i] = src[i] - facc             |
| npu.frelu      | 1      | 0x00   | FPR(res)    | FPR(src)  | f0 (ign)  | res = max(src, +0.0)               |
| npu.fgelu      | 4      | 0x00   | FPR(res)    | FPR(src)  | f0 (ign)  | res = gelu(src)                    |
| npu.frstacc    | 5      | 0x00   | FPR(res)    | f0        | f0        | res = (f32)facc; facc = 0          |
