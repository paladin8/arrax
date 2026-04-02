# arrax: Design Document

An MLIR-based compiler that fuses array expressions in Python and emits optimized RISC-V assembly targeting a custom neural processing unit.

## Architecture

```
Python DSL (tracing)
  │
  ▼
array dialect (fused expression DAGs on abstract arrays)
  │
  ▼
linalg dialect (structured ops on tensors with indexing maps)
  │  ── tiling pass (decompose to fit memory budget)
  │  ── fusion pass (merge adjacent elementwise ops)
  │
  ▼
linalg dialect on memrefs (after bufferization)
  │
  ▼
npu dialect (NPU-specific ops: npu.fvmac, npu.fvexp, etc.)
  │
  ▼
riscv dialect (RISC-V base instructions + NPU custom encodings)
  │
  ▼
Assembly text (.S file)
  │
  ▼
riscv32 toolchain (GNU as + ld via .insn directives)
  │
  ▼
ELF binary → run on riscv-npu emulator
```

The compiler is built on xDSL (Python-based MLIR framework). Phase 2 replaces the assembly text emitter with a proper LLVM RISC-V backend extension.

## Target Hardware

The target is the FP32 instruction set of the NPU coprocessor defined in [riscv-npu](https://github.com/paladin8/riscv-npu). The NPU provides memory-to-memory vector operations (address + length from registers), a double-precision accumulator for dot products, and scalar activation functions. It does not have vector registers for FP — all FP vector ops read/write memory directly.

This memory-to-memory model means the compilation path goes directly from `linalg` (structured ops in destination-passing style) to NPU instructions, skipping the MLIR `vector` dialect entirely. The `vector` dialect models fixed-width SIMD register files, which doesn't match this hardware.

## Phase 1: xDSL Pipeline to Assembly

### 1.1 Python Tracing DSL

A Python library that captures array computation graphs by overloading operators on a lazy `Array` class.

```python
from arrax import Array, compile

def my_kernel(A: Array, B: Array, C: Array) -> Array:
    return (A + B) * C

binary = compile(my_kernel, shapes={"A": (1024,), "B": (1024,), "C": (1024,)})
```

`Array` objects record operations into a DAG instead of executing them. Each operation creates a node. `compile()` walks the DAG and emits the `array` dialect IR.

Supported operations:

| Category | Operations | Notes |
|----------|-----------|-------|
| Elementwise binary | add, sub, mul, div | Array-array and scalar-array |
| Elementwise unary | neg, relu, gelu, exp | Maps to NPU activation instructions |
| Reductions | sum, max | Full-array reductions |
| Dot product | dot(A, B) | 1D, maps to FVMAC + FRSTACC |
| Matmul | A @ B | 2D matrix multiply via tiled dot products |
| Composite | softmax(A), rmsnorm(A, gamma) | Multi-instruction lowering patterns |

All shapes are static (known at compile time). Dynamic shapes are out of scope.

### 1.2 The `array` Dialect

High-level, value-semantic dialect representing array computations. Defined in xDSL using IRDL.

```
array.add      : (tensor<NxF32>, tensor<NxF32>) -> tensor<NxF32>
array.sub      : (tensor<NxF32>, tensor<NxF32>) -> tensor<NxF32>
array.mul      : (tensor<NxF32>, tensor<NxF32>) -> tensor<NxF32>
array.div      : (tensor<NxF32>, tensor<NxF32>) -> tensor<NxF32>
array.neg      : (tensor<NxF32>) -> tensor<NxF32>
array.exp      : (tensor<NxF32>) -> tensor<NxF32>
array.relu     : (tensor<NxF32>) -> tensor<NxF32>
array.gelu     : (tensor<NxF32>) -> tensor<NxF32>
array.sum      : (tensor<NxF32>) -> f32
array.max      : (tensor<NxF32>) -> f32
array.dot      : (tensor<NxF32>, tensor<NxF32>) -> f32
array.matmul   : (tensor<MxKxF32>, tensor<KxNxF32>) -> tensor<MxNxF32>
array.softmax  : (tensor<NxF32>) -> tensor<NxF32>
array.rmsnorm  : (tensor<NxF32>, tensor<NxF32>) -> tensor<NxF32>  // (input, gamma)
array.splat    : (f32) -> tensor<NxF32>                           // scalar broadcast
```

All operations use MLIR `tensor` types (value semantics, no memory). `array.softmax` and `array.rmsnorm` are kept as single operations at this level because they have multi-instruction lowering patterns that benefit from being recognized as a unit. Lowering to linalg primitives too early scatters the pattern — the same design principle as the WSE stencil paper (Stawinoga et al., ASPLOS 2026), which retains stencil semantics deep into the pipeline rather than lowering to generic loops early.

### 1.3 Lowering: array → linalg

**Elementwise ops** lower to `linalg.generic` with identity indexing maps:

```
array.add(%A, %B) → linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
} ins(%A, %B) outs(%init) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum
}
```

**Reductions** lower to `linalg.generic` with reduction iterator types.

**Dot product** lowers to `linalg.generic` with a contraction pattern (parallel + reduction).

**Matmul** lowers to `linalg.matmul` (standard matmul indexing maps).

**Softmax** decomposes into a sequence of linalg operations:
1. `linalg.generic` (reduction, max) → scalar max
2. `linalg.generic` (elementwise, sub max) → shifted values
3. `linalg.generic` (elementwise, exp) → exponentials
4. `linalg.generic` (reduction, sum) → denominator
5. `linalg.generic` (elementwise, mul by 1/denom) → normalized output

**RMSNorm** decomposes into:
1. `linalg.generic` (reduction, sum of squares via contraction of x with x) → sum_sq
2. Scalar: `mean_sq = sum_sq / N + eps`
3. Scalar: `rsqrt(mean_sq)` → scale
4. `linalg.generic` (elementwise, x × gamma × scale) → output

The decomposed operations remain adjacent in the IR, which the fusion pass and later NPU pattern matching exploit.

### 1.4 Tiling Pass

Decomposes operations on large arrays into loops over tiles that fit within a configurable memory budget.

**Configuration**: `--memory-budget=65536` (default 64KB). Disable with `--no-tiling`.

**Mechanism**: Standard linalg tiling — rewrites `linalg.generic` on a large tensor into `scf.for` containing `linalg.generic` on `tensor.extract_slice`. Tile size:

```
tile_size = memory_budget / (num_live_tensors * sizeof(f32))
```

For a binary elementwise op (3 live tensors) at 64KB: `tile_size = 65536 / 12 = 5461`. Rounded down to alignment (64 elements).

Tiling applies to parallel dimensions only. Reduction dimensions are not tiled — the NPU's FVMAC handles arbitrarily long reductions in a single instruction.

### 1.5 Fusion Pass

Merges adjacent elementwise `linalg.generic` operations to eliminate intermediate buffer materialization.

`(A + B) * C` before fusion:
```
%t = linalg.generic [addf] ins(%A, %B) outs(%tmp)
%r = linalg.generic [mulf] ins(%t, %C) outs(%out)
```

After fusion:
```
%r = linalg.generic {
  ^bb0(%a, %b, %c, %out):
    %t = arith.addf %a, %b
    %r = arith.mulf %t, %c
    linalg.yield %r
} ins(%A, %B, %C) outs(%out)
```

The intermediate `%tmp` is eliminated entirely. This is the primary advantage over NumPy, which materializes every intermediate.

**Legality**: Two `linalg.generic` ops fuse if they have compatible indexing maps and the producer has a single consumer.

**Boundary**: Fusion does not apply across reductions. A reduction produces a scalar that breaks the elementwise chain.

### 1.6 Bufferization

Converts `tensor` types to `memref` types using a custom bufferization pass (xDSL has no general-purpose one-shot bufferization).

- Output buffers become function arguments (destination-passing style)
- Intermediate buffers use `memref.alloc`
- Stack allocation for tile-sized intermediates (when tiling is enabled)
- Static allocation for input/output arrays in the firmware memory map

**Buffer reuse and deallocation**: The initial implementation allocates a fresh buffer for each intermediate and never deallocates. A future optimization pass can perform liveness analysis to reuse buffers (e.g., in `A + B + C + D`, the first intermediate is dead before the second is needed) and insert `memref.dealloc` where appropriate. This is not needed for Milestone 1 since the NPU firmware runs once on a flat memory region.

After bufferization, `linalg.generic` operates on memrefs in destination-passing style:
```
linalg.generic ins(%A, %B : memref<1024xf32>, memref<1024xf32>)
               outs(%out : memref<1024xf32>)
```

### 1.7 The `npu` Dialect

Hardware-specific dialect. Each operation maps 1:1 to an NPU instruction.

```
// Accumulator operations
npu.fmacc        %rs1, %rs2           : (f32_reg, f32_reg) -> ()
npu.fvmac        %src1, %src2, %n     : (memref, memref, index) -> ()
npu.frstacc                           : () -> f32_reg

// Elementwise memory-to-memory (binary)
npu.fvadd        %src1, %src2, %dst, %n : (memref, memref, memref, index) -> ()
npu.fvsub        %src1, %src2, %dst, %n : (memref, memref, memref, index) -> ()
npu.fvmul_scale  %src, %dst, %n         : (memref, memref, index) -> ()
npu.fvdiv_scale  %src, %dst, %n         : (memref, memref, index) -> ()
npu.fvsub_scalar %src, %dst, %n         : (memref, memref, index) -> ()

// Elementwise memory-to-memory (unary)
npu.fvexp        %src, %dst, %n       : (memref, memref, index) -> ()
npu.fvrelu       %src, %dst, %n       : (memref, memref, index) -> ()
npu.fvgelu       %src, %dst, %n       : (memref, memref, index) -> ()

// Reductions
npu.fvreduce     %src, %n             : (memref, index) -> f32_reg
npu.fvmax        %src, %n             : (memref, index) -> f32_reg

// Scalar
npu.frelu        %src                 : (f32_reg) -> f32_reg
npu.fgelu        %src                 : (f32_reg) -> f32_reg
npu.fvrsqrt      %addr                : (memref) -> f32_reg
```

Operations use memref operands for arrays and f32 register operands for scalars. The NPU's `fvmul_scale`, `fvdiv_scale`, and `fvsub_scalar` operations implicitly read the scalar from the FP accumulator.

### 1.8 Lowering: linalg on memrefs → npu

Pattern-matching pass. Recognizes linalg operations and maps them to NPU instruction sequences.

**Elementwise add/sub**:
```
linalg.generic [addf] ins(%A, %B) outs(%C)  →  npu.fvadd %A, %B, %C, %n
```

**Scalar broadcast multiply** (one operand is a splat constant in the accumulator):
```
linalg.generic [mulf] ins(%A, %splat_s) outs(%C)  →  npu.fvmul_scale %A, %C, %n
```

**Dot product**:
```
linalg.generic [contraction: mulf + addf] ins(%A, %B) outs(%init)
  →  npu.fvmac %A, %B, %n ; npu.frstacc → %result
```

**Matmul (MxK @ KxN → MxN)**:
```
// Transpose B into scratch buffer (columns non-contiguous in row-major)
for i in 0..M:
  for j in 0..N:
    npu.fvmac &A[i,0], &B_T[j,0], K
    npu.frstacc → C[i,j]
```

If B is already column-major or K=1, skip the transpose.

**Softmax**:
```
npu.fvmax %src, %n → %max_val
// store max_val to facc
npu.fvsub_scalar %src, %shifted, %n
npu.fvexp %shifted, %exp_buf, %n
npu.fvreduce %exp_buf, %n → %sum
// store sum to facc
npu.fvdiv_scale %exp_buf, %dst, %n
```

**RMSNorm**:
```
npu.fvmac %x, %x, %n                      // facc = sum(x²)
npu.frstacc → %sum_sq
// scalar: mean_sq = sum_sq / N + eps (FDIV.S, FADD.S)
npu.fvrsqrt %mean_sq_addr → %scale
// store scale to facc via FMACC
npu.fvmul_scale %x, %tmp, %n              // tmp = x * scale
// gamma multiplication: element-by-element (scalar loop or staged FVMUL)
```

**Fused elementwise chains**: A fused `linalg.generic` with body `{ mulf(addf(a, b), c) }` that doesn't match a single NPU instruction is decomposed into a sequence of NPU ops with intermediate scratch buffers. The fusion still helps by reducing the number of passes over data compared to unfused code, even if each sub-operation needs its own NPU call.

**Fallback**: Any `linalg.generic` that doesn't match a known NPU pattern lowers to a scalar loop using base RISC-V FP instructions (FLW, FSW, FADD.S, FMUL.S, etc.). This guarantees completeness.

### 1.9 Lowering: npu → Assembly

Each `npu` operation emits one or more `.insn` directives:

```
npu.fvmac %addr1, %addr2, %n
```
emits:
```asm
    # NPU.FVMAC rd=a2, rs1=a0, rs2=a1
    .insn r 0x2B, 0, 1, a2, a0, a1
```

Register allocation is straightforward: function arguments arrive in a0-a7/fa0-fa7 per RISC-V calling convention. Tile loop induction variables use s-registers. The register pressure is very low because NPU instructions mostly operate on memory, not registers.

### 1.10 Firmware Integration

The compiler emits a `.S` file containing the kernel function:

```asm
    .globl npu_kernel
    .type npu_kernel, @function
npu_kernel:
    # a0 = output array pointer
    # a1 = input array A pointer
    # a2 = input array B pointer
    # a3 = array length
    # ... compiler-generated instruction sequence ...
    ret
```

A generated C harness (`main.c`) based on `firmware/common/` from riscv-npu:
- Declares array buffers
- Initializes input data
- Calls `npu_kernel()`
- Prints output via UART for verification

Build pipeline:
1. Compiler produces `kernel.S`
2. `riscv32-unknown-elf-as` assembles (`.insn` handles NPU encodings)
3. `riscv32-unknown-elf-gcc` compiles `main.c`, links with `kernel.S`
4. Produces ELF binary
5. `riscv-npu` emulator executes it

### 1.11 Verification and Benchmarks

**Correctness**: Run each test case in NumPy (float32) and compare against emulator output, with tolerance for floating-point ordering differences.

**Performance metric**: Instruction count comparison between compiler output and hand-written C firmware using NPU intrinsics. The emulator reports:
- Total RISC-V instructions executed
- Total NPU instructions executed (by type)
- Total memory loads/stores

**Test matrix**:

| Test | Expression | Pattern exercised |
|------|-----------|-------------------|
| 1 | `A + B` | Single elementwise |
| 2 | `(A + B) * C` | Fusion |
| 3 | `(A + B) * 0.5 + C` | Fusion + scalar broadcast |
| 4 | `dot(A, B)` | FVMAC + FRSTACC |
| 5 | `relu(A + B)` | Fused elementwise + activation |
| 6 | `A @ B` (small matrices) | Tiled matmul |
| 7 | `softmax(A)` | Multi-instruction composite |
| 8 | `rmsnorm(A, gamma)` | Multi-instruction composite |
| 9 | Large fused chain, 100K elements | Tiling + fusion |

**Optional x86 comparison**: Compile the same expressions to native x86 via xDSL's standard LLVM path. Compare fused (compiler) vs unfused (NumPy) wall-clock time. Both run natively, so the comparison is fair.

## Phase 2: LLVM Backend Extension

Replace assembly text emission with a proper LLVM RISC-V backend extension.

### 2.1 Pipeline Change

```
npu dialect
  → LLVM dialect (with NPU intrinsic calls)
  → LLVM IR
  → Modified LLVM RISC-V backend
  → Assembly / Object code
```

### 2.2 LLVM Intrinsic Definitions

In `llvm/include/llvm/IR/IntrinsicsRISCVNPU.td`:

```tablegen
def int_riscv_npu_fvmac : Intrinsic<
  [],
  [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
  [IntrWriteMem, IntrArgMemOnly]
>;

def int_riscv_npu_frstacc : Intrinsic<
  [llvm_float_ty],
  [],
  [IntrHasSideEffects]
>;

// one intrinsic per NPU instruction
```

### 2.3 RISC-V Backend TableGen

In `llvm/lib/Target/RISCV/`:

NPU accumulator modeled as an implicit register (not allocatable):
```tablegen
def FPUACC : RISCVReg<0, "facc">;
```

Instructions in `RISCVInstrInfoNPU.td`:
```tablegen
def NPU_FVMAC : RVInst<
  (outs),
  (ins GPR:$rd, GPR:$rs1, GPR:$rs2),
  "npu.fvmac", "$rd, $rs1, $rs2",
  [(int_riscv_npu_fvmac GPR:$rs1, GPR:$rs2, GPR:$rd)]
> {
  let Opcode = 0x2B;
  let funct3 = 0b000;
  let funct7 = 0b0000001;
}
```

### 2.4 xDSL npu → LLVM Dialect Lowering

```
npu.fvmac %addr1, %addr2, %n
  → llvm.call @llvm.riscv.npu.fvmac(%addr1, %addr2, %n)
```

### 2.5 Build Integration

Build modified LLVM from source. xDSL pipeline emits LLVM IR text, invokes custom `llc` for RISC-V compilation.

## New Emulator Instructions

The following instructions should be added to `riscv-npu` to support efficient compilation. All use opcode `0x2B` (FP NPU custom-1 space).

### NPU.FVADD — FP Vector Add

```
funct3 = 000, funct7 = 0000111    Format: R-type
Syntax: NPU.FVADD rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] + mem_f32[regs[rs2] + i*4]
```

Element-wise addition. Result stored in-place at rs2's address. Source and destination overlap is the intended usage.

### NPU.FVSUB — FP Vector Subtract

```
funct3 = 000, funct7 = 0001000    Format: R-type
Syntax: NPU.FVSUB rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] - mem_f32[regs[rs2] + i*4]
```

Element-wise subtraction. `rs1 - rs2`, result stored at rs2.

### NPU.FVRELU — FP Vector ReLU

```
funct3 = 000, funct7 = 0001001    Format: R-type
Syntax: NPU.FVRELU rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    val = mem_f32[regs[rs1] + i*4]
    mem_f32[regs[rs2] + i*4] = max(val, 0.0)
```

Source and destination may overlap.

### NPU.FVGELU — FP Vector GELU

```
funct3 = 000, funct7 = 0001010    Format: R-type
Syntax: NPU.FVGELU rd, rs1, rs2
```

```
n = regs[rd]
for i in 0..n-1:
    x = mem_f32[regs[rs1] + i*4]
    mem_f32[regs[rs2] + i*4] = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Source and destination may overlap.

### NPU.FVDIV — FP Vector Divide by Accumulator Scalar

```
funct3 = 000, funct7 = 0001011    Format: R-type
Syntax: NPU.FVDIV rd, rs1, rs2
```

```
n = regs[rd]
divisor = (float32)facc
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] / divisor
```

Counterpart to FVMUL (which multiplies by the accumulator). The accumulator is NOT modified.

### NPU.FVSUB_SCALAR — FP Vector Subtract Accumulator Scalar

```
funct3 = 000, funct7 = 0001100    Format: R-type
Syntax: NPU.FVSUB_SCALAR rd, rs1, rs2
```

```
n = regs[rd]
scalar = (float32)facc
for i in 0..n-1:
    mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] - scalar
```

Subtracts the accumulator scalar from each element. Source and destination may overlap.

### Summary

| Instruction | funct7 | Purpose | Priority |
|------------|--------|---------|----------|
| FVADD | 0000111 | Elementwise array add | Required |
| FVSUB | 0001000 | Elementwise array sub | Required |
| FVRELU | 0001001 | Vectorized ReLU | Required |
| FVGELU | 0001010 | Vectorized GELU | Required |
| FVDIV | 0001011 | Divide by accumulator scalar | Nice-to-have |
| FVSUB_SCALAR | 0001100 | Subtract accumulator scalar | Required |

## Project Structure

```
arrax/
├── README.md
├── pyproject.toml
├── src/
│   └── arrax/
│       ├── __init__.py
│       ├── dsl/
│       │   ├── __init__.py
│       │   ├── array.py              # Array class with operator overloading
│       │   └── tracer.py             # DAG capture and shape inference
│       ├── dialects/
│       │   ├── __init__.py
│       │   ├── array_dialect.py      # array dialect (IRDL)
│       │   └── npu_dialect.py        # npu dialect (IRDL)
│       ├── lowering/
│       │   ├── __init__.py
│       │   ├── dsl_to_array.py       # traced DAG → array dialect IR
│       │   ├── array_to_linalg.py    # array → linalg on tensors
│       │   ├── tiling.py             # linalg tiling pass
│       │   ├── fusion.py             # linalg elementwise fusion
│       │   ├── bufferize.py          # tensor → memref
│       │   ├── linalg_to_npu.py      # linalg on memrefs → npu dialect
│       │   └── npu_to_asm.py         # npu → RISC-V assembly text
│       ├── pipeline.py               # full pass pipeline
│       └── codegen/
│           ├── __init__.py
│           ├── asm_emitter.py        # assembly text generation
│           ├── firmware_harness.py   # main.c harness generation
│           └── build.py              # toolchain invocation (as, gcc, ld)
├── tests/
│   ├── test_dsl.py
│   ├── test_lowering.py
│   ├── test_end_to_end.py
│   └── test_benchmarks.py
├── firmware/
│   └── harness/
│       ├── main.c.template
│       ├── linker.ld
│       └── Makefile
└── examples/
    ├── elementwise.py
    ├── dot_product.py
    ├── matmul.py
    ├── softmax.py
    └── rmsnorm.py
```

## Implementation Order

### Milestone 1: Minimal end-to-end (Week 1)

`A + B` compiling from Python to assembly, assembling, and running on the emulator with correct output.

1. Repo setup, xDSL install, RISC-V toolchain verification
2. Minimal `array` dialect (`array.add` only)
3. `Array` class with `__add__` tracing
4. `array.add` → `linalg.generic`
5. Bufferize (tensor → memref), skip tiling/fusion
6. `linalg.generic [addf] on memrefs` → `npu.fvadd`
7. `npu.fvadd` → assembly
8. Firmware harness, compile, execute on emulator
9. Verify output matches NumPy

### Milestone 2: Fusion and elementwise ops (Week 2)

1. Remaining elementwise ops (sub, mul, div, neg)
2. Elementwise fusion pass
3. Fused expression tests: `(A + B) * C`, `(A + B) * 0.5 + C`
4. Unary ops (relu, gelu, exp) with NPU lowering
5. Buffer reuse pass: liveness analysis to share intermediate memref.alloc buffers
6. Verify fused chains produce minimal instruction counts

### Milestone 3: Reductions and dot product (Week 2-3)

1. `array.sum`, `array.max` → `npu.fvreduce`, `npu.fvmax`
2. `array.dot` → `npu.fvmac` + `npu.frstacc`
3. Matmul (outer loops + dot product)
4. Instruction count benchmarks vs hand-written firmware

### Milestone 4: Composites (Week 3)

1. Softmax end-to-end
2. RMSNorm end-to-end
3. Verification against NumPy/PyTorch reference

### Milestone 5: Tiling (Week 3-4)

1. Tiling pass with configurable memory budget
2. Large array tests (100K+ elements)
3. Correctness verification: tiled == untiled
4. Instruction count comparison: tiled vs untiled

### Milestone 6: LLVM backend (Week 5-7)

1. Build LLVM from source with RISC-V target
2. TableGen NPU instruction definitions
3. LLVM intrinsic definitions
4. Instruction selection patterns
5. xDSL pipeline emitting LLVM dialect instead of assembly
6. End-to-end through LLVM backend
7. Output parity with Phase 1 assembly emitter

## Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Development language |
| xDSL | latest | MLIR framework |
| riscv-npu | local (with new instructions) | Target emulator |
| riscv32-unknown-elf-gcc | 13+ | Cross-compiler toolchain |
| NumPy | latest | Reference + verification |
| LLVM source | 18+ | Phase 2 only |

## Key Design Decisions

**xDSL over C++ MLIR**: Faster iteration, Python-native, 1:1 compatible with MLIR IR. Same choice as the WSE stencil paper.

**Direct linalg → NPU lowering, no vector dialect**: The NPU's memory-to-memory model (address + length operands) maps to linalg's destination-passing style, not to the vector dialect's register-to-register model. Structurally identical to how the WSE stencil paper lowers linalg to CSL DSD builtins.

**Composite ops retained in the array dialect**: Softmax and RMSNorm have multi-instruction lowering patterns where recognizing the operation as a unit produces better code than decomposing early and trying to re-fuse.

**Assembly emission first, LLVM backend second**: `.insn` directives are production-quality (used by the existing riscv-npu C firmware) and the assembly path serves as a reference implementation and fallback after the LLVM backend exists.
