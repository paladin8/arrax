# Milestone 5: LLVM Backend

## Goal

Add an LLVM-based codegen path alongside the existing assembly emitter. This involves two bodies of work: (1) an LLVM RISC-V vendor extension (`Xnpu`) that teaches LLVM about the NPU's 16 FP instructions, and (2) a direct LLVM IR emitter via llvmlite that walks the existing npu dialect IR and produces `.ll` text.

The assembly emitter remains the default. The LLVM path is opt-in via `backend="llvm"` and produces LLVM IR text that can be compiled by a patched `llc`.

## Architecture overview

```
                     existing path (default)
                    ┌───────────────────────────────┐
                    │ npu dialect IR                │
                    │   → asm_emitter.py            │
                    │   → .S text                   │
                    │   → riscv-gcc → ELF           │
                    └───────────────────────────────┘

                     new LLVM path (backend="llvm")
                    ┌─────────────────────────────────┐
                    │ npu dialect IR                  │
                    │   → llvm_emitter.py (llvmlite)  │
                    │   → .ll text                    │
                    │   → llc (patched) -filetype=obj │
                    │   → kernel.o                    │
                    │   → riscv-gcc link → ELF        │
                    └─────────────────────────────────┘
```

Both paths share all passes up through NpuCanonicalize + verify. They diverge only at the final codegen stage.

## Scope

### In scope

- RISC-V vendor extension `Xnpu0p1` with all 16 FP NPU instructions
- TableGen instruction definitions (R-type encoding, opcode 0x2B)
- LLVM intrinsic declarations (`@llvm.riscv.npu.*`)
- Instruction selection patterns (intrinsic → instruction, 1:1)
- Direct LLVM IR emitter via llvmlite (walks npu dialect IR, emits `.ll` text)
- Pipeline integration with `backend="llvm"` flag
- Build tooling: `build-llvm.sh` script with idempotent patching
- LLVM IR golden tests (21 tests covering all ops, tiling, composites, structure)
- Output parity tests (7 tests, gated on `llc` availability)

### Out of scope

- Integer NPU instructions (14 ops) — FP only
- Replacing the assembly emitter — both paths coexist
- LLVM optimization passes — rely on `llc -O2` defaults
- Custom LLVM register classes for `facc` — modeled as implicit state
- xDSL LLVM dialect lowering passes — xDSL lacks arith/memref/scf/func→LLVM conversions

## LLVM vendor extension: `Xnpu`

### Instruction encoding

All FP NPU instructions use R-type encoding with opcode `0x2B`. Vector ops use funct3=0 with varying funct7. Scalar ops (FRELU, FGELU, FRSTACC) use varying funct3.

| Instruction    | funct3 | funct7 | Operands (LLVM)                    |
|----------------|--------|--------|------------------------------------|
| NPU.FMACC      | 0      | 0x00   | FPR rs1, FPR rs2 (facc implicit)   |
| NPU.FVMAC      | 0      | 0x01   | GPR rs1, GPR rs2, GPR rd (n)       |
| NPU.FVEXP      | 0      | 0x02   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FRSQRT     | 0      | 0x03   | FPR rs1 → FPR rd                  |
| NPU.FVMUL      | 0      | 0x04   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVREDUCE   | 0      | 0x05   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVMAX      | 0      | 0x06   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVADD      | 0      | 0x07   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVSUB      | 0      | 0x08   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVRELU     | 0      | 0x09   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVGELU     | 0      | 0x0A   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVDIV      | 0      | 0x0B   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FVSUB_SC   | 0      | 0x0C   | GPR rs1, GPR rs2 (dst), GPR rd (n) |
| NPU.FRELU      | 1      | —      | FPR rs1 → FPR rd                  |
| NPU.FGELU      | 4      | —      | FPR rs1 → FPR rd                  |
| NPU.FRSTACC    | 5      | —      | → FPR rd (facc implicit)           |

### LLVM intrinsic signatures

Each intrinsic maps 1:1 to a hardware instruction. The LLVM IR emitter handles all higher-level translation (memcpy for in-place ops, facc load sequences for scalar-vector ops, FRSTACC brackets for dot products).

All vector ops share the signature `(ptr, ptr, i32) -> void`, mapping to `(rs1, rs2, rd)` in the instruction encoding:
```
; Binary in-place: dst[i] = src1[i] op dst[i]
declare void @llvm.riscv.npu.fvadd(ptr %src1, ptr %src2_dst, i32 %n)
declare void @llvm.riscv.npu.fvsub(ptr %src1, ptr %src2_dst, i32 %n)

; Unary: dst[i] = f(src[i])
declare void @llvm.riscv.npu.fvexp(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvrelu(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvgelu(ptr %src, ptr %dst, i32 %n)

; Scalar-vector: dst[i] = f(src[i], facc)  — facc loaded separately
declare void @llvm.riscv.npu.fvmul(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvdiv(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvsub_scalar(ptr %src, ptr %dst, i32 %n)

; Dot product: facc += dot(lhs[0..n], rhs[0..n])
declare void @llvm.riscv.npu.fvmac(ptr %lhs, ptr %rhs, i32 %n)
```

Reductions return `float` (FPR result in rd):
```
declare float @llvm.riscv.npu.fvreduce(ptr %src, i32 %n)
declare float @llvm.riscv.npu.fvmax(ptr %src, i32 %n)
```

Scalar FP ops:
```
declare void @llvm.riscv.npu.fmacc(float %a, float %b)
declare float @llvm.riscv.npu.frsqrt(float %x)
declare float @llvm.riscv.npu.frstacc()
declare float @llvm.riscv.npu.frelu(float %x)
declare float @llvm.riscv.npu.fgelu(float %x)
```

All instruction selection patterns are trivial 1:1 `Pat<>` records — no custom `ISelLowering` needed.

### Patch structure

```
llvm-npu/
  README.md                  — quick start + manual setup instructions
  build-llvm.sh              — automated: clone → copy → patch → build → test
  RISCVInstrInfoXnpu.td      — new file → lib/Target/RISCV/
  IntrinsicsRISCVXnpu.td     — new file → include/llvm/IR/
  tests/
    test_vector_ops.ll        — 9 vector instruction tests
    test_reductions.ll        — FVREDUCE + FVMAX tests
    test_scalar_ops.ll        — FMACC, FRSQRT, FRSTACC, FRELU, FGELU tests
    run_tests.sh              — standalone test runner (grep-based, 16 checks)
```

`build-llvm.sh` handles the full build pipeline with idempotent patching (grep guards + post-patch verification). Uses `ninja -j2` to avoid OOM on constrained systems. The `JOBS` env var overrides parallelism.

Files modified in the LLVM tree (one-line additions each, applied by `build-llvm.sh`):

| File                                           | Change                                    |
|------------------------------------------------|-------------------------------------------|
| `lib/Target/RISCV/RISCVFeatures.td`           | `HasVendorXnpu` feature + predicate       |
| `lib/Target/RISCV/RISCVInstrInfo.td`           | `include "RISCVInstrInfoXnpu.td"`         |
| `include/llvm/IR/IntrinsicsRISCV.td`           | `include "IntrinsicsRISCVXnpu.td"`        |
| `lib/Target/RISCV/RISCVSubtarget.h`           | `HasVendorXnpu` member + accessor         |

## LLVM IR emitter (`src/arrax/codegen/llvm_emitter.py`)

### Design decision: direct emitter vs xDSL lowering passes

The original design called for 5 xDSL lowering passes (NpuToLlvm, ArithToLlvm, ScfToLlvm, MemrefToLlvm, FuncToLlvm) that would convert npu dialect IR into xDSL's LLVM dialect, then use `convert_module()` to emit `.ll` text.

This was abandoned because xDSL 0.59.0 lacks the necessary conversion passes (`arith-to-llvm`, `memref-to-llvm`, `scf-to-llvm`, `func-to-llvm`) and its LLVM dialect has no unconditional branch operation. Writing all 5 passes from scratch would have been substantial work with no reuse benefit.

Instead, a direct llvmlite emitter walks the post-NpuCanonicalize IR (the same input as `asm_emitter.py`) and builds LLVM IR using the llvmlite library. This mirrors the asm emitter's traversal pattern but outputs LLVM IR instead of assembly text, keeping the two backends structurally parallel.

### Architecture

```python
def emit_llvm_ir(module: ModuleOp) -> str:
    """Walk npu dialect IR, build LLVM IR via llvmlite, return .ll text."""
    emitter = _LlvmEmitter()
    emitter.emit_module(module)
    return str(emitter.get_module())
```

`_LlvmEmitter` is a stateful class that maintains:
- An SSA value map (`id(SSAValue) → ir.Value`) for translating xDSL values to llvmlite values
- An intrinsic cache for deduplicating `@llvm.riscv.npu.*` declarations
- An `ir.IRBuilder` that advances through basic blocks

### Op coverage

| xDSL dialect | Ops handled                                                               |
|--------------|---------------------------------------------------------------------------|
| npu          | FVAdd, FVSub, FVRelu, FVExp, FVMul, FVDiv, FVSubScalar, FRsqrt,         |
|              | FVReduce, FVMax, FVMac                                                    |
| arith        | ConstantOp (i32, f32), SubiOp, MinSIOp, AddfOp, DivfOp                   |
| memref       | AllocOp, AllocaOp, SubviewOp, StoreOp, LoadOp                            |
| scf          | ForOp (with iter_args + phi nodes), YieldOp                               |
| func         | FuncOp, ReturnOp                                                          |

### Key emission patterns

**Binary vector ops (fvadd, fvsub):** Hardware writes in-place to rs2. If `src2 != dst`, emit `llvm.memcpy` first to copy src2 → dst, then call the intrinsic with `(src1, dst, n)`.

**Scalar-vector ops (fvmul, fvdiv, fvsub.scalar):** Load scalar into facc via `frstacc` (zero) + `fmacc(scalar, 1.0)`, then call the vector intrinsic.

**Dot product (fvmac):** Untiled: emit full FRSTACC bracket inline (zero → fvmac → read). Tiled: `_emit_for` wraps the loop with FRSTACC bracket (zero before header, read after exit).

**scf.for:** Lowered to `preheader → header → body → header / exit` basic blocks with phi nodes for IV and iter_args. IV advancement and yield-to-phi threading handled in the yield handler.

**Reductions (fvreduce, fvmax):** Return `float` partial result from intrinsic, then combine with accumulator phi via `fadd` (sum) or `maxnum` + NaN check (max).

## Pipeline integration

### API change

```python
def compile_to_asm(
    fn: Callable[..., Array],
    shapes: dict[str, tuple[int, ...]],
    backend: str = "asm",
) -> tuple[str, list[str]]:
```

- `backend="asm"` — existing path, returns assembly text (default, unchanged)
- `backend="llvm"` — new path, returns LLVM IR text (`.ll`)

### File layout

```
src/arrax/
  codegen/
    llvm_emitter.py          — emit_llvm_ir() entry point (direct llvmlite emitter)
    build.py                 — build_elf_from_ll(), _generate_firmware_wrapper()
    asm_emitter.py           — unchanged

llvm-npu/
  README.md                  — quick start + manual setup instructions
  build-llvm.sh              — automated clone → patch → build → test
  RISCVInstrInfoXnpu.td      — instruction definitions + isel patterns
  IntrinsicsRISCVXnpu.td     — intrinsic declarations
  tests/
    test_vector_ops.ll        — 9 vector instruction tests
    test_reductions.ll        — reduction instruction tests
    test_scalar_ops.ll        — scalar FP instruction tests
    run_tests.sh              — standalone test runner (16 mnemonic checks)

tests/codegen/
  test_llvm_emitter.py       — 21 golden tests (ops, tiling, composites, structure)
  test_llvm_parity.py        — 7 E2E parity tests (gated on llc availability)
```

## Testing strategy

### Phase 1: LLVM patch tests (standalone, shell)

Hand-written `.ll` files in `llvm-npu/tests/` that call `@llvm.riscv.npu.*` intrinsics. Validated with `run_tests.sh` which compiles each file via `llc` and greps for expected `.insn` mnemonics. 16 checks total covering all FP instructions. These live outside the Python test suite — they validate the LLVM patches with no Python dependency.

### Phase 2: LLVM IR golden tests (Python, no LLVM needed)

21 tests in `tests/codegen/test_llvm_emitter.py`:
- **Basic (9):** untiled vector ops (add, sub, relu, exp, scalar-mul, scalar-div, sum, amax, dot)
- **Tiled (3):** add, dot, sum with N=128 — verify loop structure (phi nodes, for.header blocks)
- **Composite (3):** softmax, rmsnorm, fused expression — verify multi-op sequences
- **Defensive (1):** alloca + load for ops not in current pipeline
- **Structural (5):** target triple, function signature, memcpy for in-place ops, no register names

All tests check `.ll` text via substring matching — no `llc` dependency.

### Phase 3: Output parity tests (Python, requires patched `llc`)

7 tests in `tests/codegen/test_llvm_parity.py`:
- `add_small` (N=4), `add_tiled` (N=128), `sum`, `dot`, `softmax`, `rmsnorm`, `fused_expression`
- Same expression → both backends → riscv-npu emulator → `assert_allclose(rtol=1e-5)`
- Gated on `_has_llc` via `pytest.mark.skipif`

### Success criteria (all met)

1. All 16 FP instructions defined in TableGen; `llc` compiles hand-written `.ll` to correct `.insn` encoding
2. Direct llvmlite emitter covers every op the pipeline produces (npu, arith, memref, scf, func)
3. `emit_llvm_ir()` successfully produces `.ll` text for all test expressions
4. When patched `llc` is available: output parity with asm_emitter for all 7 representative expressions
5. All 518 tests pass — the LLVM path is additive

## Implementation plan (completed)

### Phase 1: LLVM vendor extension patches

**1.1 Scaffold `llvm-npu/` directory** ✓
- `llvm-npu/README.md` with quick start + manual setup instructions
- `llvm-npu/tests/` directory with `.ll` test files and `run_tests.sh`

**1.2 Write `IntrinsicsRISCVXnpu.td`** ✓
- 16 LLVM intrinsic declarations grouped by memory effect:
  - `IntrArgMemOnly`: pure vector ops (fvadd, fvsub, fvexp, fvrelu, fvgelu, fvdiv, fvsub_scalar)
  - `IntrInaccessibleMemOrArgMemOnly`: facc-touching vector ops (fvmul, fvmac)
  - `IntrInaccessibleMemOnly`: facc-only ops (fmacc, frstacc)
  - `IntrNoMem`: pure scalar ops (frsqrt, frelu, fgelu)
- Reductions (fvreduce, fvmax): `IntrArgMemOnly` with FPR result

**1.3 Write `RISCVInstrInfoXnpu.td`** ✓
- 4 encoding classes: `XNpuVecGPR` (9 vector ops), `XNpuReduce` (2 reductions), `XNpuScalarFF` (3 scalar unary), inline FMACC/FRSTACC
- Uses `Inst{6-0} = OPC_CUSTOM_1.Value` (not `let Opcode =` which doesn't exist in `RVInst`)
- FMACC/FRSTACC set `mayLoad=1, mayStore=1` to match intrinsic memory effects
- All isel patterns are trivial `Pat<>` records — no custom `ISelLowering` needed
- Gated on `let Predicates = [HasVendorXnpu]`

**1.4 Write hand-written `.ll` test files** ✓
- `test_vector_ops.ll`: 9 vector instructions
- `test_reductions.ll`: FVREDUCE + FVMAX with accumulation patterns
- `test_scalar_ops.ll`: FMACC, FRSQRT, FRSTACC, FRELU, FGELU
- `run_tests.sh`: 16 mnemonic checks (compile + grep)

**1.5 Write `build-llvm.sh`** ✓
- Automated: clone LLVM → copy TableGen files → patch (idempotent grep guards + post-patch verification) → cmake → `ninja -j2` → run tests
- GNU sed check, `JOBS` env var for parallelism override

### Phase 2: Direct LLVM IR emitter

> Original design called for 5 xDSL lowering passes. Pivoted to direct llvmlite emitter because xDSL 0.59.0 lacks arith/memref/scf/func→LLVM conversions and has no unconditional branch in its LLVM dialect.

**2.1 `llvm_emitter.py`** ✓
- `_LlvmEmitter` class: walks xDSL IR, builds llvmlite IR
- Handles all 11 NPU ops, 5 arith ops, 5 memref ops, scf.for/yield, func.func/return
- Key patterns: `_emit_facc_load` (FRSTACC + FMACC bracket), `_emit_memcpy` (llvm.memcpy intrinsic), `_emit_for` with FRSTACC bracket for FVMac loops, phi nodes for iter_args
- Opaque pointers (`ir.PointerType()`), GEP with `source_etype=_f32`
- Target: `riscv32-unknown-none-elf`

**2.2 Pipeline integration** ✓
- `backend` parameter added to `compile_to_asm()` in `pipeline.py`
- `backend="llvm"` calls `emit_llvm_ir()` instead of `emit_assembly()`
- Both backends share all passes through NpuCanonicalize + verify

**2.3 Build tooling** ✓
- `build_elf_from_ll()` in `build.py`: `.ll` → `llc -filetype=obj -O2` → `kernel.o` → link
- `_generate_firmware_wrapper()`: main() + data declarations + memcpy stub
- Memcpy stub: byte-by-byte copy using t0/t3, preserving a0 return value
- `llc -filetype=obj` bypasses GCC assembler (which doesn't know custom mnemonics)

### Phase 3: Tests

**3.1 LLVM IR golden tests** ✓
- 21 tests in `tests/codegen/test_llvm_emitter.py`
- Coverage: basic (9), tiled (3), composite (3), defensive (1), structural (5)

**3.2 Output parity tests** ✓
- 7 tests in `tests/codegen/test_llvm_parity.py`
- add_small, add_tiled, sum, dot, softmax, rmsnorm, fused_expression
- Gated on `_has_llc` via `pytest.mark.skipif`
- `assert_allclose(rtol=1e-5)` between both backends
