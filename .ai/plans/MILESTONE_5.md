# Milestone 5: LLVM Backend

## Goal

Add an LLVM-based codegen path alongside the existing assembly emitter. This involves two bodies of work: (1) an LLVM RISC-V vendor extension (`Xnpu`) that teaches LLVM about the NPU's 16 FP instructions, and (2) xDSL lowering passes that convert the existing npu dialect IR into xDSL's LLVM dialect, which the existing llvmlite backend emits as `.ll` text.

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
                    │   → NpuToLlvm pass              │
                    │   → ArithToLlvm pass            │
                    │   → ScfToLlvm pass              │
                    │   → MemrefToLlvm pass           │
                    │   → FuncToLlvm pass             │
                    │   → xDSL LLVM dialect IR        │
                    │   → convert_module() [llvmlite] │
                    │   → .ll text                    │
                    │   → llc (patched) → .o          │
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
- xDSL lowering passes: NpuToLlvm, ArithToLlvm, ScfToLlvm, MemrefToLlvm, FuncToLlvm
- Pipeline integration with `backend="llvm"` flag
- Unit tests for each lowering pass
- LLVM IR golden tests
- Output parity tests (gated on `llc` availability)

### Out of scope

- Integer NPU instructions (14 ops) — FP only
- Automated LLVM build system — manual build with README
- Replacing the assembly emitter — both paths coexist
- LLVM optimization passes — rely on `llc -O2` defaults
- Custom LLVM register classes for `facc` — modeled as implicit state

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

Binary vector ops (FVADD, FVSUB — two source arrays):
```
declare void @llvm.riscv.npu.fvadd(ptr %src1, ptr %src2, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvsub(ptr %src1, ptr %src2, ptr %dst, i32 %n)
```

Unary vector ops (FVRELU, FVEXP, FVGELU — one source array):
```
declare void @llvm.riscv.npu.fvrelu(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvexp(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvgelu(ptr %src, ptr %dst, i32 %n)
```

Scalar-vector ops (FVMUL, FVDIV, FVSUB_SCALAR — facc is loaded before the instruction, modeled as an explicit float operand in the intrinsic):
```
declare void @llvm.riscv.npu.fvmul(ptr %src, ptr %dst, i32 %n, float %scalar)
declare void @llvm.riscv.npu.fvdiv(ptr %src, ptr %dst, i32 %n, float %scalar)
declare void @llvm.riscv.npu.fvsub_scalar(ptr %src, ptr %dst, i32 %n, float %scalar)
```

Reduction vector ops (FVREDUCE, FVMAX — chunk reduction into a scratch destination, caller reads result back):
```
declare void @llvm.riscv.npu.fvreduce(ptr %src, ptr %dst, i32 %n)
declare void @llvm.riscv.npu.fvmax(ptr %src, ptr %dst, i32 %n)
```

Dot product (FVMAC — accumulates into facc across calls):
```
declare void @llvm.riscv.npu.fvmac(ptr %lhs, ptr %rhs, i32 %n)
```

Scalar FP ops:
```
declare void @llvm.riscv.npu.fmacc(float %a, float %b)
declare float @llvm.riscv.npu.frsqrt(float %x)
declare float @llvm.riscv.npu.frstacc()
declare float @llvm.riscv.npu.frelu(float %x)
declare float @llvm.riscv.npu.fgelu(float %x)
```

All vector intrinsics have memory side effects (read src, write dst). FMACC and FVMAC write implicit facc state. FRSTACC reads and clears facc. The NpuToLlvm pass handles the translation from xDSL's SSA-threaded reduction model (where acc_in/result are explicit f32 operands) to the hardware's memory-to-memory model (scratch buffers, facc state).

### Patch structure

```
llvm-npu/
  README.md                              — clone LLVM, apply patch, cmake, build
  patches/
    0001-add-xnpu-extension.patch        — single patch against LLVM main
```

Files touched by the patch (all within `llvm/`):

| File                                           | Change                                    |
|------------------------------------------------|-------------------------------------------|
| `lib/Target/RISCV/RISCVFeatures.td`           | `HasVendorXnpu` feature + extension       |
| `lib/Target/RISCV/RISCVInstrInfoXnpu.td` (new)| 16 FP instruction defs, R-type encoding   |
| `lib/Target/RISCV/RISCVInstrInfo.td`           | `include "RISCVInstrInfoXnpu.td"`         |
| `include/llvm/IR/IntrinsicsRISCV.td`           | 16 `@llvm.riscv.npu.*` intrinsic decls    |
| `lib/Target/RISCV/RISCVISelLowering.cpp`       | Intrinsic → custom node lowering          |

## xDSL lowering passes

### NpuToLlvm (`src/arrax/lowering/npu_to_llvm.py`)

Pattern-based `ModulePass`. Each npu op becomes an `llvm.call_intrinsic`:

| npu op          | LLVM intrinsic                      |
|-----------------|-------------------------------------|
| `npu.fvadd`     | `@llvm.riscv.npu.fvadd`            |
| `npu.fvsub`     | `@llvm.riscv.npu.fvsub`            |
| `npu.fvrelu`    | `@llvm.riscv.npu.fvrelu`           |
| `npu.fvexp`     | `@llvm.riscv.npu.fvexp`            |
| `npu.fvmul`     | `@llvm.riscv.npu.fvmul`            |
| `npu.fvdiv`     | `@llvm.riscv.npu.fvdiv`            |
| `npu.fvsub_scalar` | `@llvm.riscv.npu.fvsub_scalar`  |
| `npu.frsqrt`    | `@llvm.riscv.npu.frsqrt`           |
| `npu.fvreduce`  | `@llvm.riscv.npu.fvreduce`         |
| `npu.fvmax`     | `@llvm.riscv.npu.fvmax`            |
| `npu.fvmac`     | `@llvm.riscv.npu.fvmac`            |

Memref operands are converted to `ptr` (LLVM opaque pointer). Index operands become `i32`. Float operands pass through.

### Standard dialect passes (`src/arrax/lowering/lower_to_llvm.py`)

Bundled in one file since each covers a small op subset:

**ArithToLlvm** — `arith.constant` → `llvm.mlir.constant`, `arith.subi` → `llvm.sub`, `arith.minsi` → `llvm.icmp slt` + `llvm.select`, `arith.addf` → `llvm.fadd`, `arith.divf` → `llvm.fdiv`

**ScfToLlvm** — `scf.for` + `scf.yield` → LLVM basic blocks with `llvm.br`, `llvm.cond_br`, `phi` nodes for IVs and iter_args. This is the most complex pass.

**MemrefToLlvm** — `memref.alloc`/`memref.alloca` → `llvm.alloca`, `memref.subview` → `llvm.getelementptr`, `memref.store` → `llvm.store`, `memref.load` → `llvm.load`. All memrefs lower to `ptr` (flat memory model, no descriptor).

**FuncToLlvm** — `func.func` → `llvm.func`, `func.return` → `llvm.return`. Arguments: memref → `ptr`, index → `i32`, f32 → `f32`.

### LLVM IR emitter (`src/arrax/codegen/llvm_emitter.py`)

```python
def emit_llvm_ir(module: ModuleOp) -> str:
    """Lower npu dialect IR to LLVM dialect, emit .ll text."""
    ctx = Context()
    NpuToLlvmPass().apply(ctx, module)
    LowerToLlvmPass().apply(ctx, module)
    llvm_module = convert_module(module)  # xdsl.backend.llvm.convert
    return str(llvm_module)
```

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
  lowering/
    npu_to_llvm.py          — NpuToLlvm pass
    lower_to_llvm.py        — ArithToLlvm, ScfToLlvm, MemrefToLlvm, FuncToLlvm
  codegen/
    llvm_emitter.py          — emit_llvm_ir() entry point
    asm_emitter.py           — unchanged

llvm-npu/
  README.md                  — LLVM build instructions
  patches/
    0001-add-xnpu-extension.patch
  tests/
    test_fvadd.ll            — hand-written .ll for standalone llc testing
    test_fvreduce.ll
    ...
    run_tests.sh             — shell script: llc each .ll, check .insn encoding
```

## Testing strategy

### Phase 1: LLVM patch tests (standalone, shell)

Hand-written `.ll` files in `llvm-npu/tests/` that call `@llvm.riscv.npu.*` intrinsics. Validated with:

```bash
llc -march=riscv32 -mattr=+f,+xnpu test_fvadd.ll -o - | grep ".insn r 0x2B"
```

These live outside the Python test suite — they validate the LLVM patches with no Python dependency.

### Phase 2: Lowering pass unit tests (Python, no LLVM needed)

- `tests/lowering/test_npu_to_llvm.py` — each NPU op → expected `llvm.call_intrinsic`
- `tests/lowering/test_lower_to_llvm.py` — arith/scf/memref/func ops → expected LLVM dialect IR

Construct small xDSL modules, run passes, assert on output ops. No `llc` dependency.

### Phase 3: LLVM IR golden tests (Python, no LLVM needed)

Known Python expressions → `compile_to_asm(..., backend="llvm")` → assert `.ll` text contains expected intrinsic calls, basic block structure, phi nodes. String matching on the output.

### Phase 4: Output parity tests (Python, requires patched `llc`)

Same expression through both backends → emulator → compare numerical results within f32 tolerance. Gated with `pytest.mark.skipif(not shutil.which("llc"))`.

Expressions to test: `A + B`, `softmax(A)`, `rmsnorm(A)`, `dot(A, B)`, `A * 0.5 + B`.

### Success criteria

1. All 16 FP instructions defined in TableGen; `llc` compiles hand-written `.ll` to correct `.insn` encoding
2. Full lowering from post-NpuCanonicalize IR to xDSL LLVM dialect for every op the pipeline produces
3. `convert_module()` successfully emits `.ll` text for all existing test expressions
4. When patched `llc` is available: output parity with asm_emitter for representative expressions
5. Existing 492 tests remain green — the LLVM path is additive

## Implementation plan

### Phase 1: LLVM vendor extension patches

**1.1 Scaffold `llvm-npu/` directory**
- Create `llvm-npu/README.md` with LLVM clone + build instructions
- Create `llvm-npu/patches/` directory
- Create `llvm-npu/tests/` directory with `run_tests.sh` stub

**1.2 Write `RISCVFeatures.td` patch**
- Add `HasVendorXnpu` feature predicate
- Add `Xnpu` extension to the vendor extension list
- Gate on `-march=rv32imf_xnpu0p1`

**1.3 Write `RISCVInstrInfoXnpu.td`**
- Define R-type instruction format for opcode 0x2B
- Define all 16 FP instructions with correct funct3/funct7 encoding
- Vector ops: 3 GPR operands (rs1=addr, rs2=addr/dst, rd=n)
- Scalar ops: FPR operands as appropriate
- Include from `RISCVInstrInfo.td`

**1.4 Write `IntrinsicsRISCV.td` patch**
- Declare all 16 `@llvm.riscv.npu.*` intrinsics
- Vector intrinsics: `(ptr, ptr, ptr, i32) -> void` with memory effects
- Scalar intrinsics: appropriate FP signatures
- Mark memory read/write side effects correctly

**1.5 Write instruction selection patterns**
- `Pat<>` records in `RISCVInstrInfoXnpu.td` mapping each intrinsic to its instruction
- Add `RISCVISelLowering.cpp` changes if custom lowering is needed for intrinsic → node → instruction

**1.6 Write hand-written `.ll` test files**
- One `.ll` file per instruction (or grouped by category)
- `run_tests.sh`: compile each with `llc`, grep for expected `.insn` encoding

**1.7 Generate patch file**
- Build LLVM with patches applied, verify tests pass
- `git diff` → `0001-add-xnpu-extension.patch`

### Phase 2: xDSL lowering passes

**2.1 NpuToLlvm pass**
- `src/arrax/lowering/npu_to_llvm.py`
- Pattern per NPU op: convert memref → ptr, index → i32, emit `llvm.call_intrinsic`
- `tests/lowering/test_npu_to_llvm.py` — unit test each pattern

**2.2 FuncToLlvm patterns**
- In `src/arrax/lowering/lower_to_llvm.py`
- `func.func` → `llvm.func` (memref args → ptr, index → i32)
- `func.return` → `llvm.return`
- Unit tests

**2.3 ArithToLlvm patterns**
- In `lower_to_llvm.py`
- ConstantOp, SubiOp, MinSIOp, AddfOp, DivfOp → LLVM equivalents
- Unit tests

**2.4 MemrefToLlvm patterns**
- In `lower_to_llvm.py`
- AllocOp/AllocaOp → `llvm.alloca`
- SubviewOp → `llvm.getelementptr`
- StoreOp/LoadOp → `llvm.store`/`llvm.load`
- All memrefs lower to `ptr` (flat memory, no descriptor)
- Unit tests

**2.5 ScfToLlvm patterns**
- In `lower_to_llvm.py`
- `scf.for` + `scf.yield` → basic blocks, `llvm.br`, `llvm.cond_br`, phi nodes
- Handle iter_args threading
- Unit tests with both simple loops and reduction loops with iter_args

### Phase 3: Pipeline integration and end-to-end tests

**3.1 LLVM IR emitter**
- `src/arrax/codegen/llvm_emitter.py`
- `emit_llvm_ir(module)` → runs lowering passes, calls `convert_module()`, returns `.ll` text

**3.2 Pipeline integration**
- Add `backend` parameter to `compile_to_asm()` in `pipeline.py`
- `backend="llvm"` calls `emit_llvm_ir()` instead of `emit_assembly()`
- Default remains `"asm"`

**3.3 LLVM IR golden tests**
- `tests/codegen/test_llvm_emitter.py`
- Known expressions → `backend="llvm"` → assert `.ll` contains expected intrinsic calls
- Test: `A + B`, `sum(A)`, `dot(A, B)`, `softmax(A)`, `rmsnorm(A)`

**3.4 Output parity tests (optional, requires patched `llc`)**
- `tests/codegen/test_llvm_parity.py`
- Gated on `llc` availability
- Same expression → both backends → emulator → compare results
