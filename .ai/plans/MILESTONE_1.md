# Milestone 1: Minimal End-to-End

`A + B` compiling from Python DSL to RISC-V assembly, assembling with the riscv32 toolchain, and running on the riscv-npu emulator with correct output verified against NumPy.

## Goal

Prove the full pipeline works end-to-end with the simplest possible case. Every layer exists, every interface is real, nothing is mocked. The output of `(A + B)` on the emulator matches NumPy to float32 tolerance.

## External Dependencies

**riscv-npu emulator**: FVADD is implemented in `riscv-npu/src/riscv_npu/npu/fp_instructions.py` (line 376) with passing tests in `tests/npu/test_fp_vector_ops.py`. No blocking dependency — all tasks including end-to-end testing can proceed.

**riscv-npu as a separate project**: The emulator lives at `../riscv-npu` with its own venv. The build script and end-to-end tests must invoke it cross-project:
```python
subprocess.run(
    ["uv", "run", "--directory", RISCV_NPU_DIR, "python", "-m", "riscv_npu", "run", elf_path],
    capture_output=True, text=True,
)
```
The firmware build also needs files from riscv-npu:
- `../riscv-npu/firmware/common/start.o` (precompiled startup)
- `../riscv-npu/firmware/common/syscalls.c` (UART I/O)
- `../riscv-npu/firmware/common/linker.ld` (memory map)

Configure the path via environment variable `RISCV_NPU_DIR` (default: `../riscv-npu`).

## Test Structure

Tests mirror the source layout:
```
tests/
├── helpers.py                          (make_module, lower_to_linalg, bufferize)
├── dsl/
│   ├── test_array.py                   (6 tests)
│   └── test_tracer.py                  (7 tests)
├── dialects/
│   ├── test_array_dialect.py           (11 tests)
│   └── test_npu_dialect.py             (11 tests)
└── lowering/
    ├── test_dsl_to_array.py            (7 tests)
    ├── test_array_to_linalg.py         (7 tests)
    └── test_bufferize.py               (9 tests)
```

## Tasks

### Task 1: Array dialect — `array.add` (DONE)

**File**: `src/arrax/dialects/array_dialect.py`

- `AddOp` with `operand_def(TensorType)` for lhs/rhs, `result_def(TensorType)`
- `Pure` trait, `verify_()` rejects mismatched operand types
- Declarative `assembly_format` with IR round-trip support

### Task 2: Array class with `__add__` tracing (DONE)

**File**: `src/arrax/dsl/array.py` — `Array` class with `__add__`, `is_leaf` property

**File**: `src/arrax/dsl/tracer.py` — `trace()` returns `tuple[Array, list[str]]`

**Deviation from plan**: `trace()` returns `(result_dag, param_names)` instead of bare `Array`. This is necessary because `dsl_to_array` needs param order for the FuncOp signature, and walking DAG leaves doesn't recover it reliably.

### Task 3: DSL to array dialect IR (DONE)

**File**: `src/arrax/lowering/dsl_to_array.py`

```python
def dsl_to_array(result: Array, param_names: list[str], shapes: dict) -> ModuleOp:
```

- Recursive post-order DAG walk, `id(node)` caching for diamond dedup
- Emits `func.func @kernel` with block args in signature order

**Deviation from plan**: Takes `param_names` as explicit arg (from trace output) instead of the plan's 2-arg signature.

### Task 4: array.add → linalg.generic (DONE)

**File**: `src/arrax/lowering/array_to_linalg.py`

- `AddToLinalgPattern` via `PatternRewriteWalker` + `GreedyRewritePatternApplier`
- Generalized to N-dimensional tensors (not just 1D)
- `tensor.EmptyOp` for output, identity `AffineMap`, parallel iterators

**xDSL API corrections from plan**:
- `empty.tensor` not `empty.result` (EmptyOp result is named `tensor`)
- `arith.AddfOp` not `arith.Addf`
- `IteratorTypeAttr.parallel()` factory method, not `IteratorTypeAttr(IteratorType.PARALLEL)`

### Task 5: Bufferization (DONE)

**File**: `src/arrax/lowering/bufferize.py`

Custom pass (xDSL 0.59.0 has no general-purpose bufferization). Rebuilds FuncOp from scratch:
- Tensor args → memref args, output memrefs added as function args (destination-passing)
- Final `tensor.empty` → output function arg; intermediates → `memref.alloc`
- `linalg.generic` on memrefs with no result types; body cloned via `Region.clone()`
- SSA remapping via `value_map` dict
- Raises `ValueError` on unrecognized ops in the block

**No deallocation** of intermediate allocs — documented as future work (Milestone 2).

### Task 6: NPU dialect — `npu.fvadd` (DONE)

**File**: `src/arrax/dialects/npu_dialect.py`

- 3-address form: `src1`, `src2`, `dst` (MemRefType), `n` (IndexType)
- No results (writes to dst)
- `verify_()` checks all memrefs match and have f32 element type
- Declarative `assembly_format` with IR round-trip support

### Task 7: Lowering — `linalg.generic [addf]` on memrefs to `npu.fvadd` (DONE)

**File**: `src/arrax/lowering/linalg_to_npu.py`

- `LinalgAddToNpuPattern` matches 5 criteria: 2 ins + 1 outs, 1D identity maps, parallel iterator, addf body, memref operands
- Emits `arith.constant N : index` + `npu.fvadd src1, src2, dst, n`
- Non-matching ops pass through unchanged (tested with mulf negative case)

### Task 8: Assembly emission — `npu.fvadd` to `.insn` directive

Walk the IR and emit RISC-V assembly text.

**File**: `src/arrax/codegen/asm_emitter.py`

```python
def emit_assembly(module: ModuleOp) -> str:
    """Walk a module containing npu ops and emit RISC-V assembly."""
```

The emitter walks the `func.func` and emits a complete `.S` file. For each op:
- `arith.constant` → note the value (used by npu ops for element count)
- `npu.fvadd` → copy loop (LW/SW from src2 to dst) + `.insn r 0x2B, 0x0, 0x07, rd, rs1, rs2`
- `memref.alloc` → `.comm` directive for static allocation in .bss
- `func.return` → `ret`

**Output for A + B:**
```asm
    .globl npu_kernel
    .type npu_kernel, @function
npu_kernel:
    # a0 = src1 (A), a1 = src2 (B), a2 = dst (out)

    # Copy B to dst (LW/SW loop, needed because FVADD writes in-place to rs2)
    li t0, 1024           # element count
    mv t1, a1             # src2 ptr
    mv t2, a2             # dst ptr
.Lcopy_0:
    lw t3, 0(t1)
    sw t3, 0(t2)
    addi t1, t1, 4
    addi t2, t2, 4
    addi t0, t0, -1
    bnez t0, .Lcopy_0

    # NPU.FVADD: dst[i] = src1[i] + dst[i]
    li a3, 1024
    .insn r 0x2B, 0x0, 0x07, a3, a0, a2

    ret
```

**Register assignment**: Function args in `a0`–`a7` per RISC-V calling convention. Argument order matches the FuncOp signature (inputs first, then output memref). Scratch registers `t0`–`t3` for the copy loop.

**FVADD encoding**: `.insn r 0x2B, 0x0, 0x07, rd, rs1, rs2` — rd=element count, rs1=src1 addr, rs2=dst addr (which now contains src2's data after the copy).

**Tests** (`tests/codegen/test_asm_emitter.py`):
- Generate assembly for `A + B`, verify `.insn r 0x2B, 0x0, 0x07` present
- Verify assembly contains `.globl npu_kernel`, `ret`
- Verify copy loop is present (LW/SW pattern)
- Golden-string snapshot of full output

### Task 9: End-to-end test — compile, assemble, run, verify

Uses the riscv-npu library API (`from riscv_npu import Emulator`) — no firmware harness generation, no UART parsing. riscv-npu is an editable path dependency.

**Requires**: riscv-npu Phase 12 (library API) implemented + RISC-V toolchain installed.

**File**: `tests/test_end_to_end.py`

```python
def test_add_end_to_end():
    import numpy as np
    from riscv_npu import Emulator

    N = 64
    A_data = np.arange(N, dtype=np.float32)
    B_data = np.arange(N, dtype=np.float32) * 2
    expected = A_data + B_data

    # 1. Compile: Python DSL → assembly text
    asm_text = compile_to_asm(lambda A, B: A + B, {"A": (N,), "B": (N,)})

    # 2. Assemble: .S → ELF (requires riscv64-unknown-elf-as)
    elf_path = assemble(asm_text)

    # 3. Run on emulator via library API
    emu = Emulator()
    emu.load_elf(elf_path)
    emu.write_f32("A", A_data)
    emu.write_f32("B", B_data)
    result = emu.run()
    assert result.exit_code == 0

    # 4. Read output and compare
    actual = emu.read_f32("out", N)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
```

**Key simplification over original plan**: No `firmware_harness.py`, no `build.py`, no `main.c` generation, no UART hex parsing. The emulator reads/writes arrays directly via `write_f32`/`read_f32` at symbol addresses.

**Still needed**: `assemble()` helper in `src/arrax/codegen/build.py` that invokes `riscv64-unknown-elf-as` to produce an ELF from the `.S` file. The assembly must include data section declarations for the symbol addresses (`.comm A, N*4` etc.) so the emulator can resolve them.

### Task 10: Pipeline orchestration

Wire everything into a single `compile()` entry point.

**File**: `src/arrax/pipeline.py`

```python
def compile_to_asm(fn, shapes: dict[str, tuple[int, ...]]) -> str:
    """Full pipeline: trace → lower → emit assembly text."""
    from xdsl.context import Context
    from arrax.dsl.tracer import trace
    from arrax.lowering.dsl_to_array import dsl_to_array
    from arrax.lowering.array_to_linalg import ArrayToLinalgPass
    from arrax.lowering.bufferize import BufferizePass
    from arrax.lowering.linalg_to_npu import LinalgToNpuPass
    from arrax.codegen.asm_emitter import emit_assembly

    dag, param_names = trace(fn, shapes)
    module = dsl_to_array(dag, param_names, shapes)
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    return emit_assembly(module)
```

## Task Dependency Graph

```
Task 1 (array dialect)  ──┐
Task 2 (Array tracing)  ──┼── Task 3 (DSL → array IR) ── Task 4 (array → linalg)
                           │                                    │
                           │                              Task 5 (bufferize)
                           │                                    │
Task 6 (npu dialect)  ─────┼──────────────── Task 7 (linalg → npu)
                           │                                    │
                           │                              Task 8 (npu → asm)
                           │                                    │
                           │                              Task 9 (end-to-end test)
                           │                                    │
                           └──────────────── Task 10 (pipeline)
```

Tasks 1–7: DONE.
Task 8 → 9 → 10: sequential, remaining work.
Task 9 blocked on riscv-npu Phase 12 (library API).

## NPU FVADD Encoding Reference

```
Instruction: NPU.FVADD
Format:      R-type
Opcode:      0x2B (FP NPU custom space)
funct3:      0x0
funct7:      0x07 (0000111)
Operands:    rd = element count register
             rs1 = source 1 base address register
             rs2 = source 2 base address register (also destination)
Semantics:   for i in 0..regs[rd]-1:
                 mem_f32[regs[rs2] + i*4] = mem_f32[regs[rs1] + i*4] + mem_f32[regs[rs2] + i*4]

Assembly:    .insn r 0x2B, 0x0, 0x07, a3, a1, a0
             # a3 = n, a1 = src1 addr, a0 = src2/dst addr
```

## Firmware Memory Layout Reference

```
0x10000000  UART (8 bytes, read/write)
0x80000000  RAM start (code + data loaded here by ELF loader)
              .text.init  (_start: set SP, call main, ecall 93)
              .text       (kernel code, harness code)
              .rodata     (string literals, constants)
              .data       (initialized globals — input arrays)
              .bss        (uninitialized globals — output arrays)
0x800FFFF0  Stack top (ORIGIN + LENGTH - 16, grows downward)
0x80100000  RAM end (1 MB total)
```

## Verification Strategy

1. **Unit tests per task**: Each task has its own test file mirroring the source module
2. **IR verification**: Call `module.verify()` after every pass to catch malformed IR early
3. **Golden-string snapshots**: Exact IR comparison for the basic `A + B` case at each pipeline stage
4. **NumPy reference**: End-to-end comparison with `np.testing.assert_allclose(actual, expected, rtol=1e-6)`
5. **Assembly inspection**: Verify the generated `.insn` directives match the encoding table
6. **Edge cases**: `n=0` is a no-op (zero-iteration loop in emulator FVADD). `n=1` is the minimal non-trivial case.

## Known Limitations (Milestone 1)

- **Copy overhead**: The 3-address `npu.fvadd` always emits a copy loop (LW/SW for each element) before the `.insn` FVADD, even when `dst == src2`. First optimization target for Milestone 2.
- **No buffer reuse**: Each intermediate gets its own `memref.alloc`, no deallocation. Liveness-based buffer reuse is Milestone 2.
- **No tiling**: Arrays must fit in memory. Tiling is Milestone 5.
- **No fusion**: Single operation only. Fusion is Milestone 2.
- **Single operation**: Only `array.add`. More elementwise ops in Milestone 2.
