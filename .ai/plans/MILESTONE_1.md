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

## Completed Tasks

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

## Remaining Tasks

### Task 7: Lowering — `linalg.generic [addf]` on memrefs to `npu.fvadd`

Pattern-match the bufferized `linalg.generic` and replace it with `npu.fvadd`.

**File**: `src/arrax/lowering/linalg_to_npu.py`

**Pattern matching criteria**:
1. `linalg.GenericOp` with exactly 2 `ins` operands, 1 `outs` operand
2. All three indexing maps are 1D identity: `affine_map<(d0) -> (d0)>`
3. Single iterator type: `parallel`
4. Body block has exactly: `arith.addf` + `linalg.yield`
5. All operands are `memref<Nxf32>`

**Emit**: `npu.fvadd src1, src2, dst, n` where:
- `src1` = first `ins` operand
- `src2` = second `ins` operand
- `dst` = `outs` operand
- `n` = `arith.constant` with the static dimension from memref shape

**Tests** (`tests/lowering/test_linalg_to_npu.py`):
- Full pipeline from `array.add` through linalg, bufferize, to `npu.fvadd`
- Verify the final IR contains exactly one `npu.fvadd` and no `linalg.generic`
- Golden-string IR snapshot

### Task 8: Assembly emission — `npu.fvadd` to `.insn` directive

Walk the IR and emit RISC-V assembly text.

**File**: `src/arrax/codegen/asm_emitter.py`

```python
def emit_assembly(module: ModuleOp) -> str:
    """Walk a module containing npu ops and emit RISC-V assembly."""
```

The emitter walks the `func.func` operation and emits:

```asm
    .globl npu_kernel
    .type npu_kernel, @function
npu_kernel:
    # a0 = pointer to input A (src1)
    # a1 = pointer to input B (src2)
    # a2 = pointer to output (dst)
    # a3 = element count (n)

    # Copy src2 to dst (LW/SW loop)
    # NPU.FVADD rd=a3, rs1=a0, rs2=a2
    .insn r 0x2B, 0x0, 0x07, a3, a0, a2

    ret
```

**Register assignment**: Function args arrive in `a0`–`a7` per RISC-V calling convention. The argument order matches the FuncOp signature (inputs first, then output).

**FVADD encoding**: `.insn r 0x2B, 0x0, 0x07, rd, rs1, rs2` — rd=element count, rs1=src1 addr, rs2=src2/dst addr.

For Milestone 1, always emit the copy + insn sequence (simple, correct).

**File**: `src/arrax/codegen/firmware_harness.py`

Generate a `main.c` that:
- Declares float arrays A[N], B[N], out[N]
- Initializes A and B with known values
- Calls `npu_kernel(A, B, out, N)`
- Prints output as raw IEEE 754 hex bytes via UART (freestanding, no printf)

```c
void print_float_hex(float f) {
    unsigned char *bytes = (unsigned char *)&f;
    for (int i = 3; i >= 0; i--)
        print_hex_byte(bytes[i]);
    putchar('\n');
}
```

Python `parse_output()` reconstructs via `struct.unpack('!f', bytes.fromhex(line))`.

**File**: `src/arrax/codegen/build.py`

Invoke the toolchain using riscv-npu's common firmware files:
```python
RISCV_NPU_DIR = os.environ.get("RISCV_NPU_DIR", "../riscv-npu")
COMMON_DIR = os.path.join(RISCV_NPU_DIR, "firmware", "common")
# Uses precompiled start.o, shared linker.ld, syscalls.c
```

**Tests** (`tests/codegen/test_asm_emitter.py`):
- Generate assembly for `A + B`, verify `.insn r 0x2B, 0x0, 0x07` present
- Verify assembly contains `.globl`, `ret`, register names

### Task 9: End-to-end test — compile, assemble, run, verify

**File**: `tests/test_end_to_end.py`

```python
def test_add_end_to_end():
    import numpy as np

    N = 64
    A_data = np.arange(N, dtype=np.float32)
    B_data = np.arange(N, dtype=np.float32) * 2
    expected = A_data + B_data

    # 1. Trace
    dag, param_names = trace(lambda A, B: A + B, {"A": (N,), "B": (N,)})

    # 2. Lower through full pipeline
    module = dsl_to_array(dag, param_names, {"A": (N,), "B": (N,)})
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)

    # 3. Emit assembly + firmware harness
    asm_text = emit_assembly(module)
    harness_c = generate_harness(A_data, B_data, N)

    # 4. Build and run
    elf_path = build_firmware(asm_text, harness_c)
    output = run_emulator(elf_path)

    # 5. Compare
    actual = parse_output(output)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
```

Requires riscv-npu emulator + RISC-V toolchain installed.

### Task 10: Pipeline orchestration

Wire everything into a single `compile()` entry point.

**File**: `src/arrax/pipeline.py`

```python
def compile(fn, shapes: dict[str, tuple[int, ...]]) -> CompiledKernel:
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
    asm = emit_assembly(module)
    return CompiledKernel(asm=asm, shapes=shapes)
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
                           │                              Task 8 (npu → asm + harness)
                           │                                    │
                           │                              Task 9 (end-to-end test)
                           │                                    │
                           └──────────────── Task 10 (pipeline.compile)
```

Tasks 1–6: DONE.
Task 7 → 8 → 9 → 10: sequential, remaining work.

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

Emulator: `uv run python -m riscv_npu run firmware.elf`

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
