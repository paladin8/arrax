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

## Tasks

### Task 1: Array dialect — `array.add` operation

Define the `array` dialect with a single operation: `array.add`.

**File**: `src/arrax/dialects/array_dialect.py`

```python
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation, irdl_op_definition,
    operand_def, result_def,
)
from xdsl.dialects.builtin import TensorType, Float32Type

@irdl_op_definition
class AddOp(IRDLOperation):
    name = "array.add"

    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        result_type = lhs.type
        super().__init__(operands=[lhs, rhs], result_types=[result_type])

ArrayDialect = Dialect("array", [AddOp], [])
```

**Constraints**:
- Both operands must be `tensor<NxF32>` (1D, float32, static shape)
- Result type equals the input type
- Type verification should reject mismatched shapes

**Tests** (`tests/test_dsl.py`):
- Construct `AddOp` programmatically, verify IR prints as `array.add`
- Verify type mismatch raises a verification error

### Task 2: Array class with `__add__` tracing

Implement the lazy `Array` class that records a computation DAG.

**File**: `src/arrax/dsl/array.py`

```python
class Array:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape
        self.op: str | None = None       # None for inputs
        self.operands: list[Array] = []

    def __add__(self, other: Array) -> Array:
        result = Array(name="", shape=self.shape)
        result.op = "add"
        result.operands = [self, other]
        return result
```

An `Array` with `op=None` is a leaf (function parameter). An `Array` with `op="add"` is a DAG node.

**File**: `src/arrax/dsl/tracer.py`

```python
def trace(fn, shapes: dict[str, tuple[int, ...]]) -> Array:
    """Call fn with placeholder Arrays and return the result DAG."""
    import inspect
    params = list(inspect.signature(fn).parameters.keys())
    inputs = {name: Array(name, shapes[name]) for name in params}
    result = fn(**inputs)
    return result
```

**Tests** (`tests/test_dsl.py`):
- `trace(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})` returns an `Array` with `op="add"` and two leaf operands
- Verify shapes propagate correctly

### Task 3: Lowering — traced DAG to array dialect IR

Walk the traced DAG and emit `array` dialect operations inside an xDSL `ModuleOp`.

**File**: `src/arrax/lowering/dsl_to_array.py`

```python
from xdsl.dialects.builtin import ModuleOp, Float32Type, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

def dsl_to_array(result: Array, shapes: dict[str, tuple[int, ...]]) -> ModuleOp:
    """Convert traced DAG to array dialect IR wrapped in a func.func."""
```

The output IR should look like:
```
builtin.module {
  func.func @kernel(%A: tensor<1024xf32>, %B: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = array.add %A, %B : tensor<1024xf32>
    func.return %0 : tensor<1024xf32>
  }
}
```

**Key details**:
- Function arguments correspond to leaf `Array` nodes (those with `op=None`)
- Argument order follows the original function signature (use `inspect` in the tracer)
- The DAG is walked bottom-up (post-order): emit operations for operands before the operation that uses them
- Each `Array` node maps to exactly one SSA value

**Tests** (`tests/test_lowering.py`):
- Build DAG for `A + B`, lower, verify the module prints expected IR
- Verify the module passes xDSL verification (`module.verify()`)

### Task 4: Lowering — `array.add` to `linalg.generic`

Rewrite `array.add` to a `linalg.generic` with identity indexing maps and an `arith.addf` body.

**File**: `src/arrax/lowering/array_to_linalg.py`

```python
from xdsl.passes import ModulePass
from xdsl.context import Context
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.dialects import arith, linalg, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr, Float32Type, ModuleOp, TensorType,
)
from xdsl.dialects.linalg import IteratorTypeAttr, IteratorType
from xdsl.pattern_rewriter import (
    RewritePattern, PatternRewriter,
    GreedyRewritePatternApplier, PatternRewriteWalker,
    op_type_rewrite_pattern,
)

class AddToLinalgPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter):
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        n = result_type.get_shape()[0]
        f32 = Float32Type()

        # Output init tensor (value-semantic, no allocation)
        empty = tensor.EmptyOp([], result_type)

        # Identity affine map: (d0) -> (d0)
        identity = AffineMap.identity(1)  # 1-dim identity
        maps = [AffineMapAttr(identity)] * 3  # ins0, ins1, outs0
        iters = [IteratorTypeAttr(IteratorType.PARALLEL)]

        # Build body region: block with 3 f32 args (a, b, out)
        block = Block(arg_types=[f32, f32, f32])
        add_op = arith.Addf(block.args[0], block.args[1])
        yield_op = linalg.YieldOp(add_op.result)
        block.add_ops([add_op, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.lhs, op.rhs],
            outputs=[empty.result],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[result_type],
        )
        rewriter.replace_matched_op([empty, generic], [generic.res[0]])

class ArrayToLinalgPass(ModulePass):
    name = "array-to-linalg"
    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([AddToLinalgPattern()])
        ).rewrite_module(op)
```

**Key imports** (easy to get wrong):
- `AffineMap` from `xdsl.ir.affine` (NOT `xdsl.ir`)
- `IteratorTypeAttr`, `IteratorType` from `xdsl.dialects.linalg`
- `Context` from `xdsl.context` (NOT `xdsl.ir`)

The output IR:
```
func.func @kernel(%A: tensor<1024xf32>, %B: tensor<1024xf32>) -> tensor<1024xf32> {
  %init = tensor.empty() : tensor<1024xf32>
  %0 = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>,
                       affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
  } ins(%A, %B : tensor<1024xf32>, tensor<1024xf32>)
    outs(%init : tensor<1024xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  } -> tensor<1024xf32>
  func.return %0 : tensor<1024xf32>
}
```

**Key details**:
- Need `tensor.empty()` to create the init/output tensor (value-semantic, no allocation yet)
- Affine maps are identity: `affine_map<(d0) -> (d0)>` for 1D
- The `linalg.generic` body receives scalar block arguments, one per operand + one per output
- Use `linalg.YieldOp` to yield the result

**Tests** (`tests/test_lowering.py`):
- Lower `array.add` and verify the resulting `linalg.generic` structure
- Verify the module passes xDSL verification after lowering

### Task 5: Bufferization — tensor to memref

Convert tensor-based `linalg.generic` to memref-based `linalg.generic`.

**File**: `src/arrax/lowering/bufferize.py`

xDSL 0.59.0 has no general-purpose one-shot bufferization pass. Write a custom pass scoped to the Milestone 1 IR shape. This is the highest-complexity task — it rewrites the function signature and remaps SSA values.

**Strategy**: Rebuild the `FuncOp` from scratch with memref types, rather than patching in place.

```python
class BufferizePass(ModulePass):
    name = "bufferize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, func.FuncOp):
                continue
            self._bufferize_func(func_op)

    def _bufferize_func(self, func_op: func.FuncOp) -> None:
        # 1. Build new function signature:
        #    - Convert each tensor<NxF32> input arg to memref<NxF32>
        #    - For each tensor<NxF32> result, add a memref<NxF32> output arg
        #    - New return type is void (no results)
        #
        # 2. Create new Block with memref-typed block arguments
        #    Map old_block_arg[i] -> new_block_arg[i] for inputs
        #    The output memref is new_block_arg[len(inputs)]
        #
        # 3. Walk operations in the old block:
        #    - tensor.EmptyOp: skip (output buffer is now a function arg)
        #    - linalg.GenericOp: rebuild with memref operands, no result types
        #      Replace the output operand (was tensor.empty result) with
        #      the output memref function argument
        #    - func.ReturnOp: replace with void return (no operands)
        #
        # 4. Replace the old FuncOp with the new one
```

**Key pitfalls**:
- The old `linalg.generic` has `result_types=[tensor<NxF32>]` (tensor semantics produce a value). After bufferization, `result_types=[]` — it writes to the output memref directly.
- The `func.return` in tensor form returns the `linalg.generic` result. After bufferization, `func.return` has no operands.
- SSA remapping: every use of old block args must be replaced with the corresponding new block args. Build a mapping dict and use it when reconstructing operations.

Steps in detail:

1. Replace `func.func` argument types: `tensor<1024xf32>` → `memref<1024xf32>`
2. Add output memref parameter for each function result (destination-passing style)
3. Eliminate `tensor.empty()` — its role is taken by the output memref arg
4. Rebuild `linalg.generic` with memref operands and no result types
5. Replace `func.return %tensor` with `func.return` (void)

**Output IR**:
```
func.func @kernel(%A: memref<1024xf32>, %B: memref<1024xf32>, %out: memref<1024xf32>) {
  linalg.generic {
      indexing_maps = [...],
      iterator_types = ["parallel"]
  } ins(%A, %B : memref<1024xf32>, memref<1024xf32>)
    outs(%out : memref<1024xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  }
  func.return
}
```

**Key change**: After bufferization, the function takes an output memref as a parameter (destination-passing style). The `linalg.generic` has no result types — it writes directly to the output memref.

**Tests** (`tests/test_lowering.py`):
- Bufferize the `linalg.generic` from Task 4, verify memref types
- Verify the function signature gains an output parameter

### Task 6: NPU dialect — `npu.fvadd` operation

Define the `npu` dialect with `npu.fvadd`.

**File**: `src/arrax/dialects/npu_dialect.py`

```python
@irdl_op_definition
class FVAddOp(IRDLOperation):
    name = "npu.fvadd"

    src1 = operand_def(MemRefType)    # source 1 address
    src2 = operand_def(MemRefType)    # source 2 address (also destination)
    n = operand_def(IndexType)        # element count

    def __init__(self, src1, src2, n):
        super().__init__(operands=[src1, src2, n], result_types=[])

NPUDialect = Dialect("npu", [FVAddOp], [])
```

**Semantics**: `npu.fvadd` maps 1:1 to the NPU FVADD instruction. It reads `n` float32 elements from `src1`, adds them to the elements at `src2`, and stores the result at `src2` (in-place). This matches the hardware: `dst[i] = src1[i] + src2[i]`, result at rs2's address.

**Note on in-place semantics**: The hardware stores results in-place at `src2`. For the `A + B` case, we need to copy B to the output buffer first, then FVADD A into it. OR we can model `npu.fvadd` with a separate destination operand at the dialect level and handle the copy-to-dst + in-place-add in the assembly emitter. The latter is cleaner — the npu dialect can have `src1`, `src2`, `dst`, `n`, and the asm emitter handles the mechanics.

**Revised design** (3-address form at dialect level):

```python
@irdl_op_definition
class FVAddOp(IRDLOperation):
    name = "npu.fvadd"

    src1 = operand_def(MemRefType)    # first source
    src2 = operand_def(MemRefType)    # second source
    dst = operand_def(MemRefType)     # destination
    n = operand_def(IndexType)        # element count
```

The asm emitter will:
1. If `dst == src2`: emit just `FVADD rd, rs1, rs2` (in-place)
2. If `dst != src2`: emit a memcpy of src2 to dst, then `FVADD rd, rs1, rs2` (where rs2 = dst)

For Milestone 1, the bufferization pass should try to make `dst == src2` (alias the output buffer to one of the inputs) when safe. If not, the copy fallback works.

**Tests** (`tests/test_lowering.py`):
- Construct `npu.fvadd` programmatically, verify IR prints correctly
- Verify the operation passes xDSL verification

### Task 7: Lowering — `linalg.generic [addf]` on memrefs to `npu.fvadd`

Pattern-match the bufferized `linalg.generic` and replace it with `npu.fvadd`.

**File**: `src/arrax/lowering/linalg_to_npu.py`

```python
class LinalgAddToNpuPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter):
        # Match: linalg.generic with exactly 2 inputs, 1 output,
        #        identity indexing maps, parallel iterator,
        #        body contains arith.addf + linalg.yield
        # Emit: npu.fvadd src1, src2, dst, n
        ...
```

**Pattern matching criteria**:
1. `linalg.generic` (not `linalg.matmul` or other named ops)
2. Exactly 2 `ins` operands, 1 `outs` operand
3. All three indexing maps are identity: `affine_map<(d0) -> (d0)>`
4. Single iterator type: `parallel`
5. Body block has exactly: `%r = arith.addf %arg0, %arg1`, `linalg.yield %r`
6. All operands are `memref<Nxf32>`

**Extract `n`**: Get the static dimension from the memref type's shape. Create an `arith.constant` with that value as an index.

**Tests** (`tests/test_lowering.py`):
- Full pipeline from `array.add` through linalg through bufferize to `npu.fvadd`
- Verify the final IR contains exactly one `npu.fvadd` and no `linalg.generic`

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
    # a0 = pointer to output array (dst)
    # a1 = pointer to input array A (src1)
    # a2 = pointer to input array B (src2)
    # a3 = element count (n)

    # Copy src2 to dst if dst != src2 (omit if aliased)
    # ... memcpy loop using FLW/FSW or word-sized LW/SW ...

    # NPU.FVADD rd=a3, rs1=a1, rs2=a0
    # dst[i] = src1[i] + dst[i]  (dst already contains src2 data)
    .insn r 0x2B, 0x0, 0x07, a3, a1, a0

    ret
```

**Register assignment** (RISC-V calling convention):
- Function arguments arrive in `a0`–`a7`
- For `kernel(dst, src1, src2, n)`: `a0`=dst, `a1`=src1, `a2`=src2, `a3`=n
- The `.insn r` format: `.insn r opcode, funct3, funct7, rd, rs1, rs2`
- FVADD encoding: opcode=0x2B, funct3=0x0, funct7=0x07
- rd carries the element count, rs1 is source 1 address, rs2 is source 2 / destination address

**Calling convention for the kernel function**:
The compiler chooses the argument order. For `A + B -> out`:
- `a0` = output buffer pointer (dst)
- `a1` = A pointer (src1)
- `a2` = B pointer (src2)
- `a3` = element count

The assembly emitter must handle the case where the NPU instruction writes in-place to rs2 but our destination is a different buffer. Strategy:
1. If dst aliases src2 (from bufferization): emit just the `.insn`
2. Otherwise: emit a copy loop (LW/SW) from src2 to dst, then `.insn` with rs2=dst

For Milestone 1, always emit the copy + insn sequence (simple, correct). Optimize aliasing later.

**Tests** (`tests/test_lowering.py` or `tests/test_end_to_end.py`):
- Generate assembly for `A + B`, verify it contains the `.insn r 0x2B, 0x0, 0x07` directive
- Verify the assembly text is syntactically valid (basic string checks)

**File**: `src/arrax/codegen/firmware_harness.py`

Generate a `main.c` that:
- Declares float arrays A[N], B[N], out[N]
- Initializes A and B with known values (e.g., A[i] = (float)i, B[i] = (float)(i * 2))
- Calls `npu_kernel(out, A, B, N)`
- Prints output values via UART for verification

**Float output format**: The firmware is freestanding (`-nostdlib`) — no `printf`/`sprintf`. Print each float as its raw IEEE 754 hex bytes using a simple function:

```c
void print_hex_byte(unsigned char b) {
    const char hex[] = "0123456789abcdef";
    putchar(hex[b >> 4]);
    putchar(hex[b & 0xf]);
}

void print_float_hex(float f) {
    unsigned char *bytes = (unsigned char *)&f;
    for (int i = 3; i >= 0; i--)  /* big-endian hex for readability */
        print_hex_byte(bytes[i]);
    putchar('\n');
}
```

The Python `parse_output()` function reconstructs floats via `struct.unpack('!f', bytes.fromhex(line))`. This is exact — no decimal rounding issues.

**File**: `src/arrax/codegen/build.py`

Invoke the toolchain using riscv-npu's common firmware files:

```python
RISCV_NPU_DIR = os.environ.get("RISCV_NPU_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "riscv-npu"))
COMMON_DIR = os.path.join(RISCV_NPU_DIR, "firmware", "common")

subprocess.run([
    "riscv64-unknown-elf-gcc",
    "-march=rv32imf", "-mabi=ilp32f",
    "-O2", "-nostdlib", "-ffreestanding",
    f"-T{os.path.join(COMMON_DIR, 'linker.ld')}",
    "-I", COMMON_DIR,
    "-o", elf_path,
    harness_c_path,
    kernel_s_path,
    os.path.join(COMMON_DIR, "start.o"),  # precompiled startup
    os.path.join(COMMON_DIR, "syscalls.c"),
])
```

**Note**: Uses the precompiled `start.o` and shared `linker.ld` from riscv-npu rather than duplicating them.

### Task 9: End-to-end test — compile, assemble, run, verify

**File**: `tests/test_end_to_end.py`

```python
def test_add_end_to_end():
    """A + B: Python -> assembly -> emulator -> verify against NumPy."""
    import numpy as np

    N = 64  # small for testing
    A_data = np.arange(N, dtype=np.float32)
    B_data = np.arange(N, dtype=np.float32) * 2

    expected = A_data + B_data

    # 1. Trace
    result_dag = trace(lambda A, B: A + B, {"A": (N,), "B": (N,)})

    # 2. Lower through full pipeline
    module = dsl_to_array(result_dag, {"A": (N,), "B": (N,)})
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)

    # 3. Emit assembly
    asm_text = emit_assembly(module)

    # 4. Generate firmware harness
    harness_c = generate_harness(A_data, B_data, N)

    # 5. Build ELF
    elf_path = build_firmware(asm_text, harness_c)

    # 6. Run on emulator
    output = run_emulator(elf_path)

    # 7. Compare
    actual = parse_output(output)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
```

**Tests** (`tests/test_end_to_end.py`):
- `A + B` with N=64, known input data, verify output matches NumPy
- Edge cases: N=1, N=0 (if the hardware supports it)

### Task 10: Pipeline orchestration

Wire everything into a single `compile()` entry point.

**File**: `src/arrax/pipeline.py`

```python
def compile(fn, shapes: dict[str, tuple[int, ...]]) -> CompiledKernel:
    """Full pipeline: trace -> lower -> emit -> build."""
    from arrax.dsl.tracer import trace
    from arrax.lowering.dsl_to_array import dsl_to_array
    from arrax.lowering.array_to_linalg import ArrayToLinalgPass
    from arrax.lowering.bufferize import BufferizePass
    from arrax.lowering.linalg_to_npu import LinalgToNpuPass
    from arrax.codegen.asm_emitter import emit_assembly

    from xdsl.context import Context

    dag = trace(fn, shapes)
    module = dsl_to_array(dag, shapes)
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    asm = emit_assembly(module)
    return CompiledKernel(asm=asm, shapes=shapes)
```

**File**: `src/arrax/__init__.py`

```python
from arrax.dsl.array import Array
from arrax.pipeline import compile

__all__ = ["Array", "compile"]
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

Tasks 1, 2, 6 can be done in parallel (no dependencies on each other).
Tasks 3–8 are sequential (each depends on the previous).
Task 9 depends on Task 8 + riscv-npu emulator + RISC-V toolchain installed.
Task 10 depends on all other tasks.

**Highest-risk task**: Task 5 (bufferization) — requires rebuilding FuncOp, SSA remapping, and no built-in xDSL pass to lean on. Start here if exploring ahead.

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

1. **Unit tests per task**: Each task has its own tests that verify the transformation in isolation
2. **IR verification**: Call `module.verify()` after every pass to catch malformed IR early
3. **Print-and-inspect**: Print IR after each pass during development to visually confirm structure
4. **NumPy reference**: End-to-end comparison with `np.testing.assert_allclose(actual, expected, rtol=1e-6)`
5. **Assembly inspection**: Verify the generated `.insn` directives match the encoding table
6. **Edge cases**: `n=0` is a no-op (zero-iteration loop in emulator FVADD). `n=1` is the minimal non-trivial case.

## Known Limitations (Milestone 1)

- **Copy overhead**: The 3-address `npu.fvadd` always emits a copy loop (LW/SW for each element) before the `.insn` FVADD, even when `dst == src2`. For N=1024 this doubles the instruction count. First optimization target for Milestone 2: check pointer aliasing in the emitter (`dst == src2` → skip copy).
- **No tiling**: Arrays must fit in memory. Tiling is Milestone 5.
- **No fusion**: Single operation only. Fusion is Milestone 2.
- **Single operation**: Only `array.add`. More elementwise ops in Milestone 2.
