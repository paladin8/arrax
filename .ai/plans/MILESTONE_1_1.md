# Milestone 1.1: Strip-Mine Tiling for NPU Vector Limit

The riscv-npu hardware limits vector operations to 256 bytes (64 f32 elements) per instruction. Arrays larger than 64 elements must be strip-mined into chunks. This is a correctness requirement, not an optimization.

## Goal

`A + B` works correctly for any 1D array size, including sizes that are not multiples of 64. The tiling is invisible to the DSL user.

## Approach

Insert a tiling pass after bufferize and before linalg-to-npu. The pass strip-mines `linalg.generic` ops on memrefs into `scf.for` loops over 64-element `memref.subview` slices. The existing linalg-to-npu pattern then matches each small chunk unchanged.

### Pipeline (changed steps marked)

```
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> **Tile** -> LinalgToNpu -> NpuCanonicalize -> verify -> emit_assembly
```

## Hardware Constraint

```
NPU vector op max elements:  64  (256 bytes at f32)
Constant:                     NPU_MAX_VEC_LEN = 64
```

## Tasks

### Task 1: Tiling pass — strip-mine linalg.generic on memrefs

**File**: `src/arrax/lowering/tile.py`

```python
NPU_MAX_VEC_LEN = 64

class TilePass(ModulePass):
    """Strip-mine linalg.generic ops to fit NPU vector length limit."""
```

For each 1D `linalg.generic` on memrefs where `n > NPU_MAX_VEC_LEN`:

```
scf.for %i = 0 to n step 64 {
    %chunk = arith.minsi(64, n - %i)     // handles remainder
    %sub_a  = memref.subview %A[%i][%chunk][1]
    %sub_b  = memref.subview %B[%i][%chunk][1]
    %sub_o  = memref.subview %out[%i][%chunk][1]
    linalg.generic {addf} on memref<?xf32>  // dynamic dim for chunk
}
```

Ops with `n <= 64` pass through unchanged.

**Key detail**: The subviewed memrefs have a dynamic dimension (`memref<?xf32>`) since the last chunk may be smaller than 64. This means linalg-to-npu must extract the chunk size from the subview rather than from the static memref shape.

**Tests** (`tests/lowering/test_tile.py`):
- `n=32` (below limit): pass-through, no scf.for emitted
- `n=64` (exact limit): pass-through, no scf.for emitted
- `n=128` (exact multiple): scf.for with step 64, two iterations
- `n=100` (non-multiple): scf.for with remainder handling
- Verify IR round-trips through `module.verify()`

### Task 2: Update linalg-to-npu to handle dynamic-dim subviews

**File**: `src/arrax/lowering/linalg_to_npu.py`

Currently `LinalgAddToNpuPattern` reads `n` from the static memref shape:
```python
n_val: int = src1.type.get_shape()[0]
```

After tiling, the memrefs are `memref<?xf32>` (dynamic). The element count comes from the subview's size operand, not the type. Update the pattern to:
1. Accept dynamic-dim 1D memrefs
2. Get `n` from the subview's dynamic size (passed as an SSA value) instead of the static shape

This also means the `arith.constant` for n is no longer created here — the value is already an SSA value from the tiling loop.

**Tests**: Update `tests/lowering/test_linalg_to_npu.py`:
- Existing static-shape tests still pass (n <= 64 skips tiling)
- New test with dynamic-dim memref input (post-tiling IR)

### Task 3: Assembly emission for scf.for and memref.subview

**File**: `src/arrax/codegen/asm_emitter.py`

New op handlers:

- **`scf.for`**: Emit a counted loop. Map the induction variable to a register, emit `li`/`bge` loop structure.
- **`memref.subview`**: Pointer arithmetic. Base register + offset × 4 (f32 stride). Result mapped to a temp register.
- **`arith.minsi`**: `min` via branch (`blt`/`mv`).

The FVADD `.insn` operand for n is now a register (dynamic chunk size), not a hardcoded `li` constant. The emitter already puts n in `t0` before the `.insn`, so this mostly means loading from the SSA value's register instead of a constant.

**Tests**: Update `tests/codegen/test_asm_emitter.py`:
- Assembly for n=128 contains a loop structure
- Assembly for n=100 contains remainder handling
- Existing n<=64 golden tests still pass

### Task 4: Optional — enforce n <= 64 in FVAddOp.verify_()

**File**: `src/arrax/dialects/npu_dialect.py`

When the `n` operand is a known constant (traceable to `arith.constant`), verify that `n <= NPU_MAX_VEC_LEN`. This catches tiling bugs at IR verification time rather than at emulator runtime.

Only enforced when the constant is statically resolvable — dynamic values (from scf.for induction) skip the check.

### Task 5: Pipeline integration and end-to-end tests

**File**: `src/arrax/pipeline.py` — insert `TilePass` between `BufferizePass` and `LinalgToNpuPass`.

**File**: `tests/test_end_to_end.py` — add/update:
- `test_add` bumped to N=128 (was 64) — forces tiling
- `test_add_non_multiple` with N=100 — exercises remainder path
- `test_add_large` with N=1024 — 16 iterations of tiling
- Existing N=1 and N=16 tests still pass (below limit, no tiling)

## Task Dependency Graph

```
Task 1 (tile pass) ── Task 2 (update linalg-to-npu) ── Task 3 (asm emission)
                                                              │
                                                        Task 5 (pipeline + e2e)
Task 4 (optional verify) ────────────────────────────────────┘
```

## Known Considerations

- **Subview strides**: All subviews have stride 1 (contiguous). Non-contiguous tiling is not needed for 1D elementwise ops.
- **Dynamic vs static remainder**: Could peel the remainder as a separate static-size op instead of using `arith.minsi`. Simpler to use dynamic size uniformly; optimize later if needed.
- **NPU_MAX_VEC_LEN as constant**: Hardcoded to 64. If the hardware limit changes, update in one place (`tile.py`).
- **Chained adds**: `(A + B) + C` with N=128 should produce two tiled loops (one per add), each with their own scf.for. Fusion of these loops is Milestone 2.
