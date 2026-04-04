# Milestone 2: More Ops, Fusion, Buffer Optimization

## Goal

Expand the compiler from a single op (add) to a full set of elementwise operations, then optimize chained expressions via loop fusion and buffer reuse.

## Op coverage

| DSL surface   | Array dialect op | linalg body      | NPU op | funct7 |
|---------------|------------------|------------------|--------|--------|
| `A + B`       | AddOp (existing) | `arith.addf`     | FVADD  | 0x07   |
| `A - B`       | SubOp            | `arith.subf`     | FVSUB  | 0x08   |
| `relu(A)`     | ReluOp           | `arith.maximumf` | FVRELU | 0x09   |
| `exp(A)`      | ExpOp            | `math.exp`       | FVEXP  | 0x02   |
| `A * scalar`  | MulScalarOp      | `arith.mulf`     | FVMUL  | 0x04   |
| `A / scalar`  | DivScalarOp      | `arith.divf`     | FVDIV  | 0x0B   |

Note: elementwise `A * B` (two arrays) is not supported — the NPU has no elementwise vector multiply instruction. `__mul__` and `__truediv__` only accept `float` as the right operand.

All ops go through `linalg.generic` so tiling works unchanged.

### Binary elementwise (sub)

Same pattern as add:
- `Array.__sub__` creates DAG node with `op="sub"`
- `array.SubOp(lhs, rhs) -> tensor` in array dialect
- Lowers to `linalg.generic { arith.subf }` on tensors
- Bufferize, tile, then `linalg_to_npu` matches `subf` body -> `npu.FVSubOp`
- Subtraction is NOT commutative: no swap optimization in canonicalize

### Unary elementwise (relu, exp)

New pattern: 1 input, 1 output (vs 2 inputs for binary).

DSL surface: standalone functions, not operators.
```python
from arrax import relu, exp
result = relu(A + B)
result = exp(A)
```

These create DAG nodes with `op="relu"` / `op="exp"` and one operand.

Array dialect: `array.ReluOp(input: tensor) -> tensor`, `array.ExpOp(input: tensor) -> tensor`.

Linalg lowering: `linalg.generic` with 1 input memref, 1 output memref, identity indexing maps, parallel iterator. Body:
- relu: `arith.maximumf(%in, 0.0)` + `linalg.yield`
- exp: `math.exp(%in)` + `linalg.yield`

Tiling works unchanged (strip-mines any 1D generic). `linalg_to_npu` matches:
- 1 input + 1 output + `arith.MaximumfOp` with const 0.0 -> `npu.FVReluOp`
- 1 input + 1 output + `math.ExpOp` -> `npu.FVExpOp`

NPU dialect ops:
- `npu.FVReluOp(src, dst, n)` — 3 operands (no src2, writes dst in-place from src)
- `npu.FVExpOp(src, dst, n)` — same shape

Assembly emission: unary NPU instructions use rs1=source, rs2=destination (in-place), rd=count. Same `.insn r` format, different funct7. No copy loop needed when src != dst since the hardware reads from rs1 and writes to rs2 independently.

### Scalar-vector (mul_scalar, div_scalar)

New concept: the NPU float accumulator (`facc`) holds the scalar operand.

DSL:
```python
A * 3.0   # Array.__mul__(float) -> Array
3.0 * A   # Array.__rmul__(float) -> Array (commutative, delegates to __mul__)
A / 2.0   # Array.__truediv__(float) -> Array
```

The scalar is a compile-time constant stored as an attribute on the DAG node and array dialect op.

Array dialect: `array.MulScalarOp(input: tensor, scalar: f32attr) -> tensor`, `array.DivScalarOp(input: tensor, scalar: f32attr) -> tensor`.

Linalg lowering: `linalg.generic` with body `arith.mulf(x, const)` / `arith.divf(x, const)`. The constant is materialized as `arith.constant` inside the body.

`linalg_to_npu` matches `mulf(x, const)` or `divf(x, const)` patterns and emits `npu.FVMulOp` / `npu.FVDivOp`. The scalar value is extracted from the constant and stored as an attribute on the NPU op.

NPU dialect:
- `npu.FVMulOp(src, dst, n, scalar: f32attr)` — scalar is an attribute, not SSA
- `npu.FVDivOp(src, dst, n, scalar: f32attr)` — same

Assembly emission sequence for FVMUL:
```asm
    # Inside the loop body (emitted inline with each scalar op):
    .insn r 0x2B, 0x5, 0x00, f0, f0, f0   # FRSTACC: zero facc
    li t0, <scalar_bits>                     # IEEE 754 float32 bits
    fmv.w.x f1, t0                           # integer -> float register
    lui t0, 0x3F800                           # 1.0f
    fmv.w.x f2, t0
    .insn r 0x2B, 0x0, 0x00, f0, f1, f2    # FMACC: facc = scalar * 1.0
    .insn r 0x2B, 0x0, 0x04, rd, rs1, rs2  # FVMUL: dst[i] = src[i] * facc
```

Note: facc load is emitted inline per scalar op. In tiled loops this reloads facc
each iteration (redundant but correct). A future optimization could hoist the load
before the loop since facc persists across iterations.

## Fusion

### Strategy: post-tiling loop fusion

Merge adjacent `scf.for` loops with identical iteration bounds into a single loop.

Pipeline placement:
```
Bufferize -> Tile -> Fuse -> BufferOptimize -> LinalgToNpu -> NpuCanonicalize -> emit_assembly
```

### Fusibility criteria

Two adjacent `scf.for` ops are fusible when:
1. Same lower bound, upper bound, and step (SSA values trace to same definition)
2. Producer-consumer relationship: the second loop reads a buffer written by the first
3. No intervening ops between them that depend on the first loop's complete results

### Transform

1. Look at the op immediately after an `scf.for`
2. If it's another `scf.for` with matching lb/ub/step
3. Remap the second loop's induction variable to the first loop's IV
4. Move the second loop's body ops (except yield) into the first loop's body, before its yield
5. Erase the second loop

### Example: `(A + B) + C` with N=128

Before fusion:
```
memref.alloc %tmp : memref<128xf32>
scf.for %iv = 0 to 128 step 64 {
    %a_tile = subview %A[%iv]...
    %b_tile = subview %B[%iv]...
    %tmp_tile = subview %tmp[%iv]...
    linalg.generic ins(%a_tile, %b_tile) outs(%tmp_tile) { addf }
}
scf.for %iv2 = 0 to 128 step 64 {
    %tmp_tile = subview %tmp[%iv2]...
    %c_tile = subview %C[%iv2]...
    %out_tile = subview %out[%iv2]...
    linalg.generic ins(%tmp_tile, %c_tile) outs(%out_tile) { addf }
}
```

After fusion:
```
memref.alloc %tmp : memref<128xf32>
scf.for %iv = 0 to 128 step 64 {
    %a_tile = subview %A[%iv]...
    %b_tile = subview %B[%iv]...
    %tmp_tile = subview %tmp[%iv]...
    linalg.generic ins(%a_tile, %b_tile) outs(%tmp_tile) { addf }
    %tmp_tile2 = subview %tmp[%iv]...
    %c_tile = subview %C[%iv]...
    %out_tile = subview %out[%iv]...
    linalg.generic ins(%tmp_tile2, %c_tile) outs(%out_tile) { addf }
}
```

Two NPU instructions per iteration, data stays local.

## Buffer optimization

Two transforms, applied in order after fusion.

### Buffer shrinking

After fusion, intermediate buffers are only accessed within a single loop iteration (one tile at a time). A full N-element alloc can shrink to `min(N, 64)` elements.

**Detection:** A `memref.alloc` whose only uses are `memref.subview` ops, and all those subviews are inside the same fused loop body.

**Transform:** Replace the N-element alloc with a 64-element alloc. Rewrite subviews from `subview %tmp[%iv][%n][1]` to `subview %tmp[0][%n][1]` — always at offset 0 since data doesn't persist across iterations.

### Buffer reuse (liveness-based)

Multiple shrunken intermediate buffers that don't overlap in liveness can share the same allocation.

**Detection:** Within a fused loop body, assign each alloc a liveness interval (first write op index to last read op index). Non-overlapping intervals can share.

**Transform:** Replace one alloc with the other. Adjust subview source operands.

**Pipeline placement:** Single `BufferOptimize` pass after `Fuse`:
```
Tile -> Fuse -> BufferOptimize -> LinalgToNpu -> ...
```

## Implementation phases

| Phase | Scope                          | Key files                                             |
|-------|--------------------------------|-------------------------------------------------------|
| 1     | Sub (binary elementwise)       | array.py, array_dialect.py, npu_dialect.py, lowering/ |
| 2     | Relu + Exp (unary elementwise) | Same set; new 1-input linalg.generic pattern          |
| 3     | Fusion                         | New fusion.py pass; pipeline.py update                |
| 4     | Buffer shrinking + reuse       | New buffer_optimize.py pass; pipeline.py update       |
| 5     | Scalar-vector (mul, div)       | Full stack; new accumulator management in asm_emitter |

Each phase is independently testable end-to-end.

### Test strategy

- **Unit tests** for each new dialect op, lowering pattern, and asm emission path
- **End-to-end**: Python expression -> emulator -> compare with NumPy float32 reference
- **Fusion IR tests**: verify fused loop structure (one loop, multiple ops per body)
- **Buffer tests**: verify alloc count and size reduction in lowered IR
- **Scalar-vector**: verify facc load sequence + FVMUL/FVDIV correctness

### Key end-to-end test cases

```python
# Phase 1
lambda A, B: A - B

# Phase 2
lambda A, B: relu(A + B)
lambda A: exp(A)

# Phase 3 (fusion visible in cycle count reduction)
lambda A, B, C: relu(A + B) - C

# Phase 4 (buffer reduction visible in IR)
lambda A, B, C, D: (A + B) + (C + D)   # two intermediates, non-overlapping

# Phase 5
lambda A: A * 3.0
lambda A, B: (A + B) / 2.0
```
