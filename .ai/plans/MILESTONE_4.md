# Milestone 4: Non-Terminal Reductions, Softmax, RMSNorm

## Goal

Lift the terminal-only restriction on reductions, enabling scalar reduction results to flow into subsequent operations. Use this infrastructure to implement two composite ops end-to-end: `softmax(x)` and `rmsnorm(x)` (without gamma). These are the first multi-pass array operations in arrax — each decomposes into a sequence of reductions and elementwise ops that the existing tiling and fusion passes handle automatically.

## Op coverage

| DSL surface    | Array dialect op | Decomposition in ArrayToLinalg                                      | New NPU ops used     |
|----------------|------------------|----------------------------------------------------------------------|----------------------|
| `softmax(A)`   | SoftmaxOp        | amax -> sub_broadcast -> exp -> sum -> div_broadcast                 | FVSubScalarOp        |
| `rmsnorm(A)`   | RMSNormOp        | dot(x,x) -> scalar div/add/rsqrt -> mul_broadcast                   | FRsqrtOp            |

Both ops take rank-1 `tensor<Nxf32>` input and produce rank-1 `tensor<Nxf32>` output. They are composite: ArrayToLinalg decomposes them into sequences of linalg.generic ops (reductions + elementwise with scalar broadcast). No new array-level rewrite is needed in downstream passes.

## Pipeline (changed steps marked)

```
trace
  -> dsl_to_array                        [CHANGED: terminal-only validator removed]
  -> ArrayToLinalg                       [EXTENDED: softmax/rmsnorm decomposition, broadcast binary patterns]
  -> Bufferize                           [UNCHANGED: rank-0 memref already supported from M3]
  -> Tile                                [EXTENDED: skip rank-0 broadcast operands in subview creation]
  -> Fuse                                [UNCHANGED: parallel+parallel and parallel->reduction already work]
  -> BufferOptimize                      [UNCHANGED]
  -> LinalgToNpu                         [EXTENDED: broadcast patterns, new NPU ops, scalar forwarding, rmsnorm scalar math]
  -> NpuCanonicalize                     [UNCHANGED]
  -> verify
  -> emit_assembly                       [EXTENDED: FVSubScalarOp, FRsqrtOp, scalar arith, runtime scalar facc load]
```

Four passes extended, one changed, rest unchanged.

## DSL surface

Two new free functions exported from the `arrax` package:

```python
from arrax import softmax, rmsnorm

softmax(A)    # -> Array with shape == A.shape
rmsnorm(A)    # -> Array with shape == A.shape
```

Both take a rank-1 Array and return a rank-1 Array of the same shape.

RMSNorm uses hardcoded eps=1e-5 (standard default, matches PyTorch/LLaMA). No gamma parameter in M4 — the NPU lacks an element-wise vector multiply instruction, so `x * gamma` would require a scalar loop fallback. Full rmsnorm with vector gamma is deferred to a future milestone (requires either a new NPU instruction or a scalar loop emitter).

### Terminal restriction removal

The `_validate_reductions_terminal` function in `dsl_to_array.py` is deleted. All IR-level passes were already written to handle non-terminal reductions (M3 spec mandated this). The DSL layer was the only enforcement point.

Existing terminal-restriction tests in `tests/dsl/test_reductions_terminal.py` are updated: patterns that previously raised `ValueError` (e.g., `A + sum(A)`) should now compile successfully. The test file is repurposed to verify these patterns work rather than verifying they're rejected.

## Array dialect

Two new `irdl_op_definition` ops:

```python
@irdl_op_definition
class SoftmaxOp(IRDLOperation):
    name = "array.softmax"
    input = operand_def(TensorType)
    result = result_def(TensorType)
    # verify: input rank == 1, f32; result rank == 1, f32; shapes match
    traits = frozenset([Pure()])

@irdl_op_definition
class RMSNormOp(IRDLOperation):
    name = "array.rmsnorm"
    input = operand_def(TensorType)
    result = result_def(TensorType)
    # verify: input rank == 1, f32; result rank == 1, f32; shapes match
    traits = frozenset([Pure()])
```

Both verify that input and result are rank-1 f32 tensors with matching shapes. They follow the same pattern as existing unary ops (ReluOp, ExpOp) but produce rank-1 output (not rank-0 like reductions).

## Lowering: ArrayToLinalg decomposition

### Softmax

`SoftmaxToLinalgPattern` decomposes `array.softmax(%x)` into 5 linalg.generic ops:

```
// Step 1: max reduction
%max_init = tensor.empty() : tensor<f32>
linalg.fill(%neg_inf, %max_init)
%max = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
} ins(%x) outs(%max_init) { arith.maximumf }

// Step 2: subtract max (broadcast: rank-0 scalar applied to rank-1 vector)
%shifted = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
} ins(%x, %max) outs(%empty_N) { arith.subf(%x_elem, %max_scalar) }

// Step 3: exp
%e = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
} ins(%shifted) outs(%empty_N) { math.exp }

// Step 4: sum reduction
%sum_init = tensor.empty() : tensor<f32>
linalg.fill(%zero, %sum_init)
%sum = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
} ins(%e) outs(%sum_init) { arith.addf }

// Step 5: divide by sum (broadcast)
%out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
} ins(%e, %sum) outs(%empty_N) { arith.divf(%e_elem, %sum_scalar) }
```

Key: steps 2 and 5 use mixed affine maps — `(d0) -> (d0)` for the vector operand and `(d0) -> ()` for the scalar broadcast. This is standard linalg broadcasting. The broadcast operand is a rank-0 tensor produced by a preceding reduction.

### RMSNorm

`RMSNormToLinalgPattern` decomposes `array.rmsnorm(%x)` into 2 linalg.generic ops plus attributes for scalar math:

```
// Step 1: sum of squares (dot product of x with itself)
%sumsq_init = tensor.empty() : tensor<f32>
linalg.fill(%zero, %sumsq_init)
%sumsq = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
    iterator_types = ["reduction"]
} ins(%x, %x) outs(%sumsq_init) { mulf + addf }   // same body as dot product

// Steps 2-4: multiply by scale (broadcast)
// The scalar math (sumsq/N + eps, rsqrt) is encoded as attributes on the broadcast-mul
// generic. LinalgToNpu reads these attributes and emits the scalar math sequence
// (fdiv.s, fadd.s, store-to-scratch, FRSQRT) between loading the reduction result
// and executing the FVMUL.
%out = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
    arrax.rmsnorm_divisor = N : i64,
    arrax.rmsnorm_eps = 1e-5 : f32
} ins(%x, %sumsq) outs(%empty_N) { arith.mulf(%x_elem, %scale_scalar) }
```

The attribute-based approach follows the precedent set by `arrax.mean_divisor` in M3: ArrayToLinalg encodes the composite operation's scalar math as discardable attributes on the linalg.generic, and LinalgToNpu reads them to emit the correct sequence. This avoids representing scalar arith ops at the tensor level (arith.divf/addf work on scalars, not tensors) and keeps ArrayToLinalg clean.

The dot(x,x) reduction generic gets `arrax.facc = "persistent"` (same as DotToLinalgPattern). The broadcast-mul generic does NOT get an `arrax.facc` tag — it's a standard mulf with broadcast, not a scalar-vector multiply pattern. This means the fusion pass will not block fusion of the broadcast-mul with adjacent ops.

### Tiling + fusion behavior

**Softmax (N > 64):**
After tiling, 5 linalg ops become 5 scf.for loops. After fusion:
- Loop 1: amax reduction (must see all data before subtracting)
- Loop 2: sub_scalar + exp + sum — three ops fused (parallel + parallel + reduction)
- Loop 3: div by sum (elementwise, separate because reduction->parallel not fused)

Three passes over data. Optimal for numerically stable softmax.

**With preceding elementwise ops**, e.g. `softmax(A + B)`:
- Loop 1: add + amax fused (parallel -> reduction) — already supported
- Loop 2: sub_scalar + exp + sum fused (parallel + parallel + reduction)
- Loop 3: div by sum

Still three passes. The add gets fused into the amax reduction.

**RMSNorm (N > 64):**
After tiling, dot + mul become 2 scf.for loops. Scalar math is not tiled. After fusion:
- Loop 1: dot(x,x) reduction
- Scalar: fdiv.s, fadd.s, frsqrt (not looped, emitted by LinalgToNpu)
- Loop 2: mul by scale (elementwise)

Two passes over data.

**Small N (N <= 64):**
No tiling. Operations remain as individual linalg.generic ops, lowered directly to NPU instructions. No loops.

### facc conflict model: read-write lock

The NPU's facc register is used in two fundamentally different ways:

1. **Persistent accumulation** (FVMAC/dot product): facc holds a running total across the entire loop. The asm emitter brackets the loop with FRSTACC — no per-iteration save/restore. Any other facc usage within the loop would clobber the accumulator.
2. **Ephemeral parameter** (FVMUL, FVDIV, FVSUB_SCALAR, and the FRSTACC+FMACC bracket in FVReduceOp): facc is loaded, consumed by one instruction, then available for reuse. Multiple ephemeral uses are safe within the same iteration because they're sequential.

This maps naturally to a **read-write lock** model:

| Tag value     | Lock  | Meaning                                                          | Examples                                       |
|---------------|-------|------------------------------------------------------------------|-------------------------------------------------|
| (no tag)      | none  | doesn't touch facc                                               | FVADD, FVSUB, FVEXP, FVRELU                   |
| `"ephemeral"` | read  | uses facc within a single iteration, consumed immediately        | scalar-vec mul/div, broadcast ops, sum/amax     |
| `"persistent"` | write | facc accumulates across the entire loop, exclusive access needed | dot product (FVMAC)                             |

**Fusion rule**: block if either loop has `persistent`. Two `ephemeral` loops can fuse freely (sequential within an iteration, no conflict).

This replaces the M3 `arrax.uses_facc` UnitAttr with a `StringAttr`:
- M3's scalar-vec mul/div: was `arrax.uses_facc` (UnitAttr) → now `arrax.facc = "ephemeral"`
- M3's dot product: was `arrax.uses_facc` (UnitAttr) → now `arrax.facc = "persistent"`

The `_has_facc_conflict` function in `fusion.py` changes from "both have the tag" to "either has persistent":

```python
def _has_facc_conflict(first: scf.ForOp, second: scf.ForOp) -> bool:
    def _facc_level(loop: scf.ForOp) -> str:
        # Returns "persistent", "ephemeral", or "none"
        ...
    return "persistent" in (_facc_level(first), _facc_level(second))
```

**Tagging in M4:**

| Op pattern                  | Tag value     | Why                                                      |
|-----------------------------|---------------|----------------------------------------------------------|
| scalar-vec mul/div (M3)     | `ephemeral`   | loads constant into facc, FVMUL/FVDIV, done             |
| dot product (M3)            | `persistent`  | FVMAC accumulates across all iterations                  |
| broadcast sub/div/mul (M4)  | `ephemeral`   | loads runtime scalar into facc, one NPU op, done         |
| sum/amax/mean reduction     | (no tag)      | FVREDUCE/FVMAX don't touch facc; iter_args bracket is internal |

**Softmax fusion analysis with the lock model:**

Fused loop 2 (sub_scalar + exp + sum):
- FVSUB_SCALAR: `ephemeral` — loads max into facc, uses it, done
- FVEXP: no tag — doesn't touch facc
- FVREDUCE (sum): no tag — doesn't touch facc (uses local accumulation)

No `persistent` → fusion allowed. ✓

Unfused `dot(A * 2.0, B)`:
- scalar-vec mul: `ephemeral`
- dot: `persistent`

Has `persistent` → fusion blocked. ✓

**Benefit over M3 design:** The lock model makes the fusion rule principled rather than ad-hoc. The M3 UnitAttr conflated ephemeral and persistent usage into one tag, which happened to produce correct results but for the wrong reason (scalar-vec mul was "blocked from fusing with dot" because both had the tag, but it was equally "blocked from fusing with another scalar-vec mul" which would have been safe). The read-write model expresses the actual hardware constraint.

## Tile pass: rank-0 broadcast operand handling

The `_tile_parallel` method in `tile.py` currently creates subviews for ALL operands. For broadcast generics with rank-0 inputs (scalar broadcast), the rank-0 operand must pass through without a subview — you cannot create a dynamic subview of a rank-0 memref.

Change: when iterating over inputs to create subviews, check rank. If rank-0, pass the operand through directly. The indexing maps are preserved unchanged — `(d0) -> ()` still correctly broadcasts the scalar.

```python
# Current (creates subview for everything):
for memref_val in list(inputs) + list(outputs):
    sv = memref.SubviewOp.get(source=memref_val, ...)
    subviews.append(sv)

# New (skip rank-0):
tiled_inputs = []
subview_ops = []
for memref_val in inputs:
    mt = memref_val.type
    if isinstance(mt, MemRefType) and len(mt.get_shape()) == 0:
        tiled_inputs.append(memref_val)  # pass through
    else:
        sv = memref.SubviewOp.get(source=memref_val, ...)
        subview_ops.append(sv)
        tiled_inputs.append(sv.result)
# Output is always rank-1 — subview as before
```

This is ~10 lines of change. _tile_reduction does not need changes — reduction generics don't have broadcast inputs in M4.

## NPU dialect: new ops

### FVSubScalarOp

Vector subtract by runtime scalar. Hardware instruction: FVSUB_SCALAR (funct7=0x0C).

```python
@irdl_op_definition
class FVSubScalarOp(IRDLOperation):
    name = "npu.fvsub_scalar"
    src = operand_def(MemRefType)          # rank-1 source
    dst = operand_def(MemRefType)          # rank-1 destination
    n = operand_def(IndexType)             # element count
    scalar = operand_def(Float32Type)      # runtime f32 scalar (from reduction)
    traits = frozenset()
```

Semantics: `dst[i] = src[i] - scalar` for i in 0..n-1. The scalar is loaded into facc before the instruction executes.

### FRsqrtOp

Reciprocal square root. Hardware instruction: FRSQRT (funct7=0x03).

```python
@irdl_op_definition
class FRsqrtOp(IRDLOperation):
    name = "npu.frsqrt"
    src = operand_def(MemRefType)          # rank-0 memref (single float address)
    result = result_def(Float32Type)       # scalar result: 1/sqrt(src)
    traits = frozenset()
```

Semantics: reads one f32 from memory at src address, returns 1/sqrt(value) as scalar f32. This is a scalar instruction despite the `fv` prefix — it operates on a single memory location.

### FVMulOp/FVDivOp refactor: property → SSA operand

Currently these ops store the scalar as a `FloatAttr` property (compile-time constant). We refactor to take the scalar as an SSA `f32` operand instead. For compile-time constants, the caller emits an `arith.ConstantOp(FloatAttr)` and passes its result. For runtime scalars (from reduction results), the caller passes the SSA value directly.

```python
# Before:
class FVMulOp(IRDLOperation):
    src, dst, n = operand_def(MemRefType), operand_def(MemRefType), operand_def(IndexType)
    scalar = prop_def(FloatAttr)  # compile-time constant

# After:
class FVMulOp(IRDLOperation):
    src, dst, n = operand_def(MemRefType), operand_def(MemRefType), operand_def(IndexType)
    scalar = operand_def(Float32Type)  # SSA f32 value (constant or runtime)
```

This unifies compile-time and runtime scalars. The asm emitter doesn't need to distinguish — it looks up the scalar in the register pool and loads it into facc. For constants, the pool register was populated when the asm emitter processed the `arith.ConstantOp`.

Impact on existing code:
- `npu_dialect.py`: FVMulOp/FVDivOp constructor changes (property → operand)
- `linalg_to_npu.py`: emit `arith.ConstantOp` + pass result to FVMulOp/FVDivOp
- `asm_emitter.py`: read scalar from pool instead of property. Already handles f32 constants in `_emit_constant` (stores IEEE bits in `_fconst_map`). The pool's `_materialize_f32_into` handles materialization.

## LinalgToNpu: new patterns

### Broadcast binary elementwise

New pattern recognizes linalg.generic with mixed affine maps — one rank-1 operand and one rank-0 broadcast operand:

```
linalg.generic {
    indexing_maps = [(d0) -> (d0), (d0) -> (), (d0) -> (d0)],
    iterator_types = ["parallel"]
} ins(%vec, %scalar_memref) outs(%out) {
    arith.subf %vec_elem, %scalar_elem -> npu.fvsub_scalar
    arith.divf %vec_elem, %scalar_elem -> npu.fvdiv (with runtime scalar)
    arith.mulf %vec_elem, %scalar_elem -> npu.fvmul (with runtime scalar)
}
```

The pattern:
1. Detects the `(d0) -> ()` map indicating a broadcast operand.
2. **Scalar forwarding**: walks backwards from the generic to find a `memref.store` to the rank-0 memref. If found, uses the stored SSA value directly instead of emitting a `memref.load`. This avoids a memory round-trip — the reduction result stays in a register (managed by the FP pool). If no dominating store is found, emits a `memref.load` as fallback.
3. Emits the appropriate NPU op with the scalar as an SSA f32 operand.

This scalar forwarding is analogous to `find_preceding_fill` (shared utility from M3) — walk backwards to find a matching store pattern.

### RMSNorm scalar math (attributed broadcast-mul)

When LinalgToNpu encounters a broadcast-mul generic with `arrax.rmsnorm_divisor` attribute:

1. Load the dot result from rank-0 memref (scalar forwarding if possible)
2. Emit `arith.DivfOp(%sumsq, N_const)` → `%meansq` (becomes fdiv.s at asm time)
3. Emit `arith.AddfOp(%meansq, eps_const)` → `%shifted` (becomes fadd.s at asm time)
4. Store `%shifted` to a `memref.alloca<f32>` scratch (FRSQRT reads from memory)
5. Emit `npu.FRsqrtOp(%scratch)` → `%scale`
6. Emit `npu.FVMulOp(%src, %dst, %n, %scale)` (runtime scalar)

This follows the `arrax.mean_divisor` precedent: attributes encode the composite operation's inter-step scalar math, LinalgToNpu emits the actual instructions.

## Assembly emission

### FVSubScalarOp

```asm
    # Load scalar into facc
    .insn r 0x2B, 0x5, 0x00, f0, f0, f0     # FRSTACC: zero facc
    lui t0, 0x3F800                           # 1.0f upper bits
    fmv.w.x ft0, t0                           # ft0 = 1.0
    .insn r 0x2B, 0x0, 0x00, f0, fs?, ft0    # FMACC: facc += scalar * 1.0
    # Execute FVSUB_SCALAR
    li t0, <n>
    .insn r 0x2B, 0x0, 0x0C, t0, rs1, rs2   # FVSUB_SCALAR rd=n, rs1=src, rs2=dst
```

The scalar register (fs?) is looked up from the register pool.

### FRsqrtOp

```asm
    # addr already in a general-purpose register (from rank-0 memref base pointer)
    .insn r 0x2B, 0x0, 0x03, fd, rs1, x0    # FRSQRT: f[rd] = 1/sqrt(mem[rs1])
```

Result goes to a float register managed by the pool.

### Scalar arith ops

`arith.DivfOp`, `arith.AddfOp` on f32 values (from rmsnorm scalar math) emit standard RISC-V F instructions:

```asm
    fdiv.s fs?, fs?, fs?   # divf
    fadd.s fs?, fs?, fs?   # addf
```

Register pool manages the f32 values. These are the first scalar arith ops arrax emits (previously all scalar math was NPU instructions or materialization sequences).

### Runtime scalar facc load (refactored FVMulOp/FVDivOp)

After the refactor, both compile-time and runtime scalars are SSA f32 values. The asm emitter loads them into facc identically:

```asm
    .insn r 0x2B, 0x5, 0x00, f0, f0, f0     # FRSTACC: zero facc
    lui t0, 0x3F800                           # 1.0f
    fmv.w.x ft0, t0
    .insn r 0x2B, 0x0, 0x00, f0, fs?, ft0   # FMACC: facc += pool_reg * 1.0
```

For compile-time constants, the pool register was populated when the asm emitter processed the `arith.ConstantOp(FloatAttr)`. For runtime values, the pool register was populated when the reduction result was processed. No special-casing needed.

## Implementation phases

| Phase | Scope                                                    | Key files                                                      |
|-------|----------------------------------------------------------|----------------------------------------------------------------|
| 1     | Non-terminal reductions + tile broadcast + facc lock     | dsl_to_array.py, tile.py, fusion.py, array_to_linalg.py       |
| 2     | NPU ops + FVMul/FVDiv refactor + asm emission            | npu_dialect.py, linalg_to_npu.py, asm_emitter.py              |
| 3     | Softmax end-to-end                            | array_dialect.py, array_to_linalg.py, linalg_to_npu.py, tests |
| 4     | RMSNorm end-to-end                            | array_dialect.py, array_to_linalg.py, linalg_to_npu.py, tests |
| 5     | Polish: E2E tests, review, refactor           | tests/test_end_to_end.py, all files                            |

Each phase is independently testable. Phases 1-2 are infrastructure. Phase 3 is the first composite. Phase 4 adds rmsnorm. Phase 5 is integration testing and cleanup.

### Test strategy

**Unit tests** for each new component:
- Broadcast binary linalg patterns (IR structure tests)
- FVSubScalarOp/FRsqrtOp verify_ and asm emission
- Tile pass with rank-0 broadcast operands
- Softmax and rmsnorm ArrayToLinalg decomposition (IR structure)
- Runtime scalar facc load in asm emitter
- Scalar forwarding (rank-0 memref.store → direct SSA)

**Integration tests:**
- Non-terminal reduction patterns that were previously rejected now compile
- Softmax IR after each pipeline stage (correct loop structure, fusion behavior)
- RMSNorm IR after each pipeline stage

**End-to-end tests** (Python -> emulator -> compare with NumPy):
- `softmax(A)` — small (N=32, untiled), medium (N=64, exact), large (N=128, tiled), non-multiple (N=135)
- `rmsnorm(A)` — same size variants
- `softmax(A + B)` — elementwise producer fused before softmax
- `rmsnorm(relu(A))` — elementwise producer fused before rmsnorm
- Edge cases: all-same values (softmax should produce uniform 1/N), near-zero values (rmsnorm stability)

**Numerical tolerance:** Softmax and rmsnorm involve multiple FP operations (exp, div, rsqrt) that compound rounding differences. Tests compare against NumPy float32 reference with `rtol=1e-5, atol=1e-6` (slightly relaxed from elementwise `atol=1e-7`).

## Key risks and mitigations

**Risk: broadcast affine maps not supported by xDSL tiling.**
The `(d0) -> ()` broadcast map on a linalg.generic input is standard MLIR but may not be handled by xDSL's pattern rewriter for tiling. Mitigation: tile pass is extended to skip rank-0 operands. Verify this works in Phase 1 with a hand-crafted broadcast generic before building softmax on it.

**Risk: FVMulOp/FVDivOp refactor breaks existing tests.**
Changing from property to SSA operand touches working code. Mitigation: Phase 2 refactors the ops and immediately runs the full test suite to verify no regressions. The change is mechanical — every call site just wraps the float value in an arith.ConstantOp.

**Risk: FRSQRT memory operand after scalar forwarding.**
FRSQRT reads from a memory address. After scalar forwarding, the value is SSA f32 (no memref). Mitigation: LinalgToNpu emits `memref.alloca + memref.store` to put the value in memory before FRSQRT. One extra store, acceptable overhead for one instruction.

**Risk: softmax numerical stability for large values.**
The max-subtraction step prevents overflow in exp(). If the max is not correctly computed (e.g., due to a tiling bug), exp() produces inf. Mitigation: E2E tests include large-value inputs where stability matters.

---

## Detailed implementation plan

### Phase 1: Non-terminal reductions + tile broadcast + facc lock refactor

**Goal**: Remove the terminal restriction, enable tiling of broadcast generics, and refactor the facc conflict model to read-write locks. After this phase, non-terminal reduction patterns compile through bufferize+tile (but not yet through LinalgToNpu or asm).

#### Step 1.1: Remove terminal-only validator

**Files**: `src/arrax/lowering/dsl_to_array.py`, `tests/dsl/test_reductions_terminal.py`

1. **RED**: Write test in `tests/dsl/test_reductions_terminal.py` that `A + sum(A)` compiles through `dsl_to_array` without raising. Currently fails because the validator rejects it.
2. **GREEN**: In `dsl_to_array.py`:
   - Delete `_REDUCTION_OPS` (line 25) and `_validate_reductions_terminal` (lines 28-56).
   - Delete the call `_validate_reductions_terminal(result)` on line 86.
   - Keep `visited_nodes` — it's used elsewhere.
3. **REFACTOR**: Update the existing rejection tests in `test_reductions_terminal.py`. Previously-rejected patterns (`A + sum(A)`, `sum(A) + sum(B)`, `A * mean(A)`) should now be acceptance tests that verify these patterns produce valid array dialect IR. Remove or invert the `pytest.raises` assertions.
4. **VERIFY**: `uv run pytest -x` — all tests pass.

#### Step 1.2: Tile pass: handle rank-0 broadcast operands

**Files**: `src/arrax/lowering/tile.py`, `tests/lowering/test_tile.py`

1. **RED**: Write a test in `tests/lowering/test_tile.py` that constructs a broadcast binary generic (rank-1 + rank-0 inputs, parallel iterator, mixed affine maps) and runs the tile pass on it. Verify the tiled IR has one scf.for with the rank-0 operand passed through (not subviewed). Currently fails because `_tile_parallel` tries to subview the rank-0 operand.
2. **GREEN**: In `tile.py`, modify `_tile_parallel`:
   - When iterating over inputs to create subviews, check if the input type is a rank-0 MemRefType. If so, skip the subview and pass the operand through directly.
   - Build the tiled generic's input list from a mix of subview results (rank-1) and pass-through values (rank-0).
   - Keep the output subview unchanged (output is always rank-1).
3. **VERIFY**: The new test passes. `uv run pytest -x` — no regressions.

#### Step 1.3: Refactor facc conflict model to read-write locks

**Files**: `src/arrax/lowering/fusion.py`, `src/arrax/lowering/array_to_linalg.py`, `tests/lowering/test_fusion.py`

1. **RED**: Write test that `dot(A * 2.0, B)` still does NOT fuse (persistent + ephemeral → blocked). Write test that `sum(A * 2.0)` still fuses (ephemeral + none → allowed). These should already pass — we're changing representation, not behavior. Additionally, write a test that two ephemeral ops fuse: this verifies the new model allows it (currently blocked by M3's UnitAttr approach where both having the tag blocks fusion). Example: `dot(A * 2.0, B)` keeps 2 loops; `sum(A * 2.0)` has 1 loop.
2. **GREEN**:
   - `array_to_linalg.py`: Replace `arrax.uses_facc` UnitAttr with `arrax.facc` StringAttr:
     - MulScalarToLinalgPattern, DivScalarToLinalgPattern: set `arrax.facc = "ephemeral"`
     - DotToLinalgPattern: set `arrax.facc = "persistent"`
   - `fusion.py`: Rewrite `_has_facc_conflict` to implement the lock rule:
     - Extract `_facc_level(loop)` → returns `"persistent"`, `"ephemeral"`, or `"none"` by checking `arrax.facc` attribute on linalg.generic ops in the loop body.
     - Block fusion if either loop has `"persistent"`. Allow if both are `"ephemeral"` or `"none"`.
   - `tile.py`: Update attribute propagation — `arrax.facc` is a discardable attribute, so the existing `for name, attr in op.attributes.items()` loop already copies it. No changes needed.
3. **REFACTOR**: Update existing test names/assertions in `test_fusion.py` if any reference the old UnitAttr by name.
4. **VERIFY**: `uv run pytest -x` — all tests pass. Existing `test_facc_conflict_dot_scalar_not_fused` and `test_facc_conflict_non_dot_ok` still pass with the new model.

### Phase 2: NPU ops + FVMul/FVDiv refactor + asm emission

**Goal**: Define the new NPU ops, refactor existing scalar-vector ops to use SSA operands, and add asm emission for all new/changed ops. After this phase, existing scalar-vector E2E tests still pass with the refactored ops.

#### Step 2.1: FVMulOp/FVDivOp refactor to SSA operand

**Files**: `src/arrax/dialects/npu_dialect.py`, `src/arrax/lowering/linalg_to_npu.py`, `src/arrax/codegen/asm_emitter.py`

1. **RED**: Existing tests should still pass after refactoring — this is a behavior-preserving change. Write a small unit test that creates FVMulOp with an SSA f32 operand (instead of a float property) and verifies it round-trips.
2. **GREEN**: Refactor in three files:
   - `npu_dialect.py`: Change `FVMulOp.scalar` from `prop_def(FloatAttr)` to `operand_def(Float32Type)`. Update constructor to accept SSA value instead of float. Same for `FVDivOp`.
   - `linalg_to_npu.py`: Where FVMulOp/FVDivOp are created, emit an `arith.ConstantOp(FloatAttr(scalar_val, f32))` and pass its result as the SSA operand. The constant op is inserted before the NPU op.
   - `asm_emitter.py`: Change `_emit_fv_mul` and `_emit_fv_div` to read the scalar from the pool (via `op.scalar`, now an SSA value) instead of from a FloatAttr property. The pool register is loaded into facc via FRSTACC + FMACC as before. The constant was already materialized into the pool by `_emit_constant` (which handles FloatAttr → `_fconst_map`).
3. **VERIFY**: `uv run pytest -x` — ALL existing tests pass (especially scalar-vector mul/div E2E tests).

#### Step 2.2: FVSubScalarOp dialect + asm emission

**Files**: `src/arrax/dialects/npu_dialect.py`, `src/arrax/codegen/asm_emitter.py`, `tests/dialects/test_npu_dialect.py`

1. **RED**: Write unit test that creates FVSubScalarOp and verifies it. Write asm emission test that emits FVSubScalarOp and checks the output contains FRSTACC + FMACC + FVSUB_SCALAR.
2. **GREEN**:
   - `npu_dialect.py`: Add FVSubScalarOp with operands: src (MemRefType), dst (MemRefType), n (IndexType), scalar (Float32Type). Add verify_ checking element types, shape match, vector length limit.
   - `asm_emitter.py`: Add `_emit_fvsub_scalar`. Sequence: load scalar into facc (FRSTACC, materialize 1.0 into ft0, FMACC pool_reg * ft0), load n into t0, emit `.insn r 0x2B, 0x0, 0x0C, t0, src_reg, dst_reg`. Add FVSubScalarOp to the import list and dispatch.
3. **VERIFY**: Unit tests pass.

#### Step 2.3: FRsqrtOp dialect + asm emission

**Files**: `src/arrax/dialects/npu_dialect.py`, `src/arrax/codegen/asm_emitter.py`, `tests/dialects/test_npu_dialect.py`

1. **RED**: Write unit test for FRsqrtOp creation/verify. Write asm emission test checking `.insn r 0x2B, 0x0, 0x03`.
2. **GREEN**:
   - `npu_dialect.py`: Add FRsqrtOp with operands: src (MemRefType, rank-0), result (Float32Type). Verify src is rank-0 f32 memref.
   - `asm_emitter.py`: Add `_emit_frsqrt`. Load src base address into a GPR (from reg_map), allocate pool register for result, emit `.insn r 0x2B, 0x0, 0x03, fd, rs1, x0`. Bind result to pool register.
3. **VERIFY**: Unit tests pass.

#### Step 2.4: Scalar arith emission (fdiv.s, fadd.s)

**Files**: `src/arrax/codegen/asm_emitter.py`

1. **RED**: Write unit test that constructs arith.DivfOp and arith.AddfOp on f32 SSA values in a function body alongside NPU ops, emits assembly, and checks for `fdiv.s` / `fadd.s` instructions.
2. **GREEN**: Add `_emit_divf` and `_emit_addf` handlers in asm_emitter:
   - Look up both operands in the FP pool.
   - Allocate pool register for result.
   - Emit `fdiv.s result_reg, lhs_reg, rhs_reg` / `fadd.s result_reg, lhs_reg, rhs_reg`.
   - Add arith.DivfOp and arith.AddfOp to the op dispatch.
3. **VERIFY**: Unit tests pass. `uv run pytest -x` — no regressions.

### Phase 3: Softmax end-to-end

**Goal**: Full softmax pipeline from DSL to emulator. After this phase, `softmax(A)` compiles and produces correct results for all sizes.

#### Step 3.1: SoftmaxOp in array dialect

**Files**: `src/arrax/dialects/array_dialect.py`, `tests/dialects/test_array_dialect.py`

1. **RED**: Write test that creates SoftmaxOp with rank-1 f32 tensor input and verifies it.
2. **GREEN**: Add SoftmaxOp to array_dialect.py. Operand: input (TensorType). Result: result (TensorType). Verify: input is rank-1 f32, result is rank-1 f32, shapes match. Add to dialect registration.
3. **VERIFY**: Test passes.

#### Step 3.2: SoftmaxToLinalgPattern

**Files**: `src/arrax/lowering/array_to_linalg.py`, `tests/lowering/test_array_to_linalg.py`

1. **RED**: Write IR structure test that creates array.softmax, runs ArrayToLinalgPass, and verifies the output contains:
   - 2 linalg.fill ops (neg_inf for max, 0.0 for sum)
   - 2 reduction generics (amax, sum)
   - 3 parallel generics (sub broadcast, exp, div broadcast)
   - Correct affine maps (mixed `(d0)->(d0)` and `(d0)->()`)
2. **GREEN**: Implement SoftmaxToLinalgPattern:
   - Emit 5 linalg.generic ops as specified in the decomposition section.
   - Reduction generics: same structure as AmaxToLinalgPattern and SumToLinalgPattern.
   - Broadcast binary generics: 2 inputs (rank-1 + rank-0), 1 output (rank-1), mixed affine maps.
   - Register the pattern in `ArrayToLinalgPass.apply()`.
3. **VERIFY**: IR structure test passes.

#### Step 3.3: LinalgToNpu broadcast binary pattern

**Files**: `src/arrax/lowering/linalg_to_npu.py`, `tests/lowering/test_linalg_to_npu.py`

1. **RED**: Write test that constructs a broadcast-sub generic (rank-1 + rank-0 inputs, subf body, mixed maps) after bufferize, runs LinalgToNpuPass, and verifies the output contains `npu.fvsub_scalar` with the scalar as an SSA operand. Similarly for broadcast-div → npu.fvdiv and broadcast-mul → npu.fvmul.
2. **GREEN**: Add `LinalgBroadcastBinaryToNpuPattern`:
   - Match linalg.generic with 2 inputs + 1 output, parallel iterator, and one `(d0)->()` affine map.
   - Identify which input is the broadcast scalar (the one with the scalar map).
   - **Scalar forwarding**: use a `_find_dominating_store(scalar_memref)` helper (similar to `find_preceding_fill`) to walk backwards and find a `memref.store` to the rank-0 memref. If found, use the stored value directly. Otherwise emit `memref.LoadOp`.
   - Match body op: subf → FVSubScalarOp, divf → FVDivOp (runtime scalar), mulf → FVMulOp (runtime scalar).
   - Emit the NPU op with the scalar as SSA operand.
   - Handle `arrax.rmsnorm_divisor` attribute (skip for Phase 3 — no attribute means plain broadcast).
3. **VERIFY**: Tests pass.

#### Step 3.4: DSL softmax function + dsl_to_array

**Files**: `src/arrax/dsl/array.py`, `src/arrax/lowering/dsl_to_array.py`, `tests/dsl/test_dsl.py`

1. **RED**: Write DSL test: `softmax(Array("A", (128,)))` produces a DAG node with op="softmax" and shape == (128,).
2. **GREEN**:
   - `array.py`: Add `softmax(x: Array) -> Array` function. Creates DAG node with `op="softmax"`, one operand, shape == input shape.
   - `dsl_to_array.py`: Add `elif node.op == "softmax":` case that creates `SoftmaxOp(operand)`.
   - Import SoftmaxOp in dsl_to_array.py.
3. **VERIFY**: DSL test passes.

#### Step 3.5: Softmax integration + E2E tests

**Files**: `tests/lowering/test_softmax.py` (new), `tests/test_end_to_end.py`

1. **RED**: Write E2E test `test_softmax_basic` with N=128 (tiled). Compile `softmax(A)`, run on emulator, compare with `scipy.special.softmax` or manual `np.exp(x - x.max()) / np.exp(x - x.max()).sum()`.
2. **GREEN**: If all prior steps are correct, the E2E test should pass. Debug any issues through the pipeline.
3. **Additional tests** (write RED, verify GREEN for each):
   - `test_softmax_small` (N=32, untiled)
   - `test_softmax_exact_tile` (N=64)
   - `test_softmax_non_multiple` (N=135)
   - `test_softmax_of_add` (softmax(A + B) — tests producer fusion)
   - `test_softmax_uniform` (all-same input → output should be 1/N)
   - IR structure test: verify fused loop count (3 loops for N=128)
4. **VERIFY**: `uv run pytest -x` — all tests pass.

### Phase 4: RMSNorm end-to-end

**Goal**: Full rmsnorm pipeline. After this phase, `rmsnorm(A)` compiles and produces correct results.

#### Step 4.1: RMSNormOp in array dialect

**Files**: `src/arrax/dialects/array_dialect.py`, `tests/dialects/test_array_dialect.py`

1. **RED**: Write test that creates RMSNormOp with rank-1 f32 tensor input and verifies it.
2. **GREEN**: Add RMSNormOp to array_dialect.py. Same structure as SoftmaxOp.
3. **VERIFY**: Test passes.

#### Step 4.2: RMSNormToLinalgPattern

**Files**: `src/arrax/lowering/array_to_linalg.py`, `tests/lowering/test_array_to_linalg.py`

1. **RED**: Write IR structure test that creates array.rmsnorm, runs ArrayToLinalgPass, and verifies:
   - 1 linalg.fill (zero for dot accumulator)
   - 1 reduction generic with `arrax.facc = "persistent"` (dot body: mulf + addf)
   - 1 parallel generic with `arrax.rmsnorm_divisor` and `arrax.rmsnorm_eps` attributes (broadcast-mul)
2. **GREEN**: Implement RMSNormToLinalgPattern:
   - Step 1: dot(x,x) — same pattern as DotToLinalgPattern but with `ins(%x, %x)` (both inputs are the same tensor). Tag with `arrax.facc = "persistent"`.
   - Step 2: broadcast-mul with attributes `arrax.rmsnorm_divisor = IntegerAttr(N, i64)` and `arrax.rmsnorm_eps = FloatAttr(1e-5, f32)`. Body: mulf with broadcast map.
   - Register the pattern in `ArrayToLinalgPass.apply()`.
3. **VERIFY**: IR structure test passes.

#### Step 4.3: LinalgToNpu rmsnorm scalar math handling

**Files**: `src/arrax/lowering/linalg_to_npu.py`, `tests/lowering/test_linalg_to_npu.py`

1. **RED**: Write test that constructs a broadcast-mul generic with `arrax.rmsnorm_divisor` and `arrax.rmsnorm_eps` attributes (post-bufferize), runs LinalgToNpuPass, and verifies the output contains: `memref.load` (or forwarded SSA), `arith.divf`, `arith.addf`, `memref.alloca`, `memref.store`, `npu.frsqrt`, `npu.fvmul`.
2. **GREEN**: Extend `LinalgBroadcastBinaryToNpuPattern` (from Phase 3):
   - After loading the scalar from the rank-0 memref (or forwarding), check for `arrax.rmsnorm_divisor` attribute.
   - If present: extract N and eps from attributes. Emit:
     - `arith.ConstantOp(FloatAttr(float(N), f32))` → `%N`
     - `arith.DivfOp(%scalar, %N)` → `%meansq`
     - `arith.ConstantOp(FloatAttr(eps, f32))` → `%eps`
     - `arith.AddfOp(%meansq, %eps)` → `%shifted`
     - `memref.AllocaOp(f32, shape=[])` → `%scratch`
     - `memref.StoreOp(%shifted, %scratch, [])` (FRSQRT reads from memory)
     - `FRsqrtOp(%scratch)` → `%scale`
     - `FVMulOp(%src, %dst, %n, %scale)` (runtime scalar)
   - If not present: emit the plain broadcast pattern (from Phase 3).
3. **VERIFY**: Tests pass.

#### Step 4.4: DSL rmsnorm function + dsl_to_array

**Files**: `src/arrax/dsl/array.py`, `src/arrax/lowering/dsl_to_array.py`, `tests/dsl/test_dsl.py`

1. **RED**: Write DSL test: `rmsnorm(Array("A", (128,)))` produces correct DAG node.
2. **GREEN**:
   - `array.py`: Add `rmsnorm(x: Array) -> Array` function. op="rmsnorm", shape == input shape.
   - `dsl_to_array.py`: Add `elif node.op == "rmsnorm":` case that creates `RMSNormOp(operand)`.
3. **VERIFY**: DSL test passes.

#### Step 4.5: RMSNorm E2E tests

**Files**: `tests/test_end_to_end.py`

1. **RED**: Write E2E test `test_rmsnorm_basic` with N=128. Compare with NumPy: `x / np.sqrt(np.mean(x**2) + 1e-5)`.
2. **GREEN**: Debug through pipeline if needed.
3. **Additional tests**:
   - `test_rmsnorm_small` (N=32, untiled)
   - `test_rmsnorm_exact_tile` (N=64)
   - `test_rmsnorm_non_multiple` (N=135)
   - `test_rmsnorm_of_relu` (rmsnorm(relu(A)) — tests producer fusion)
   - `test_rmsnorm_near_zero` (small values — tests eps stability)
4. **VERIFY**: `uv run pytest -x` — all tests pass.

### Phase 5: Polish

**Goal**: Comprehensive E2E testing, code review, refactoring, memory update.

#### Step 5.1: Additional E2E tests

- `test_softmax_large` (N=256 or larger)
- `test_rmsnorm_large` (N=256 or larger)
- `test_softmax_of_mul_scalar` (softmax(A * 2.0) — tests facc-using producer)
- Any edge cases discovered during implementation

#### Step 5.2: Code review

Review all M4 code for:
- Consistent patterns across new and existing code
- No dead code or unused imports
- Docstrings on all public functions
- Attribute propagation through tile pass for new ops
- Pool register lifecycle (allocate, use, release) for new scalar values

#### Step 5.3: IR structure verification

Write tests verifying the fused loop structure for both softmax and rmsnorm at N=128:
- Softmax: exactly 3 scf.for loops after fusion
- RMSNorm: exactly 2 scf.for loops after fusion
- Buffer optimization: intermediate buffers shrunk to 64 elements

#### Step 5.4: Update memory.md, commit

- Update `.ai/memory.md` with M4 status, new patterns, any blockers for M5
- `uv run pytest` — all passing
- Commit
