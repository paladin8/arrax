# Milestone 3: Reductions, Dot Product, and Reduction Fusion

## Goal

Introduce the reduction class of operations end-to-end: `sum`, `dot`, `amax`, and `mean` over 1D f32 arrays. Each reduction produces a scalar result delivered as a rank-0 memref output. Loop fusion is extended to merge elementwise producers with reduction consumers into a single tile loop, so expressions like `sum(A + B)` and `dot(A + B, C)` compile to one pass over memory.

Reductions are **terminal only** in this milestone: they must be the function's return value and cannot feed other ops inside the same compiled function. This restriction is enforced in a single DSL-layer validator and is designed to be lifted cleanly in Milestone 4 (composites) without changes to any IR-level pass.

## Op coverage

| DSL surface    | Array dialect op | linalg body                                  | NPU op                 | NPU funct7          |
|----------------|------------------|----------------------------------------------|------------------------|---------------------|
| `sum(A)`       | SumOp            | reduction `addf`                             | FVReduceOp             | 0x05                |
| `dot(A, B)`    | DotOp            | reduction `mulf + addf` (or `linalg.dot`)    | FVMacOp                | 0x01                |
| `amax(A)`      | AmaxOp           | reduction `maximumf`                         | FVMaxOp                | 0x06                |
| `mean(A)`      | MeanOp           | reduction `addf` + `arrax.mean_divisor` attr | FVReduceOp + `divisor` | 0x05 (then fdiv.s)  |

All four reductions take rank-1 `tensor<Nxf32>` input(s) and produce rank-0 `tensor<f32>`. `dot` takes two inputs with identical shape; the other three take one input.

New capability beyond op coverage: reduction/elementwise loop fusion in the `Fuse` pass, producing single-loop code for chained expressions.

## Pipeline (changed steps marked)

```
trace
  -> dsl_to_array                        [EXTENDED: validates terminal-only]
  -> ArrayToLinalg                       [EXTENDED: reduction lowering + linalg.fill init]
  -> Bufferize                           [EXTENDED: rank-0 memref support]
  -> Tile                                [EXTENDED: reduction strip-mine with iter_args]
  -> Fuse                                [EXTENDED: parallel -> reduction merge + facc conflict guard]
  -> BufferOptimize                      [UNCHANGED: shrink already covers fused intermediates]
  -> LinalgToNpu                         [EXTENDED: reduction patterns]
  -> NpuCanonicalize                     [UNCHANGED]
  -> verify
  -> emit_assembly                       [EXTENDED: iter_args, scalar FP register pool, facc handling, mean fdiv]
```

Five passes extended, one unchanged, plus a unified scalar FP register pool in the asm emitter.

## DSL surface

Four new free functions exported from the `arrax` package:

```python
from arrax import sum, dot, amax, mean

sum(A)        # -> Array with shape=()
dot(A, B)     # -> Array with shape=() ; requires A.shape == B.shape and 1D
amax(A)       # -> Array with shape=()
mean(A)       # -> Array with shape=()
```

The names shadow Python's `sum` and `max` builtins when star-imported; users who need the builtins must access them via `builtins.sum`. `amax` is named after NumPy's convention to avoid the `max` name entirely. The naming matches M2's pattern of `from arrax import relu, exp`.

`Array` accepts `shape=()` without structural changes. `is_leaf`, `__repr__`, and the DAG machinery already treat shape as an opaque tuple; rank-0 is a valid case by inspection. Minor audit of a few asserts (if any) that assume `len(shape) >= 1`.

### Validator: reductions must be terminal

A new validator runs inside `dsl_to_array` (or a sibling module `dsl/validate.py` — decide at implementation). It walks the traced DAG once and checks that any node whose `op` is `sum`, `dot`, `amax`, or `mean` has no non-root users. Non-root = the node is referenced as an operand by any other node in the DAG.

On violation it raises `ValueError` with a message naming the offending reduction and pointing to the M3 restriction:

```
ValueError: reduction `sum` is used non-terminally at node <id>; milestone 3
only supports reductions as the function's return value. Compile a separate
function for intermediate scalars.
```

Accepted in M3:
```python
lambda A: sum(A)
lambda A: dot(A, B)
lambda A: amax(A)
lambda A: mean(A)
lambda A, B: sum(A + B)          # reduction is the only root; fused at Fuse pass
lambda A, B: dot(A + B, A - B)   # reduction is the root; elementwise inputs fused
lambda A: relu(A + 1.0)          # unchanged — no reductions involved
```

Rejected in M3:
```python
lambda A: A + sum(A)            # sum has a non-root user (the add)
lambda A, B: sum(A) + sum(B)    # both sums feed an add; neither is terminal
lambda A: A * mean(A)           # mean feeds scalar-vector mul
```

**Scope principle: the terminal restriction lives only in this validator.** All IR-level passes are written as if reductions could appear anywhere; the DSL layer simply prevents them from doing so. Milestone 4 lifts the restriction by deleting the validator and adding its dependent scalar-consumer ops, with no structural changes to any M3 pass. A hand-constructed non-terminal reduction IR test in `tests/lowering/test_bufferize.py` will verify this invariant for M3's bufferize pass.

## Array dialect

Four new `irdl_op_definition` ops in `src/arrax/dialects/array_dialect.py`:

```python
@irdl_op_definition
class SumOp(IRDLOperation):
    name = "array.sum"
    input = operand_def(TensorType)
    result = result_def(TensorType)
    # verify: input rank == 1, f32; result rank == 0, f32
    traits = frozenset([Pure()])

@irdl_op_definition
class DotOp(IRDLOperation):
    name = "array.dot"
    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)
    # verify: lhs/rhs identical rank-1 f32 tensors; result rank-0 f32
    traits = frozenset([Pure()])

@irdl_op_definition
class AmaxOp(IRDLOperation):
    name = "array.amax"
    input = operand_def(TensorType)
    result = result_def(TensorType)
    traits = frozenset([Pure()])

@irdl_op_definition
class MeanOp(IRDLOperation):
    name = "array.mean"
    input = operand_def(TensorType)
    result = result_def(TensorType)
    traits = frozenset([Pure()])
```

All four carry `assembly_format` strings so they round-trip through IR text like M1/M2 ops. MeanOp is a distinct op (not Sum plus post-divide) so the divisor flows through the stack as a single unit.

## Array → Linalg lowering

Four new `RewritePattern` classes in `src/arrax/lowering/array_to_linalg.py`. Each reduction lowers to a `linalg.generic` with:

- `iterator_types = ["reduction"]`
- Input indexing map `(d0) -> (d0)` (identity; for dot the two inputs share this map)
- Output indexing map `(d0) -> ()` (scalar sink)
- Output tensor initialized via `linalg.fill` with the reduction's identity
- Body computing the combiner over `%in` and `%acc`

### Init values and bodies

| Op     | Identity (arith.constant f32)   | Body                                                     |
|--------|---------------------------------|----------------------------------------------------------|
| sum    | `0.0`                           | `%s = arith.addf %acc, %in`                              |
| dot    | `0.0`                           | `%p = arith.mulf %lhs, %rhs`; `%s = arith.addf %acc, %p` |
| mean   | `0.0`                           | `%s = arith.addf %acc, %in` (same as sum)                |
| amax   | `-inf` (IEEE bits `0xFF800000`) | `%m = arith.maximumf %acc, %in`                          |

`linalg.fill` seeds the rank-0 output tensor with the identity:
```mlir
%empty = tensor.empty() : tensor<f32>
%c_id  = arith.constant 0.0 : f32
%init  = linalg.fill ins(%c_id : f32) outs(%empty : tensor<f32>) -> tensor<f32>
```

`%init` is passed as the `outs` operand of the generic. This is the standard MLIR idiom — the generic body reads the current accumulator value from `outs` on entry, produces a new value, and yields it.

### Example: sum(A) where A : tensor<128xf32>

```mlir
%empty = tensor.empty() : tensor<f32>
%c0    = arith.constant 0.0 : f32
%init  = linalg.fill ins(%c0 : f32) outs(%empty : tensor<f32>) -> tensor<f32>
%result = linalg.generic {
    indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
    ],
    iterator_types = ["reduction"]
} ins(%A : tensor<128xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %acc: f32):
        %s = arith.addf %acc, %in : f32
        linalg.yield %s : f32
} -> tensor<f32>
```

### Mean: divisor as a discardable attribute

`MeanOp` lowers to the same generic as sum, but the pattern attaches an `arrax.mean_divisor` discardable attribute holding the input length as an `IntegerAttr : i64`:

```mlir
%result = linalg.generic {
    indexing_maps = [...],
    iterator_types = ["reduction"],
    arrax.mean_divisor = 128 : i64
} ins(%A : tensor<128xf32>) outs(%init : tensor<f32>) { addf } -> tensor<f32>
```

The attribute is preserved by bufferize and tile (both clone op attributes unchanged) and is read by `LinalgToNpu` where it becomes a property on `npu.fvreduce`, which directs the asm emitter to emit a trailing `fdiv.s`.

### Dot: named op vs body-shape match

At the start of Phase 3, verify whether `xdsl.dialects.linalg.DotOp` exists in xDSL 0.59. The check is `grep -rn "class DotOp\|name = \"linalg.dot\"" path/to/xdsl/dialects/linalg*`.

- **If `linalg.dot` exists**: `array.dot` lowers to it directly. `LinalgToNpu` matches by op type, not body shape.
- **If not**: `array.dot` lowers to `linalg.generic` with `mulf + addf` body (as shown above). `LinalgToNpu` matches the body shape. Add a comment noting the preference for the named op.

Either path is correct; the named op is cleaner and more IR-readable when available.

### xDSL API notes (verify at implementation time)

- `linalg.fill` (class `FillOp`): confirm presence. Fallback: `tensor.from_elements` plus assigning it as the generic's `outs` — slightly less idiomatic.
- `IteratorTypeAttr.reduction()` factory: confirm presence. Fallback: `IteratorTypeAttr(IteratorType.REDUCTION)`.
- `arith.maximumf` (NaN-propagating): confirm presence. Fallback: `arith.maxnumf` (NaN-ignoring) with a comment that the semantic difference affects tests with NaN inputs.

## Bufferize extension (rank-0 memref support)

Current `Bufferize` walks the function, converting `tensor.empty` to `memref.alloc` and `linalg.generic` on tensors to `linalg.generic` on memrefs. For rank-0, the type converter must handle:

- `tensor<f32>` -> `memref<f32>`
- `memref.alloc() : memref<f32>` (no dimensions in the call)
- `linalg.fill ins(%c : f32) outs(%m : memref<f32>)` (rank-0 output)

The existing "promote final `tensor.empty` to the function's output arg" logic already examines the terminal value of the function; it just needs to accept rank-0 tensors as a valid terminal type. Concretely: the type-equality check between the terminal tensor type and the appended output memref type should naturally match `tensor<f32>` to `memref<f32>` after the type converter runs.

The `arrax.mean_divisor` attribute on `linalg.generic` is preserved automatically because `Region.clone()` copies the entire op including its attributes.

**Forward-compat test**: one hand-constructed bufferize test where a rank-0 `linalg.generic` reduction has a non-terminal use (wired into a hypothetical downstream op), asserting that bufferize allocates a rank-0 `memref.alloc()` for it as an intermediate rather than promoting to the function output. This pins down the invariant that bufferize does not assume reductions are terminal. (DSL will not produce this pattern in M3; the test builds IR directly.)

## Tile pass extension (reduction strip-mining with iter_args)

The new case lives alongside the existing parallel strip-mine in `src/arrax/lowering/tile.py`.

**Detection**: `linalg.generic` on memrefs whose `iterator_types` contains `"reduction"`. For N <= 64, pass through unchanged (the linalg-to-npu pattern handles untiled reductions fine). For N > 64, apply the transform below.

**Transform**: wrap the reduction in an `scf.for` that carries the scalar accumulator across iterations via `iter_args`. Each tile allocates a stack scratch for the inner generic, fills it with the current accumulator, runs a chunk-sized reduction into it, loads the new accumulator out, and yields.

### Before tiling (sum, N=128)

```mlir
%c0   = arith.constant 0.0 : f32
linalg.fill ins(%c0 : f32) outs(%out : memref<f32>)
linalg.generic {
    iterator_types = ["reduction"],
    indexing_maps = [(d0) -> (d0), (d0) -> ()]
} ins(%A : memref<128xf32>) outs(%out : memref<f32>) { addf }
```

### After tiling

```mlir
%c0    = arith.constant 0.0 : f32
%c0idx = arith.constant 0 : index
%c64   = arith.constant 64 : index
%c128  = arith.constant 128 : index

%final = scf.for %iv = %c0idx to %c128 step %c64
         iter_args(%acc = %c0) -> (f32) {
    %chunk = arith.minsi %c64, (%c128 - %iv) : index
    %sub_a = memref.subview %A[%iv][%chunk][1]
               : memref<128xf32> to memref<?xf32, strided<[1], offset: ?>>
    %scratch = memref.alloca() : memref<f32>
    linalg.fill ins(%acc : f32) outs(%scratch : memref<f32>)
    linalg.generic {
        iterator_types = ["reduction"],
        indexing_maps = [(d0) -> (d0), (d0) -> ()]
    } ins(%sub_a : memref<?xf32, strided<[1], offset: ?>>)
      outs(%scratch : memref<f32>) { addf }
    %new_acc = memref.load %scratch[] : memref<f32>
    scf.yield %new_acc : f32
}
memref.store %final, %out[] : memref<f32>
```

Key choices:

1. **`iter_args` thread carries the accumulator in SSA f32**, not a heap memref. The asm emitter register-allocates this directly into the scalar FP register pool — no per-iteration load/store for the accumulator itself.
2. **`memref.alloca` (stack), not `memref.alloc`** for the inner scratch. Per-iteration, no buffer optimization interaction.
3. **`linalg.fill` inside the loop body seeds the inner generic with `%acc`.** Dataflow is explicit: each tile reads the current accumulator, adds its contribution, writes back, yields.
4. **Final `memref.store` lives outside the loop.** Single memory write to the output.
5. **Dot uses the same structure** with two input subviews and the `mulf + addf` body (or `linalg.dot`). The asm emitter will later special-case dot to use `facc` as the real accumulator — at the IR level it still threads through `iter_args` for uniformity.
6. **`arrax.mean_divisor` attribute migrates to the inner (post-tile) linalg.generic** so that LinalgToNpu, which walks inside the loop body, can see it.
7. **N <= 64 passes through unchanged** — the untiled linalg.generic reduction is matched directly by LinalgToNpu.

## NPU dialect: three new reduction ops

New ops in `src/arrax/dialects/npu_dialect.py`. Each takes a prior-accumulator f32 SSA value and produces a new-accumulator f32 SSA value, matching the `iter_args` threading pattern from the tile pass.

```python
@irdl_op_definition
class FVReduceOp(IRDLOperation):
    """Chunk sum: result = acc_in + sum(src[0..n]).
    Maps to NPU.FVREDUCE (opcode=0x2B, funct7=0x05) + fadd.s with prior accumulator.
    Optional `divisor` property signals mean semantics (trailing fdiv.s in asm).
    """
    name = "npu.fvreduce"
    src     = operand_def(MemRefType)
    n       = operand_def(IndexType)
    acc_in  = operand_def(Float32Type)
    result  = result_def(Float32Type)
    divisor = opt_prop_def(IntegerAttr)   # i64, present only for mean
    irdl_options = (ParsePropInAttrDict(),)
    # verify: src is 1D f32 memref; n <= 64 when constant

@irdl_op_definition
class FVMacOp(IRDLOperation):
    """Chunk dot: result = acc_in + dot(lhs[0..n], rhs[0..n]).
    Maps to NPU.FVMAC (funct7=0x01) accumulating into facc across calls.
    """
    name = "npu.fvmac"
    lhs     = operand_def(MemRefType)
    rhs     = operand_def(MemRefType)
    n       = operand_def(IndexType)
    acc_in  = operand_def(Float32Type)
    result  = result_def(Float32Type)

@irdl_op_definition
class FVMaxOp(IRDLOperation):
    """Chunk max: result = max(acc_in, max(src[0..n])).
    Maps to NPU.FVMAX (funct7=0x06) + fmax.s with prior accumulator.
    """
    name = "npu.fvmax"
    src     = operand_def(MemRefType)
    n       = operand_def(IndexType)
    acc_in  = operand_def(Float32Type)
    result  = result_def(Float32Type)
```

Each op's `verify_()` enforces the same `n <= NPU_MAX_VEC_LEN` rule on constant `n` that existing NPU ops do, plus type checks (1D f32 memref inputs, identical lhs/rhs shape for `FVMacOp`).

**Rationale for SSA f32 threading instead of rank-0 memref output**: the NPU reduction instructions produce values in FP registers. Writing through a rank-0 memref would pay an `fsw` + `flw` round trip per loop iteration. Threading the accumulator through SSA keeps it in a register across the whole loop; the memref write happens exactly once, after the loop.

## LinalgToNpu: reduction pattern matching

New patterns in `src/arrax/lowering/linalg_to_npu.py` that recognize the tiled reduction shape produced by the Tile pass and collapse it to a single NPU op.

### Matched cluster

Inside an `scf.for` body with `iter_args(%acc = %init) -> (f32)`:

```mlir
%scratch = memref.alloca() : memref<f32>
linalg.fill ins(%acc : f32) outs(%scratch : memref<f32>)
linalg.generic {
    iterator_types = ["reduction"],
    indexing_maps = [...]
} ins(%src : ...) outs(%scratch : memref<f32>) { <body> }
%new_acc = memref.load %scratch[] : memref<f32>
scf.yield %new_acc : f32
```

### Rewrite target

```mlir
%new_acc = npu.fvreduce %src, %n, %acc : ... -> f32
  (or fvmac / fvmax based on body shape)
scf.yield %new_acc : f32
```

The `alloca`, `fill`, `linalg.generic`, and `load` are erased. The `scf.yield` and the `scf.for` structure are preserved.

### Body-shape dispatch

| Linalg body (reduction iterator)                                   | Resulting NPU op |
|--------------------------------------------------------------------|------------------|
| `arith.addf(%acc, %in)` yielding to output                         | `npu.fvreduce`   |
| `arith.mulf(%a, %b)` + `arith.addf(%acc, %prod)` (or `linalg.dot`) | `npu.fvmac`      |
| `arith.maximumf(%acc, %in)` yielding to output                     | `npu.fvmax`      |

Order-of-operands tolerance: commutative combiners may appear with operands swapped post-canonicalization (e.g., `addf(%in, %acc)` vs `addf(%acc, %in)`). The matcher accepts both.

### Mean's divisor

If the matched `linalg.generic` carries the `arrax.mean_divisor` discardable attribute, the resulting `npu.fvreduce` sets its `divisor` property to the same IntegerAttr. No intermediate IR ops.

### N <= 64 fast path (no tile loop)

When the reduction was not wrapped in a tile loop (N <= 64 at the input), the matched cluster appears in straight-line code with `%acc = arith.constant 0.0` and the result feeding directly into `memref.store %result, %out[]`. Same pattern, no structural special case — the matcher walks its operands back to find `%acc` whether it originates from a constant or an `iter_args`.

## Asm emission

The largest chunk of new code. Lives in `src/arrax/codegen/asm_emitter.py`.

### 1. `scf.for` with `iter_args`

Current emitter handles iter_args-free `scf.for` only. The new case:

- Before loop emission, run a **last-use pre-walk** over the function body to compute `last_use_index[ssa_value] -> op_index` for every f32 SSA value. This drives the scalar FP register pool's release logic.
- For each `scf.for`'s `iter_args`, allocate an FP register from the **scalar FP register pool** (see next subsection) for each iter_arg. The same register represents the SSA value throughout the loop.
- Emit code to materialize the `iter_args` init operand into the allocated register before the loop label. For a literal constant (`arith.constant 0.0`), this is `li` + `fmv.w.x`. The existing f32-constant-to-register helper (used by M2's FVMUL scalar seed) is reused.
- Inside the loop body, the yielded value must end up in the iter_args register. The NPU reduction op emitters target the allocated register directly so no `fmv.s` is needed at yield time.
- After the loop, the iter_args register holds the final value. Subsequent ops (the `fdiv.s` for mean, the final `memref.store`) read from it.

### 2. Scalar FP register pool

A new helper `ScalarFPRegisterPool` with:

- **Pool**: `fs0` through `fs11` (RISC-V callee-saved FP regs, 12 entries).
- **Map**: `dict[SSAValue, str]` of allocated registers.
- **`allocate(ssa_value) -> str`**: picks the next free register, marks it in use, records in the map. Returns the register name.
- **`release(ssa_value)`**: frees the register associated with `ssa_value`.
- **`last_use_index`**: precomputed per function. After emitting the op at index `i`, any SSA value whose `last_use_index == i` is released.
- **Hard failure** if the pool is exhausted (clear error message; no spill logic).

Used only for f32 SSA values that represent scalars (reduction accumulators, mean divisors, loop-yield values). Ephemeral scratch FP registers used inside a single op emission routine (e.g., `ft0` for FVREDUCE's immediate destination) are **not** managed by the pool — they're local, single-op use.

M3 will allocate exactly one scalar at a time in practice, resolving to `fs0`. The pool exists to keep the machinery general from day one so M4 can add additional live scalars without retrofitting.

### 3. New NPU reduction op handlers

**`npu.fvreduce`** (sum / mean):
```asm
    # acc_in already in <acc_reg>, chunk n in <n_reg>, src base in <src_reg>
    .insn r 0x2B, 0x0, 0x05, ft0, <src_reg>, <n_reg>   # FVREDUCE ft0 = sum(...)
    fadd.s <acc_reg>, <acc_reg>, ft0                    # combine into iter_args reg
```

**`npu.fvmax`**:
```asm
    .insn r 0x2B, 0x0, 0x06, ft0, <src_reg>, <n_reg>
    fmax.s <acc_reg>, <acc_reg>, ft0
```

**`npu.fvmac`** (dot — uses facc as the real accumulator):
```asm
    # Once, before the loop label (emitted when the pool sees an fvmac op):
    .insn r 0x2B, 0x5, 0x00, x0, f0, f0        # FRSTACC x0: zero facc, discard

    # Inside the loop body:
    .insn r 0x2B, 0x0, 0x01, <n_reg>, <lhs_reg>, <rhs_reg>  # FVMAC facc += dot

    # Once, after the loop closes:
    .insn r 0x2B, 0x5, 0x00, <acc_reg>, f0, f0  # FRSTACC <acc_reg>: read facc, zero
```

**The `fvmac` asymmetry, documented in a code comment**: for sum/mean/amax the iter_args SSA thread maps to a pooled FP register updated via `fadd.s` / `fmax.s` each iteration. For dot the iter_args thread is *cosmetic* — the real state is `facc`, and the emitter bypasses the pooled register during the loop, materializing the final value in the pooled register only at loop exit via `FRSTACC <acc_reg>`. M3 asserts `acc_in` is the constant 0.0 for every fvmac (enforced at array -> linalg lowering).

### 4. Mean's trailing `fdiv.s`

When emitting a reduction loop whose body contains a single `npu.fvreduce` op with a non-empty `divisor` property, after the loop closes and before any subsequent store:

```asm
    # Materialize N_f32_bits as f32 once
    li t0, <N as IEEE f32 bits>
    fmv.w.x ft1, t0
    fdiv.s <acc_reg>, <acc_reg>, ft1    # <acc_reg> = sum / N = mean
```

`ft1` is ephemeral scratch (not pool-managed). The divisor is a compile-time constant, so the bit pattern is a literal in the emitted assembly.

### 5. Scalar store to rank-0 output memref

New handler for `memref.store %val : f32, %out[] : memref<f32>`:

```asm
    fsw <val_reg>, 0(<out_base_reg>)
```

Where `<val_reg>` is whatever pool register holds `%val` and `<out_base_reg>` is the integer argument register holding the output memref base (e.g., `a1` or later, depending on arg order).

Because the LinalgToNpu pattern collapses the `alloca + fill + linalg.generic + load` cluster to a single NPU op, the asm emitter never sees `memref.alloca` or intermediate scalar `memref.load`/`memref.store` inside a loop — only the single outer `memref.store` to `%out` after the loop.

### 6. Files touched in codegen

- `src/arrax/codegen/asm_emitter.py`:
  - New `ScalarFPRegisterPool` helper
  - `iter_args` path in the `scf.for` emitter
  - Three new NPU op handlers (`npu.fvreduce`, `npu.fvmac`, `npu.fvmax`)
  - Mean divisor post-loop `fdiv.s` emission
  - Rank-0 `memref.store` handler
  - Last-use pre-walk over the function
- No changes to `build.py` or `firmware_harness.py` — the output symbol is still a rank-0 memref in the BSS, addressed the same way by the emulator.

### What is not added in M3 emission

- No general `arith.divf` / `arith.constant` f32 handler. Mean's divide is folded into the NPU op emitter via the `divisor` property. General scalar arith is an M4 concern.
- No integer-register allocator changes.
- No changes to Fuse or BufferOptimize (their codegen-relevant behavior is unchanged; the Fuse transform extension happens at the IR level, upstream of asm emission).

## Reduction/elementwise loop fusion

This is new scope for M3 that extends the existing Fuse pass.

### What fuses

Any elementwise `scf.for` (no `iter_args`) that produces a buffer read by an immediately-following reduction `scf.for` (one `iter_args` carrying a scalar f32), provided they share iteration bounds, producer-consumer matches, and no `facc` conflict arises (see below).

Examples that compile to a single fused loop:

```python
sum(A + B)                     # FVADD then FVREDUCE in one loop
sum(relu(A))                   # FVRELU then FVREDUCE
sum((A + B) - C)               # two FVADDs (pre-fused by M2's parallel-parallel) then FVREDUCE
amax(A - B)                    # FVSUB then FVMAX
mean(A + B)                    # FVADD then FVREDUCE + trailing fdiv
dot(A + B, C)                  # FVADD then FVMAC
dot(A - B, A + B)              # two FVSUBs/FVADDs then FVMAC
```

### Transform mechanics

**Before**:
```mlir
scf.for %iv = ... step %c64 {
    ... elementwise body writing %tmp_sub ...
}

%final = scf.for %iv2 = ... step %c64 iter_args(%acc = %init) -> (f32) {
    ... reduction body reading %tmp_sub ...
    scf.yield %new_acc : f32
}
memref.store %final, %out[]
```

**After**:
```mlir
%final = scf.for %iv = ... step %c64 iter_args(%acc = %init) -> (f32) {
    ... elementwise body (IV remapped) writing %tmp_sub ...
    ... reduction body reading %tmp_sub ...
    scf.yield %new_acc : f32
}
memref.store %final, %out[]
```

Steps in the transform:

1. **Detection**: same producer-consumer check as M2 fusion, with one relaxation — the second loop may have `iter_args` while the first has none. Both loops still share `lb`, `ub`, `step` via SSA-identical operands.
2. **Adopt the reduction loop's iter_args** on the merged loop (signature, init operands, yield type).
3. **Remap the elementwise loop's induction variable** to the reduction loop's IV. Walk every use inside the elementwise body and rewrite.
4. **Splice the elementwise body before the reduction body.** All ops from the elementwise loop's body (minus its implicit terminator) are inserted at the top of the reduction loop's body. Existing op-move primitive from M2 fusion.
5. **Erase the elementwise loop.**
6. **Fixed-point iteration.** The Fuse pass runs to a fixed point so chains like `sum((A+B) - C)` fuse fully: first the two elementwise loops merge via M2's parallel-parallel case, then that merged loop fuses with the reduction via the new case.

### Buffer shrinking interaction

After reduction/elementwise fusion, the intermediate `%tmp` buffer (originally `memref.alloc` of N elements from the elementwise producer) is accessed only through subviews inside a single fused loop body. That is exactly M2's buffer shrink detection condition, so it fires without modification and `%tmp` shrinks from `memref<Nxf32>` to `memref<64xf32>`. Subview offsets rewrite from `%tmp[%iv]` to `%tmp[0]`. No change to `BufferOptimize`.

One robustness test: confirm the shrink detector works when the producer subview's offset is the fused loop's IV (which may differ in SSA identity from the original elementwise IV because of remapping). Covered by a new test in `tests/lowering/test_buffer_optimize.py`.

### Facc conflict guard

`npu.fvmul` / `npu.fvdiv` (M2's scalar-vector ops) use `facc` as a scalar holder. `npu.fvmac` (dot) uses `facc` as the running dot accumulator. A fused loop containing both in the same iteration would corrupt the dot running sum.

**Conflict pattern (detected at the linalg level, pre-NPU)**:

- Any `linalg.generic { parallel }` in the candidate merged body has a body that references an `arith.constant` produced inside the block (the M2 scalar-vector body shape: `arith.mulf %x, %const` or `arith.divf %x, %const`).
- AND the candidate merged body contains a `linalg.generic { reduction }` with `mulf + addf` body (or a `linalg.dot` named op) — i.e., a dot reduction.

When both conditions hold, **fusion is skipped** for that candidate pair. Both loops remain separate and the result compiles correctly (as two separate tile loops).

Non-dot reductions (sum, amax, mean) are unaffected — FVREDUCE, FVMAX, and sum's trailing `fdiv.s` do not touch `facc`. Fusing scalar-vector ops with sum/amax/mean is always safe.

**Known M3 limitation** (documented in code comment + user-facing error context): expressions of the form `dot(A * c, B)` or `dot(A, B * c)` compile correctly but do not fuse across the scalar-vector/reduction boundary. They run as two tile loops. M4 will revisit this once scalar arithmetic on reduction outputs is available (`c * dot(A, B)` becomes a natural rewrite).

### Fusion cases handled after M3

| Producer         | Consumer         | Status                                         |
|------------------|------------------|------------------------------------------------|
| parallel         | parallel         | M2 (unchanged)                                 |
| parallel         | reduction        | M3 (new, with facc conflict guard)             |
| reduction        | parallel         | Not fused (reduction must complete first)      |
| reduction        | reduction        | Not possible in M3 (terminal-only restriction) |

### Safety assertion

Add an explicit early-return in the Fuse pass for "iter_args shapes differ and neither is the parallel-reduction case we handle", with a comment explaining that M3 reductions are terminal and that more exotic iter_args mismatches have no valid merge. This prevents future optimization work from silently relaxing the match predicate in a way that corrupts correctness.

## Testing strategy

Tests mirror the M1/M2 structure — one test file per module.

### Unit tests by layer

**`tests/dsl/test_array.py`** (new cases):
- `Array(shape=())` constructs without error
- `is_leaf` and `__repr__` work on rank-0 arrays
- Rank-0 Array in a DAG position is accepted by the tracer

**`tests/dsl/test_tracer.py`** (new cases):
- `sum(A)`, `dot(A, B)`, `amax(A)`, `mean(A)` produce DAG nodes with expected op names and `shape == ()`
- `dot(A, B)` raises `ValueError` on mismatched shapes (1D different lengths, or non-1D)

**`tests/dsl/test_reductions_terminal.py`** (new file):
- Accepted: `lambda A: sum(A)`, `lambda A: dot(A, B)`, `lambda A: amax(A)`, `lambda A: mean(A)`, `lambda A, B: sum(A + B)`, `lambda A, B: dot(A + B, A - B)`
- Rejected with ValueError: `lambda A: A + sum(A)`, `lambda A, B: sum(A) + sum(B)`, `lambda A: A * mean(A)`
- Error message contains "terminal" and the offending reduction name

**`tests/dialects/test_array_dialect.py`** (new cases):
- `SumOp`, `AmaxOp`, `MeanOp` verify rank-1 f32 input -> rank-0 f32 result
- `DotOp` verifies identical rank-1 f32 inputs -> rank-0 f32 result
- Shape mismatch in `DotOp` raises `VerifyException`
- All four round-trip through IR parse/print

**`tests/dialects/test_npu_dialect.py`** (new cases):
- `FVReduceOp` verifies 1D f32 memref src + f32 acc_in -> f32 result
- `FVMacOp` verifies matching lhs/rhs memref types
- `FVMaxOp` same as FVReduceOp
- All three enforce `n <= 64` when `n` is a constant
- `FVReduceOp` accepts and preserves the optional `divisor` property
- All three round-trip through IR parse/print

**`tests/lowering/test_dsl_to_array.py`** (new cases):
- Each reduction produces the corresponding `array.*` op with rank-0 result
- Terminal validator integration: rejected cases raise ValueError during `dsl_to_array`

**`tests/lowering/test_array_to_linalg.py`** (new cases):
- Each reduction becomes `linalg.generic` with `reduction` iterator + `linalg.fill` init + expected body shape
- `MeanOp` carries `arrax.mean_divisor` discardable attribute after lowering
- `DotOp` produces `mulf + addf` body (or `linalg.dot` named op, depending on xDSL availability — both branches tested)
- Init values: 0.0 for sum/dot/mean, -inf for amax

**`tests/lowering/test_bufferize.py`** (new cases):
- Rank-0 `tensor.empty` -> `memref.alloc() : memref<f32>`
- Rank-0 linalg.generic reduction -> on-memref form with output promoted to function arg
- `arrax.mean_divisor` attribute survives bufferize unchanged
- **Forward-compat test**: hand-constructed IR with a rank-0 reduction having a non-terminal use -> bufferize allocates an intermediate rank-0 memref rather than promoting to the function output

**`tests/lowering/test_tile.py`** (new cases):
- N=32, reduction: pass-through (untiled)
- N=64, reduction: pass-through (at limit)
- N=128, reduction: scf.for with iter_args over 2 tiles, iter_args init = identity, inner alloca + fill + linalg.generic + load + yield
- N=100, reduction: scf.for with iter_args + remainder chunk handling via arith.minsi
- Verify iter_args f32 type, yield operand, terminal memref.store outside the loop
- Each of sum, amax, mean, dot (different body shapes)

**`tests/lowering/test_linalg_to_npu.py`** (new cases):
- Matched pattern: `alloca + fill + linalg.generic + load` cluster in a loop body -> single `npu.fvreduce`/`fvmac`/`fvmax` op
- Untiled case (N<=64): same pattern match in straight-line code
- `divisor` property transfer for mean
- Commutative body operand orderings both match (e.g., `addf(%acc, %in)` vs `addf(%in, %acc)`)
- Dot via named `linalg.dot` op (if xDSL provides it) and via body-shape match (fallback path)

**`tests/codegen/test_asm_emitter.py`** (new cases):
- `sum(A)` N=64: golden-string snapshot containing FVREDUCE .insn + `fadd.s` + final `fsw`
- `sum(A)` N=128: FVREDUCE inside a loop, iter_args materialization before loop, final `fsw` after
- `dot(A, B)` N=64 and N=128: FRSTACC bracketing the loop, FVMAC .insn inside
- `amax(A)` N=128: FVMAX + `fmax.s`, iter_args init to -inf
- `mean(A)` N=128: trailing `fdiv.s` after the loop, before the final `fsw`
- Scalar FP register pool: hand-built IR with two simultaneously-live scalars -> distinct `fs*` registers allocated
- Last-use release: a scalar freed after its last use, its register reusable by a later allocation

**`tests/lowering/test_fusion.py`** (new cases):
- `sum(A + B)`: single fused loop with iter_args, FVADD body followed by FVREDUCE body
- `sum(relu(A))`: same with FVRELU producer
- `dot(A + B, C)`: single fused loop with FVADD + FVMAC bodies
- `amax(A - B)`: FVSUB + FVMAX bodies
- `mean((A + B) - C)`: chain of three ops fused (two parallel fuses happen first, then the parallel-reduction fuse)
- **Conflict guard**: `dot(A * 2.0, B)` -> two separate loops (fusion skipped); assert both loops present in the post-Fuse IR
- **Conflict guard (non-dot ok)**: `sum(A * 2.0)` -> single fused loop (facc is used by FVMUL, but FVREDUCE doesn't touch facc, so fusion is safe)

**`tests/lowering/test_buffer_optimize.py`** (new cases):
- Post-fusion intermediate buffer shrinks to 64 elements
- Subview offsets rewrite correctly when the fused loop's IV differs in SSA from the original elementwise IV

### End-to-end tests (`tests/test_end_to_end.py`)

Each case compiles through the full pipeline, assembles, runs on the emulator, and compares to NumPy with `rtol=1e-5` (relaxed from 1e-6 for reductions because of double-precision-vs-float accumulation differences).

```python
# Primitive reductions
lambda A: sum(A)          # N=16, N=64, N=100, N=1024
lambda A, B: dot(A, B)    # N=16, N=64, N=128, N=1024
lambda A: amax(A)         # N=16, N=64, N=100, with all-negative and mixed-sign inputs
lambda A: mean(A)         # N=16, N=64, N=100, N=1024

# Fused reduction + elementwise
lambda A, B: sum(A + B)
lambda A: sum(relu(A))
lambda A, B: amax(A - B)
lambda A, B: dot(A + B, A - B)
lambda A, B, C: mean((A + B) - C)

# Non-fused due to facc conflict (correctness check, not speed)
lambda A, B: dot(A * 2.0, B)
```

### Negative tests (DSL-level validation)

In `tests/dsl/test_reductions_terminal.py`:
```python
pytest.raises(ValueError, compile_to_asm, lambda A: A + sum(A), ...)
pytest.raises(ValueError, compile_to_asm, lambda A, B: sum(A) + sum(B), ...)
pytest.raises(ValueError, compile_to_asm, lambda A: A * mean(A), ...)
```

Shape mismatches in `dot`:
```python
pytest.raises(ValueError, compile_to_asm, lambda A, B: dot(A, B), {"A": (32,), "B": (64,)})
```

## Implementation phases

Six phases, each independently testable end-to-end. Each phase commit includes `.ai/memory.md` updates per session-end rule.

| Phase | Scope                                                                                                                                                                                                                                |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | `sum(A)` end-to-end + all infrastructure: terminal validator, rank-0 bufferize, reduction tile-extension with iter_args, scalar FP register pool, FVReduceOp, LinalgToNpu reduction pattern, last-use pre-walk, asm emission for sum |
| 2     | `amax(A)` — add AmaxOp + FVMaxOp + array-to-linalg + linalg-to-npu dispatch + asm emission (small delta on Phase 1 infrastructure)                                                                                                   |
| 3     | `dot(A, B)` — add DotOp + FVMacOp + array-to-linalg (check `linalg.dot` first) + linalg-to-npu dispatch + facc-based asm emission with the `fvmac` asymmetry                                                                         |
| 4     | `mean(A)` — add MeanOp + `arrax.mean_divisor` attribute plumbing + `divisor` property on FVReduceOp + trailing `fdiv.s` in asm                                                                                                       |
| 5     | Reduction/elementwise fusion extension — new case in Fuse pass (parallel -> reduction merge), facc conflict guard, fixed-point iteration if not already present, buffer-optimize post-fusion verification                            |
| 6     | Polish: fusion safety early-return with documentation comment, examples (`examples/dot_product.py`, expand `examples/elementwise.py`), full negative test suite, any remaining cleanup                                               |

### Phase dependency graph

```
Phase 1 (sum + all infrastructure)
  -> Phase 2 (amax)
       -> Phase 3 (dot + facc asymmetry)
            -> Phase 4 (mean + divisor)
                 -> Phase 5 (reduction/elementwise fusion)
                      -> Phase 6 (polish)
```

### Rationale for ordering

- **Phase 1 carries the heaviest machinery.** Terminal validator, reduction tiling with iter_args, scalar FP register pool, rank-0 bufferize, last-use pre-walk. Everything else is incremental after this. A working `sum(N=128)` end-to-end is the most valuable early milestone.
- **Phase 2 (amax) is the cheapest add.** It validates that the infrastructure generalizes to a different combiner with no new machinery beyond an op and a pattern case.
- **Phase 3 (dot) introduces the one real asymmetry** (facc-as-real-accumulator with cosmetic SSA thread). Isolating it in its own phase means the infrastructure from P1/P2 is already stable when the unusual case lands. Also where the `linalg.dot` vs body-match decision is made.
- **Phase 4 (mean) is a tiny delta** on top of Phase 1's sum machinery — one attribute, one asm post-step.
- **Phase 5 (fusion) comes after all four reductions** so fusion tests can exercise every reduction combined with every elementwise producer. Comes before polish because fusion changes what's visible in the IR, and polish tests depend on the fused shape.
- **Phase 6 is low-risk cleanup** — examples, docs, extra guards, any remaining test coverage.

## Files touched (summary)

### New files
- `src/arrax/dsl/validate.py` *(or validator function added to `dsl_to_array.py` — decide at impl time)*
- `tests/dsl/test_reductions_terminal.py`

### Modified — DSL
- `src/arrax/__init__.py` — export `sum`, `dot`, `amax`, `mean`
- `src/arrax/dsl/array.py` — `sum`, `dot`, `amax`, `mean` free functions; rank-0 shape audit
- `src/arrax/dsl/tracer.py` — tracer already shape-polymorphic; verify rank-0 path
- `src/arrax/lowering/dsl_to_array.py` — wire up four new reduction ops; call validator

### Modified — Dialects
- `src/arrax/dialects/array_dialect.py` — SumOp, DotOp, AmaxOp, MeanOp
- `src/arrax/dialects/npu_dialect.py` — FVReduceOp, FVMacOp, FVMaxOp

### Modified — Lowering
- `src/arrax/lowering/array_to_linalg.py` — four new patterns with `linalg.fill` init
- `src/arrax/lowering/bufferize.py` — rank-0 tensor/memref handling
- `src/arrax/lowering/tile.py` — reduction strip-mine with iter_args
- `src/arrax/lowering/fusion.py` — parallel -> reduction merge case + facc conflict guard
- `src/arrax/lowering/linalg_to_npu.py` — reduction pattern matching for FVReduceOp/FVMacOp/FVMaxOp + divisor plumbing

### Modified — Codegen
- `src/arrax/codegen/asm_emitter.py` — `ScalarFPRegisterPool`, `iter_args` path, three new NPU op emitters, mean trailing fdiv, rank-0 memref.store, last-use pre-walk

### Modified — Pipeline / Tests
- `src/arrax/pipeline.py` — no new passes to register (all extensions are to existing passes); re-verify pass ordering
- `tests/...` — per test layer above
- `examples/dot_product.py`, `examples/elementwise.py` — populate with working reduction examples

## Known limitations (documented with M3)

- `dot(A * c, B)` compiles correctly but does not fuse across the scalar-vector/reduction boundary (facc conflict). Runs as two tile loops instead of one. M4 revisits.
- Reductions are terminal only. Intermediate scalars are not supported in a single compiled function. M4 lifts this.
- Multi-output functions (tuple return) not supported. Each reduction is its own function. No M3 work to enable it.
- Reduction + reduction fusion not possible (would require multi-reduction support).
- 2D arrays and matmul out of scope.

## Deferred to Milestone 4 (composites)

- Composite ops: softmax, rmsnorm, layernorm
- Dynamic scalars feeding vector ops (FVMUL/FVDIV/FVSUB_SCALAR with scalar operand from a reduction)
- Scalar arithmetic on f32 (`arith.addf`, `arith.divf`, `math.sqrt`, `math.rsqrt` on SSA scalars)
- Lifting the terminal-reduction restriction (delete the validator)
- DAG-level peephole: `sum(A * B)` -> `dot(A, B)`, `mean(A * A)` -> `dot(A, A) / N`

## Target hardware reference

NPU reduction instructions in `riscv-npu/src/riscv_npu/npu/fp_instructions.py`:

```
FVREDUCE: f[rd] = sum(mem_f32[rs1 .. rs1 + n*4])
          opcode 0x2B, funct3 0x0, funct7 0x05
          rs1 = src base, rs2 = n, rd = dest FP reg
          Uses double-precision accumulation, rounds to f32 at the end.

FVMAC:    facc += dot(mem_f32[rs1 .. +n], mem_f32[rs2 .. +n])
          opcode 0x2B, funct3 0x0, funct7 0x01
          rs1 = lhs base, rs2 = rhs base, rd = n
          Double-precision accumulation into facc. Does NOT reset facc.

FVMAX:    f[rd] = max(mem_f32[rs1 .. rs1 + n*4])
          opcode 0x2B, funct3 0x0, funct7 0x06
          rs1 = src base, rs2 = n, rd = dest FP reg
          NaN propagating.

FRSTACC:  f[rd] = (f32)facc; facc = 0.0
          opcode 0x2B, funct3 0x5, funct7 0x00
          rd = dest FP reg (use x0 to discard the old value)
          Reads and zeros facc in one instruction.
```

All three reduction instructions are subject to the 64-element (`NPU_MAX_VEC_LEN`) vector length limit, the same as all other FP NPU vector instructions.
