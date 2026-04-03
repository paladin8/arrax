# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 Phase 1 complete: A - B (subtraction) across full stack
- 137 tests passing across 12 test files
- Dependencies: xdsl 0.59.0, numpy, riscv-npu (editable path dep at ../riscv-npu)
- Python 3.14.2 via uv

## Pipeline
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> Tile -> LinalgToNpu -> NpuCanonicalize -> verify -> emit_assembly

## Key patterns
- xDSL ops: `arith.SubiOp` not `SubIOp`, `arith.AddfOp` not `Addf`, `IteratorTypeAttr.parallel()` factory
- `tensor.EmptyOp` result is `.tensor` not `.result`
- Custom bufferize pass (xDSL 0.59.0 has no one-shot bufferize)
- NPU_MAX_VEC_LEN = 64 defined in npu_dialect.py, imported by tile.py
- Sequential loops reuse s-reg range (save/restore _s_reg_count in _emit_for)
- _emit_fv_binop: shared asm emission for binary NPU ops (funct7 parameterized)
- LinalgElementwiseToNpuPattern: dispatches on body op type (addf->FVAdd, subf->FVSub)

## Next
- Milestone 2 Phase 2: relu + exp (unary elementwise)
- Milestone 2 Phase 3: post-tiling loop fusion
- Milestone 2 Phase 4: buffer shrinking + reuse
- Milestone 2 Phase 5: scalar-vector mul/div
