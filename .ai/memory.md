# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 Phase 1 complete: A - B (subtraction)
- Milestone 2 Phase 2 complete: relu(A), exp(A) (unary elementwise)
- 176 tests passing across 12 test files
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
- _emit_fv_unop: shared asm emission for unary NPU ops (no copy loop)
- LinalgElementwiseToNpuPattern: _match_binary (addf/subf) + _match_unary (maximumf/exp)
- Unary linalg.generic: 1 input, 1 output, 2 identity maps (not 3)
- arith.MaximumfOp for relu body, math.ExpOp for exp body
- `FloatAttr(0.0, f32)` for the zero constant in relu body

## Next
- Milestone 2 Phase 3: post-tiling loop fusion
- Milestone 2 Phase 4: buffer shrinking + reuse
- Milestone 2 Phase 5: scalar-vector mul/div
