# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 Phases 1-4 complete: sub, relu, exp, fusion, buffer optimization
- 201 tests passing across 14 test files
- Dependencies: xdsl 0.59.0, numpy, riscv-npu (editable path dep at ../riscv-npu)
- Python 3.14.2 via uv

## Pipeline
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> Tile -> Fuse -> BufferOptimize -> LinalgToNpu -> NpuCanonicalize -> verify -> emit_assembly

## Key patterns
- xDSL ops: `arith.SubiOp` not `SubIOp`, `arith.AddfOp` not `Addf`
- Custom bufferize pass (xDSL 0.59.0 has no one-shot bufferize)
- NPU_MAX_VEC_LEN = 64 defined in npu_dialect.py
- Sequential loops reuse s-reg range (save/restore _s_reg_count in _emit_for)
- _emit_fv_binop / _emit_fv_unop: shared asm emission parameterized by name + funct7
- LinalgElementwiseToNpuPattern: _match_binary (addf/subf) + _match_unary (maximumf/exp)
- Fusion: _same_bounds checks constant lb/ub/step, CSE includes attributes in key
- Buffer optimize: shrink intermediates to 64 elements, reuse non-overlapping (start >= merged_end)
- SSAValue.replace_all_uses_with (not replace_by, deprecated)

## Next
- Milestone 2 Phase 5: scalar-vector mul/div (accumulator management)
