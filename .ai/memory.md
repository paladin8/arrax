# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 complete: all 5 phases (sub, relu, exp, fusion, buffer opt, scalar mul/div)
- 221 tests passing across 14 test files
- Dependencies: xdsl 0.59.0, numpy, riscv-npu (editable path dep at ../riscv-npu)
- Python 3.14.2 via uv

## Pipeline
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> Tile -> Fuse -> BufferOptimize -> LinalgToNpu -> NpuCanonicalize -> verify -> emit_assembly

## Key patterns
- xDSL ops: `arith.SubiOp` not `SubIOp`, `arith.AddfOp` not `Addf`
- Custom bufferize pass (xDSL 0.59.0 has no one-shot bufferize)
- NPU_MAX_VEC_LEN = 64 defined in npu_dialect.py
- Sequential loops reuse s-reg range (save/restore _s_reg_count in _emit_for)
- _emit_fv_binop / _emit_fv_unop / _emit_fv_scalar_op: shared asm emission
- LinalgElementwiseToNpuPattern: _match_binary + _match_unary (dispatches all ops)
- Fusion CSE key includes attributes/properties (prevents merging different constants)
- Buffer optimize: shrink to 64 elements, greedy interval coloring for reuse
- Scalar-vector: facc load via FRSTACC + FMACC, then FVMUL/FVDIV
- prop_def + ParsePropInAttrDict for float attributes on ops (tuple not list for irdl_options)
- SSAValue.replace_all_uses_with (not replace_by, deprecated)

## Next
- Milestone 3+ (not yet planned)
