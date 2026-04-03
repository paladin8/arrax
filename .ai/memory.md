# arrax memory

## Status
- Milestone 1 complete: A + B end-to-end (Python DSL -> xDSL IR -> NPU assembly -> ELF -> riscv-npu emulator -> correct output)
- Milestone 1.1 complete: strip-mine tiling for NPU 64-element vector limit (scf.for + memref.subview)
- 109 tests passing across 12 test files
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

## Next
- Milestone 2: fusion, more elementwise ops, buffer reuse
