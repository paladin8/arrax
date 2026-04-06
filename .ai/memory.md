# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 complete: all 5 phases (sub, relu, exp, fusion, buffer opt, scalar mul/div)
- Milestone 3 complete: all 6 phases + comprehensive review + refactoring
- 444 tests passing
- Dependencies: xdsl 0.59.0, numpy, riscv-npu (editable path dep at ../riscv-npu)
- Python 3.14.2 via uv

## Pipeline
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> Tile -> Fuse -> BufferOptimize -> LinalgToNpu -> NpuCanonicalize -> verify -> emit_assembly

## Key patterns
- xDSL ops: `arith.SubiOp` not `SubIOp`, `arith.AddfOp` not `Addf`
- Custom bufferize pass (xDSL 0.59.0 has no one-shot bufferize)
- NPU_MAX_VEC_LEN = 64 defined in npu_dialect.py
- Sequential loops reuse s-reg range (save/restore _s_reg_count in _emit_for)
- Fusion CSE key includes attributes/properties (prevents merging different constants)
- Scalar FP values use ScalarFPRegisterPool (fs0-fs11, LIFO reuse, alias-aware release)
- _bind_reduction_acc: shared pool-binding helper for all reduction emitters (fvreduce/fvmax/fvmac)
- find_preceding_fill: shared utility in lowering/utils.py (used by tile.py and linalg_to_npu.py)
- Buffer reuse: liveness interval extends +1 for ins[0] of binary linalg.generic (NPU copy-aliasing hazard)
- _emit_fv_mac: FRSTACC bracket (zero before, FVMAC .insn body, read after); facc accumulates directly
- _emit_fv_max: feq.s/bnez/fmv.s NaN forwarder after fmax.s (RISC-V NaN-suppressing vs NPU NaN-propagating)
- Mean: arrax.mean_divisor discardable attr on linalg.generic → FVReduceOp divisor property → fdiv.s
- _verify_rank1_to_rank0_f32 shared helper for SumOp/AmaxOp/MeanOp verify_
- _has_facc_conflict: pattern-matches linalg.generic body to detect scalar-vec + dot conflict
