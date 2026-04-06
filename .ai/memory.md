# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 complete: all 5 phases (sub, relu, exp, fusion, buffer opt, scalar mul/div)
- Milestone 3 Phases 1-3 complete: sum, amax, dot end-to-end
- 395 tests passing
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
- Fusion refuses to merge loops with iter_args (safety guard; parallel→reduction fusion is Phase 5)
- Scalar FP values use ScalarFPRegisterPool (fs0-fs11, LIFO reuse, alias-aware release)
- _emit_fv_reduce / _emit_fv_max: materialize acc_in BEFORE loading n into t0 (shared scratch)
- _emit_fv_max: feq.s/bnez/fmv.s NaN forwarder after fmax.s (RISC-V NaN-suppressing vs NPU NaN-propagating)
- _emit_fv_mac: FRSTACC bracket (zero before, FVMAC .insn body, read after); facc accumulates directly
  - Untiled: inline bracket in _emit_fv_mac
  - Tiled: _emit_for detects FVMacOp, emits FRSTACC zero/read around loop; iter_args thread cosmetic
- Tile pass _tile_reduction: generalized for N inputs (creates subviews for each)
- LinalgToNpu _match_reduction_body: dispatches on body shape via builder lambda
- Terminal validator: reductions must be the func return in M3 (lifted in M4)
- Bufferize promotes rank-0 tensor.empty reachable from return; non-terminal ones become memref.alloc

## Next
- Milestone 3 Phase 4: mean (sum + post-loop divide by N)
- Milestone 3 Phase 5: parallel→reduction fusion (lifting the safety guard in fusion.py)

## Follow-ups (carry-forwards from Phase 2/3 reviews)
- I1: _emit_fv_reduce/_emit_fv_max/_emit_fv_mac share pool-binding boilerplate; extract _bind_reduction_acc helper
- I2: _find_preceding_fill (linalg_to_npu) and _find_init_fill (tile) are near-duplicates; unify
- N2: SumOp/AmaxOp verify_ duplicated; extract _verify_rank1_to_rank0_f32 when MeanOp adds 3rd copy
