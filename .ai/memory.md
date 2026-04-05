# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 complete: all 5 phases (sub, relu, exp, fusion, buffer opt, scalar mul/div)
- Milestone 3 Phase 1 complete: sum(A) end-to-end + all supporting infra
- 293 tests passing
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
- Pool.release: unconditionally pops val's entry; frees register only when no alias survives
- _emit_yield looks up target reg via parent.results[i], not body_arg (body_arg may be released)
- _emit_fv_reduce: materialize acc_in BEFORE loading n into t0 (t0 is shared scratch)
- compute_last_use: forward walk across nested regions; drives _release_dead_fp_regs per op
- Reduction tiling: scf.for with f32 iter_args threading accumulator through SSA
- LinalgReductionToNpu matches both untiled (fill+generic) and tiled (alloca+fill+generic+load) shapes
- Terminal validator: reductions must be the func return in M3 (lifted in M4)
- Bufferize promotes the rank-0 `tensor.empty` reachable from return; non-terminal ones become memref.alloc
- SSAValue.replace_all_uses_with (not replace_by, deprecated)

## Next
- Milestone 3 Phase 2: amax reduction (FVMAX funct7=0x06, body = maximumf, identity = -inf)
- Milestone 3 Phase 3: dot product (two-input reduction, fmul+fadd body or FVMAC)
- Milestone 3 Phase 4: mean (sum + post-loop divide by N)
- Milestone 3 Phase 5: parallel→reduction fusion (lifting the safety guard in fusion.py)
