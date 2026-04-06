# arrax memory

## Status
- Milestone 1 + 1.1 complete: A + B end-to-end with strip-mine tiling
- Milestone 2 complete: all 5 phases (sub, relu, exp, fusion, buffer opt, scalar mul/div)
- Milestone 3 Phase 1 complete: sum(A) end-to-end + all supporting infra
- Milestone 3 Phase 2 complete: amax(A) end-to-end (FVMAX funct7=0x06, maximumf body, -inf identity)
- 343 tests passing
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
- _emit_fv_reduce / _emit_fv_max: materialize acc_in BEFORE loading n into t0 (t0 is shared scratch)
- _emit_fv_max: follows fmax.s combine with feq.s/bnez/fmv.s NaN forwarder —
  RISC-V fmax.s is NaN-suppressing but NPU.FVMAX propagates NaN, so the combine
  must force acc:=ft0 when ft0 is NaN to match np.amax semantics end-to-end
- compute_last_use: forward walk across nested regions; drives _release_dead_fp_regs per op
- Reduction tiling: scf.for with f32 iter_args threading accumulator through SSA
- LinalgReductionToNpu matches both untiled (fill+generic) and tiled (alloca+fill+generic+load) shapes
- Terminal validator: reductions must be the func return in M3 (lifted in M4)
- Bufferize promotes the rank-0 `tensor.empty` reachable from return; non-terminal ones become memref.alloc
- SSAValue.replace_all_uses_with (not replace_by, deprecated)

## Next
- Milestone 3 Phase 3: dot product (two-input reduction, fmul+fadd body or FVMAC)
- Milestone 3 Phase 4: mean (sum + post-loop divide by N)
- Milestone 3 Phase 5: parallel→reduction fusion (lifting the safety guard in fusion.py)

## Follow-ups (from Phase 2 code review)
- I1: _emit_fv_max and _emit_fv_reduce are ~30 lines of copy-paste that differ only in
  mnemonic + funct7 + combine insn. Collapse into a shared helper before Phase 3 copies
  the template a third time for FVMAC.
- N6: _find_preceding_fill (linalg_to_npu.py) and _find_init_fill (tile.py) are
  near-duplicates predating M3. Unify during the Phase 3 refactor above.
- N2: AmaxOp.verify_ duplicates SumOp.verify_; extract a _verify_rank1_to_rank0_f32
  helper once Phase 4 adds the 3rd copy (MeanOp).
