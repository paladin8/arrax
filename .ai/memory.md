# arrax memory

## Status
- Milestones 1-4 complete
- 492 tests passing
- Dependencies: xdsl 0.59.0, numpy, riscv-npu (editable path dep at ../riscv-npu)
- Python 3.14 via uv

## Pipeline
trace -> dsl_to_array -> ArrayToLinalg -> Bufferize -> Tile -> Fuse -> BufferOptimize -> LinalgToNpu (+ rank-0 forwarding) -> NpuCanonicalize -> verify -> emit_assembly

## Key patterns
- xDSL ops: `arith.SubiOp` not `SubIOp`, `arith.AddfOp` not `Addf`
- Custom bufferize pass (xDSL 0.59.0 has no one-shot bufferize)
- NPU_MAX_VEC_LEN = 64 defined in npu_dialect.py
- Sequential loops reuse s-reg range (save/restore _s_reg_count in _emit_for)
- Scalar FP values use ScalarFPRegisterPool (fs0-fs11, LIFO reuse, alias-aware release)
- _bind_reduction_acc: shared pool-binding helper for all reduction emitters (fvreduce/fvmax/fvmac)
- find_preceding_fill: shared utility in lowering/utils.py (used by tile.py and linalg_to_npu.py)
- Buffer reuse: liveness interval extends +1 for ins[0] of binary linalg.generic (NPU copy-aliasing hazard)
- _emit_fv_mac: FRSTACC bracket (zero before, FVMAC .insn body, read after); facc accumulates directly
- _emit_fv_max: feq.s/bnez/fmv.s NaN forwarder after fmax.s
- facc read-write lock model: arrax.facc StringAttr ("ephemeral" / "persistent") on linalg.generic; fusion blocked when persistent + non-none
- FVMulOp/FVDivOp: scalar is SSA f32 operand (not FloatAttr property); unifies compile-time and runtime scalars
- Rank-0 linalg.generic (0 iterators): used for scalar math between reductions and broadcast ops
- LinalgRank0ToScalarPattern: inlines rank-0 generic bodies (divf/addf → load+arith+store; rsqrt → frsqrt)
- _forward_rank0_stores: eliminates store→load pairs for rank-0 memrefs (walks full module)
- Mean: sum reduction + rank-0 divf generic (no attributes)
- Softmax: amax + broadcast-sub + exp + sum + broadcast-div (5 linalg generics, 3 loops when tiled)
- RMSNorm (no gamma): dot(x,x) + rank-0 divf + rank-0 addf + rank-0 rsqrt + broadcast-mul (2 loops when tiled)
- Broadcast binary pattern: mixed affine maps (d0)->(d0) + (d0)->(); scalar forwarding via _find_dominating_store
- _emit_facc_load_from_pool: loads SSA f32 from register pool into facc (FRSTACC + FMACC)
