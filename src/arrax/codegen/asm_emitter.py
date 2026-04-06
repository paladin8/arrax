"""Assembly text generation for RISC-V + NPU .insn directives.

Walks NPU-dialect IR and emits a complete .S file with:
- Function prologue (.globl, .type)
- Copy loops for npu.fvadd in-place semantics (src2 -> dst)
- .insn r directives for NPU instructions
- .comm directives for intermediate buffer allocations
- scf.for loops (with or without iter_args) for tiled operations
- memref.subview pointer arithmetic
- npu.fvreduce handler for rank-0 reductions

Register allocation strategy:
- a0-a7: function arguments (RISC-V calling convention)
- s0-s11: callee-saved, used for values persisting across loops:
  memref.alloc addresses, loop IVs, memref.subview pointers
- t4-t5: loop-body-local values (arith.subi, arith.minsi) — reset each iteration
- t0-t3: scratch for copy loops, constant loading, .insn operands
- fs0-fs11: scalar FP SSA values (reduction accumulators, iter_args) managed
  by ScalarFPRegisterPool
- ft0, ft1: ephemeral FP scratch for .insn destinations, constant materialization
"""

from __future__ import annotations

import struct

from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Operation, SSAValue

from arrax.dialects.npu_dialect import (
    FVAddOp,
    FVDivOp,
    FVExpOp,
    FVMacOp,
    FVMaxOp,
    FVMulOp,
    FVReduceOp,
    FVReluOp,
    FVSubOp,
)

# t-registers for loop-body-local values that die before the copy loop.
# t0-t3 are reserved as scratch for copy loops and constant loading.
_LOOP_BODY_T_REGS = ["t4", "t5"]

# Scalar FP register pool: RISC-V callee-saved FP registers fs0-fs11.
_SCALAR_FP_POOL = [f"fs{i}" for i in range(12)]


class ScalarFPRegisterPool:
    """Manages fs0-fs11 allocation for f32 SSA values that persist across ops.

    Used for reduction accumulators, iter_args, and any f32 scalar that must
    stay live across multiple emitted ops (including entire loop bodies).
    Ephemeral scratch FP registers used inside a single op emission routine
    (e.g., ``ft0`` for FVREDUCE's immediate destination) are NOT managed here.
    """

    def __init__(self) -> None:
        self._free: list[str] = list(_SCALAR_FP_POOL)
        self._map: dict[int, str] = {}

    def allocate(self, val: SSAValue) -> str:
        """Allocate the next free fs-register for ``val`` and return its name."""
        if not self._free:
            raise ValueError(
                "asm_emitter: scalar FP register pool exhausted "
                f"(all of {_SCALAR_FP_POOL} in use)"
            )
        reg = self._free.pop(0)
        self._map[id(val)] = reg
        return reg

    def bind(self, val: SSAValue, reg: str) -> None:
        """Alias an already-allocated ``reg`` to another SSA value.

        Used for iter_args threading where a loop yield's SSA value must
        share the same register as its corresponding iter_arg.
        """
        assert reg in _SCALAR_FP_POOL, (
            f"ScalarFPRegisterPool.bind: {reg!r} is not a pooled register"
        )
        self._map[id(val)] = reg

    def get(self, val: SSAValue) -> str:
        return self._map[id(val)]

    def contains(self, val: SSAValue) -> bool:
        return id(val) in self._map

    def release(self, val: SSAValue) -> None:
        """Drop ``val``'s binding from the map; return the register to the
        free list only once every aliased SSA value has been released.

        A register may be aliased (via :meth:`bind`) to multiple SSA values —
        for instance, an ``scf.for`` iter_arg body-arg and its matching
        loop-result share a register. This method unconditionally removes
        ``val``'s entry; the register returns to the free list only if no
        other SSA value in the map still references it. Callers that need
        to look up the target register for a yielded value must query the
        surviving alias (e.g. ``parent.results[i]``), not the released one.
        """
        reg = self._map.pop(id(val), None)
        if reg is None:
            return
        # If any other SSA value still maps to this register, it remains held.
        if reg in self._map.values():
            return
        if reg not in self._free:
            # LIFO: the just-released register is reused next.
            self._free.insert(0, reg)


def compute_last_use(root_block: Block) -> dict[int, int]:
    """Forward walk that returns ``{id(SSAValue): last_op_index}`` per value.

    Op indices number every op in IR order including ops inside nested
    regions. A value's last-use index is the largest op index at which it
    appears as an operand. Definitions that are never used do not appear
    in the map.
    """
    last_use: dict[int, int] = {}
    idx = [0]

    def visit(block: Block) -> None:
        for op in block.ops:
            cur = idx[0]
            idx[0] += 1
            for operand in op.operands:
                last_use[id(operand)] = cur
            for region in op.regions:
                for sub in region.blocks:
                    visit(sub)

    visit(root_block)
    return last_use


def emit_assembly(module: ModuleOp) -> str:
    """Walk a module containing npu ops and emit RISC-V assembly text."""
    emitter = _AsmEmitter()
    emitter.emit_module(module)
    return emitter.get_output()


class _AsmEmitter:
    """Stateful assembly emitter that walks IR and builds output text."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._bss: list[str] = []
        self._reg_map: dict[int, str] = {}  # id(SSAValue) -> register name
        self._const_map: dict[int, int] = {}  # id(SSAValue) -> integer value
        self._fconst_map: dict[int, int] = {}  # id(SSAValue) -> f32 IEEE bits
        self._s_reg_count: int = 0
        self._label_count: int = 0
        self._loop_t_reg_idx: int = 0  # index into _LOOP_BODY_T_REGS
        self._fp_pool: ScalarFPRegisterPool = ScalarFPRegisterPool()
        self._last_use: dict[int, int] = {}
        self._op_index: int = 0

    def get_output(self) -> str:
        result = "\n".join(self._lines) + "\n"
        if self._bss:
            result += "\n    .section .bss\n" + "\n".join(self._bss) + "\n"
        return result

    def emit_module(self, module: ModuleOp) -> None:
        for op in module.body.block.ops:
            if isinstance(op, func.FuncOp):
                self._emit_func(op)

    def _reg(self, val: SSAValue) -> str:
        """Look up the register assigned to an SSA value."""
        return self._reg_map[id(val)]

    def _const(self, val: SSAValue) -> int:
        """Look up the integer constant for an SSA value."""
        return self._const_map[id(val)]

    def _alloc_s_reg(self) -> str:
        """Allocate the next callee-saved s-register."""
        if self._s_reg_count >= 12:
            raise ValueError(
                f"asm_emitter: s-register overflow (need s{self._s_reg_count}, "
                f"RISC-V only has s0-s11)"
            )
        reg = f"s{self._s_reg_count}"
        self._s_reg_count += 1
        return reg

    def _alloc_loop_t_reg(self) -> str:
        """Allocate a t-register for loop-body-local values (t4-t5)."""
        if self._loop_t_reg_idx >= len(_LOOP_BODY_T_REGS):
            raise ValueError(
                f"asm_emitter: loop t-register overflow "
                f"(need more than {len(_LOOP_BODY_T_REGS)} loop-body registers)"
            )
        reg = _LOOP_BODY_T_REGS[self._loop_t_reg_idx]
        self._loop_t_reg_idx += 1
        return reg

    def _load_operand(self, val: SSAValue, scratch: str = "t0") -> str:
        """Get a register for an SSA value, loading constants into scratch."""
        if id(val) in self._reg_map:
            return self._reg_map[id(val)]
        if id(val) in self._const_map:
            self._lines.append(f"    li {scratch}, {self._const_map[id(val)]}")
            return scratch
        raise ValueError("asm_emitter: unknown SSA value")

    def _count_s_regs_in_block(self, block: Block) -> int:
        """Count max concurrent s-registers needed in a block.

        Persistent values (allocs) accumulate. Sequential loops take the max
        (they reuse the same s-reg slots). Each loop body uses s-regs for the
        IV and subview pointers.
        """
        persistent = 0
        max_loop_regs = 0
        for op in block.ops:
            if isinstance(op, memref.AllocOp):
                persistent += 1
            elif isinstance(op, scf.ForOp):
                loop_regs = self._count_for_s_regs(op)
                max_loop_regs = max(max_loop_regs, loop_regs)
        return persistent + max_loop_regs

    def _count_for_s_regs(self, for_op: scf.ForOp) -> int:
        """Count s-regs needed by a single for loop (IV + subviews)."""
        count = 1  # induction variable
        body = for_op.body.blocks.first
        assert body is not None
        for op in body.ops:
            if isinstance(op, memref.SubviewOp):
                count += 1
            elif isinstance(op, scf.ForOp):
                count += self._count_for_s_regs(op)
        return count

    def _emit_func(self, func_op: func.FuncOp) -> None:
        name = func_op.sym_name.data
        self._lines.append("    .text")
        self._lines.append(f"    .globl {name}")
        self._lines.append(f"    .type {name}, @function")
        self._lines.append(f"{name}:")

        block = func_op.body.blocks.first
        assert block is not None

        # Function args -> a0, a1, ...
        for i, arg in enumerate(block.args):
            self._reg_map[id(arg)] = f"a{i}"

        # Fresh per-function state for the scalar FP register pool and
        # last-use table. Op indices number every op in visit order.
        self._fp_pool = ScalarFPRegisterPool()
        self._last_use = compute_last_use(block)
        self._op_index = 0

        # Count s-registers needed (max concurrent)
        num_s_regs = self._count_s_regs_in_block(block)

        # Prologue: save callee-saved s-registers
        if num_s_regs > 0:
            frame_size = num_s_regs * 4
            self._lines.append(f"    addi sp, sp, -{frame_size}")
            for i in range(num_s_regs):
                self._lines.append(f"    sw s{i}, {i * 4}(sp)")

        # Walk body ops
        for op in block.ops:
            if isinstance(op, func.ReturnOp):
                if num_s_regs > 0:
                    for i in range(num_s_regs):
                        self._lines.append(f"    lw s{i}, {i * 4}(sp)")
                    self._lines.append(f"    addi sp, sp, {num_s_regs * 4}")
                self._lines.append("    ret")
            else:
                self._emit_op(op)

    def _emit_op(self, op: object) -> None:
        """Dispatch to the appropriate emission method for an op."""
        # Record visit order for last-use bookkeeping.
        cur_idx = self._op_index
        self._op_index += 1

        if isinstance(op, arith.ConstantOp):
            self._emit_constant(op)
        elif isinstance(op, memref.AllocOp):
            self._emit_alloc(op)
        elif isinstance(op, FVAddOp):
            self._emit_fv_binop(op, "FVADD", 0x07)
        elif isinstance(op, FVSubOp):
            self._emit_fv_binop(op, "FVSUB", 0x08)
        elif isinstance(op, FVReluOp):
            self._emit_fv_unop(op, "FVRELU", 0x09)
        elif isinstance(op, FVExpOp):
            self._emit_fv_unop(op, "FVEXP", 0x02)
        elif isinstance(op, FVMulOp):
            self._emit_fv_scalar_op(op, "FVMUL", 0x04)
        elif isinstance(op, FVDivOp):
            self._emit_fv_scalar_op(op, "FVDIV", 0x0B)
        elif isinstance(op, FVReduceOp):
            self._emit_fv_reduce(op)
        elif isinstance(op, FVMaxOp):
            self._emit_fv_max(op)
        elif isinstance(op, FVMacOp):
            self._emit_fv_mac(op)
        elif isinstance(op, scf.ForOp):
            self._emit_for(op)
        elif isinstance(op, memref.SubviewOp):
            self._emit_subview(op)
        elif isinstance(op, memref.StoreOp):
            self._emit_store(op)
        elif isinstance(op, arith.SubiOp):
            self._emit_subi(op)
        elif isinstance(op, arith.MinSIOp):
            self._emit_minsi(op)
        elif isinstance(op, scf.YieldOp):
            self._emit_yield(op)
        elif isinstance(op, Operation):
            raise ValueError(f"asm_emitter: unsupported op {op.name}")
        else:
            raise ValueError(f"asm_emitter: unsupported op {type(op)}")

        # Release any scalar FP register whose last use is this op.
        self._release_dead_fp_regs(op, cur_idx)

    def _release_dead_fp_regs(self, op: object, cur_idx: int) -> None:
        """Free any pooled fs-register whose final use was this op."""
        if not isinstance(op, Operation):
            return
        for operand in op.operands:
            if not self._fp_pool.contains(operand):
                continue
            if self._last_use.get(id(operand)) == cur_idx:
                self._fp_pool.release(operand)

    def _emit_constant(self, op: arith.ConstantOp) -> None:
        """Record constant value for later use (emitted inline when needed).

        Integer constants go into ``_const_map``. Float constants get their
        IEEE bits stored in ``_fconst_map`` — materialization happens on
        demand at the first use site (e.g. as an ``iter_args`` init).
        """
        if isinstance(op.value, IntegerAttr):
            self._const_map[id(op.result)] = op.value.value.data
        elif isinstance(op.value, FloatAttr) and isinstance(
            op.value.type, Float32Type
        ):
            fval = op.value.value.data
            bits = struct.unpack("<I", struct.pack("<f", fval))[0]
            self._fconst_map[id(op.result)] = bits
        else:
            raise ValueError(
                f"asm_emitter: unsupported constant type {op.value}"
            )

    def _emit_alloc(self, op: memref.AllocOp) -> None:
        """Emit .comm for static buffer allocation, load address into s-register."""
        assert isinstance(op.memref.type, MemRefType)
        shape = op.memref.type.get_shape()
        size = 1
        for dim in shape:
            size *= dim
        size *= 4  # f32 = 4 bytes per element

        sym = f".Lbuf_{self._label_count}"
        self._label_count += 1
        sreg = self._alloc_s_reg()
        self._bss.append(f"    .comm {sym}, {size}, 4")
        self._lines.append(f"    la {sreg}, {sym}")
        self._reg_map[id(op.memref)] = sreg

    def _emit_for(self, op: scf.ForOp) -> None:
        """Emit a counted loop for scf.for, with optional f32 iter_args.

        For each iter_arg:
        - Allocate one fs-register from the scalar FP pool
        - Materialize the init operand into it before the loop label
        - Bind the body block arg to the same fs-register
        - Bind the corresponding scf.for result to the same fs-register too
          (post-loop uses read from it directly)

        The yield operand's value must already live in the iter_arg's fs
        register when the yield executes; reduction op emitters enforce
        this by writing into the bound fs-register directly.
        """
        body = op.body.blocks.first
        assert body is not None

        # Save s-reg counter — sequential loops reuse the same s-reg range
        loop_s_reg_start = self._s_reg_count

        # Assign IV to an s-register (persists across iterations)
        iv_reg = self._alloc_s_reg()
        self._reg_map[id(body.args[0])] = iv_reg

        # --- iter_args handling ---
        iter_arg_regs: list[str] = []
        # body.args[0] is the IV; iter_args begin at index 1.
        for i, init_val in enumerate(op.iter_args):
            body_arg = body.args[i + 1]
            fp_reg = self._fp_pool.allocate(body_arg)
            iter_arg_regs.append(fp_reg)
            # The loop result (read after the loop) reads from the same reg.
            self._fp_pool.bind(op.results[i], fp_reg)
            # Materialize init value into the register before the loop.
            self._materialize_f32_into(init_val, fp_reg)

        # Detect FVMacOp in the body — dot uses hardware facc, which
        # accumulates across iterations.  Bracket the loop with FRSTACC.
        has_fvmac = any(isinstance(bop, FVMacOp) for bop in body.ops)

        # Initialize IV from lower bound
        if id(op.lb) in self._const_map and self._const_map[id(op.lb)] == 0:
            self._lines.append(f"    li {iv_reg}, 0")
        else:
            lb_reg = self._load_operand(op.lb, "t0")
            self._lines.append(f"    mv {iv_reg}, {lb_reg}")

        if has_fvmac:
            self._lines.append(f"    # FRSTACC: zero facc before dot loop")
            self._lines.append(
                f"    .insn r 0x2B, 0x5, 0x00, f0, f0, f0"
            )

        loop_label = f".Lfor_{self._label_count}"
        end_label = f".Lfor_end_{self._label_count}"
        self._label_count += 1

        self._lines.append(f"{loop_label}:")
        # Compare IV to upper bound
        ub_reg = self._load_operand(op.ub, "t0")
        self._lines.append(f"    bge {iv_reg}, {ub_reg}, {end_label}")

        # Reset loop-body t-register allocation for each iteration
        saved_t_idx = self._loop_t_reg_idx
        self._loop_t_reg_idx = 0

        # Emit body ops
        for body_op in body.ops:
            self._emit_op(body_op)

        self._loop_t_reg_idx = saved_t_idx

        # Increment IV by step
        if id(op.step) in self._const_map:
            step_val = self._const_map[id(op.step)]
            if -2048 <= step_val <= 2047:
                self._lines.append(f"    addi {iv_reg}, {iv_reg}, {step_val}")
            else:
                self._lines.append(f"    li t0, {step_val}")
                self._lines.append(f"    add {iv_reg}, {iv_reg}, t0")
        else:
            step_reg = self._load_operand(op.step, "t0")
            self._lines.append(f"    add {iv_reg}, {iv_reg}, {step_reg}")

        self._lines.append(f"    j {loop_label}")
        self._lines.append(f"{end_label}:")

        if has_fvmac:
            # Read facc into the iter_arg's pool register. This overwrites
            # the cosmetic 0.0 that _materialize_f32_into loaded earlier.
            acc_reg = self._fp_pool.get(op.results[0])
            self._lines.append(f"    # FRSTACC: read facc into {acc_reg}")
            self._lines.append(
                f"    .insn r 0x2B, 0x5, 0x00, {acc_reg}, f0, f0"
            )

        # Restore s-reg counter for sequential loop reuse
        self._s_reg_count = loop_s_reg_start

    def _materialize_f32_into(self, val: SSAValue, fp_reg: str) -> None:
        """Load the f32 value ``val`` into ``fp_reg`` (callee-saved or scratch).

        Supports f32 arith.constant (tracked in ``_fconst_map``) and already-
        pooled scalars (emits ``fmv.s`` aliasing).
        """
        if id(val) in self._fconst_map:
            bits = self._fconst_map[id(val)]
            self._lines.append(f"    li t0, {bits}")
            self._lines.append(f"    fmv.w.x {fp_reg}, t0")
            return
        if self._fp_pool.contains(val):
            src = self._fp_pool.get(val)
            if src != fp_reg:
                self._lines.append(f"    fmv.s {fp_reg}, {src}")
            return
        raise ValueError(
            f"asm_emitter: cannot materialize f32 SSA value into {fp_reg}"
        )

    def _emit_yield(self, op: scf.YieldOp) -> None:
        """For loops with iter_args, ensure yielded f32 values land in the
        iter_arg's pooled register. Reduction ops write directly into it,
        so this is typically a no-op; guard against the alias mismatch case
        where the yielded value is a different SSA (e.g., a bare constant).

        Resolution of the target register goes through ``parent.results[i]``
        rather than the body arg. Both are bound to the same pool register
        during ``_emit_for``, but the body arg may already have been released
        by ``_release_dead_fp_regs`` once its final in-body operand use has
        passed — so looking it up there would miss.
        """
        parent = op.parent_op()
        if not isinstance(parent, scf.ForOp):
            return
        for i, yield_val in enumerate(op.operands):
            if not isinstance(yield_val.type, Float32Type):
                continue
            target_reg = self._fp_pool.get(parent.results[i])
            if self._fp_pool.contains(yield_val):
                src_reg = self._fp_pool.get(yield_val)
                if src_reg != target_reg:
                    self._lines.append(f"    fmv.s {target_reg}, {src_reg}")
            else:
                # Yielded value must be materializable (e.g. an f32 constant).
                self._materialize_f32_into(yield_val, target_reg)

    def _emit_fv_reduce(self, op: FVReduceOp) -> None:
        """Emit NPU.FVREDUCE + fadd.s into the pooled accumulator register.

        The ``acc_in`` SSA value already lives in an fs-register (either an
        iter_args reg inside a loop or a freshly-materialized constant
        outside). The ``result`` SSA value is bound to the SAME register,
        so subsequent ops read the updated accumulator without extra moves.
        """
        src = self._reg(op.src)

        # Acc_in comes either from an iter_arg (already in the pool) or a
        # bare f32 constant (e.g. untiled N<=64 case); for the latter we
        # must allocate a pool register and materialize it now. Do this
        # BEFORE loading n into t0: materialization uses t0 as a scratch
        # for the IEEE bits and would otherwise clobber the loaded n.
        if self._fp_pool.contains(op.acc_in):
            acc_reg = self._fp_pool.get(op.acc_in)
        else:
            acc_reg = self._fp_pool.allocate(op.acc_in)
            self._materialize_f32_into(op.acc_in, acc_reg)
        # The result SSA value aliases the accumulator register in-place.
        self._fp_pool.bind(op.result, acc_reg)

        # Resolve n: either a constant (static shape) or a known register.
        n_is_const = id(op.n) in self._const_map
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            n_reg = "t0"
        else:
            n_reg = self._reg(op.n)

        self._lines.append(f"    # NPU.FVREDUCE ft0 = sum({src}[0..{n_reg}])")
        # FVREDUCE funct7 = 0x05; rd = ft0 (ephemeral), rs1 = src, rs2 = n.
        self._lines.append(
            f"    .insn r 0x2B, 0x0, 0x05, ft0, {src}, {n_reg}"
        )
        # Combine the per-call partial into the running accumulator.
        self._lines.append(f"    fadd.s {acc_reg}, {acc_reg}, ft0")

    def _emit_fv_max(self, op: FVMaxOp) -> None:
        """Emit NPU.FVMAX + fmax.s into the pooled accumulator register.

        Mirrors :meth:`_emit_fv_reduce` exactly for the max reduction: the
        ``acc_in`` SSA value already lives in an fs-register (iter_arg or
        freshly-materialized -inf constant), ``result`` is bound to that
        same register, and the per-chunk partial in ft0 is combined via
        NaN-propagating ``fmax.s`` — matching NPU FVMAX semantics.
        """
        src = self._reg(op.src)

        # Materialize acc_in into a pool register if needed. Do this BEFORE
        # loading n into t0, since materialization uses t0 as scratch for
        # the IEEE bits and would otherwise clobber the element count.
        if self._fp_pool.contains(op.acc_in):
            acc_reg = self._fp_pool.get(op.acc_in)
        else:
            acc_reg = self._fp_pool.allocate(op.acc_in)
            self._materialize_f32_into(op.acc_in, acc_reg)
        self._fp_pool.bind(op.result, acc_reg)

        n_is_const = id(op.n) in self._const_map
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            n_reg = "t0"
        else:
            n_reg = self._reg(op.n)

        self._lines.append(f"    # NPU.FVMAX ft0 = max({src}[0..{n_reg}])")
        # FVMAX funct7 = 0x06; rd = ft0 (ephemeral), rs1 = src, rs2 = n.
        self._lines.append(
            f"    .insn r 0x2B, 0x0, 0x06, ft0, {src}, {n_reg}"
        )
        # Combine the per-call partial into the running accumulator.
        # RISC-V fmax.s is NaN-suppressing (returns the non-NaN operand when
        # exactly one is NaN), but np.amax / arith.maximumf are NaN-propagating.
        # FVMAX already returns NaN in ft0 whenever any input element is NaN,
        # so we force ``acc_reg := ft0`` when ft0 is NaN, otherwise take the
        # normal fmax.s combine.
        nan_skip_label = f".Lfvmax_nonan_{self._label_count}"
        self._label_count += 1
        self._lines.append(f"    fmax.s {acc_reg}, {acc_reg}, ft0")
        self._lines.append(f"    feq.s t0, ft0, ft0")
        self._lines.append(f"    bnez t0, {nan_skip_label}")
        self._lines.append(f"    fmv.s {acc_reg}, ft0")
        self._lines.append(f"{nan_skip_label}:")

    def _emit_fv_mac(self, op: FVMacOp) -> None:
        """Emit NPU.FVMAC with FRSTACC bracketing.

        Unlike sum/amax which combine per-chunk results via scalar ``fadd.s``
        / ``fmax.s``, dot accumulates directly in the hardware ``facc``
        register across iterations. The ``acc_in`` / ``result`` SSA thread
        is cosmetic — the real state lives in ``facc``.

        Untiled (straight-line): FRSTACC zero, FVMAC .insn, FRSTACC read.
        Tiled (inside scf.for): only FVMAC .insn; the FRSTACC bracket is
        emitted by ``_emit_for`` around the loop.
        """
        lhs = self._reg(op.lhs)
        rhs = self._reg(op.rhs)

        # Pool register handling: bind result to the same register as acc_in.
        if self._fp_pool.contains(op.acc_in):
            acc_reg = self._fp_pool.get(op.acc_in)
        else:
            acc_reg = self._fp_pool.allocate(op.acc_in)
        self._fp_pool.bind(op.result, acc_reg)

        # Check if inside a tiled loop (facc persists across iterations).
        in_loop = isinstance(op.parent_op(), scf.ForOp)

        # Resolve n: constant → "t0" (loaded inline), register → its name.
        n_is_const = id(op.n) in self._const_map

        if not in_loop:
            # Untiled: full FRSTACC bracket inline.
            self._lines.append(f"    # FRSTACC: zero facc")
            self._lines.append(
                f"    .insn r 0x2B, 0x5, 0x00, f0, f0, f0"
            )

        self._lines.append(
            f"    # NPU.FVMAC facc += dot({lhs}, {rhs})"
        )
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            self._lines.append(
                f"    .insn r 0x2B, 0x0, 0x01, t0, {lhs}, {rhs}"
            )
        else:
            n_reg = self._reg(op.n)
            self._lines.append(
                f"    .insn r 0x2B, 0x0, 0x01, {n_reg}, {lhs}, {rhs}"
            )

        if not in_loop:
            # Untiled: read facc into the pool register.
            self._lines.append(f"    # FRSTACC: read facc into {acc_reg}")
            self._lines.append(
                f"    .insn r 0x2B, 0x5, 0x00, {acc_reg}, f0, f0"
            )

    def _emit_store(self, op: memref.StoreOp) -> None:
        """Emit a rank-0 f32 store: ``fsw <val>, 0(<base>)``.

        Only rank-0 stores are supported in M3 (the terminal write of a
        reduction result). Other memref.store shapes are not produced by
        the pipeline.
        """
        if op.indices:
            raise ValueError(
                "asm_emitter: only rank-0 memref.store is supported in M3"
            )
        val = op.value
        ref = op.memref
        if not isinstance(val.type, Float32Type):
            raise ValueError(
                f"asm_emitter: memref.store expected f32 value, got {val.type}"
            )
        val_reg = self._fp_pool.get(val)
        base_reg = self._reg(ref)
        self._lines.append(f"    fsw {val_reg}, 0({base_reg})")

    def _emit_subview(self, op: memref.SubviewOp) -> None:
        """Emit pointer arithmetic for a 1D subview: base + offset * 4.

        Subview results use s-registers because they must survive through
        the copy loop in _emit_fvadd (which clobbers t0-t3).
        """
        base_reg = self._reg(op.source)
        offset_reg = self._load_operand(op.offsets[0], "t0")
        result_reg = self._alloc_s_reg()
        self._lines.append(f"    slli t1, {offset_reg}, 2")
        self._lines.append(f"    add {result_reg}, {base_reg}, t1")
        self._reg_map[id(op.result)] = result_reg

    def _emit_subi(self, op: arith.SubiOp) -> None:
        """Emit integer subtraction. Result in a loop-body t-register."""
        lhs_reg = self._load_operand(op.lhs, "t0")
        rhs_reg = self._load_operand(op.rhs, "t1")
        result_reg = self._alloc_loop_t_reg()
        self._lines.append(f"    sub {result_reg}, {lhs_reg}, {rhs_reg}")
        self._reg_map[id(op.result)] = result_reg

    def _emit_minsi(self, op: arith.MinSIOp) -> None:
        """Emit signed integer minimum via conditional branch.

        Result in a loop-body t-register. The result survives through the
        copy loop because t4-t5 are not clobbered by _emit_fvadd.
        """
        lhs_reg = self._load_operand(op.lhs, "t0")
        rhs_reg = self._load_operand(op.rhs, "t1")
        result_reg = self._alloc_loop_t_reg()
        done_label = f".Lmin_done_{self._label_count}"
        self._label_count += 1
        # result = lhs; if lhs < rhs, skip; else result = rhs
        self._lines.append(f"    mv {result_reg}, {lhs_reg}")
        self._lines.append(f"    blt {result_reg}, {rhs_reg}, {done_label}")
        self._lines.append(f"    mv {result_reg}, {rhs_reg}")
        self._lines.append(f"{done_label}:")
        self._reg_map[id(op.result)] = result_reg

    def _emit_fv_unop(
        self, op: FVReluOp | FVExpOp, name: str, funct7: int
    ) -> None:
        """Emit .insn for a unary NPU op (no copy loop needed).

        Hardware reads from rs1 and writes to rs2 independently.
        """
        src = self._reg(op.src)
        dst = self._reg(op.dst)
        n_is_const = id(op.n) in self._const_map
        funct7_hex = f"0x{funct7:02X}"

        self._lines.append(f"    # NPU.{name} {dst}[i] = {name.lower()}({src}[i])")
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, t0, {src}, {dst}"
            )
        else:
            n_reg = self._reg(op.n)
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, {n_reg}, {src}, {dst}"
            )

    def _emit_facc_load(self, scalar_bits: int) -> None:
        """Emit the instruction sequence to load a float32 scalar into facc.

        Sequence: FRSTACC (zero facc), load scalar bits into f1,
        load 1.0 into f2, FMACC (facc = f1 * f2 = scalar).
        """
        self._lines.append(f"    # Load scalar into facc")
        # FRSTACC: f[rd] = facc; facc = 0.0
        self._lines.append(f"    .insn r 0x2B, 0x5, 0x00, f0, f0, f0")
        # Load scalar float bits into f1
        self._lines.append(f"    li t0, {scalar_bits}")
        self._lines.append(f"    fmv.w.x f1, t0")
        # Load 1.0f into f2
        self._lines.append(f"    lui t0, 0x3F800")
        self._lines.append(f"    fmv.w.x f2, t0")
        # FMACC: facc += f1 * f2 = scalar * 1.0
        self._lines.append(f"    .insn r 0x2B, 0x0, 0x00, f0, f1, f2")

    def _emit_fv_scalar_op(
        self, op: FVMulOp | FVDivOp, name: str, funct7: int
    ) -> None:
        """Emit facc load + .insn for a scalar-vector NPU op.

        NOTE: facc load is emitted inline with each scalar op. In a tiled
        loop, this reloads facc each iteration. A future optimization could
        hoist the load before the loop since facc persists across iterations.
        """
        src = self._reg(op.src)
        dst = self._reg(op.dst)
        n_is_const = id(op.n) in self._const_map
        funct7_hex = f"0x{funct7:02X}"
        op_symbol = "*" if isinstance(op, FVMulOp) else "/"

        # Convert float scalar to IEEE 754 bits
        scalar_val = op.scalar.value.data
        scalar_bits = struct.unpack("<I", struct.pack("<f", scalar_val))[0]

        # Load scalar into facc
        self._emit_facc_load(scalar_bits)

        # NPU instruction: rd=n, rs1=src, rs2=dst
        self._lines.append(
            f"    # NPU.{name} {dst}[i] = {src}[i] {op_symbol} {scalar_val}"
        )
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, t0, {src}, {dst}"
            )
        else:
            n_reg = self._reg(op.n)
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, {n_reg}, {src}, {dst}"
            )

    def _emit_fv_binop(
        self, op: FVAddOp | FVSubOp, name: str, funct7: int
    ) -> None:
        """Emit .insn for a binary NPU op, with copy loop if src2 != dst.

        The hardware writes in-place to rs2. If src2 and dst are the same
        buffer, no copy is needed. Otherwise, copy src2 -> dst first.
        """
        src1 = self._reg(op.src1)
        src2 = self._reg(op.src2)
        dst = self._reg(op.dst)
        n_is_const = id(op.n) in self._const_map
        funct7_hex = f"0x{funct7:02X}"

        # Copy src2 -> dst only if they're different buffers
        if op.src2 is not op.dst:
            idx = self._label_count
            self._label_count += 1
            loop_label = f".Lcopy_{idx}"
            done_label = f".Lcopy_done_{idx}"
            if n_is_const:
                self._lines.append(
                    f"    # copy {src2} -> {dst} ({self._const(op.n)} words)"
                )
                self._lines.append(f"    li t0, {self._const(op.n)}")
            else:
                self._lines.append(f"    # copy {src2} -> {dst}")
                self._lines.append(f"    mv t0, {self._reg(op.n)}")
            self._lines.append(f"    beqz t0, {done_label}")
            self._lines.append(f"    mv t1, {src2}")
            self._lines.append(f"    mv t2, {dst}")
            self._lines.append(f"{loop_label}:")
            self._lines.append(f"    lw t3, 0(t1)")
            self._lines.append(f"    sw t3, 0(t2)")
            self._lines.append(f"    addi t1, t1, 4")
            self._lines.append(f"    addi t2, t2, 4")
            self._lines.append(f"    addi t0, t0, -1")
            self._lines.append(f"    bnez t0, {loop_label}")
            self._lines.append(f"{done_label}:")

        # NPU instruction: rd=n, rs1=src1, rs2=dst (now contains src2 data)
        op_symbol = "+" if isinstance(op, FVAddOp) else "-"
        self._lines.append(
            f"    # NPU.{name} {dst}[i] = {src1}[i] {op_symbol} {dst}[i]"
        )
        if n_is_const:
            self._lines.append(f"    li t0, {self._const(op.n)}")
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, t0, {src1}, {dst}"
            )
        else:
            n_reg = self._reg(op.n)
            self._lines.append(
                f"    .insn r 0x2B, 0x0, {funct7_hex}, {n_reg}, {src1}, {dst}"
            )
