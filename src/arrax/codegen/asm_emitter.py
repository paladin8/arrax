"""Assembly text generation for RISC-V + NPU .insn directives.

Walks NPU-dialect IR and emits a complete .S file with:
- Function prologue (.globl, .type)
- Copy loops for npu.fvadd in-place semantics (src2 -> dst)
- .insn r directives for NPU instructions
- .comm directives for intermediate buffer allocations
- scf.for loops for tiled operations
- memref.subview pointer arithmetic

Register allocation strategy:
- a0-a7: function arguments (RISC-V calling convention)
- s0-s11: callee-saved, used for values persisting across loops:
  memref.alloc addresses, loop IVs, memref.subview pointers
- t4-t5: loop-body-local values (arith.subi, arith.minsi) — reset each iteration
- t0-t3: scratch for copy loops, constant loading, .insn operands
"""

from __future__ import annotations

from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp
from xdsl.ir import Block, Operation, SSAValue

from arrax.dialects.npu_dialect import FVAddOp, FVExpOp, FVReluOp, FVSubOp

# t-registers for loop-body-local values that die before the copy loop.
# t0-t3 are reserved as scratch for copy loops and constant loading.
_LOOP_BODY_T_REGS = ["t4", "t5"]


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
        self._s_reg_count: int = 0
        self._label_count: int = 0
        self._loop_t_reg_idx: int = 0  # index into _LOOP_BODY_T_REGS

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
        elif isinstance(op, scf.ForOp):
            self._emit_for(op)
        elif isinstance(op, memref.SubviewOp):
            self._emit_subview(op)
        elif isinstance(op, arith.SubiOp):
            self._emit_subi(op)
        elif isinstance(op, arith.MinSIOp):
            self._emit_minsi(op)
        elif isinstance(op, scf.YieldOp):
            pass  # no-op for loops without iter_args
        elif isinstance(op, Operation):
            raise ValueError(f"asm_emitter: unsupported op {op.name}")
        else:
            raise ValueError(f"asm_emitter: unsupported op {type(op)}")

    def _emit_constant(self, op: arith.ConstantOp) -> None:
        """Record constant value for later use (emitted inline when needed)."""
        assert isinstance(op.value, IntegerAttr)
        self._const_map[id(op.result)] = op.value.value.data

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
        """Emit a counted loop for scf.for."""
        body = op.body.blocks.first
        assert body is not None

        # Save s-reg counter — sequential loops reuse the same s-reg range
        loop_s_reg_start = self._s_reg_count

        # Assign IV to an s-register (persists across iterations)
        iv_reg = self._alloc_s_reg()
        self._reg_map[id(body.args[0])] = iv_reg

        # Initialize IV from lower bound
        if id(op.lb) in self._const_map and self._const_map[id(op.lb)] == 0:
            self._lines.append(f"    li {iv_reg}, 0")
        else:
            lb_reg = self._load_operand(op.lb, "t0")
            self._lines.append(f"    mv {iv_reg}, {lb_reg}")

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

        # Restore s-reg counter for sequential loop reuse
        self._s_reg_count = loop_s_reg_start

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
