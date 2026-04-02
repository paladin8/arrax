"""Assembly text generation for RISC-V + NPU .insn directives.

Walks NPU-dialect IR and emits a complete .S file with:
- Function prologue (.globl, .type)
- Copy loops for npu.fvadd in-place semantics (src2 → dst)
- .insn r directives for NPU instructions
- .comm directives for intermediate buffer allocations
"""

from __future__ import annotations

from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp
from xdsl.ir import SSAValue

from arrax.dialects.npu_dialect import FVAddOp


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
        self._reg_map: dict[int, str] = {}  # id(SSAValue) → register name
        self._const_map: dict[int, int] = {}  # id(SSAValue) → integer value
        self._alloc_count: int = 0
        self._label_count: int = 0

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

    def _emit_func(self, func_op: func.FuncOp) -> None:
        name = func_op.sym_name.data
        self._lines.append("    .text")
        self._lines.append(f"    .globl {name}")
        self._lines.append(f"    .type {name}, @function")
        self._lines.append(f"{name}:")

        block = func_op.body.blocks.first
        assert block is not None

        # Function args → a0, a1, ...
        for i, arg in enumerate(block.args):
            self._reg_map[id(arg)] = f"a{i}"

        # Count s-registers used (for callee-save prologue/epilogue)
        s_regs_used: list[str] = []
        for op in block.ops:
            if isinstance(op, memref.AllocOp):
                s_regs_used.append(f"s{len(s_regs_used)}")

        # Prologue: save callee-saved s-registers
        if s_regs_used:
            frame_size = len(s_regs_used) * 4
            self._lines.append(f"    addi sp, sp, -{frame_size}")
            for i, sreg in enumerate(s_regs_used):
                self._lines.append(f"    sw {sreg}, {i * 4}(sp)")

        # Walk body ops
        for op in block.ops:
            if isinstance(op, arith.ConstantOp):
                self._emit_constant(op)
            elif isinstance(op, memref.AllocOp):
                self._emit_alloc(op)
            elif isinstance(op, FVAddOp):
                self._emit_fvadd(op)
            elif isinstance(op, func.ReturnOp):
                # Epilogue: restore callee-saved s-registers
                if s_regs_used:
                    for i, sreg in enumerate(s_regs_used):
                        self._lines.append(f"    lw {sreg}, {i * 4}(sp)")
                    frame_size = len(s_regs_used) * 4
                    self._lines.append(f"    addi sp, sp, {frame_size}")
                self._lines.append("    ret")
            else:
                raise ValueError(f"asm_emitter: unsupported op {op.name}")

    def _emit_constant(self, op: arith.ConstantOp) -> None:
        """Record constant value for later use (emitted inline when needed)."""
        assert isinstance(op.value, IntegerAttr)
        self._const_map[id(op.result)] = op.value.value.data

    def _emit_alloc(self, op: memref.AllocOp) -> None:
        """Emit .comm for static buffer allocation, load address into s-register."""
        assert isinstance(op.memref.type, MemRefType)
        shape = op.memref.type.get_shape()
        # Total size in bytes (f32 = 4 bytes per element)
        size = 1
        for dim in shape:
            size *= dim
        size *= 4

        sym = f".Lbuf_{self._alloc_count}"
        sreg = f"s{self._alloc_count}"
        self._bss.append(f"    .comm {sym}, {size}, 4")
        self._lines.append(f"    la {sreg}, {sym}")
        self._reg_map[id(op.memref)] = sreg
        self._alloc_count += 1

    def _emit_fvadd(self, op: FVAddOp) -> None:
        """Emit .insn for NPU FVADD, with copy loop if src2 != dst.

        The hardware writes in-place to rs2. If src2 and dst are the same
        buffer, no copy is needed. Otherwise, copy src2 → dst first.
        """
        src1 = self._reg(op.src1)
        src2 = self._reg(op.src2)
        dst = self._reg(op.dst)
        n = self._const(op.n)

        # Copy src2 → dst only if they're different buffers
        if op.src2 is not op.dst:
            idx = self._label_count
            self._label_count += 1
            loop_label = f".Lcopy_{idx}"
            done_label = f".Lcopy_done_{idx}"
            self._lines.append(f"    # copy {src2} -> {dst} ({n} words)")
            self._lines.append(f"    li t0, {n}")
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
        # NPU.FVADD: rd=n, rs1=src1, rs2=dst (now contains src2 data)
        self._lines.append(f"    # NPU.FVADD {dst}[i] = {src1}[i] + {dst}[i]")
        self._lines.append(f"    li t0, {n}")
        self._lines.append(f"    .insn r 0x2B, 0x0, 0x07, t0, {src1}, {dst}")
