"""Tests for arrax.codegen.asm_emitter — RISC-V assembly generation."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    Float32Type,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region

from arrax.codegen.asm_emitter import emit_assembly
from arrax.dialects.npu_dialect import FVAddOp, FVSubOp
from arrax.dsl.array import Array
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.linalg_to_npu import LinalgToNpuPass
from arrax.lowering.npu_canonicalize import NpuCanonicalizePass
from arrax.lowering.tile import TilePass
from tests.helpers import make_module


def _to_asm(module: ModuleOp) -> str:
    """Apply full lowering pipeline (no tiling) and emit assembly."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    return emit_assembly(module)


def _to_asm_tiled(module: ModuleOp) -> str:
    """Apply full lowering pipeline with tiling and emit assembly."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    TilePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    NpuCanonicalizePass().apply(ctx, module)
    module.verify()
    return emit_assembly(module)


class TestAsmEmitter:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)

        expected = """\
    .text
    .globl kernel
    .type kernel, @function
kernel:
    # copy a1 -> a2 (64 words)
    li t0, 64
    beqz t0, .Lcopy_done_0
    mv t1, a1
    mv t2, a2
.Lcopy_0:
    lw t3, 0(t1)
    sw t3, 0(t2)
    addi t1, t1, 4
    addi t2, t2, 4
    addi t0, t0, -1
    bnez t0, .Lcopy_0
.Lcopy_done_0:
    # NPU.FVADD a2[i] = a0[i] + a2[i]
    li t0, 64
    .insn r 0x2B, 0x0, 0x07, t0, a0, a2
    ret
"""
        assert asm == expected

    def test_contains_insn_directive(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x07" in asm

    def test_text_section(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        asm = _to_asm(module)
        assert asm.startswith("    .text\n")

    def test_function_header(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        asm = _to_asm(module)
        assert ".globl kernel" in asm
        assert ".type kernel, @function" in asm
        assert "kernel:" in asm
        assert "ret" in asm

    def test_copy_loop_with_guard(self) -> None:
        """Copy loop has beqz guard for n=0 safety."""
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert "beqz t0, .Lcopy_done_0" in asm
        assert ".Lcopy_0:" in asm
        assert "lw t3, 0(t1)" in asm
        assert "sw t3, 0(t2)" in asm
        assert "bnez t0, .Lcopy_0" in asm
        assert ".Lcopy_done_0:" in asm

    def test_n_matches_shape(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (48,), "B": (48,)})
        asm = _to_asm(module)
        assert "li t0, 48" in asm

    def test_chained_add(self) -> None:
        """(A + B) + C: two FVADDs, .comm allocation, s-register save/restore."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        asm = _to_asm(module)

        # Two FVADD instructions
        assert asm.count(".insn r 0x2B, 0x0, 0x07") == 2
        # Two copy loops
        assert asm.count(".Lcopy_") >= 2
        # Intermediate buffer in .bss
        assert ".section .bss" in asm
        assert ".comm .Lbuf_0, 128, 4" in asm
        assert "la s0, .Lbuf_0" in asm
        # Callee-save prologue/epilogue for s0
        assert "sw s0, 0(sp)" in asm
        assert "lw s0, 0(sp)" in asm

    def test_no_bss_for_simple_case(self) -> None:
        """Simple A + B has no intermediate allocs, so no .bss section."""
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert ".section .bss" not in asm
        assert ".comm" not in asm

    def test_no_prologue_for_simple_case(self) -> None:
        """Simple A + B uses no s-registers, so no save/restore."""
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert "addi sp" not in asm
        assert "sw s0" not in asm

    def test_diamond_dag(self) -> None:
        """A + A: copy and FVADD use the same register for src1 and src2."""
        module = make_module(lambda A: A + A, {"A": (16,)})
        asm = _to_asm(module)
        # src1 and src2 are both a0
        assert "copy a0 -> a1" in asm
        assert ".insn r 0x2B, 0x0, 0x07, t0, a0, a1" in asm

    def test_skip_copy_when_src2_is_dst(self) -> None:
        """When src2 and dst are the same SSA value, no copy loop is needed."""
        f32_mem = MemRefType(Float32Type(), [64])
        # func.func @test(%src1: memref, %src2_dst: memref)
        # npu.fvadd %src1, %src2_dst, %src2_dst, 64  ← src2 == dst
        block = Block(arg_types=[f32_mem, f32_mem])
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
        fvadd = FVAddOp(block.args[0], block.args[1], block.args[1], n_const.result)
        block.add_ops([n_const, fvadd, func.ReturnOp()])
        func_op = func.FuncOp(
            name="test_kernel",
            function_type=([f32_mem, f32_mem], []),
            region=Region([block]),
        )
        module = ModuleOp([func_op])

        asm = emit_assembly(module)
        # Should have FVADD but NO copy loop
        assert ".insn r 0x2B, 0x0, 0x07" in asm
        assert "lw t3" not in asm
        assert "sw t3" not in asm
        assert ".Lcopy" not in asm

    def test_skip_copy_after_canonicalize(self) -> None:
        """Canonicalize swaps src1/src2 when src1==dst; emitter then skips copy."""
        f32_mem = MemRefType(Float32Type(), [64])
        # npu.fvadd %0, %1, %0, 64  ← src1 == dst
        block = Block(arg_types=[f32_mem, f32_mem])
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
        fvadd = FVAddOp(block.args[0], block.args[1], block.args[0], n_const.result)
        block.add_ops([n_const, fvadd, func.ReturnOp()])
        func_op = func.FuncOp(
            name="test_kernel",
            function_type=([f32_mem, f32_mem], []),
            region=Region([block]),
        )
        module = ModuleOp([func_op])

        # Canonicalize first (swaps operands), then emit
        ctx = Context()
        NpuCanonicalizePass().apply(ctx, module)
        asm = emit_assembly(module)
        # After canonicalize: src2 == dst, so no copy
        assert ".insn r 0x2B, 0x0, 0x07" in asm
        assert "lw t3" not in asm
        assert ".Lcopy" not in asm
        # rs1 = a1 (swapped), rs2 = a0 (dst)
        assert ".insn r 0x2B, 0x0, 0x07, t0, a1, a0" in asm

    def test_tiled_has_loop(self) -> None:
        """n=128 with tiling: assembly contains a for loop."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".Lfor_end_" in asm
        assert "bge" in asm
        assert ".insn r 0x2B, 0x0, 0x07" in asm
        assert "ret" in asm

    def test_tiled_has_subview_pointer_math(self) -> None:
        """Tiled assembly computes subview pointers via slli + add."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert "slli t1" in asm  # offset * 4
        assert "add s" in asm  # base + offset

    def test_tiled_has_minsi(self) -> None:
        """Tiled assembly computes min for remainder handling."""
        module = make_module(lambda A, B: A + B, {"A": (100,), "B": (100,)})
        asm = _to_asm_tiled(module)
        assert ".Lmin_done_" in asm
        assert "blt" in asm

    def test_tiled_small_unchanged(self) -> None:
        """n=32 (below limit): no loop, same as untiled."""
        module_a = make_module(lambda A, B: A + B, {"A": (32,), "B": (32,)})
        module_b = make_module(lambda A, B: A + B, {"A": (32,), "B": (32,)})
        asm_tiled = _to_asm_tiled(module_a)
        asm_untiled = _to_asm(module_b)
        assert asm_tiled == asm_untiled

    def test_tiled_s_regs_saved(self) -> None:
        """Tiled emission uses s-registers and saves/restores them."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        # Must save/restore s-registers
        assert "addi sp, sp, -" in asm
        assert "sw s0" in asm
        assert "lw s0" in asm

    def test_tiled_chained_add(self) -> None:
        """(A + B) + C with n=128: two tiled loops, s-regs reused."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        asm = _to_asm_tiled(module)
        # Two for loops (one per add)
        assert asm.count(".Lfor_") >= 2
        # Two FVADDs
        assert asm.count(".insn r 0x2B, 0x0, 0x07") == 2
        # Intermediate buffer
        assert ".comm" in asm
        # S-regs must stay within s0-s11 (no s12+)
        assert "s12" not in asm

    def test_basic_sub(self) -> None:
        """A - B: uses FVSUB funct7=0x08."""
        module = make_module(lambda A, B: A - B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x08" in asm
        assert "NPU.FVSUB" in asm

    def test_sub_has_copy_loop(self) -> None:
        """FVSUB needs copy loop like FVADD (hardware writes in-place to rs2)."""
        module = make_module(lambda A, B: A - B, {"A": (64,), "B": (64,)})
        asm = _to_asm(module)
        assert ".Lcopy_" in asm

    def test_mixed_add_sub(self) -> None:
        """(A + B) - C: one FVADD and one FVSUB."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) - C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x07" in asm  # FVADD
        assert ".insn r 0x2B, 0x0, 0x08" in asm  # FVSUB

    def test_tiled_sub(self) -> None:
        """Tiled subtraction produces loop with FVSUB."""
        module = make_module(lambda A, B: A - B, {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x08" in asm
