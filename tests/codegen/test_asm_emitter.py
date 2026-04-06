"""Tests for arrax.codegen.asm_emitter — RISC-V assembly generation."""

from __future__ import annotations

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region

from arrax.codegen.asm_emitter import (
    ScalarFPRegisterPool,
    compute_last_use,
    emit_assembly,
)
from arrax.dialects.npu_dialect import FVAddOp, FVSubOp
from arrax.dsl.array import Array, amax, dot, exp, mean, relu, sum
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

    def test_mul_scalar(self) -> None:
        """A * 3.0: facc load + FVMUL."""
        module = make_module(lambda A: A * 3.0, {"A": (64,)})
        asm = _to_asm(module)
        # FRSTACC to zero facc
        assert ".insn r 0x2B, 0x5, 0x00" in asm
        # FMACC to load scalar
        assert ".insn r 0x2B, 0x0, 0x00" in asm
        # FVMUL
        assert ".insn r 0x2B, 0x0, 0x04" in asm
        assert "NPU.FVMUL" in asm

    def test_div_scalar(self) -> None:
        """A / 2.0: facc load + FVDIV."""
        module = make_module(lambda A: A / 2.0, {"A": (64,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x0B" in asm
        assert "NPU.FVDIV" in asm

    def test_tiled_mul_scalar(self) -> None:
        """Tiled A * 3.0: facc load inside loop + FVMUL."""
        module = make_module(lambda A: A * 3.0, {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x04" in asm

    def test_tiled_sub(self) -> None:
        """Tiled subtraction produces loop with FVSUB."""
        module = make_module(lambda A, B: A - B, {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x08" in asm

    def test_relu(self) -> None:
        """relu(A): uses FVRELU funct7=0x09, no copy loop."""
        module = make_module(lambda A: relu(A), {"A": (64,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x09" in asm
        assert "NPU.FVRELU" in asm
        # No copy loop for unary ops
        assert ".Lcopy_" not in asm

    def test_exp(self) -> None:
        """exp(A): uses FVEXP funct7=0x02, no copy loop."""
        module = make_module(lambda A: exp(A), {"A": (64,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x02" in asm
        assert "NPU.FVEXP" in asm
        assert ".Lcopy_" not in asm

    def test_relu_of_add(self) -> None:
        """relu(A + B): one FVADD with copy + one FVRELU without copy."""
        module = make_module(lambda A, B: relu(A + B), {"A": (32,), "B": (32,)})
        asm = _to_asm(module)
        assert ".insn r 0x2B, 0x0, 0x07" in asm  # FVADD
        assert ".insn r 0x2B, 0x0, 0x09" in asm  # FVRELU
        # Only one copy loop (for FVADD), not two
        assert asm.count("lw t3, 0(t1)") == 1

    def test_tiled_relu(self) -> None:
        module = make_module(lambda A: relu(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x09" in asm

    # --- sum reduction ---

    def test_sum_untiled(self) -> None:
        """sum(A), N=64: FVREDUCE .insn + fadd.s into fs0 + fsw terminal store."""
        module = make_module(lambda A: sum(A), {"A": (64,)})
        asm = _to_asm_tiled(module)
        # FVREDUCE funct7 = 0x05
        assert ".insn r 0x2B, 0x0, 0x05" in asm
        # Accumulator materialized into scalar FP register fs0
        assert "fs0" in asm
        # Identity seeding: 0.0 loaded into fs0 via li+fmv.w.x
        assert "fmv.w.x fs0" in asm
        # Combine the per-call partial into fs0
        assert "fadd.s fs0, fs0" in asm
        # Terminal store to the rank-0 output memref (a1)
        assert "fsw fs0, 0(a1)" in asm

    def test_sum_small(self) -> None:
        module = make_module(lambda A: sum(A), {"A": (16,)})
        asm = _to_asm_tiled(module)
        assert ".insn r 0x2B, 0x0, 0x05" in asm
        assert "fsw fs0, 0(a1)" in asm

    def test_sum_untiled_n_not_clobbered_by_acc_materialization(self) -> None:
        """Regression: untiled fvreduce must load n into t0 AFTER materializing
        the f32 acc_in — otherwise the ``li t0, <f32 bits>`` emitted by
        materialization overwrites the just-loaded element count.
        """
        module = make_module(lambda A: sum(A), {"A": (16,)})
        asm = _to_asm_tiled(module)
        # The last `li t0, ...` before the FVREDUCE .insn must load the
        # element count (16), not the f32 bits for 0.0.
        insn_idx = asm.index(".insn r 0x2B, 0x0, 0x05")
        pre_insn = asm[:insn_idx]
        last_li_t0 = pre_insn.rfind("li t0,")
        assert last_li_t0 != -1
        last_li_line = pre_insn[last_li_t0:].splitlines()[0]
        assert "li t0, 16" in last_li_line, (
            f"n clobbered before FVREDUCE; last li t0 before .insn was: "
            f"{last_li_line!r}"
        )

    def test_sum_tiled_n_from_minsi_register(self) -> None:
        """In the tiled case, FVREDUCE's n operand is the chunk size
        produced by ``arith.minsi`` inside the loop body. It lives in a
        loop-body t-register (not t0), so there is no scratch aliasing
        with acc_in materialization — but the emission must actually
        reference that register, not t0.
        """
        module = make_module(lambda A: sum(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        # The FVREDUCE emission is inside the loop body; its rs2 (n) must
        # be a loop-body t-register (t4 or t5), not t0.
        lines = asm.splitlines()
        insn_lines = [line for line in lines if ".insn r 0x2B, 0x0, 0x05" in line]
        assert len(insn_lines) == 1, f"expected one FVREDUCE, got {insn_lines}"
        insn_line = insn_lines[0]
        # rs2 is the last comma-separated operand on the line.
        rs2 = insn_line.rsplit(",", 1)[1].strip()
        assert rs2 in {"t4", "t5"}, (
            f"tiled FVREDUCE n register should be a loop-body t-reg, got {rs2!r}"
        )

    def test_sum_tiled(self) -> None:
        """sum(A), N=128: FVREDUCE inside scf.for; fs0 init before loop; fsw after."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x05" in asm  # FVREDUCE
        assert "fadd.s fs0, fs0" in asm
        assert "fsw fs0, 0(a1)" in asm
        # iter_args init (0.0) emitted before the loop label
        pre_loop = asm.split(".Lfor_")[0]
        assert "fmv.w.x fs0" in pre_loop
        # Terminal store lives after the loop end label (last .Lfor_end
        # occurrence is the label; earlier ones are branch targets).
        post_loop = asm.split(".Lfor_end")[-1]
        assert "fsw fs0, 0(a1)" in post_loop

    def test_sum_non_multiple(self) -> None:
        """sum(A), N=100: remainder tolerated by FVREDUCE inside tiled loop."""
        module = make_module(lambda A: sum(A), {"A": (100,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x05" in asm
        assert "fsw fs0, 0(a1)" in asm

    def test_sum_large(self) -> None:
        """sum(A), N=1024."""
        module = make_module(lambda A: sum(A), {"A": (1024,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x05" in asm

    # --- amax reduction ---

    def test_amax_untiled(self) -> None:
        """amax(A), N=64: FVMAX .insn + fmax.s into fs0 + fsw terminal store.

        FVMAX funct7 = 0x06; acc seed is -inf (IEEE 0xff800000 == -8388608
        as a signed 32-bit integer — the li immediate encodes the bit
        pattern, xDSL/GAS accept the signed form).
        """
        module = make_module(lambda A: amax(A), {"A": (64,)})
        asm = _to_asm_tiled(module)
        # FVMAX funct7 = 0x06
        assert ".insn r 0x2B, 0x0, 0x06" in asm
        # Accumulator materialized into scalar FP register fs0
        assert "fs0" in asm
        # Identity seeding: -inf loaded into fs0 via li+fmv.w.x
        assert "fmv.w.x fs0" in asm
        # Combine the per-call partial into fs0 via NaN-propagating max
        assert "fmax.s fs0, fs0" in asm
        # Terminal store to the rank-0 output memref (a1)
        assert "fsw fs0, 0(a1)" in asm

    def test_amax_small(self) -> None:
        module = make_module(lambda A: amax(A), {"A": (16,)})
        asm = _to_asm_tiled(module)
        assert ".insn r 0x2B, 0x0, 0x06" in asm
        assert "fmax.s fs0, fs0" in asm
        assert "fsw fs0, 0(a1)" in asm

    def test_amax_untiled_n_not_clobbered_by_acc_materialization(self) -> None:
        """Regression: untiled fvmax must load n into t0 AFTER materializing
        the f32 acc_in (-inf) — otherwise the ``li t0, <-inf bits>`` emitted
        by materialization overwrites the just-loaded element count.
        """
        module = make_module(lambda A: amax(A), {"A": (16,)})
        asm = _to_asm_tiled(module)
        insn_idx = asm.index(".insn r 0x2B, 0x0, 0x06")
        pre_insn = asm[:insn_idx]
        last_li_t0 = pre_insn.rfind("li t0,")
        assert last_li_t0 != -1
        last_li_line = pre_insn[last_li_t0:].splitlines()[0]
        assert "li t0, 16" in last_li_line, (
            f"n clobbered before FVMAX; last li t0 before .insn was: "
            f"{last_li_line!r}"
        )

    def test_amax_tiled_n_from_minsi_register(self) -> None:
        """In the tiled case, FVMAX's n operand is the chunk size produced
        by ``arith.minsi`` inside the loop body. It must live in a loop-body
        t-register (t4/t5), not t0.
        """
        module = make_module(lambda A: amax(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        lines = asm.splitlines()
        insn_lines = [line for line in lines if ".insn r 0x2B, 0x0, 0x06" in line]
        assert len(insn_lines) == 1, f"expected one FVMAX, got {insn_lines}"
        insn_line = insn_lines[0]
        rs2 = insn_line.rsplit(",", 1)[1].strip()
        assert rs2 in {"t4", "t5"}, (
            f"tiled FVMAX n register should be a loop-body t-reg, got {rs2!r}"
        )

    def test_amax_tiled(self) -> None:
        """amax(A), N=128: FVMAX inside scf.for; fs0 init (-inf) before loop;
        fsw after."""
        module = make_module(lambda A: amax(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x06" in asm  # FVMAX
        assert "fmax.s fs0, fs0" in asm
        assert "fsw fs0, 0(a1)" in asm
        # iter_args init (-inf) emitted before the loop label
        pre_loop = asm.split(".Lfor_")[0]
        assert "fmv.w.x fs0" in pre_loop
        # Terminal store lives after the loop end label.
        post_loop = asm.split(".Lfor_end")[-1]
        assert "fsw fs0, 0(a1)" in post_loop

    def test_amax_non_multiple(self) -> None:
        """amax(A), N=100: remainder tolerated by FVMAX inside tiled loop."""
        module = make_module(lambda A: amax(A), {"A": (100,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x06" in asm
        assert "fsw fs0, 0(a1)" in asm

    def test_amax_large(self) -> None:
        """amax(A), N=1024."""
        module = make_module(lambda A: amax(A), {"A": (1024,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x06" in asm

    # --- dot (fvmac) asm emission ---

    def test_dot_untiled(self) -> None:
        """dot(A, B), N=64: FRSTACC bracket + FVMAC .insn + fsw terminal store."""
        module = make_module(lambda A, B: dot(A, B), {"A": (64,), "B": (64,)})
        asm = _to_asm_tiled(module)
        # FRSTACC zero before FVMAC (funct3=0x5, funct7=0x00)
        assert ".insn r 0x2B, 0x5, 0x00" in asm
        # FVMAC funct7 = 0x01
        assert ".insn r 0x2B, 0x0, 0x01" in asm
        # Result lands in fs0 via FRSTACC read
        assert "fs0" in asm
        # Terminal store to the rank-0 output memref (a2)
        assert "fsw fs0, 0(a2)" in asm

    def test_dot_untiled_small(self) -> None:
        module = make_module(lambda A, B: dot(A, B), {"A": (16,), "B": (16,)})
        asm = _to_asm_tiled(module)
        assert ".insn r 0x2B, 0x0, 0x01" in asm
        assert "fsw fs0, 0(a2)" in asm

    def test_dot_untiled_frstacc_bracket(self) -> None:
        """Untiled dot: FRSTACC zero appears before FVMAC, FRSTACC read after."""
        module = make_module(lambda A, B: dot(A, B), {"A": (32,), "B": (32,)})
        asm = _to_asm_tiled(module)
        lines = asm.splitlines()
        # Find FRSTACC and FVMAC instruction lines
        frstacc_lines = [i for i, l in enumerate(lines) if ".insn r 0x2B, 0x5, 0x00" in l]
        fvmac_lines = [i for i, l in enumerate(lines) if ".insn r 0x2B, 0x0, 0x01" in l]
        assert len(frstacc_lines) == 2, f"expected 2 FRSTACC, got {frstacc_lines}"
        assert len(fvmac_lines) == 1, f"expected 1 FVMAC, got {fvmac_lines}"
        # FRSTACC zero < FVMAC < FRSTACC read
        assert frstacc_lines[0] < fvmac_lines[0] < frstacc_lines[1]

    def test_dot_tiled(self) -> None:
        """dot(A, B), N=128: FRSTACC zero before loop, FVMAC inside, FRSTACC read after."""
        module = make_module(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x01" in asm  # FVMAC
        assert "fsw fs0, 0(a2)" in asm
        # FRSTACC zero before the loop label
        pre_loop = asm.split(".Lfor_")[0]
        assert ".insn r 0x2B, 0x5, 0x00" in pre_loop
        # FRSTACC read + terminal store after the loop
        post_loop = asm.split(".Lfor_end")[-1]
        assert ".insn r 0x2B, 0x5, 0x00" in post_loop
        assert "fsw fs0, 0(a2)" in post_loop

    def test_dot_tiled_no_fadd_combine(self) -> None:
        """Tiled dot does NOT use fadd.s or fmax.s combine — facc accumulates directly."""
        module = make_module(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        asm = _to_asm_tiled(module)
        assert "fadd.s" not in asm
        assert "fmax.s" not in asm

    def test_dot_non_multiple(self) -> None:
        """dot(A, B), N=100: remainder handled."""
        module = make_module(lambda A, B: dot(A, B), {"A": (100,), "B": (100,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x01" in asm
        assert "fsw fs0, 0(a2)" in asm

    def test_dot_large(self) -> None:
        """dot(A, B), N=1024."""
        module = make_module(lambda A, B: dot(A, B), {"A": (1024,), "B": (1024,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x01" in asm

    # --- mean reduction ---

    def test_mean_untiled(self) -> None:
        """mean(A), N=64: FVREDUCE + fadd.s + trailing fdiv.s + fsw."""
        module = make_module(lambda A: mean(A), {"A": (64,)})
        asm = _to_asm_tiled(module)
        assert ".insn r 0x2B, 0x0, 0x05" in asm  # FVREDUCE
        assert "fadd.s fs0, fs0" in asm
        assert "fdiv.s fs0, fs0, ft1" in asm
        assert "fsw fs0, 0(a1)" in asm

    def test_mean_tiled(self) -> None:
        """mean(A), N=128: FVREDUCE in loop, fdiv.s after loop, fsw at end."""
        module = make_module(lambda A: mean(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x05" in asm  # FVREDUCE
        assert "fdiv.s fs0, fs0, ft1" in asm
        assert "fsw fs0, 0(a1)" in asm
        # fdiv.s must come AFTER the loop, not inside it
        post_loop = asm.split(".Lfor_end")[-1]
        assert "fdiv.s" in post_loop

    def test_mean_tiled_fdiv_before_store(self) -> None:
        """The fdiv.s must appear between the loop end and the terminal fsw."""
        module = make_module(lambda A: mean(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        fdiv_idx = asm.index("fdiv.s")
        fsw_idx = asm.index("fsw fs0")
        assert fdiv_idx < fsw_idx

    def test_mean_non_multiple(self) -> None:
        """mean(A), N=100: remainder handled, fdiv.s still present."""
        module = make_module(lambda A: mean(A), {"A": (100,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert ".insn r 0x2B, 0x0, 0x05" in asm
        assert "fdiv.s fs0, fs0, ft1" in asm

    def test_mean_large(self) -> None:
        """mean(A), N=1024."""
        module = make_module(lambda A: mean(A), {"A": (1024,)})
        asm = _to_asm_tiled(module)
        assert ".Lfor_" in asm
        assert "fdiv.s" in asm

    def test_sum_no_fdiv(self) -> None:
        """sum(A) must NOT have fdiv.s (no divisor)."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        asm = _to_asm_tiled(module)
        assert "fdiv.s" not in asm


class TestScalarFPRegisterPool:
    @staticmethod
    def _mk() -> arith.ConstantOp:
        return arith.ConstantOp(FloatAttr(0.0, Float32Type()))

    def test_first_allocation_is_fs0(self) -> None:
        pool = ScalarFPRegisterPool()
        c = self._mk()
        assert pool.allocate(c.result) == "fs0"

    def test_two_live_scalars_get_distinct_registers(self) -> None:
        pool = ScalarFPRegisterPool()
        a = self._mk()
        b = self._mk()
        r1 = pool.allocate(a.result)
        r2 = pool.allocate(b.result)
        assert r1 != r2
        assert pool.get(a.result) == r1
        assert pool.get(b.result) == r2

    def test_release_returns_register_to_pool(self) -> None:
        pool = ScalarFPRegisterPool()
        a = self._mk()
        b = self._mk()
        r1 = pool.allocate(a.result)
        pool.release(a.result)
        r2 = pool.allocate(b.result)
        assert r1 == r2  # reused

    def test_bind_aliases_existing_register(self) -> None:
        """bind() maps a new SSA value to an already-allocated register."""
        pool = ScalarFPRegisterPool()
        a = self._mk()
        b = self._mk()
        r1 = pool.allocate(a.result)
        pool.bind(b.result, r1)
        assert pool.get(b.result) == r1

    def test_release_after_bind_frees_when_all_aliases_gone(self) -> None:
        """Regression: alias tracking must unconditionally pop each released
        SSA value from the map and only free the underlying register once
        every aliased reference has been dropped. Previously the release
        path kept val's entry in the map while aliases existed, which
        left neither alias ever reaching the 'sole holder' branch and
        leaked the register permanently.
        """
        pool = ScalarFPRegisterPool()
        a = self._mk()
        b = self._mk()
        c = self._mk()
        r1 = pool.allocate(a.result)
        pool.bind(b.result, r1)
        # Drop one alias: register still held by b, must not be freed yet.
        pool.release(a.result)
        # Drop the other: register is now unreferenced and must return to
        # the free list in LIFO order.
        pool.release(b.result)
        r2 = pool.allocate(c.result)
        assert r1 == r2, (
            f"expected {r1} to be reused after all aliases released, "
            f"but got {r2}"
        )

    def test_lifo_reuse(self) -> None:
        """Just-released registers are reused before lower-numbered ones."""
        pool = ScalarFPRegisterPool()
        a, b, c = self._mk(), self._mk(), self._mk()
        r1 = pool.allocate(a.result)  # fs0
        r2 = pool.allocate(b.result)  # fs1
        pool.release(b.result)
        r3 = pool.allocate(c.result)
        assert r3 == r2  # fs1 comes back first, not fs2
        assert r1 == "fs0"

    def test_exhaustion_raises(self) -> None:
        pool = ScalarFPRegisterPool()
        # Allocate all 12 regs
        for _ in range(12):
            c = self._mk()
            pool.allocate(c.result)
        overflow = self._mk()
        with pytest.raises(ValueError, match="scalar FP register pool"):
            pool.allocate(overflow.result)


class TestComputeLastUse:
    def test_last_use_is_index_of_final_operand_reference(self) -> None:
        """A value's last-use index equals the index of the last op that
        references it as an operand, numbered in IR visit order.
        """
        # Build: c0 = const; c1 = const; s0 = addi c0, c1; s1 = subi s0, c0
        # Visit order indices: c0=0, c1=1, s0=2, s1=3.
        # Last uses: c0 -> 3 (operand of s1), c1 -> 2 (operand of s0),
        # s0 -> 3 (operand of s1); s1 has no users.
        c0 = arith.ConstantOp(IntegerAttr(10, IndexType()))
        c1 = arith.ConstantOp(IntegerAttr(20, IndexType()))
        s0 = arith.AddiOp(c0.result, c1.result)
        s1 = arith.SubiOp(s0.result, c0.result)
        block = Block()
        block.add_ops([c0, c1, s0, s1])

        lu = compute_last_use(block)
        assert lu[id(c0.result)] == 3
        assert lu[id(c1.result)] == 2
        assert lu[id(s0.result)] == 3
        assert id(s1.result) not in lu  # never used

    def test_last_use_crosses_nested_regions(self) -> None:
        """Ops inside a nested region are numbered in the same index sequence
        as their enclosing block, so a value defined outside the region and
        used inside it has its last-use at the inner op's index.
        """
        # Build: c0 = const; for-like op with body that references c0.
        # We use scf.ForOp so the region walk in compute_last_use is real.
        from xdsl.dialects import scf

        lb = arith.ConstantOp(IntegerAttr(0, IndexType()))
        ub = arith.ConstantOp(IntegerAttr(4, IndexType()))
        step = arith.ConstantOp(IntegerAttr(1, IndexType()))
        outer = arith.ConstantOp(IntegerAttr(42, IndexType()))

        body_block = Block(arg_types=[IndexType()])
        inner = arith.AddiOp(body_block.args[0], outer.result)
        yield_op = scf.YieldOp()
        body_block.add_ops([inner, yield_op])
        for_op = scf.ForOp(
            lb.result, ub.result, step.result, [], Region([body_block])
        )

        outer_block = Block()
        outer_block.add_ops([lb, ub, step, outer, for_op])

        lu = compute_last_use(outer_block)
        # Visit order: lb=0, ub=1, step=2, outer=3, for_op=4, inner=5, yield=6
        # outer.result is referenced by inner → last use = 5.
        assert lu[id(outer.result)] == 5
        # lb/ub/step last-used at for_op (index 4).
        assert lu[id(lb.result)] == 4
        assert lu[id(ub.result)] == 4
        assert lu[id(step.result)] == 4
