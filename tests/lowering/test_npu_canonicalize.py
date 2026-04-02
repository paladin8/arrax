"""Tests for arrax.lowering.npu_canonicalize — NPU dialect optimizations."""

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

from arrax.dialects.npu_dialect import FVAddOp
from arrax.lowering.npu_canonicalize import NpuCanonicalizePass


def _make_fvadd_module(src1_idx: int, src2_idx: int, dst_idx: int) -> ModuleOp:
    """Build a module with a single npu.fvadd using block args by index.

    Creates a func with 3 memref args. src1_idx, src2_idx, dst_idx
    select which block arg is used for each fvadd operand.
    """
    f32_mem = MemRefType(Float32Type(), [64])
    block = Block(arg_types=[f32_mem, f32_mem, f32_mem])
    n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
    fvadd = FVAddOp(
        block.args[src1_idx], block.args[src2_idx], block.args[dst_idx], n_const.result
    )
    block.add_ops([n_const, fvadd, func.ReturnOp()])
    func_op = func.FuncOp(
        name="test_kernel",
        function_type=([f32_mem, f32_mem, f32_mem], []),
        region=Region([block]),
    )
    return ModuleOp([func_op])


class TestFVAddSwapForInPlace:
    def test_swap_when_src1_is_dst(self) -> None:
        """src1 == dst, src2 != dst → swap so src2 == dst."""
        # fvadd %0, %1, %0 → fvadd %1, %0, %0
        module = _make_fvadd_module(src1_idx=0, src2_idx=1, dst_idx=0)
        ctx = Context()
        NpuCanonicalizePass().apply(ctx, module)
        module.verify()

        ir = str(module)
        # After swap: src1=%1, src2=%0, dst=%0
        assert "npu.fvadd %1, %0, %0" in ir

    def test_no_swap_when_src2_is_dst(self) -> None:
        """src2 == dst already → no change needed."""
        # fvadd %0, %1, %1 → unchanged
        module = _make_fvadd_module(src1_idx=0, src2_idx=1, dst_idx=1)
        ctx = Context()
        NpuCanonicalizePass().apply(ctx, module)
        module.verify()

        ir = str(module)
        assert "npu.fvadd %0, %1, %1" in ir

    def test_no_swap_when_all_different(self) -> None:
        """All operands different → no change."""
        # fvadd %0, %1, %2 → unchanged
        module = _make_fvadd_module(src1_idx=0, src2_idx=1, dst_idx=2)
        ctx = Context()
        NpuCanonicalizePass().apply(ctx, module)
        module.verify()

        ir = str(module)
        assert "npu.fvadd %0, %1, %2" in ir

    def test_no_swap_when_both_are_dst(self) -> None:
        """src1 == src2 == dst → no swap needed (already in-place)."""
        # fvadd %0, %0, %0 → unchanged
        module = _make_fvadd_module(src1_idx=0, src2_idx=0, dst_idx=0)
        ctx = Context()
        NpuCanonicalizePass().apply(ctx, module)
        module.verify()

        ir = str(module)
        assert "npu.fvadd %0, %0, %0" in ir
