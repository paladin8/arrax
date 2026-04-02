"""Lowering: linalg on memrefs -> npu dialect."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.npu_dialect import FVAddOp


class LinalgAddToNpuPattern(RewritePattern):
    """Match linalg.generic [addf] on memrefs and replace with npu.fvadd."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # Must have 2 inputs, 1 output
        inputs = list(op.inputs)
        outputs = list(op.outputs)
        if len(inputs) != 2 or len(outputs) != 1:
            return

        # All operands must be 1D memref (f32 enforced by FVAddOp.verify_())
        for operand in inputs + outputs:
            if not isinstance(operand.type, MemRefType):
                return
            if len(operand.type.get_shape()) != 1:
                return

        # Indexing maps must all be 1D identity
        identity_1d = AffineMap.identity(1)
        for map_attr in op.indexing_maps.data:
            if map_attr.data != identity_1d:
                return

        # Single parallel iterator
        if len(op.iterator_types.data) != 1:
            return
        from xdsl.dialects.linalg import IteratorType

        if op.iterator_types.data[0].data != IteratorType.PARALLEL:
            return

        # Body must be exactly: addf + yield
        body_block = op.body.blocks.first
        assert body_block is not None
        body_ops = list(body_block.ops)
        if len(body_ops) != 2:
            return
        if not isinstance(body_ops[0], arith.AddfOp):
            return
        if not isinstance(body_ops[1], linalg.YieldOp):
            return

        # Extract n from the memref shape
        src1 = inputs[0]
        src2 = inputs[1]
        dst = outputs[0]
        assert isinstance(src1.type, MemRefType)
        n_val: int = src1.type.get_shape()[0]

        # Create arith.constant for n
        n_const = arith.ConstantOp(IntegerAttr(n_val, IndexType()))

        # Create npu.fvadd
        fvadd = FVAddOp(src1, src2, dst, n_const.result)

        rewriter.replace_matched_op([n_const, fvadd], [])


@dataclass(frozen=True)
class LinalgToNpuPass(ModulePass):
    """Lower linalg.generic ops on memrefs to NPU dialect operations."""

    name = "linalg-to-npu"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LinalgAddToNpuPattern()])
        ).rewrite_module(op)
