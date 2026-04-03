"""Lowering: linalg on memrefs -> npu dialect."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.dialects.linalg import IteratorType
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.npu_dialect import FVAddOp, FVSubOp


class LinalgElementwiseToNpuPattern(RewritePattern):
    """Match linalg.generic elementwise ops on memrefs and replace with NPU ops."""

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
        if op.iterator_types.data[0].data != IteratorType.PARALLEL:
            return

        # Body must be exactly: one compute op + yield
        body_block = op.body.blocks.first
        assert body_block is not None
        body_ops = list(body_block.ops)
        if len(body_ops) != 2:
            return
        if not isinstance(body_ops[1], linalg.YieldOp):
            return

        # Determine which NPU op to emit based on the body compute op
        compute_op = body_ops[0]
        if isinstance(compute_op, arith.AddfOp):
            npu_op_cls = FVAddOp
        elif isinstance(compute_op, arith.SubfOp):
            npu_op_cls = FVSubOp
        else:
            return

        # Extract n: static shape → constant, dynamic shape → subview size
        src1 = inputs[0]
        src2 = inputs[1]
        dst = outputs[0]
        assert isinstance(src1.type, MemRefType)
        shape_dim: int = src1.type.get_shape()[0]

        if shape_dim != DYNAMIC_INDEX:
            n_const = arith.ConstantOp(IntegerAttr(shape_dim, IndexType()))
            npu_op = npu_op_cls(src1, src2, dst, n_const.result)
            rewriter.replace_matched_op([n_const, npu_op], [])
        else:
            if not isinstance(src1.owner, memref.SubviewOp):
                return
            n_ssa = src1.owner.sizes[0]
            npu_op = npu_op_cls(src1, src2, dst, n_ssa)
            rewriter.replace_matched_op([npu_op], [])


@dataclass(frozen=True)
class LinalgToNpuPass(ModulePass):
    """Lower linalg.generic ops on memrefs to NPU dialect operations."""

    name = "linalg-to-npu"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LinalgElementwiseToNpuPattern()])
        ).rewrite_module(op)
