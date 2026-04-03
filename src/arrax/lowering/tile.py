"""Tiling: strip-mine linalg.generic ops to fit NPU vector length limit.

The NPU hardware limits vector operations to 256 bytes (64 f32 elements)
per instruction. This pass splits linalg.generic ops on memrefs into
scf.for loops over 64-element chunks using memref.subview.

Ops with n <= NPU_MAX_VEC_LEN pass through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref, scf
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.npu_dialect import NPU_MAX_VEC_LEN


def _dynamic_subview_type(source_type: MemRefType) -> MemRefType:
    """Build the result type for a 1D subview with dynamic offset and size.

    For memref<Nxf32>, the subview result is memref<?xf32, strided<[1], offset: ?>>.
    """
    return MemRefType(
        source_type.element_type,
        [DYNAMIC_INDEX],
        StridedLayoutAttr([1], NoneAttr()),
    )


class TileLinalgPattern(RewritePattern):
    """Strip-mine 1D linalg.generic ops exceeding NPU vector length."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # Only tile ops with memref operands (post-bufferization)
        outputs = list(op.outputs)
        if len(outputs) != 1:
            return
        out_type = outputs[0].type
        if not isinstance(out_type, MemRefType):
            return
        shape = out_type.get_shape()
        if len(shape) != 1:
            return

        n = shape[0]
        if n <= NPU_MAX_VEC_LEN:
            return

        inputs = list(op.inputs)

        # Constants for the loop bounds
        index = IndexType()
        c0 = arith.ConstantOp(IntegerAttr(0, index))
        cn = arith.ConstantOp(IntegerAttr(n, index))
        cstep = arith.ConstantOp(IntegerAttr(NPU_MAX_VEC_LEN, index))

        # Build loop body
        body_block = Block(arg_types=[index])
        iv = body_block.args[0]

        # %remaining = n - %i
        remaining = arith.SubiOp(cn.result, iv)
        # %chunk = min(step, remaining)
        chunk = arith.MinSIOp(cstep.result, remaining)

        subview_type = _dynamic_subview_type(out_type)

        # Create subviews for all inputs and the output
        subviews = []
        for memref_val in list(inputs) + list(outputs):
            sv = memref.SubviewOp.get(
                source=memref_val,
                offsets=[iv],
                sizes=[chunk.result],
                strides=[1],
                result_type=subview_type,
            )
            subviews.append(sv)

        # Clone the linalg.generic with subviewed operands
        new_body = op.body.clone()
        new_generic = linalg.GenericOp(
            inputs=[sv.result for sv in subviews[: len(inputs)]],
            outputs=[subviews[-1].result],
            body=new_body,
            indexing_maps=op.indexing_maps,
            iterator_types=op.iterator_types,
            result_types=[],
        )

        yield_op = scf.YieldOp()

        body_block.add_ops([remaining, chunk, *subviews, new_generic, yield_op])

        for_op = scf.ForOp(c0.result, cn.result, cstep.result, [], Region([body_block]))

        rewriter.replace_matched_op([c0, cn, cstep, for_op], [])


@dataclass(frozen=True)
class TilePass(ModulePass):
    """Strip-mine linalg.generic ops to fit NPU vector length limit."""

    name = "tile"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([TileLinalgPattern()])
        ).rewrite_module(op)
