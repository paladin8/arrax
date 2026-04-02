"""Lowering: array dialect -> linalg on tensors."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.linalg import IteratorTypeAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.array_dialect import AddOp


class AddToLinalgPattern(RewritePattern):
    """Rewrite array.add to linalg.generic with arith.addf body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        # Output init tensor (value-semantic, no allocation yet)
        empty = tensor.EmptyOp([], result_type)

        # Identity affine map: (d0) -> (d0) for each operand
        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 3  # ins0, ins1, outs0
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        # Body region: block with scalar args (a, b, out) -> addf -> yield
        scalar_types = [f32] * 3
        block = Block(arg_types=scalar_types)
        add = arith.AddfOp(block.args[0], block.args[1])
        yield_op = linalg.YieldOp(add.result)
        block.add_ops([add, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.lhs, op.rhs],
            outputs=[empty.tensor],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[result_type],
        )
        rewriter.replace_matched_op([empty, generic], [generic.res[0]])


@dataclass(frozen=True)
class ArrayToLinalgPass(ModulePass):
    """Lower array dialect ops to linalg.generic on tensors."""

    name = "array-to-linalg"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([AddToLinalgPattern()])
        ).rewrite_module(op)
