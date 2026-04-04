"""Lowering: array dialect -> linalg on tensors."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    FloatAttr,
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

from arrax.dialects.array_dialect import (
    AddOp,
    DivScalarOp,
    ExpOp,
    MulScalarOp,
    ReluOp,
    SubOp,
)


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


class SubToLinalgPattern(RewritePattern):
    """Rewrite array.sub to linalg.generic with arith.subf body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SubOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        empty = tensor.EmptyOp([], result_type)

        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 3
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        scalar_types = [f32] * 3
        block = Block(arg_types=scalar_types)
        sub = arith.SubfOp(block.args[0], block.args[1])
        yield_op = linalg.YieldOp(sub.result)
        block.add_ops([sub, yield_op])
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


class ReluToLinalgPattern(RewritePattern):
    """Rewrite array.relu to linalg.generic with arith.maximumf(x, 0.0) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReluOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        empty = tensor.EmptyOp([], result_type)

        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 2  # ins0, outs0
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        # Body: constant 0.0, maximumf(in, 0.0), yield
        scalar_types = [f32] * 2  # (in, out)
        block = Block(arg_types=scalar_types)
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        maximum = arith.MaximumfOp(block.args[0], zero.result)
        yield_op = linalg.YieldOp(maximum.result)
        block.add_ops([zero, maximum, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.input],
            outputs=[empty.tensor],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[result_type],
        )
        rewriter.replace_matched_op([empty, generic], [generic.res[0]])


class ExpToLinalgPattern(RewritePattern):
    """Rewrite array.exp to linalg.generic with math.exp body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExpOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        empty = tensor.EmptyOp([], result_type)

        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 2  # ins0, outs0
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        # Body: math.exp(in), yield
        scalar_types = [f32] * 2  # (in, out)
        block = Block(arg_types=scalar_types)
        exp_op = math.ExpOp(block.args[0])
        yield_op = linalg.YieldOp(exp_op.result)
        block.add_ops([exp_op, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.input],
            outputs=[empty.tensor],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[result_type],
        )
        rewriter.replace_matched_op([empty, generic], [generic.res[0]])


class MulScalarToLinalgPattern(RewritePattern):
    """Rewrite array.mul_scalar to linalg.generic with arith.mulf(x, const) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MulScalarOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        empty = tensor.EmptyOp([], result_type)

        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 2
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        scalar_types = [f32] * 2
        block = Block(arg_types=scalar_types)
        const = arith.ConstantOp(op.scalar)
        mul = arith.MulfOp(block.args[0], const.result)
        yield_op = linalg.YieldOp(mul.result)
        block.add_ops([const, mul, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.input],
            outputs=[empty.tensor],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[result_type],
        )
        rewriter.replace_matched_op([empty, generic], [generic.res[0]])


class DivScalarToLinalgPattern(RewritePattern):
    """Rewrite array.div_scalar to linalg.generic with arith.divf(x, const) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DivScalarOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        f32 = Float32Type()

        empty = tensor.EmptyOp([], result_type)

        ndims = len(result_type.get_shape())
        identity = AffineMap.identity(ndims)
        maps = [AffineMapAttr(identity)] * 2
        iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

        scalar_types = [f32] * 2
        block = Block(arg_types=scalar_types)
        const = arith.ConstantOp(op.scalar)
        div = arith.DivfOp(block.args[0], const.result)
        yield_op = linalg.YieldOp(div.result)
        block.add_ops([const, div, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[op.input],
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
            GreedyRewritePatternApplier([
                AddToLinalgPattern(),
                SubToLinalgPattern(),
                ReluToLinalgPattern(),
                ExpToLinalgPattern(),
                MulScalarToLinalgPattern(),
                DivScalarToLinalgPattern(),
            ])
        ).rewrite_module(op)
