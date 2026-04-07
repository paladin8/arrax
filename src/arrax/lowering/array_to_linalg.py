"""Lowering: array dialect -> linalg on tensors.

Each array dialect op is rewritten to one or more linalg.generic ops.
Composite ops (softmax, rmsnorm) decompose into sequences of generics.

Shared helpers factor out the four recurring linalg emission patterns:
  - _emit_binary_elementwise: 2-input parallel (add, sub)
  - _emit_unary_elementwise: 1-input parallel (exp, relu, scalar mul/div)
  - _emit_reduction: fill + reduction generic (sum, amax, dot, mean)
  - _emit_broadcast_binary: vec + scalar broadcast parallel (softmax sub/div)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import IteratorTypeAttr
from xdsl.ir import Block, Operation, Region, SSAValue
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
    AmaxOp,
    DivScalarOp,
    DotOp,
    ExpOp,
    MeanOp,
    MulScalarOp,
    RMSNormOp,
    ReluOp,
    SoftmaxOp,
    SubOp,
    SumOp,
)

_F32_NEG_INF = float("-inf")

# Type alias for body-builder callbacks.
# Takes a list of block args (SSAValue), returns (body_ops, yield_value).
BodyBuilder: TypeAlias = Callable[[list[SSAValue]], tuple[list[Operation], SSAValue]]


# ---------------------------------------------------------------------------
# Shared linalg emission helpers
# ---------------------------------------------------------------------------

def _emit_binary_elementwise(
    lhs: SSAValue,
    rhs: SSAValue,
    body_op_cls: type,
    result_type: TensorType,
) -> tuple[list[Operation], SSAValue]:
    """Emit a 2-input parallel linalg.generic with identity maps.

    Body: ``body_op_cls(lhs_scalar, rhs_scalar)`` → yield.
    Used by add, sub.
    """
    f32 = Float32Type()
    empty = tensor.EmptyOp([], result_type)
    ndims = len(result_type.get_shape())
    identity = AffineMap.identity(ndims)
    maps = [AffineMapAttr(identity)] * 3
    iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

    block = Block(arg_types=[f32, f32, f32])
    compute = body_op_cls(block.args[0], block.args[1])
    yield_op = linalg.YieldOp(compute.result)
    block.add_ops([compute, yield_op])

    generic = linalg.GenericOp(
        inputs=[lhs, rhs],
        outputs=[empty.tensor],
        body=Region([block]),
        indexing_maps=maps,
        iterator_types=iters,
        result_types=[result_type],
    )
    return [empty, generic], generic.res[0]


def _emit_unary_elementwise(
    input_val: SSAValue,
    body_builder: BodyBuilder,
    result_type: TensorType,
) -> tuple[list[Operation], SSAValue]:
    """Emit a 1-input parallel linalg.generic with identity maps.

    ``body_builder`` receives ``[in_scalar, out_scalar]`` and returns
    ``(body_ops, yield_value)``.
    Used by exp, relu, scalar mul/div.
    """
    f32 = Float32Type()
    empty = tensor.EmptyOp([], result_type)
    ndims = len(result_type.get_shape())
    identity = AffineMap.identity(ndims)
    maps = [AffineMapAttr(identity)] * 2
    iters = [IteratorTypeAttr.parallel() for _ in range(ndims)]

    block = Block(arg_types=[f32, f32])
    body_ops, result_val = body_builder(list(block.args))
    yield_op = linalg.YieldOp(result_val)
    block.add_ops([*body_ops, yield_op])

    generic = linalg.GenericOp(
        inputs=[input_val],
        outputs=[empty.tensor],
        body=Region([block]),
        indexing_maps=maps,
        iterator_types=iters,
        result_types=[result_type],
    )
    return [empty, generic], generic.res[0]


def _emit_reduction(
    inputs: list[SSAValue],
    identity_val: float,
    body_builder: BodyBuilder,
    result_type: TensorType,
) -> tuple[list[Operation], SSAValue]:
    """Emit a linalg.fill + reduction linalg.generic.

    ``inputs`` are rank-1 tensors; output is rank-0 (scalar sink).
    ``body_builder`` receives ``[*input_scalars, acc_scalar]`` and returns
    ``(body_ops, yield_value)``.
    Used by sum, amax, dot, mean, and softmax sub-reductions.
    """
    f32 = Float32Type()
    empty = tensor.EmptyOp([], result_type)
    init = arith.ConstantOp(FloatAttr(identity_val, f32))
    fill = linalg.FillOp(
        inputs=[init.result], outputs=[empty.tensor], res=[result_type],
    )

    d0 = AffineMap.identity(1)
    scalar_sink = AffineMap.from_callable(lambda d0: ())
    maps = [AffineMapAttr(d0)] * len(inputs) + [AffineMapAttr(scalar_sink)]
    iters = [IteratorTypeAttr.reduction()]

    n_inputs = len(inputs)
    block = Block(arg_types=[f32] * (n_inputs + 1))
    body_ops, result_val = body_builder(list(block.args))
    yield_op = linalg.YieldOp(result_val)
    block.add_ops([*body_ops, yield_op])

    generic = linalg.GenericOp(
        inputs=inputs,
        outputs=[fill.res[0]],
        body=Region([block]),
        indexing_maps=maps,
        iterator_types=iters,
        result_types=[result_type],
    )
    return [empty, init, fill, generic], generic.res[0]


def _emit_broadcast_binary(
    vec_input: SSAValue,
    scalar_input: SSAValue,
    body_op_cls: type,
    result_type: TensorType,
) -> tuple[list[Operation], SSAValue]:
    """Emit a parallel generic with one rank-1 and one rank-0 (broadcast) input.

    Body: ``body_op_cls(vec_scalar, broadcast_scalar)`` → yield.
    Used by softmax's sub-broadcast and div-broadcast steps.
    """
    f32 = Float32Type()
    d0 = AffineMap.identity(1)
    scalar_map = AffineMap.from_callable(lambda d0: ())

    empty = tensor.EmptyOp([], result_type)
    block = Block(arg_types=[f32, f32, f32])
    compute = body_op_cls(block.args[0], block.args[1])
    yield_op = linalg.YieldOp(compute.result)
    block.add_ops([compute, yield_op])

    generic = linalg.GenericOp(
        inputs=[vec_input, scalar_input],
        outputs=[empty.tensor],
        body=Region([block]),
        indexing_maps=[
            AffineMapAttr(d0), AffineMapAttr(scalar_map), AffineMapAttr(d0),
        ],
        iterator_types=[IteratorTypeAttr.parallel()],
        result_types=[result_type],
    )
    return [empty, generic], generic.res[0]


# ---------------------------------------------------------------------------
# Pattern classes (one per array dialect op)
# ---------------------------------------------------------------------------

class AddToLinalgPattern(RewritePattern):
    """Rewrite array.add to linalg.generic with arith.addf body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        ops, result = _emit_binary_elementwise(
            op.lhs, op.rhs, arith.AddfOp, result_type,
        )
        rewriter.replace_matched_op(ops, [result])


class SubToLinalgPattern(RewritePattern):
    """Rewrite array.sub to linalg.generic with arith.subf body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SubOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        ops, result = _emit_binary_elementwise(
            op.lhs, op.rhs, arith.SubfOp, result_type,
        )
        rewriter.replace_matched_op(ops, [result])


class ReluToLinalgPattern(RewritePattern):
    """Rewrite array.relu to linalg.generic with arith.maximumf(x, 0.0) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReluOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            zero = arith.ConstantOp(FloatAttr(0.0, Float32Type()))
            maximum = arith.MaximumfOp(args[0], zero.result)
            return [zero, maximum], maximum.result

        ops, result = _emit_unary_elementwise(op.input, body, result_type)
        rewriter.replace_matched_op(ops, [result])


class ExpToLinalgPattern(RewritePattern):
    """Rewrite array.exp to linalg.generic with math.exp body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExpOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            exp = math.ExpOp(args[0])
            return [exp], exp.result

        ops, result = _emit_unary_elementwise(op.input, body, result_type)
        rewriter.replace_matched_op(ops, [result])


class MulScalarToLinalgPattern(RewritePattern):
    """Rewrite array.mul_scalar to linalg.generic with arith.mulf(x, const) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MulScalarOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            const = arith.ConstantOp(op.scalar)
            mul = arith.MulfOp(args[0], const.result)
            return [const, mul], mul.result

        ops, result = _emit_unary_elementwise(op.input, body, result_type)
        ops[-1].attributes["arrax.facc"] = StringAttr("ephemeral")
        rewriter.replace_matched_op(ops, [result])


class DivScalarToLinalgPattern(RewritePattern):
    """Rewrite array.div_scalar to linalg.generic with arith.divf(x, const) body."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DivScalarOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            const = arith.ConstantOp(op.scalar)
            div = arith.DivfOp(args[0], const.result)
            return [const, div], div.result

        ops, result = _emit_unary_elementwise(op.input, body, result_type)
        ops[-1].attributes["arrax.facc"] = StringAttr("ephemeral")
        rewriter.replace_matched_op(ops, [result])


class SumToLinalgPattern(RewritePattern):
    """Rewrite array.sum to linalg.fill(0.0) + linalg.generic reduction."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: SumOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            # args: [in, acc]
            add = arith.AddfOp(args[1], args[0])
            return [add], add.result

        ops, result = _emit_reduction([op.input], 0.0, body, result_type)
        rewriter.replace_matched_op(ops, [result])


class AmaxToLinalgPattern(RewritePattern):
    """Rewrite array.amax to linalg.fill(-inf) + linalg.generic reduction."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AmaxOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            # args: [in, acc]
            maximum = arith.MaximumfOp(args[1], args[0])
            return [maximum], maximum.result

        ops, result = _emit_reduction(
            [op.input], _F32_NEG_INF, body, result_type,
        )
        rewriter.replace_matched_op(ops, [result])


class DotToLinalgPattern(RewritePattern):
    """Rewrite array.dot to linalg.fill(0.0) + linalg.generic reduction."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DotOp, rewriter: PatternRewriter) -> None:
        result_type = op.result.type
        assert isinstance(result_type, TensorType)

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            # args: [in_a, in_b, acc]
            mul = arith.MulfOp(args[0], args[1])
            add = arith.AddfOp(args[2], mul.result)
            return [mul, add], add.result

        ops, result = _emit_reduction(
            [op.lhs, op.rhs], 0.0, body, result_type,
        )
        ops[-1].attributes["arrax.facc"] = StringAttr("persistent")
        rewriter.replace_matched_op(ops, [result])


class MeanToLinalgPattern(RewritePattern):
    """Rewrite array.mean to linalg.fill(0.0) + linalg.generic reduction.

    The generic carries an ``arrax.mean_divisor`` discardable attribute
    holding the input length. Downstream passes read this to emit a
    trailing ``fdiv.s`` after the reduction loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MeanOp, rewriter: PatternRewriter) -> None:
        input_type = op.input.type
        assert isinstance(input_type, TensorType)
        result_type = op.result.type
        assert isinstance(result_type, TensorType)
        n = input_type.get_shape()[0]

        def body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            add = arith.AddfOp(args[1], args[0])
            return [add], add.result

        ops, result = _emit_reduction([op.input], 0.0, body, result_type)
        ops[-1].attributes["arrax.mean_divisor"] = IntegerAttr(n, i64)
        rewriter.replace_matched_op(ops, [result])


class SoftmaxToLinalgPattern(RewritePattern):
    """Decompose array.softmax into 5 linalg.generic ops.

    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    Emits: amax reduction, broadcast-sub, exp, sum reduction, broadcast-div.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: SoftmaxOp, rewriter: PatternRewriter
    ) -> None:
        input_type = op.input.type
        assert isinstance(input_type, TensorType)
        f32 = Float32Type()
        vec_type = input_type
        scalar_type = TensorType(f32, [])

        all_ops: list[Operation] = []

        def _sum_body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            add = arith.AddfOp(args[1], args[0])
            return [add], add.result

        def _amax_body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            maximum = arith.MaximumfOp(args[1], args[0])
            return [maximum], maximum.result

        def _exp_body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            exp = math.ExpOp(args[0])
            return [exp], exp.result

        # Step 1: max reduction
        max_ops, max_result = _emit_reduction(
            [op.input], _F32_NEG_INF, _amax_body, scalar_type,
        )
        all_ops.extend(max_ops)

        # Step 2: subtract max (broadcast)
        sub_ops, sub_result = _emit_broadcast_binary(
            op.input, max_result, arith.SubfOp, vec_type,
        )
        all_ops.extend(sub_ops)

        # Step 3: exp
        exp_ops, exp_result = _emit_unary_elementwise(
            sub_result, _exp_body, vec_type,
        )
        all_ops.extend(exp_ops)

        # Step 4: sum reduction
        sum_ops, sum_result = _emit_reduction(
            [exp_result], 0.0, _sum_body, scalar_type,
        )
        all_ops.extend(sum_ops)

        # Step 5: divide by sum (broadcast)
        div_ops, div_result = _emit_broadcast_binary(
            exp_result, sum_result, arith.DivfOp, vec_type,
        )
        all_ops.extend(div_ops)

        rewriter.replace_matched_op(all_ops, [div_result])


class RMSNormToLinalgPattern(RewritePattern):
    """Decompose array.rmsnorm into dot(x,x) + attributed broadcast-mul.

    rmsnorm(x) = x / sqrt(mean(x^2) + eps)

    Step 1: dot(x, x) reduction (sum of squares), tagged arrax.facc="persistent".
    Step 2: broadcast-mul with arrax.rmsnorm_divisor and arrax.rmsnorm_eps
    attributes. LinalgToNpu reads these to emit the scalar math (divf, addf,
    frsqrt) between loading the reduction result and executing fvmul.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: RMSNormOp, rewriter: PatternRewriter
    ) -> None:
        input_type = op.input.type
        assert isinstance(input_type, TensorType)
        f32 = Float32Type()
        n = input_type.get_shape()[0]
        vec_type = input_type
        scalar_type = TensorType(f32, [])

        all_ops: list[Operation] = []

        # Step 1: dot(x, x) — sum of squares
        def _dot_body(args: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
            # args: [in_a, in_b, acc]
            mul = arith.MulfOp(args[0], args[1])
            add = arith.AddfOp(args[2], mul.result)
            return [mul, add], add.result

        dot_ops, dot_result = _emit_reduction(
            [op.input, op.input], 0.0, _dot_body, scalar_type,
        )
        dot_ops[-1].attributes["arrax.facc"] = StringAttr("persistent")
        all_ops.extend(dot_ops)

        # Step 2: broadcast-mul with rmsnorm attributes
        mul_ops, mul_result = _emit_broadcast_binary(
            op.input, dot_result, arith.MulfOp, vec_type,
        )
        mul_ops[-1].attributes["arrax.rmsnorm_divisor"] = IntegerAttr(n, i64)
        mul_ops[-1].attributes["arrax.rmsnorm_eps"] = FloatAttr(1e-5, f32)
        all_ops.extend(mul_ops)

        rewriter.replace_matched_op(all_ops, [mul_result])


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
                SumToLinalgPattern(),
                AmaxToLinalgPattern(),
                DotToLinalgPattern(),
                MeanToLinalgPattern(),
                SoftmaxToLinalgPattern(),
                RMSNormToLinalgPattern(),
            ])
        ).rewrite_module(op)
