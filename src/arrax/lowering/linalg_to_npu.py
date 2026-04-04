"""Lowering: linalg on memrefs -> npu dialect."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.dialects.linalg import IteratorType
from xdsl.ir import SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.npu_dialect import FVAddOp, FVExpOp, FVReluOp, FVSubOp


def _extract_n(
    first_input: SSAValue,
    rewriter: PatternRewriter,
) -> tuple[list[arith.ConstantOp], SSAValue] | None:
    """Extract the element count n from the first input operand.

    Returns (extra_ops, n_ssa) or None if extraction fails.
    Static shape: creates an arith.constant.
    Dynamic shape (post-tiling): traces to SubviewOp.sizes[0].
    """
    assert isinstance(first_input.type, MemRefType)
    shape_dim: int = first_input.type.get_shape()[0]

    if shape_dim != DYNAMIC_INDEX:
        n_const = arith.ConstantOp(IntegerAttr(shape_dim, IndexType()))
        return [n_const], n_const.result
    else:
        if not isinstance(first_input.owner, memref.SubviewOp):
            return None
        return [], first_input.owner.sizes[0]


def _is_1d_parallel_memref(op: linalg.GenericOp) -> bool:
    """Check that the generic is 1D, parallel, with identity maps on memrefs."""
    inputs = list(op.inputs)
    outputs = list(op.outputs)

    for operand in inputs + outputs:
        if not isinstance(operand.type, MemRefType):
            return False
        if len(operand.type.get_shape()) != 1:
            return False

    identity_1d = AffineMap.identity(1)
    for map_attr in op.indexing_maps.data:
        if map_attr.data != identity_1d:
            return False

    if len(op.iterator_types.data) != 1:
        return False
    if op.iterator_types.data[0].data != IteratorType.PARALLEL:
        return False

    return True


class LinalgElementwiseToNpuPattern(RewritePattern):
    """Match linalg.generic elementwise ops on memrefs and replace with NPU ops."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        if not _is_1d_parallel_memref(op):
            return

        inputs = list(op.inputs)
        outputs = list(op.outputs)
        if len(outputs) != 1:
            return

        body_block = op.body.blocks.first
        assert body_block is not None
        body_ops = list(body_block.ops)

        if len(inputs) == 2:
            self._match_binary(op, inputs, outputs, body_ops, rewriter)
        elif len(inputs) == 1:
            self._match_unary(op, inputs, outputs, body_ops, rewriter)

    def _match_binary(
        self,
        op: linalg.GenericOp,
        inputs: list[SSAValue],
        outputs: list[SSAValue],
        body_ops: list,
        rewriter: PatternRewriter,
    ) -> None:
        """Match 2-input generics: addf → FVAdd, subf → FVSub."""
        if len(body_ops) != 2:
            return
        if not isinstance(body_ops[1], linalg.YieldOp):
            return

        compute_op = body_ops[0]
        if isinstance(compute_op, arith.AddfOp):
            npu_op_cls = FVAddOp
        elif isinstance(compute_op, arith.SubfOp):
            npu_op_cls = FVSubOp
        else:
            return

        result = _extract_n(inputs[0], rewriter)
        if result is None:
            return
        extra_ops, n_ssa = result

        npu_op = npu_op_cls(inputs[0], inputs[1], outputs[0], n_ssa)
        rewriter.replace_matched_op([*extra_ops, npu_op], [])

    def _match_unary(
        self,
        op: linalg.GenericOp,
        inputs: list[SSAValue],
        outputs: list[SSAValue],
        body_ops: list,
        rewriter: PatternRewriter,
    ) -> None:
        """Match 1-input generics: maximumf(x,0)→FVRelu, exp→FVExp."""
        # Relu: constant 0.0, maximumf, yield (3 ops)
        if len(body_ops) == 3:
            if (
                isinstance(body_ops[0], arith.ConstantOp)
                and isinstance(body_ops[0].value, FloatAttr)
                and body_ops[0].value.value.data == 0.0
                and isinstance(body_ops[1], arith.MaximumfOp)
                and isinstance(body_ops[2], linalg.YieldOp)
            ):
                npu_op_cls = FVReluOp
            else:
                return
        # Exp: math.exp, yield (2 ops)
        elif len(body_ops) == 2:
            if not isinstance(body_ops[1], linalg.YieldOp):
                return
            if isinstance(body_ops[0], math.ExpOp):
                npu_op_cls = FVExpOp
            else:
                return
        else:
            return

        result = _extract_n(inputs[0], rewriter)
        if result is None:
            return
        extra_ops, n_ssa = result

        npu_op = npu_op_cls(inputs[0], outputs[0], n_ssa)
        rewriter.replace_matched_op([*extra_ops, npu_op], [])


@dataclass(frozen=True)
class LinalgToNpuPass(ModulePass):
    """Lower linalg.generic ops on memrefs to NPU dialect operations."""

    name = "linalg-to-npu"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LinalgElementwiseToNpuPattern()])
        ).rewrite_module(op)
