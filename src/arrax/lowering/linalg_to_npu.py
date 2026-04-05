"""Lowering: linalg on memrefs -> npu dialect."""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, math, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    Float32Type,
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

from arrax.dialects.npu_dialect import (
    FVAddOp,
    FVDivOp,
    FVExpOp,
    FVMulOp,
    FVReduceOp,
    FVReluOp,
    FVSubOp,
)


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
        """Match 1-input generics to NPU ops.

        Patterns:
        - [constant 0.0, maximumf, yield] → FVRelu
        - [constant, mulf, yield]         → FVMul (scalar-vector)
        - [constant, divf, yield]         → FVDiv (scalar-vector)
        - [math.exp, yield]               → FVExp
        """
        scalar_val: float | None = None

        if len(body_ops) == 3:
            if not isinstance(body_ops[0], arith.ConstantOp):
                return
            if not isinstance(body_ops[0].value, FloatAttr):
                return
            if not isinstance(body_ops[2], linalg.YieldOp):
                return
            const_val = body_ops[0].value.value.data
            compute = body_ops[1]

            if isinstance(compute, arith.MaximumfOp) and const_val == 0.0:
                npu_op_type = "relu"
            elif isinstance(compute, arith.MulfOp):
                npu_op_type = "mul_scalar"
                scalar_val = const_val
            elif isinstance(compute, arith.DivfOp):
                npu_op_type = "div_scalar"
                scalar_val = const_val
            else:
                return
        elif len(body_ops) == 2:
            if not isinstance(body_ops[1], linalg.YieldOp):
                return
            if isinstance(body_ops[0], math.ExpOp):
                npu_op_type = "exp"
            else:
                return
        else:
            return

        result = _extract_n(inputs[0], rewriter)
        if result is None:
            return
        extra_ops, n_ssa = result

        if npu_op_type == "relu":
            npu_op = FVReluOp(inputs[0], outputs[0], n_ssa)
        elif npu_op_type == "exp":
            npu_op = FVExpOp(inputs[0], outputs[0], n_ssa)
        elif npu_op_type == "mul_scalar":
            assert scalar_val is not None
            npu_op = FVMulOp(inputs[0], outputs[0], n_ssa, scalar_val)
        elif npu_op_type == "div_scalar":
            assert scalar_val is not None
            npu_op = FVDivOp(inputs[0], outputs[0], n_ssa, scalar_val)
        else:
            return

        rewriter.replace_matched_op([*extra_ops, npu_op], [])


class LinalgReductionToNpuPattern(RewritePattern):
    """Match a rank-0 reduction cluster and collapse to an npu reduction op.

    Two input shapes are matched, anchored on the reduction linalg.generic:

    1. Untiled (N <= 64): the generic is in straight-line code with a
       preceding ``linalg.fill`` seeding its rank-0 output with the identity
       constant. The cluster collapses to a single ``npu.fvreduce`` whose
       ``acc_in`` is the identity and whose result is stored to the output
       memref after the op via ``memref.store``.

    2. Tiled (N > 64): the generic lives inside an ``scf.for`` body. Its
       ``outs`` is a per-iteration rank-0 ``memref.alloca`` seeded via
       ``linalg.fill(acc)`` where ``acc`` is the loop's iter_arg. A
       subsequent ``memref.load`` reads the new accumulator and feeds
       ``scf.yield``. The ``alloca + fill + generic + load`` cluster
       collapses to a single ``npu.fvreduce`` whose ``result`` replaces
       the load and flows directly into ``scf.yield``.

    Body shape currently handled: ``arith.addf(a, b), linalg.yield`` where
    ``{a, b}`` is the pair of block arguments (order irrelevant). That is
    the sum body; amax/dot/mean will be added in later phases.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # Single reduction iterator
        if len(op.iterator_types.data) != 1:
            return
        if op.iterator_types.data[0].data != IteratorType.REDUCTION:
            return

        # Rank-0 f32 memref output
        outputs = list(op.outputs)
        if len(outputs) != 1:
            return
        out_val = outputs[0]
        out_type = out_val.type
        if not isinstance(out_type, MemRefType):
            return
        if len(out_type.get_shape()) != 0:
            return
        if not isinstance(out_type.element_type, Float32Type):
            return

        # Single rank-1 f32 memref input (sum/amax/mean form)
        inputs = list(op.inputs)
        if len(inputs) != 1:
            return
        src = inputs[0]
        src_type = src.type
        if not isinstance(src_type, MemRefType):
            return
        if len(src_type.get_shape()) != 1:
            return
        if not isinstance(src_type.element_type, Float32Type):
            return

        # Body must be: addf(bb0, bb1), yield
        body_block = op.body.blocks.first
        if body_block is None:
            return
        body_ops = list(body_block.ops)
        if len(body_ops) != 2:
            return
        if not isinstance(body_ops[0], arith.AddfOp):
            return
        if not isinstance(body_ops[1], linalg.YieldOp):
            return
        addf = body_ops[0]
        bargs = set(body_block.args)
        if {addf.lhs, addf.rhs} != bargs:
            return
        yield_op = body_ops[1]
        if list(yield_op.operands) != [addf.result]:
            return

        # Find the fill that seeds our outs (must be in the same block,
        # directly before the generic after the alloca in the tiled case).
        fill = _find_preceding_fill(op, out_val)
        if fill is None:
            return
        acc_in = list(fill.inputs)[0]

        # Extract the element count
        extracted = _extract_n(src, rewriter)
        if extracted is None:
            return
        extra_ops, n_ssa = extracted

        # Build the npu.fvreduce op
        fvreduce = FVReduceOp(src, n_ssa, acc_in)

        load = _find_following_load(op, out_val)
        if load is not None:
            # Tiled case: fvreduce.result takes the load's place.
            alloca_op: memref.AllocaOp | None = None
            if isinstance(out_val.owner, memref.AllocaOp):
                alloca_op = out_val.owner

            # 1. Replace generic with [extra_ops, fvreduce].
            rewriter.replace_matched_op([*extra_ops, fvreduce], [])
            # 2. Erase the fill (its role is subsumed by acc_in thread).
            rewriter.erase_op(fill)
            # 3. Replace the load with fvreduce.result and erase it.
            rewriter.replace_op(load, [], [fvreduce.result])
            # 4. Erase the now-dead scratch alloca.
            if alloca_op is not None and not alloca_op.memref.uses:
                rewriter.erase_op(alloca_op)
        else:
            # Untiled case: emit a terminal store of the fvreduce result.
            store = memref.StoreOp.get(fvreduce.result, out_val, [])
            rewriter.replace_matched_op([*extra_ops, fvreduce, store], [])
            rewriter.erase_op(fill)


def _find_preceding_fill(
    generic: linalg.GenericOp, out_val: SSAValue
) -> linalg.FillOp | None:
    """Walk backwards in the parent block for a linalg.fill writing out_val."""
    cur = generic.prev_op
    while cur is not None:
        if isinstance(cur, linalg.FillOp):
            for o in cur.outputs:
                if o is out_val:
                    return cur
        cur = cur.prev_op
    return None


def _find_following_load(
    generic: linalg.GenericOp, out_val: SSAValue
) -> memref.LoadOp | None:
    """Walk forwards in the parent block for a memref.load from out_val."""
    cur = generic.next_op
    while cur is not None:
        if isinstance(cur, memref.LoadOp) and cur.memref is out_val:
            return cur
        cur = cur.next_op
    return None


@dataclass(frozen=True)
class LinalgToNpuPass(ModulePass):
    """Lower linalg.generic ops on memrefs to NPU dialect operations."""

    name = "linalg-to-npu"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [LinalgElementwiseToNpuPattern(), LinalgReductionToNpuPattern()]
            )
        ).rewrite_module(op)
