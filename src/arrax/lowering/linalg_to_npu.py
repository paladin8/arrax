"""Lowering: linalg on memrefs -> npu dialect."""

from __future__ import annotations

from collections.abc import Callable
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
from xdsl.ir import Block, Operation, SSAValue
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
    FVMacOp,
    FVMaxOp,
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
       constant. The cluster collapses to a single npu reduction op whose
       ``acc_in`` is the identity and whose result is stored to the output
       memref after the op via ``memref.store``.

    2. Tiled (N > 64): the generic lives inside an ``scf.for`` body. Its
       ``outs`` is a per-iteration rank-0 ``memref.alloca`` seeded via
       ``linalg.fill(acc)`` where ``acc`` is the loop's iter_arg. A
       subsequent ``memref.load`` reads the new accumulator and feeds
       ``scf.yield``. The ``alloca + fill + generic + load`` cluster
       collapses to a single npu reduction op whose ``result`` replaces
       the load and flows directly into ``scf.yield``.

    Body-shape dispatch:
      - ``arith.addf(a, b), linalg.yield`` -> ``npu.fvreduce``  (sum/mean)
      - ``arith.maximumf(a, b), linalg.yield`` -> ``npu.fvmax``  (amax)
      - ``arith.mulf(a, b), arith.addf(acc, prod), linalg.yield`` -> ``npu.fvmac``  (dot)

    For single-input reductions, operand order inside the combiner is
    irrelevant: ``{a, b}`` must be the pair of block arguments.
    For dot, the two data block args feed mulf and the accumulator block arg
    feeds addf.
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

        # Rank-1 f32 memref inputs (1 for sum/amax, 2 for dot)
        inputs = list(op.inputs)
        if len(inputs) not in (1, 2):
            return
        for inp in inputs:
            inp_type = inp.type
            if not isinstance(inp_type, MemRefType):
                return
            if len(inp_type.get_shape()) != 1:
                return
            if not isinstance(inp_type.element_type, Float32Type):
                return

        # Dispatch on body shape to select the NPU op.
        body_block = op.body.blocks.first
        if body_block is None:
            return
        body_ops = list(body_block.ops)

        npu_op_builder = self._match_reduction_body(
            inputs, body_block, body_ops,
        )
        if npu_op_builder is None:
            return

        # Find the fill that seeds our outs (must be in the same block,
        # directly before the generic after the alloca in the tiled case).
        fill = _find_preceding_fill(op, out_val)
        if fill is None:
            return
        acc_in = list(fill.inputs)[0]

        # Extract the element count from the first input
        extracted = _extract_n(inputs[0], rewriter)
        if extracted is None:
            return
        extra_ops, n_ssa = extracted

        # Build the npu reduction op.
        reduction_op = npu_op_builder(n_ssa, acc_in)

        load = _find_following_load(op, out_val)
        if load is not None:
            # Tiled case: reduction_op.result takes the load's place.
            alloca_op: memref.AllocaOp | None = None
            if isinstance(out_val.owner, memref.AllocaOp):
                alloca_op = out_val.owner

            # 1. Replace generic with [extra_ops, reduction_op].
            rewriter.replace_matched_op([*extra_ops, reduction_op], [])
            # 2. Erase the fill (its role is subsumed by acc_in thread).
            rewriter.erase_op(fill)
            # 3. Replace the load with reduction_op.result and erase it.
            rewriter.replace_op(load, [], [reduction_op.result])
            # 4. Erase the now-dead scratch alloca.
            if alloca_op is not None and not alloca_op.memref.uses:
                rewriter.erase_op(alloca_op)
        else:
            # Untiled case: emit a terminal store of the result.
            store = memref.StoreOp.get(reduction_op.result, out_val, [])
            rewriter.replace_matched_op([*extra_ops, reduction_op, store], [])
            rewriter.erase_op(fill)


    @staticmethod
    def _match_reduction_body(
        inputs: list[SSAValue],
        body_block: Block,
        body_ops: list,
    ) -> Callable[[SSAValue, SSAValue], Operation] | None:
        """Match the body ops and return a builder ``(n, acc_in) -> npu_op``.

        Single-input (sum/amax): 2 ops — combiner(bb0, bb1), yield
        Two-input (dot):         3 ops — mulf(bb0, bb1), addf(bb2, prod), yield
        """
        bargs = list(body_block.args)

        if len(inputs) == 1 and len(body_ops) == 2:
            combiner = body_ops[0]
            if isinstance(combiner, arith.AddfOp):
                npu_op_cls: type = FVReduceOp
            elif isinstance(combiner, arith.MaximumfOp):
                npu_op_cls = FVMaxOp
            else:
                return None
            if not isinstance(body_ops[1], linalg.YieldOp):
                return None
            if {combiner.lhs, combiner.rhs} != set(bargs):
                return None
            if list(body_ops[1].operands) != [combiner.result]:
                return None
            src = inputs[0]
            return lambda n, acc: npu_op_cls(src, n, acc)

        if len(inputs) == 2 and len(body_ops) == 3:
            # dot: mulf(data_a, data_b), addf(acc, prod), yield
            mulf_op = body_ops[0]
            addf_op = body_ops[1]
            yield_op = body_ops[2]
            if not isinstance(mulf_op, arith.MulfOp):
                return None
            if not isinstance(addf_op, arith.AddfOp):
                return None
            if not isinstance(yield_op, linalg.YieldOp):
                return None
            # mulf operands must be the two data block args (bb0, bb1)
            if {mulf_op.lhs, mulf_op.rhs} != {bargs[0], bargs[1]}:
                return None
            # addf operands must be the acc block arg (bb2) and mulf result
            if {addf_op.lhs, addf_op.rhs} != {bargs[2], mulf_op.result}:
                return None
            if list(yield_op.operands) != [addf_op.result]:
                return None
            lhs, rhs = inputs[0], inputs[1]
            return lambda n, acc: FVMacOp(lhs, rhs, n, acc)

        return None


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
