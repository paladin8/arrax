"""Tiling: strip-mine linalg.generic ops to fit NPU vector length limit.

The NPU hardware limits vector operations to 256 bytes (64 f32 elements)
per instruction. This pass splits linalg.generic ops on memrefs into
scf.for loops over 64-element chunks using memref.subview.

Parallel generics (elementwise) produce an iter-args-free loop whose body
is a cloned linalg.generic on dynamic subviews of the original operands.

Reduction generics (rank-0 memref output) produce a loop whose iter_args
thread the scalar accumulator in SSA f32. Each tile allocates an inner
rank-0 scratch, fills it with the current accumulator, runs a chunk-sized
reduction into it, loads the updated accumulator out, and yields. A single
terminal memref.store writes the final value to the output memref after
the loop closes.

Ops with n <= NPU_MAX_VEC_LEN pass through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref, scf
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    Float32Type,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.dialects.linalg import IteratorType
from xdsl.ir import Block, Region, SSAValue
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
    """Strip-mine linalg.generic ops exceeding NPU vector length.

    Dispatches on iterator_types: all-parallel uses the elementwise path,
    a single 'reduction' iterator uses the scalar-accumulator iter_args
    path. Mixed iterator kinds are not handled (M3 has no 2D ops).
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        outputs = list(op.outputs)
        if len(outputs) != 1:
            return
        out_type = outputs[0].type
        if not isinstance(out_type, MemRefType):
            return

        iter_kinds = [a.data for a in op.iterator_types.data]
        if all(k == IteratorType.PARALLEL for k in iter_kinds):
            self._tile_parallel(op, rewriter, out_type)
        elif iter_kinds == [IteratorType.REDUCTION]:
            self._tile_reduction(op, rewriter, out_type)

    def _tile_parallel(
        self,
        op: linalg.GenericOp,
        rewriter: PatternRewriter,
        out_type: MemRefType,
    ) -> None:
        shape = out_type.get_shape()
        if len(shape) != 1:
            return
        n = shape[0]
        if n <= NPU_MAX_VEC_LEN:
            return

        inputs = list(op.inputs)
        outputs = list(op.outputs)

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

    def _tile_reduction(
        self,
        op: linalg.GenericOp,
        rewriter: PatternRewriter,
        out_type: MemRefType,
    ) -> None:
        """Strip-mine a 1D reduction into an scf.for with scalar iter_args.

        Expects exactly one rank-1 input and the pre-bufferize linalg.fill
        writing the identity into the rank-0 output. The fill is replaced
        by iter_args init; a terminal memref.store writes the final
        accumulator to the original output memref after the loop.
        """
        if len(out_type.get_shape()) != 0:
            return  # reduction must produce a rank-0 sink in M3
        if not isinstance(out_type.element_type, Float32Type):
            return

        inputs = list(op.inputs)
        if not inputs:
            return
        # All inputs must be rank-1 memrefs of the same length.
        src_type = inputs[0].type
        if not isinstance(src_type, MemRefType):
            return
        src_shape = src_type.get_shape()
        if len(src_shape) != 1:
            return
        n = src_shape[0]
        for inp in inputs[1:]:
            inp_type = inp.type
            if not isinstance(inp_type, MemRefType):
                return
            if inp_type.get_shape() != src_shape:
                return
        if n <= NPU_MAX_VEC_LEN:
            return  # fast path: untiled reduction handled by LinalgToNpu

        out_memref = list(op.outputs)[0]

        # Find the preceding linalg.fill that seeds the output; its scalar
        # input becomes the iter_args init. Raise if absent — the array->
        # linalg pattern always emits one.
        init_fill = _find_init_fill(op, out_memref)
        if init_fill is None:
            return
        init_val = list(init_fill.inputs)[0]

        index = IndexType()
        f32 = Float32Type()

        c0 = arith.ConstantOp(IntegerAttr(0, index))
        cn = arith.ConstantOp(IntegerAttr(n, index))
        cstep = arith.ConstantOp(IntegerAttr(NPU_MAX_VEC_LEN, index))

        # Body block args: (iv, acc)
        body_block = Block(arg_types=[index, f32])
        iv = body_block.args[0]
        acc = body_block.args[1]

        remaining = arith.SubiOp(cn.result, iv)
        chunk = arith.MinSIOp(cstep.result, remaining)

        # Subview each rank-1 input for this tile
        sub_inputs = []
        for inp in inputs:
            sv = memref.SubviewOp.get(
                source=inp,
                offsets=[iv],
                sizes=[chunk.result],
                strides=[1],
                result_type=_dynamic_subview_type(inp.type),
            )
            sub_inputs.append(sv)

        # Rank-0 scratch on the stack, seeded with the current accumulator
        scratch = memref.AllocaOp.get(f32, shape=[])
        inner_fill = linalg.FillOp(
            inputs=[acc],
            outputs=[scratch.memref],
            res=[],
        )

        # Inner reduction generic on the tile, writing into scratch
        new_body = op.body.clone()
        inner_generic = linalg.GenericOp(
            inputs=[sv.result for sv in sub_inputs],
            outputs=[scratch.memref],
            body=new_body,
            indexing_maps=op.indexing_maps,
            iterator_types=op.iterator_types,
            result_types=[],
        )

        new_acc = memref.LoadOp.get(scratch.memref, [])
        yield_op = scf.YieldOp(new_acc.res)

        body_block.add_ops(
            [remaining, chunk, *sub_inputs, scratch, inner_fill, inner_generic, new_acc, yield_op]
        )

        for_op = scf.ForOp(
            c0.result,
            cn.result,
            cstep.result,
            [init_val],
            Region([body_block]),
        )

        # Terminal store of the final accumulator to the output memref.
        final_store = memref.StoreOp.get(for_op.results[0], out_memref, [])

        # Erase the original fill — its role is taken over by iter_args init.
        rewriter.erase_op(init_fill)
        rewriter.replace_matched_op(
            [c0, cn, cstep, for_op, final_store], [],
        )


def _find_init_fill(
    generic: linalg.GenericOp, out_memref: SSAValue
) -> linalg.FillOp | None:
    """Walk backwards in the parent block for the fill that seeds the output."""
    parent = generic.parent
    if parent is None:
        return None
    cur = generic.prev_op
    while cur is not None:
        if isinstance(cur, linalg.FillOp):
            for o in cur.outputs:
                if o is out_memref:
                    return cur
        cur = cur.prev_op
    return None


@dataclass(frozen=True)
class TilePass(ModulePass):
    """Strip-mine linalg.generic ops to fit NPU vector length limit."""

    name = "tile"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([TileLinalgPattern()])
        ).rewrite_module(op)
