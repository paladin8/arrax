"""Buffer optimization: shrink and reuse intermediate allocations.

After fusion, intermediate buffers are only accessed within a single loop
iteration (one tile at a time). This pass:

1. Shrinks N-element allocs to tile-sized (64-element) allocs.
2. Rewrites subview offsets from the loop IV to 0 (data doesn't persist).
3. Merges non-overlapping intermediate allocs into a single allocation.

Pipeline placement: after Fuse, before LinalgToNpu.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    NoneAttr,
    StridedLayoutAttr,
)
from xdsl.ir import Block, Operation
from xdsl.passes import ModulePass

from arrax.dialects.npu_dialect import NPU_MAX_VEC_LEN


def _is_intermediate_alloc(alloc: memref.AllocOp, loop: scf.ForOp) -> bool:
    """Check if an alloc is only used via subviews inside the given loop."""
    loop_body = loop.body.blocks.first
    assert loop_body is not None

    for use in alloc.memref.uses:
        op = use.operation
        if not isinstance(op, memref.SubviewOp):
            return False
        # The subview must be inside this loop's body
        if op.parent != loop_body:
            return False
    return True


def _shrink_alloc(
    alloc: memref.AllocOp,
    c0: arith.ConstantOp,
) -> None:
    """Shrink an intermediate alloc to tile size and zero its subview offsets.

    Replaces memref<Nxf32> with memref<tile_size x f32> and rewrites all
    subviews from subview[%iv][%n][1] to subview[0][%n][1].
    """
    old_type = alloc.memref.type
    assert isinstance(old_type, MemRefType)
    old_size = old_type.get_shape()[0]
    tile_size = NPU_MAX_VEC_LEN
    if old_size <= tile_size:
        return  # already tile-sized or smaller
    new_type = MemRefType(old_type.element_type, [tile_size])

    # Replace the alloc with a smaller one
    new_alloc = memref.AllocOp([], [], new_type)
    alloc_parent = alloc.parent
    assert alloc_parent is not None
    alloc_parent.insert_op_before(new_alloc, alloc)
    alloc.memref.replace_all_uses_with(new_alloc.memref)
    alloc.detach()
    alloc.erase()

    # Rewrite subviews: offset becomes 0, source type changes
    for use in list(new_alloc.memref.uses):
        sv = use.operation
        if not isinstance(sv, memref.SubviewOp):
            continue
        # Build new subview with offset=0, same dynamic size, stride=1
        new_sv = memref.SubviewOp.get(
            source=new_alloc.memref,
            offsets=[c0.result],
            sizes=list(sv.sizes),
            strides=[1],
            result_type=MemRefType(
                old_type.element_type,
                [DYNAMIC_INDEX],
                StridedLayoutAttr([1], NoneAttr()),
            ),
        )
        sv_parent = sv.parent
        assert sv_parent is not None
        sv_parent.insert_op_before(new_sv, sv)
        sv.result.replace_all_uses_with(new_sv.result)
        sv.detach()
        sv.erase()


def _op_index(block: Block, op: Operation) -> int:
    """Return the positional index of an op within a block."""
    for i, block_op in enumerate(block.ops):
        if block_op is op:
            return i
    raise ValueError("op not found in block")


def _liveness_interval(
    alloc: memref.AllocOp, loop_body: Block
) -> tuple[int, int]:
    """Compute the liveness interval [first_use, last_use] for an alloc.

    Measured by op index within the loop body. Traces through subviews
    to find the linalg.generic ops that read/write the buffer.
    """
    first = len(list(loop_body.ops))
    last = 0
    for use in alloc.memref.uses:
        sv = use.operation
        if not isinstance(sv, memref.SubviewOp):
            continue
        # Find ops that use this subview
        for sv_use in sv.result.uses:
            user = sv_use.operation
            idx = _op_index(loop_body, user)
            first = min(first, idx)
            last = max(last, idx)
    assert first <= last, "alloc has subview uses but no transitive users"
    return (first, last)


def _reuse_buffers(
    allocs: list[memref.AllocOp], loop: scf.ForOp
) -> None:
    """Replace non-overlapping intermediate allocs with shared allocations.

    Uses greedy interval coloring: each alloc is assigned to the first
    reuse slot whose liveness doesn't overlap. Allocs in the same slot
    share one physical allocation.
    """
    if len(allocs) < 2:
        return

    loop_body = loop.body.blocks.first
    assert loop_body is not None

    # Compute liveness intervals
    intervals = [(alloc, _liveness_interval(alloc, loop_body)) for alloc in allocs]
    intervals.sort(key=lambda x: x[1][0])

    # Greedy interval coloring: each slot tracks its kept alloc and end time
    slots: list[tuple[memref.AllocOp, int]] = []  # (kept_alloc, slot_end)

    for alloc, (start, end) in intervals:
        assigned = False
        for i, (kept, slot_end) in enumerate(slots):
            if start >= slot_end:
                # Fits in this slot — reuse
                alloc.memref.replace_all_uses_with(kept.memref)
                alloc.detach()
                alloc.erase()
                slots[i] = (kept, end)
                assigned = True
                break
        if not assigned:
            # No existing slot fits — start a new one
            slots.append((alloc, end))


def _optimize_func(func_op: func.FuncOp) -> None:
    """Apply buffer shrinking and reuse to a function."""
    body = func_op.body.blocks.first
    if body is None:
        return

    # Find the fused loop (if any)
    loops = [op for op in body.ops if isinstance(op, scf.ForOp)]
    if not loops:
        return

    # Find intermediate allocs (used only via subviews inside a loop)
    for loop in loops:
        intermediates = [
            op for op in body.ops
            if isinstance(op, memref.AllocOp) and _is_intermediate_alloc(op, loop)
        ]
        if not intermediates:
            continue

        # Ensure a constant 0 exists for the new subview offsets
        c0 = arith.ConstantOp(IntegerAttr(0, IndexType()))
        body.insert_op_before(c0, intermediates[0])

        # Phase 1: shrink each intermediate
        for alloc in intermediates:
            _shrink_alloc(alloc, c0)

        # Phase 2: reuse non-overlapping buffers
        # Re-collect intermediates (shrink replaced the ops)
        new_intermediates = [
            op for op in body.ops
            if isinstance(op, memref.AllocOp) and _is_intermediate_alloc(op, loop)
        ]
        _reuse_buffers(new_intermediates, loop)


@dataclass(frozen=True)
class BufferOptimizePass(ModulePass):
    """Shrink and reuse intermediate buffer allocations."""

    name = "buffer-optimize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in op.body.block.ops:
            if isinstance(func_op, func.FuncOp):
                _optimize_func(func_op)
