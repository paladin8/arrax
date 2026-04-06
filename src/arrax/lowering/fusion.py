"""Fusion pass: merge adjacent scf.for loops with identical bounds.

Post-tiling, each elementwise op becomes its own scf.for loop. When two
loops iterate over the same range and the second consumes data produced
by the first, they can be fused into a single loop — reducing loop overhead
and enabling buffer optimizations.

Two fusion cases are handled:
  1. Parallel + parallel (M2): both loops have no iter_args. The second
     loop's body is appended to the first loop's body.
  2. Parallel + reduction (M3): the first loop has no iter_args, the second
     carries iter_args (scalar f32 accumulator). The first loop's body is
     spliced before the second loop's body; the second loop (with iter_args)
     survives.

A facc conflict guard prevents fusing two loops that both contain ops
tagged ``arrax.uses_facc`` (set by ArrayToLinalg for ops that require
exclusive use of the hardware facc register).

Pipeline placement: after Tile, before LinalgToNpu.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp
from xdsl.ir import Block, Operation, SSAValue
from xdsl.passes import ModulePass


def _get_const_int(val: SSAValue) -> int | None:
    """If val is defined by arith.constant, return its integer value."""
    if isinstance(val.owner, arith.ConstantOp):
        attr = val.owner.value
        if isinstance(attr, IntegerAttr):
            return attr.value.data
    return None


def _same_bounds(a: scf.ForOp, b: scf.ForOp) -> bool:
    """Check if two for loops have the same constant lb, ub, and step."""
    for va, vb in [(a.lb, b.lb), (a.ub, b.ub), (a.step, b.step)]:
        ca = _get_const_int(va)
        cb = _get_const_int(vb)
        if ca is None or cb is None or ca != cb:
            return False
    return True


def _fuse_loops(first: scf.ForOp, second: scf.ForOp) -> None:
    """Merge second loop's body into first loop's body.

    1. Remap second loop's IV to first loop's IV.
    2. Move body ops (except yield) into first loop before its yield.
    3. Erase the second loop and its now-dead bound constants.
    """
    first_body = first.body.blocks.first
    second_body = second.body.blocks.first
    assert first_body is not None
    assert second_body is not None

    # Collect second loop's bound constants BEFORE remapping (after remapping,
    # second.lb/ub/step return first's values since the operand slots are rewritten).
    bound_consts: list[arith.ConstantOp] = []
    for val in [second.lb, second.ub, second.step]:
        if isinstance(val.owner, arith.ConstantOp):
            bound_consts.append(val.owner)

    # Remap IV: all uses of second's IV now refer to first's IV
    second_body.args[0].replace_all_uses_with(first_body.args[0])

    # Remap bound references: second loop's lb/ub/step -> first loop's lb/ub/step.
    # SAFETY: This is a global replace. Safe because tiling creates fresh
    # constant ops per loop, so second.{lb,ub,step} are not shared with
    # any op outside the second loop.
    second.lb.replace_all_uses_with(first.lb)
    second.ub.replace_all_uses_with(first.ub)
    second.step.replace_all_uses_with(first.step)

    # Find first loop's yield (last op in body)
    first_yield = first_body.last_op
    assert isinstance(first_yield, scf.YieldOp)

    # Collect second loop's body ops (skip yield)
    ops_to_move = [op for op in second_body.ops if not isinstance(op, scf.YieldOp)]

    # Detach and insert before first's yield
    for op in ops_to_move:
        op.detach()
        first_body.insert_op_before(op, first_yield)

    # CSE: deduplicate redundant subi/minsi in the fused body.
    # After remapping, the second body's subi(ub, iv) and minsi(step, remaining)
    # have identical operands to the first body's — replace and erase duplicates.
    _cse_body(first_body)

    # Erase the second loop
    second.detach()
    second.erase()

    # Clean up dead bound constants (only if they have no remaining uses)
    for op in bound_consts:
        if not op.result.uses:
            op.detach()
            op.erase()


def _cse_key(op: Operation) -> tuple | None:
    """Return a hashable key for CSE, or None if the op is not eligible.

    An op is CSE-eligible if it has no regions (rules out linalg.generic,
    scf.for) and produces at least one result. The key includes the op name,
    operand identities, and all attributes/properties (to distinguish e.g.
    constants with different values or subviews with different static strides).
    """
    if op.regions or not op.results:
        return None
    attr_key = tuple(sorted(
        (k, str(v)) for k, v in
        (*op.attributes.items(), *op.properties.items())
    ))
    return (op.name, *(id(v) for v in op.operands), attr_key)


def _cse_body(block: Block) -> None:
    """Eliminate duplicate pure ops with identical operands in a block."""
    seen: dict[tuple[str | int, ...], Operation] = {}
    to_erase: list[Operation] = []

    for op in block.ops:
        key = _cse_key(op)
        if key is None:
            continue
        if key in seen:
            for old_res, new_res in zip(op.results, seen[key].results):
                old_res.replace_all_uses_with(new_res)
            to_erase.append(op)
        else:
            seen[key] = op

    for op in to_erase:
        op.detach()
        op.erase()


def _fuse_parallel_into_reduction(
    parallel: scf.ForOp, reduction: scf.ForOp
) -> None:
    """Merge a parallel (no iter_args) loop's body into a reduction loop.

    The parallel body is spliced before the reduction body. The reduction
    loop survives with its iter_args intact. The parallel loop is erased.
    """
    par_body = parallel.body.blocks.first
    red_body = reduction.body.blocks.first
    assert par_body is not None
    assert red_body is not None

    # Collect parallel loop's bound constants before remapping.
    bound_consts: list[arith.ConstantOp] = []
    for val in [parallel.lb, parallel.ub, parallel.step]:
        if isinstance(val.owner, arith.ConstantOp):
            bound_consts.append(val.owner)

    # Remap: parallel IV → reduction IV
    par_body.args[0].replace_all_uses_with(red_body.args[0])

    # Remap bounds: parallel's lb/ub/step → reduction's
    parallel.lb.replace_all_uses_with(reduction.lb)
    parallel.ub.replace_all_uses_with(reduction.ub)
    parallel.step.replace_all_uses_with(reduction.step)

    # Collect parallel body ops (skip yield)
    ops_to_move = [op for op in par_body.ops if not isinstance(op, scf.YieldOp)]

    # Insert at the top of the reduction body (before first existing op)
    first_red_op = red_body.first_op
    for op in ops_to_move:
        op.detach()
        red_body.insert_op_before(op, first_red_op)

    # CSE: deduplicate subi/minsi/subview pairs that now have identical operands.
    _cse_body(red_body)

    # Erase the parallel loop.
    parallel.detach()
    parallel.erase()

    # Clean up dead bound constants.
    for op in bound_consts:
        if not op.result.uses:
            op.detach()
            op.erase()


def _has_facc_conflict(first: scf.ForOp, second: scf.ForOp) -> bool:
    """Check if fusing would create a facc register conflict.

    Ops tagged with the ``arrax.uses_facc`` discardable attribute (set by
    ArrayToLinalg) require exclusive use of the hardware facc register.
    If both loops contain such ops, fusing them would corrupt facc.
    """
    def _uses_facc(loop: scf.ForOp) -> bool:
        body = loop.body.blocks.first
        if body is None:
            return False
        return any(
            isinstance(op, linalg.GenericOp)
            and "arrax.uses_facc" in op.attributes
            for op in body.ops
        )

    return _uses_facc(first) and _uses_facc(second)


def _fuse_block(block: Block) -> bool:
    """Fuse adjacent scf.for loops in a block. Returns True if any fusion happened."""
    fused = False
    changed = True
    while changed:
        changed = False
        ops = list(block.ops)
        for i in range(len(ops)):
            first = ops[i]
            if not isinstance(first, scf.ForOp):
                continue
            # Look ahead past intervening non-loop ops (constants, allocs)
            j = i + 1
            while j < len(ops) and not isinstance(ops[j], scf.ForOp):
                if not isinstance(ops[j], (arith.ConstantOp, memref.AllocOp)):
                    break  # non-skippable op — stop looking
                j += 1
            second = ops[j] if j < len(ops) else None
            if not isinstance(second, scf.ForOp):
                continue
            if not _same_bounds(first, second):
                continue

            # Case 1: both parallel (no iter_args) — M2 parallel-parallel fusion.
            if not first.iter_args and not second.iter_args:
                for k in range(i + 1, j):
                    if isinstance(ops[k], memref.AllocOp):
                        ops[k].detach()
                        block.insert_op_before(ops[k], first)
                _fuse_loops(first, second)
                changed = True
                fused = True
                break

            # Case 2: parallel → reduction fusion.
            # First has no iter_args, second carries iter_args.
            if not first.iter_args and second.iter_args:
                if _has_facc_conflict(first, second):
                    continue
                for k in range(i + 1, j):
                    if isinstance(ops[k], memref.AllocOp):
                        ops[k].detach()
                        block.insert_op_before(ops[k], first)
                _fuse_parallel_into_reduction(first, second)
                changed = True
                fused = True
                break

            # Other iter_args combinations (reduction→parallel, reduction→reduction)
            # are not fused in M3.
    return fused


@dataclass(frozen=True)
class FusionPass(ModulePass):
    """Fuse adjacent scf.for loops with identical iteration bounds."""

    name = "fusion"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in op.body.block.ops:
            if not func_op.regions:
                continue
            body = func_op.regions[0].blocks.first
            if body is not None:
                _fuse_block(body)
