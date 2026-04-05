"""Fusion pass: merge adjacent scf.for loops with identical bounds.

Post-tiling, each elementwise op becomes its own scf.for loop. When two
loops iterate over the same range and the second consumes data produced
by the first, they can be fused into a single loop — reducing loop overhead
and enabling buffer optimizations (Phase 4).

Pipeline placement: after Tile, before LinalgToNpu.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, memref, scf
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
            # Safety: refuse to fuse if either loop carries iter_args.
            # Parallel→reduction fusion (matching bounds, different iter_arg
            # shapes) lands in Phase 5; Phase 1 must not silently merge a
            # parallel elementwise loop with a reduction loop whose
            # iter_args and scf.yield would be dropped on the floor.
            if first.iter_args or second.iter_args:
                continue
            # Hoist any alloc ops between the loops to before first
            for k in range(i + 1, j):
                if isinstance(ops[k], memref.AllocOp):
                    ops[k].detach()
                    block.insert_op_before(ops[k], first)
            _fuse_loops(first, second)
            changed = True
            fused = True
            break
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
