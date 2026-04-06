"""Shared utilities for lowering passes."""

from __future__ import annotations

from xdsl.dialects import linalg
from xdsl.ir import SSAValue


def find_preceding_fill(
    generic: linalg.GenericOp, out_val: SSAValue
) -> linalg.FillOp | None:
    """Walk backwards from ``generic`` for a ``linalg.fill`` writing ``out_val``."""
    cur = generic.prev_op
    while cur is not None:
        if isinstance(cur, linalg.FillOp):
            for o in cur.outputs:
                if o is out_val:
                    return cur
        cur = cur.prev_op
    return None
