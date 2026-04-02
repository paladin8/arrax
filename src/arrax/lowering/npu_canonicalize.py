"""Canonicalization pass for the NPU dialect.

Applies peephole optimizations to NPU ops before assembly emission.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from arrax.dialects.npu_dialect import FVAddOp


class FVAddSwapForInPlace(RewritePattern):
    """Swap fvadd operands when src1 == dst to enable in-place execution.

    Addition is commutative. The hardware writes in-place to rs2 (src2/dst).
    If src1 is the same buffer as dst but src2 is not, swapping src1 and src2
    makes src2 == dst, which lets the asm emitter skip the copy loop.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FVAddOp, rewriter: PatternRewriter) -> None:
        if op.src1 is op.dst and op.src2 is not op.dst:
            new_op = FVAddOp(op.src2, op.src1, op.dst, op.n)
            rewriter.replace_matched_op(new_op)


@dataclass(frozen=True)
class NpuCanonicalizePass(ModulePass):
    """Canonicalize NPU dialect ops for more efficient code generation."""

    name = "npu-canonicalize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([FVAddSwapForInPlace()])
        ).rewrite_module(op)
