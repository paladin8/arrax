"""Full compiler pass pipeline: Python DSL -> assembly."""

from __future__ import annotations

from typing import Callable

from xdsl.context import Context

from arrax.codegen.asm_emitter import emit_assembly
from arrax.dsl.array import Array
from arrax.dsl.tracer import trace
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.buffer_optimize import BufferOptimizePass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.dsl_to_array import dsl_to_array
from arrax.lowering.fusion import FusionPass
from arrax.lowering.linalg_to_npu import LinalgToNpuPass
from arrax.lowering.npu_canonicalize import NpuCanonicalizePass
from arrax.lowering.tile import TilePass


def compile_to_asm(
    fn: Callable[..., Array],
    shapes: dict[str, tuple[int, ...]],
) -> tuple[str, list[str]]:
    """Full pipeline: trace -> lower -> emit assembly text.

    Returns (assembly_text, param_names).
    """
    dag, param_names = trace(fn, shapes)
    module = dsl_to_array(dag, param_names, shapes)

    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    TilePass().apply(ctx, module)
    FusionPass().apply(ctx, module)
    BufferOptimizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    NpuCanonicalizePass().apply(ctx, module)
    module.verify()

    asm = emit_assembly(module)
    return asm, param_names
