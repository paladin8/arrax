"""Full compiler pass pipeline: Python DSL -> assembly."""

from __future__ import annotations

from typing import Callable

from xdsl.context import Context

from arrax.codegen.asm_emitter import emit_assembly
from arrax.codegen.llvm_emitter import emit_llvm_ir
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
    backend: str = "asm",
) -> tuple[str, list[str]]:
    """Full pipeline: trace -> lower -> emit assembly or LLVM IR text.

    Args:
        fn: Python function using arrax DSL.
        shapes: Shape dict mapping parameter names to shapes.
        backend: ``"asm"`` (default) for RISC-V assembly, ``"llvm"`` for
            LLVM IR text (``.ll``).

    Returns (text, param_names) where text is assembly or LLVM IR.
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

    if backend == "asm":
        text = emit_assembly(module)
    elif backend == "llvm":
        text = emit_llvm_ir(module)
    else:
        raise ValueError(f"Unknown backend {backend!r}, expected 'asm' or 'llvm'")
    return text, param_names
