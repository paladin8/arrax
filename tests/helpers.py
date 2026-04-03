"""Shared test helpers for pipeline construction."""

from __future__ import annotations

from typing import Callable

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from arrax.dsl.array import Array
from arrax.dsl.tracer import trace
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.dsl_to_array import dsl_to_array
from arrax.lowering.tile import TilePass


def make_module(
    fn: Callable[..., Array], shapes: dict[str, tuple[int, ...]]
) -> ModuleOp:
    """Trace fn and lower to array dialect IR."""
    result, params = trace(fn, shapes)
    return dsl_to_array(result, params, shapes)


def lower_to_linalg(module: ModuleOp) -> ModuleOp:
    """Apply array-to-linalg pass in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    module.verify()
    return module


def bufferize(module: ModuleOp) -> ModuleOp:
    """Apply array-to-linalg then bufferize in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    module.verify()
    return module


def tile(module: ModuleOp) -> ModuleOp:
    """Apply array-to-linalg, bufferize, then tile in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    TilePass().apply(ctx, module)
    module.verify()
    return module
