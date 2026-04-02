"""Shared test helpers for pipeline construction."""

from __future__ import annotations

from xdsl.context import Context

from arrax.dsl.tracer import trace
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.dsl_to_array import dsl_to_array


def make_module(fn, shapes):
    """Trace fn and lower to array dialect IR."""
    result, params = trace(fn, shapes)
    return dsl_to_array(result, params, shapes)


def lower_to_linalg(module):
    """Apply array-to-linalg pass in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    module.verify()
    return module


def bufferize(module):
    """Apply array-to-linalg then bufferize in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    module.verify()
    return module
