"""Tests for arrax.dsl.tracer — DAG capture via tracing."""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array
from arrax.dsl.tracer import trace


class TestTrace:
    def test_basic_trace(self) -> None:
        result, params = trace(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})

        assert params == ["A", "B"]
        assert result.op == "add"
        assert result.operands[0].is_leaf
        assert result.operands[1].is_leaf
        assert result.operands[0].name == "A"
        assert result.operands[1].name == "B"

    def test_trace_shapes(self) -> None:
        result, _ = trace(lambda X, Y: X + Y, {"X": (512,), "Y": (512,)})
        assert result.shape == (512,)
        assert result.operands[0].shape == (512,)
        assert result.operands[1].shape == (512,)

    def test_trace_preserves_param_order(self) -> None:
        def kernel(C: Array, A: Array, B: Array) -> Array:
            return A + B

        _, params = trace(kernel, {"A": (10,), "B": (10,), "C": (10,)})
        assert params == ["C", "A", "B"]

    def test_trace_chained(self) -> None:
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        result, params = trace(
            kernel, {"A": (64,), "B": (64,), "C": (64,)}
        )
        assert params == ["A", "B", "C"]
        assert result.op == "add"
        inner = result.operands[0]
        assert inner.op == "add"
        assert inner.operands[0].name == "A"
        assert inner.operands[1].name == "B"
        assert result.operands[1].name == "C"

    def test_trace_with_def_function(self) -> None:
        def my_kernel(X: Array, Y: Array) -> Array:
            return X + Y

        result, params = trace(my_kernel, {"X": (256,), "Y": (256,)})
        assert params == ["X", "Y"]
        assert result.op == "add"

    def test_trace_missing_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="missing shape for parameter 'B'"):
            trace(lambda A, B: A + B, {"A": (10,)})

    def test_trace_unused_param(self) -> None:
        """Unused params still appear in param_names."""
        result, params = trace(
            lambda A, B, C: A + B, {"A": (10,), "B": (10,), "C": (10,)}
        )
        assert params == ["A", "B", "C"]
        assert result.op == "add"
        assert result.operands[0].name == "A"
        assert result.operands[1].name == "B"
