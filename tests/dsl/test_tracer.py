"""Tests for arrax.dsl.tracer — DAG capture via tracing."""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array, amax, dot, mean, sum
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

    def test_trace_sum(self) -> None:
        """sum(A) produces a rank-0 root node."""
        result, params = trace(lambda A: sum(A), {"A": (128,)})
        assert params == ["A"]
        assert result.op == "sum"
        assert result.shape == ()
        assert result.operands[0].is_leaf
        assert result.operands[0].name == "A"

    def test_trace_sum_of_add(self) -> None:
        """sum(A + B) produces a two-level DAG with a rank-0 root."""
        result, params = trace(lambda A, B: sum(A + B), {"A": (64,), "B": (64,)})
        assert params == ["A", "B"]
        assert result.op == "sum"
        assert result.shape == ()
        inner = result.operands[0]
        assert inner.op == "add"

    def test_trace_amax(self) -> None:
        """amax(A) produces a rank-0 root node."""
        result, params = trace(lambda A: amax(A), {"A": (128,)})
        assert params == ["A"]
        assert result.op == "amax"
        assert result.shape == ()
        assert result.operands[0].is_leaf
        assert result.operands[0].name == "A"

    def test_trace_amax_of_sub(self) -> None:
        """amax(A - B) produces a two-level DAG with a rank-0 root."""
        result, params = trace(lambda A, B: amax(A - B), {"A": (64,), "B": (64,)})
        assert params == ["A", "B"]
        assert result.op == "amax"
        assert result.shape == ()
        inner = result.operands[0]
        assert inner.op == "sub"

    def test_trace_dot(self) -> None:
        """dot(A, B) produces a rank-0 root node with two operands."""
        result, params = trace(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        assert params == ["A", "B"]
        assert result.op == "dot"
        assert result.shape == ()
        assert len(result.operands) == 2
        assert result.operands[0].is_leaf
        assert result.operands[1].is_leaf

    def test_trace_dot_of_add(self) -> None:
        """dot(A + B, A - B) produces a rank-0 root with elementwise inputs."""
        result, params = trace(
            lambda A, B: dot(A + B, A - B), {"A": (64,), "B": (64,)}
        )
        assert params == ["A", "B"]
        assert result.op == "dot"
        assert result.shape == ()
        assert result.operands[0].op == "add"
        assert result.operands[1].op == "sub"

    def test_trace_mean(self) -> None:
        """mean(A) produces a rank-0 root node."""
        result, params = trace(lambda A: mean(A), {"A": (128,)})
        assert params == ["A"]
        assert result.op == "mean"
        assert result.shape == ()
        assert len(result.operands) == 1
        assert result.operands[0].is_leaf
