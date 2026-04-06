"""Tests for the reduction terminal-only validator.

Milestone 3 restricts reductions (sum, dot, amax, mean) to be the root of
the traced DAG — the function's return value. Any use as an operand to
another op raises ValueError. This test pins that restriction and exists
so that Milestone 4 can delete both the validator and this test together.
"""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array, amax, dot, mean, sum
from arrax.dsl.tracer import trace
from arrax.lowering.dsl_to_array import dsl_to_array


def _compile_to_array(fn, shapes: dict[str, tuple[int, ...]]) -> None:
    """Trace and lower to array dialect, invoking the validator."""
    result, params = trace(fn, shapes)
    dsl_to_array(result, params, shapes)


class TestAcceptedReductions:
    def test_bare_sum(self) -> None:
        _compile_to_array(lambda A: sum(A), {"A": (64,)})

    def test_sum_of_add(self) -> None:
        _compile_to_array(lambda A, B: sum(A + B), {"A": (64,), "B": (64,)})

    def test_sum_of_chain(self) -> None:
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return sum((A + B) - C)

        _compile_to_array(kernel, {"A": (64,), "B": (64,), "C": (64,)})

    def test_bare_amax(self) -> None:
        _compile_to_array(lambda A: amax(A), {"A": (64,)})

    def test_amax_of_sub(self) -> None:
        _compile_to_array(lambda A, B: amax(A - B), {"A": (64,), "B": (64,)})

    def test_bare_dot(self) -> None:
        _compile_to_array(lambda A, B: dot(A, B), {"A": (64,), "B": (64,)})

    def test_dot_of_add(self) -> None:
        _compile_to_array(
            lambda A, B, C: dot(A + B, C), {"A": (64,), "B": (64,), "C": (64,)}
        )

    def test_bare_mean(self) -> None:
        _compile_to_array(lambda A: mean(A), {"A": (64,)})

    def test_mean_of_add(self) -> None:
        _compile_to_array(lambda A, B: mean(A + B), {"A": (64,), "B": (64,)})


class TestRejectedReductions:
    def test_sum_fed_into_add_raises(self) -> None:
        """A + sum(A) — sum has a non-root user (the add)."""
        a = Array("A", (64,))
        # Build DAG manually so we can pass a non-terminal sum directly.
        s = sum(a)
        # A + s — add's second operand is a rank-0 value, a DAG violation for M3.
        root = Array(name="", shape=(64,))
        root.op = "add"
        root.operands = [a, s]
        with pytest.raises(ValueError, match="terminal"):
            dsl_to_array(root, ["A"], {"A": (64,)})

    def test_error_names_offending_reduction(self) -> None:
        a = Array("A", (64,))
        s = sum(a)
        root = Array(name="", shape=(64,))
        root.op = "add"
        root.operands = [a, s]
        with pytest.raises(ValueError, match="sum"):
            dsl_to_array(root, ["A"], {"A": (64,)})

    def test_amax_fed_into_add_raises(self) -> None:
        """A + amax(A) — amax has a non-root user (the add)."""
        a = Array("A", (64,))
        m = amax(a)
        root = Array(name="", shape=(64,))
        root.op = "add"
        root.operands = [a, m]
        with pytest.raises(ValueError, match="amax"):
            dsl_to_array(root, ["A"], {"A": (64,)})

    def test_dot_fed_into_add_raises(self) -> None:
        """A + dot(A, B) — dot has a non-root user."""
        a = Array("A", (64,))
        b = Array("B", (64,))
        d = dot(a, b)
        root = Array(name="", shape=(64,))
        root.op = "add"
        root.operands = [a, d]
        with pytest.raises(ValueError, match="dot"):
            dsl_to_array(root, ["A", "B"], {"A": (64,), "B": (64,)})

    def test_mean_fed_into_add_raises(self) -> None:
        """A + mean(A) — mean has a non-root user."""
        a = Array("A", (64,))
        m = mean(a)
        root = Array(name="", shape=(64,))
        root.op = "add"
        root.operands = [a, m]
        with pytest.raises(ValueError, match="mean"):
            dsl_to_array(root, ["A"], {"A": (64,)})
