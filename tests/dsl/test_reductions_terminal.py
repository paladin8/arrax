"""Tests for reduction usage patterns in dsl_to_array.

M3 restricted reductions to terminal (root-only) positions. M4 lifts that
restriction — reductions can now appear anywhere in the DAG and feed into
subsequent ops. This file tests both terminal (accepted since M3) and
non-terminal (accepted since M4) patterns.
"""

from __future__ import annotations

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


class TestNonTerminalReductions:
    """M4: non-terminal reductions pass through dsl_to_array without raising.

    The validator that blocked these patterns in M3 is removed. The resulting
    IR may contain shape mismatches at the array dialect level (e.g., AddOp
    with rank-1 and rank-0 operands) — that's expected. Non-terminal reductions
    are used internally by composite op decompositions (softmax, rmsnorm) at
    the linalg level, not at the array dialect level.
    """

    def _build_non_terminal(
        self, reduction_node: Array, other: Array, params: list[str], shapes: dict
    ) -> None:
        """Build a DAG where a reduction feeds into an add, then lower."""
        root = Array(name="", shape=other.shape)
        root.op = "add"
        root.operands = [other, reduction_node]
        # Should not raise ValueError — the M3 terminal validator is gone.
        dsl_to_array(root, params, shapes)

    def test_sum_fed_into_add(self) -> None:
        """A + sum(A) — sum is non-terminal, dsl_to_array should not reject."""
        a = Array("A", (64,))
        s = sum(a)
        self._build_non_terminal(s, a, ["A"], {"A": (64,)})

    def test_amax_fed_into_add(self) -> None:
        """A + amax(A) — amax is non-terminal, dsl_to_array should not reject."""
        a = Array("A", (64,))
        m = amax(a)
        self._build_non_terminal(m, a, ["A"], {"A": (64,)})

    def test_dot_fed_into_add(self) -> None:
        """A + dot(A, B) — dot is non-terminal, dsl_to_array should not reject."""
        a = Array("A", (64,))
        b = Array("B", (64,))
        d = dot(a, b)
        self._build_non_terminal(d, a, ["A", "B"], {"A": (64,), "B": (64,)})

    def test_mean_fed_into_add(self) -> None:
        """A + mean(A) — mean is non-terminal, dsl_to_array should not reject."""
        a = Array("A", (64,))
        m = mean(a)
        self._build_non_terminal(m, a, ["A"], {"A": (64,)})
