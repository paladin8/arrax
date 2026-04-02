"""Tests for dialect lowering passes."""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array
from arrax.dsl.tracer import trace
from arrax.lowering.dsl_to_array import dsl_to_array


class TestDslToArray:
    def test_basic_add(self) -> None:
        result, params = trace(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        module = dsl_to_array(result, params, {"A": (1024,), "B": (1024,)})
        module.verify()

        expected = """\
builtin.module {
  func.func @kernel(%0: tensor<1024xf32>, %1: tensor<1024xf32>) -> tensor<1024xf32> {
    %2 = array.add %0, %1 : tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
    func.return %2 : tensor<1024xf32>
  }
}"""
        assert str(module) == expected

    def test_param_order_preserved(self) -> None:
        """Function args follow signature order, not shapes dict order."""
        def kernel(B: Array, A: Array) -> Array:
            return A + B

        result, params = trace(kernel, {"A": (10,), "B": (10,)})
        module = dsl_to_array(result, params, {"A": (10,), "B": (10,)})
        module.verify()

        assert params == ["B", "A"]
        ir = str(module)
        # Param order is B, A — so in "A + B", the add uses %1 (A) and %0 (B).
        # The add operands should reference the second arg (%1) before the first (%0).
        assert "array.add %1, %0" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two array.add ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        result, params = trace(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        module = dsl_to_array(result, params, {"A": (32,), "B": (32,), "C": (32,)})
        module.verify()

        ir = str(module)
        assert ir.count("array.add") == 2

    def test_different_shape(self) -> None:
        result, params = trace(lambda X, Y: X + Y, {"X": (256,), "Y": (256,)})
        module = dsl_to_array(result, params, {"X": (256,), "Y": (256,)})
        module.verify()

        ir = str(module)
        assert "tensor<256xf32>" in ir

    def test_diamond_dag(self) -> None:
        """A + A reuses the same leaf — should not duplicate ops."""
        result, params = trace(lambda A: A + A, {"A": (16,)})
        module = dsl_to_array(result, params, {"A": (16,)})
        module.verify()

        ir = str(module)
        assert ir.count("array.add") == 1

    def test_unused_param(self) -> None:
        """Unused params still become function arguments."""
        result, params = trace(
            lambda A, B, C: A + B, {"A": (10,), "B": (10,), "C": (10,)}
        )
        module = dsl_to_array(result, params, {"A": (10,), "B": (10,), "C": (10,)})
        module.verify()

        ir = str(module)
        # Three params in signature: %0, %1, %2
        assert "%0: tensor<10xf32>, %1: tensor<10xf32>, %2: tensor<10xf32>" in ir
        # But only one add op (C/%2 is unused)
        assert ir.count("array.add") == 1

    def test_unsupported_op_raises(self) -> None:
        """An Array with an unknown op should raise ValueError."""
        a = Array("A", (10,))
        b = Array("B", (10,))
        bad = Array("", (10,))
        bad.op = "unknown"
        bad.operands = [a, b]

        with pytest.raises(ValueError, match="unsupported operation: unknown"):
            dsl_to_array(bad, ["A", "B"], {"A": (10,), "B": (10,)})
