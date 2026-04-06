"""Tests for arrax.lowering.dsl_to_array — traced DAG to array dialect IR."""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array, amax, sum
from arrax.dsl.tracer import trace
from arrax.lowering.dsl_to_array import dsl_to_array
from tests.helpers import make_module


class TestDslToArray:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
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
        assert "array.add %1, %0" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two array.add ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        module.verify()
        assert str(module).count("array.add") == 2

    def test_different_shape(self) -> None:
        module = make_module(lambda X, Y: X + Y, {"X": (256,), "Y": (256,)})
        module.verify()
        assert "tensor<256xf32>" in str(module)

    def test_diamond_dag(self) -> None:
        """A + A reuses the same leaf — should not duplicate ops."""
        module = make_module(lambda A: A + A, {"A": (16,)})
        module.verify()
        assert str(module).count("array.add") == 1

    def test_unused_param(self) -> None:
        """Unused params still become function arguments."""
        module = make_module(
            lambda A, B, C: A + B, {"A": (10,), "B": (10,), "C": (10,)}
        )
        module.verify()

        ir = str(module)
        assert "%0: tensor<10xf32>, %1: tensor<10xf32>, %2: tensor<10xf32>" in ir
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

    def test_sum_rank0_result(self) -> None:
        """sum(A) produces array.sum with a rank-0 tensor<f32> result."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        module.verify()
        ir = str(module)
        assert "array.sum" in ir
        assert "tensor<128xf32> -> tensor<f32>" in ir
        assert "func.func @kernel(%0: tensor<128xf32>) -> tensor<f32>" in ir

    def test_sum_of_add(self) -> None:
        """sum(A + B) produces array.add followed by array.sum."""
        module = make_module(
            lambda A, B: sum(A + B), {"A": (64,), "B": (64,)}
        )
        module.verify()
        ir = str(module)
        assert "array.add" in ir
        assert "array.sum" in ir

    def test_amax_rank0_result(self) -> None:
        """amax(A) produces array.amax with a rank-0 tensor<f32> result."""
        module = make_module(lambda A: amax(A), {"A": (128,)})
        module.verify()
        ir = str(module)
        assert "array.amax" in ir
        assert "tensor<128xf32> -> tensor<f32>" in ir
        assert "func.func @kernel(%0: tensor<128xf32>) -> tensor<f32>" in ir

    def test_amax_of_sub(self) -> None:
        """amax(A - B) produces array.sub followed by array.amax."""
        module = make_module(
            lambda A, B: amax(A - B), {"A": (64,), "B": (64,)}
        )
        module.verify()
        ir = str(module)
        assert "array.sub" in ir
        assert "array.amax" in ir
