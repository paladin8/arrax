"""Tests for dialect lowering passes."""

from __future__ import annotations

import pytest

from xdsl.context import Context

from arrax.dsl.array import Array
from arrax.dsl.tracer import trace
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.dsl_to_array import dsl_to_array


def _make_module(fn, shapes):
    """Trace fn and lower to array dialect IR."""
    result, params = trace(fn, shapes)
    return dsl_to_array(result, params, shapes)


def _lower_to_linalg(module):
    """Apply array-to-linalg pass in place."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    module.verify()
    return module


class TestDslToArray:
    def test_basic_add(self) -> None:
        module = _make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
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
        assert "array.add %1, %0" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two array.add ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = _make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        module.verify()
        assert str(module).count("array.add") == 2

    def test_different_shape(self) -> None:
        module = _make_module(lambda X, Y: X + Y, {"X": (256,), "Y": (256,)})
        module.verify()
        assert "tensor<256xf32>" in str(module)

    def test_diamond_dag(self) -> None:
        """A + A reuses the same leaf — should not duplicate ops."""
        module = _make_module(lambda A: A + A, {"A": (16,)})
        module.verify()
        assert str(module).count("array.add") == 1

    def test_unused_param(self) -> None:
        """Unused params still become function arguments."""
        module = _make_module(
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


class TestArrayToLinalg:
    def test_basic_add(self) -> None:
        module = _make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        _lower_to_linalg(module)

        expected = """\
builtin.module {
  func.func @kernel(%0: tensor<1024xf32>, %1: tensor<1024xf32>) -> tensor<1024xf32> {
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : tensor<1024xf32>, tensor<1024xf32>) outs(%2 : tensor<1024xf32>) {
    ^bb0(%4: f32, %5: f32, %6: f32):
      %7 = arith.addf %4, %5 : f32
      linalg.yield %7 : f32
    } -> tensor<1024xf32>
    func.return %3 : tensor<1024xf32>
  }
}"""
        assert str(module) == expected

    def test_no_array_ops_remain(self) -> None:
        module = _make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        _lower_to_linalg(module)
        ir = str(module)
        assert "array.add" not in ir
        assert "linalg.generic" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two linalg.generic ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = _make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        _lower_to_linalg(module)
        ir = str(module)
        assert ir.count("linalg.generic") == 2
        assert ir.count("tensor.empty") == 2
        assert "array.add" not in ir

    def test_body_has_addf_and_yield(self) -> None:
        module = _make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        _lower_to_linalg(module)
        ir = str(module)
        assert "arith.addf" in ir
        assert "linalg.yield" in ir

    def test_different_shape(self) -> None:
        module = _make_module(lambda X, Y: X + Y, {"X": (512,), "Y": (512,)})
        _lower_to_linalg(module)
        ir = str(module)
        assert "tensor<512xf32>" in ir
        assert "linalg.generic" in ir

    def test_diamond_dag(self) -> None:
        """A + A: single input feeds both lhs and rhs of linalg.generic."""
        module = _make_module(lambda A: A + A, {"A": (16,)})
        _lower_to_linalg(module)
        ir = str(module)
        assert ir.count("linalg.generic") == 1
        assert "array.add" not in ir

    def test_2d_tensor(self) -> None:
        """Lowering generalizes to multi-dimensional tensors."""
        module = _make_module(lambda A, B: A + B, {"A": (8, 16), "B": (8, 16)})
        _lower_to_linalg(module)
        ir = str(module)
        assert "tensor<8x16xf32>" in ir
        assert "(d0, d1) -> (d0, d1)" in ir
        assert '"parallel", "parallel"' in ir
