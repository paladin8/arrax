"""Tests for arrax.lowering.array_to_linalg — array dialect to linalg.generic."""

from __future__ import annotations

from arrax.dsl.array import Array
from tests.helpers import lower_to_linalg, make_module


class TestArrayToLinalg:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        lower_to_linalg(module)

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
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        lower_to_linalg(module)
        ir = str(module)
        assert "array.add" not in ir
        assert "linalg.generic" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two linalg.generic ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        lower_to_linalg(module)
        ir = str(module)
        assert ir.count("linalg.generic") == 2
        assert ir.count("tensor.empty") == 2
        assert "array.add" not in ir

    def test_body_has_addf_and_yield(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        lower_to_linalg(module)
        ir = str(module)
        assert "arith.addf" in ir
        assert "linalg.yield" in ir

    def test_different_shape(self) -> None:
        module = make_module(lambda X, Y: X + Y, {"X": (512,), "Y": (512,)})
        lower_to_linalg(module)
        ir = str(module)
        assert "tensor<512xf32>" in ir
        assert "linalg.generic" in ir

    def test_diamond_dag(self) -> None:
        """A + A: single input feeds both lhs and rhs of linalg.generic."""
        module = make_module(lambda A: A + A, {"A": (16,)})
        lower_to_linalg(module)
        ir = str(module)
        assert ir.count("linalg.generic") == 1
        assert "array.add" not in ir

    def test_basic_sub(self) -> None:
        module = make_module(lambda A, B: A - B, {"A": (1024,), "B": (1024,)})
        lower_to_linalg(module)
        ir = str(module)
        assert "arith.subf" in ir
        assert "linalg.generic" in ir
        assert "array.sub" not in ir

    def test_mixed_add_sub(self) -> None:
        """(A + B) - C produces one addf generic and one subf generic."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) - C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        lower_to_linalg(module)
        ir = str(module)
        assert ir.count("linalg.generic") == 2
        assert "arith.addf" in ir
        assert "arith.subf" in ir

    def test_2d_tensor(self) -> None:
        """Lowering generalizes to multi-dimensional tensors."""
        module = make_module(lambda A, B: A + B, {"A": (8, 16), "B": (8, 16)})
        lower_to_linalg(module)
        ir = str(module)
        assert "tensor<8x16xf32>" in ir
        assert "(d0, d1) -> (d0, d1)" in ir
        assert '"parallel", "parallel"' in ir
