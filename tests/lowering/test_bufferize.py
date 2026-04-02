"""Tests for arrax.lowering.bufferize — tensor to memref conversion."""

from __future__ import annotations

from arrax.dsl.array import Array
from tests.helpers import bufferize, make_module


class TestBufferize:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        bufferize(module)

        expected = """\
builtin.module {
  func.func @kernel(%0: memref<1024xf32>, %1: memref<1024xf32>, %2: memref<1024xf32>) {
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<1024xf32>, memref<1024xf32>) outs(%2 : memref<1024xf32>) {
    ^bb0(%3: f32, %4: f32, %5: f32):
      %6 = arith.addf %3, %4 : f32
      linalg.yield %6 : f32
    }
    func.return
  }
}"""
        assert str(module) == expected

    def test_no_tensor_types_remain(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        assert "tensor.empty" not in ir
        assert "memref<64xf32>" in ir

    def test_void_return(self) -> None:
        """After bufferization, func.return has no operands."""
        module = make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        bufferize(module)
        ir = str(module)
        assert "func.return\n" in ir

    def test_output_is_function_arg(self) -> None:
        """Output buffer is a function argument, not an alloc."""
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        bufferize(module)
        ir = str(module)
        assert "%0: memref<64xf32>, %1: memref<64xf32>, %2: memref<64xf32>" in ir
        assert "memref.alloc" not in ir

    def test_chained_add_has_intermediate_alloc(self) -> None:
        """(A + B) + C: intermediate result needs memref.alloc."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        bufferize(module)
        ir = str(module)
        assert "%0: memref<32xf32>, %1: memref<32xf32>, %2: memref<32xf32>, %3: memref<32xf32>" in ir
        assert ir.count("memref.alloc") == 1
        assert ir.count("linalg.generic") == 2
        assert "func.return\n" in ir

    def test_body_unchanged(self) -> None:
        """linalg.generic body (scalar f32 ops) is preserved through bufferization."""
        module = make_module(lambda A, B: A + B, {"A": (16,), "B": (16,)})
        bufferize(module)
        ir = str(module)
        assert "arith.addf" in ir
        assert "linalg.yield" in ir

    def test_different_shape(self) -> None:
        module = make_module(lambda X, Y: X + Y, {"X": (256,), "Y": (256,)})
        bufferize(module)
        ir = str(module)
        assert "memref<256xf32>" in ir
        assert "linalg.generic" in ir

    def test_diamond_dag(self) -> None:
        """A + A: single input feeds both ins of linalg.generic on memref."""
        module = make_module(lambda A: A + A, {"A": (16,)})
        bufferize(module)
        ir = str(module)
        assert "%0: memref<16xf32>, %1: memref<16xf32>" in ir
        assert "ins(%0, %0" in ir
        assert "memref.alloc" not in ir

    def test_2d_tensor(self) -> None:
        """Bufferization handles multi-dimensional tensors."""
        module = make_module(lambda A, B: A + B, {"A": (8, 16), "B": (8, 16)})
        bufferize(module)
        ir = str(module)
        assert "memref<8x16xf32>" in ir
        assert "tensor<" not in ir
