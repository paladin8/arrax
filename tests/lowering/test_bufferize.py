"""Tests for arrax.lowering.bufferize — tensor to memref conversion."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import arith, func, linalg, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    Float32Type,
    FloatAttr,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.linalg import IteratorTypeAttr
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from arrax.dsl.array import Array, amax, dot, mean, sum
from arrax.lowering.bufferize import BufferizePass
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

    def test_sum_rank0_output_is_function_arg(self) -> None:
        """sum(A): rank-0 output promoted to function arg; linalg.fill survives."""
        module = make_module(lambda A: sum(A), {"A": (64,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        # Both arguments are memrefs; output is rank-0 f32.
        assert "func.func @kernel(%0: memref<64xf32>, %1: memref<f32>)" in ir
        # linalg.fill is preserved on memref semantics.
        assert "linalg.fill" in ir
        assert "memref<f32>" in ir
        # Reduction generic remains with outs = %1 (the rank-0 function arg).
        assert '"reduction"' in ir

    def test_amax_rank0_output_is_function_arg(self) -> None:
        """amax(A): same rank-0 promotion as sum, with -inf identity fill."""
        module = make_module(lambda A: amax(A), {"A": (64,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        assert "func.func @kernel(%0: memref<64xf32>, %1: memref<f32>)" in ir
        assert "linalg.fill" in ir
        assert "memref<f32>" in ir
        assert '"reduction"' in ir
        # -inf identity seed survives bufferization unchanged.
        assert "0xff800000" in ir.lower()
        assert "arith.maximumf" in ir

    def test_dot_rank0_output_is_function_arg(self) -> None:
        """dot(A, B): rank-0 output promoted to function arg with three memref args."""
        module = make_module(lambda A, B: dot(A, B), {"A": (64,), "B": (64,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        assert "func.func @kernel(%0: memref<64xf32>, %1: memref<64xf32>, %2: memref<f32>)" in ir
        assert "linalg.fill" in ir
        assert "memref<f32>" in ir
        assert '"reduction"' in ir
        assert "arith.mulf" in ir
        assert "arith.addf" in ir

    def test_mean_rank0_output_and_divisor_attr_preserved(self) -> None:
        """mean(A): same as sum but arrax.mean_divisor attr survives bufferization."""
        module = make_module(lambda A: mean(A), {"A": (64,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        assert "func.func @kernel(%0: memref<64xf32>, %1: memref<f32>)" in ir
        assert "linalg.fill" in ir
        assert '"reduction"' in ir
        assert "arrax.mean_divisor = 64 : i64" in ir

    def test_sum_of_add_has_intermediate_alloc(self) -> None:
        """sum(A + B) needs a rank-1 alloc for the add and a rank-0 out arg."""
        module = make_module(lambda A, B: sum(A + B), {"A": (32,), "B": (32,)})
        bufferize(module)
        ir = str(module)
        assert "tensor<" not in ir
        assert "memref.alloc()" in ir  # rank-1 intermediate for A+B
        assert "memref<32xf32>" in ir
        assert "memref<f32>" in ir
        assert ir.count("linalg.generic") == 2

    def test_non_terminal_rank0_reduction_allocs_intermediate(self) -> None:
        """Forward-compat: a rank-0 reduction whose result is consumed by a
        later op (rather than returned directly) must NOT be promoted to
        the function output. Bufferize should allocate a scratch
        ``memref.alloc() : memref<f32>`` for it and map the surviving
        rank-1 output into the function arg.

        Milestone 3's terminal validator currently forbids this shape from
        the DSL surface, but Milestone 4 will lift that restriction.
        Pinning the bufferize behavior here ensures the Milestone 4
        change is a validator-only edit — no bufferize rework needed.
        """
        f32 = Float32Type()
        t1 = TensorType(f32, [8])
        t0 = TensorType(f32, [])

        arg_block = Block(arg_types=[t1])
        A = arg_block.args[0]

        c0 = arith.ConstantOp(FloatAttr(0.0, f32))
        empty_scalar = tensor.EmptyOp([], t0)
        filled = linalg.FillOp(
            inputs=[c0.result],
            outputs=[empty_scalar.tensor],
            res=[t0],
        )

        map_in = AffineMapAttr(AffineMap.identity(1))
        map_out = AffineMapAttr(AffineMap(1, 0, ()))
        reduction_body = Block(arg_types=[f32, f32])
        addf = arith.AddfOp(reduction_body.args[0], reduction_body.args[1])
        reduction_body.add_ops([addf, linalg.YieldOp(addf.result)])
        sum_generic = linalg.GenericOp(
            inputs=[A],
            outputs=[filled.res[0]],
            body=Region([reduction_body]),
            indexing_maps=[map_in, map_out],
            iterator_types=[IteratorTypeAttr.reduction()],
            result_types=[t0],
        )

        empty_vec = tensor.EmptyOp([], t1)
        bcast_body = Block(arg_types=[f32, f32])
        bcast_body.add_ops([linalg.YieldOp(bcast_body.args[0])])
        bcast_generic = linalg.GenericOp(
            inputs=[sum_generic.res[0]],
            outputs=[empty_vec.tensor],
            body=Region([bcast_body]),
            indexing_maps=[map_out, map_in],
            iterator_types=[IteratorTypeAttr.parallel()],
            result_types=[t1],
        )

        arg_block.add_ops(
            [
                c0,
                empty_scalar,
                filled,
                sum_generic,
                empty_vec,
                bcast_generic,
                func.ReturnOp(bcast_generic.res[0]),
            ]
        )
        kernel = func.FuncOp(
            name="kernel",
            function_type=([t1], [t1]),
            region=Region([arg_block]),
        )
        module = ModuleOp([kernel])
        module.verify()

        BufferizePass().apply(Context(), module)
        module.verify()

        ir = str(module)
        assert "tensor<" not in ir
        # Rank-0 scratch is an intermediate alloc, not a function arg:
        # the function signature has exactly two memref args (input + output
        # rank-1), not three.
        assert (
            "func.func @kernel(%0: memref<8xf32>, %1: memref<8xf32>)" in ir
        )
        # The non-terminal rank-0 reduction materializes as memref.alloc.
        assert "memref.alloc() : memref<f32>" in ir
        # The terminal rank-1 generic writes into the function output (%1).
        assert 'outs(%1 : memref<8xf32>)' in ir
