"""Tests for arrax.lowering.linalg_to_npu — linalg.generic to npu ops."""

from __future__ import annotations

from xdsl.context import Context

from arrax.dsl.array import Array
from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.linalg_to_npu import LinalgToNpuPass
from tests.helpers import make_module


def _lower_to_npu(module):
    """Apply full pipeline: array-to-linalg, bufferize, linalg-to-npu."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    return module


class TestLinalgToNpu:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        _lower_to_npu(module)

        expected = """\
builtin.module {
  func.func @kernel(%0: memref<1024xf32>, %1: memref<1024xf32>, %2: memref<1024xf32>) {
    %3 = arith.constant 1024 : index
    npu.fvadd %0, %1, %2, %3 : memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, index
    func.return
  }
}"""
        assert str(module) == expected

    def test_no_linalg_ops_remain(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "linalg.generic" not in ir
        assert "npu.fvadd" in ir

    def test_chained_add(self) -> None:
        """(A + B) + C produces two npu.fvadd ops."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert ir.count("npu.fvadd") == 2
        assert "linalg.generic" not in ir
        # Intermediate alloc still present
        assert "memref.alloc" in ir

    def test_n_constant_matches_shape(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (256,), "B": (256,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "arith.constant 256 : index" in ir

    def test_different_shape(self) -> None:
        module = make_module(lambda X, Y: X + Y, {"X": (512,), "Y": (512,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "memref<512xf32>" in ir
        assert "npu.fvadd" in ir

    def test_diamond_dag(self) -> None:
        """A + A: single npu.fvadd with same memref for both inputs."""
        module = make_module(lambda A: A + A, {"A": (16,)})
        _lower_to_npu(module)
        ir = str(module)
        assert ir.count("npu.fvadd") == 1
        # src1 and src2 are the same memref (%0)
        assert "npu.fvadd %0, %0" in ir

    def test_void_return_preserved(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (8,), "B": (8,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "func.return\n" in ir

    def test_mulf_body_not_matched(self) -> None:
        """A linalg.generic with mulf body should pass through unchanged."""
        from xdsl.dialects import arith, func, linalg, memref
        from xdsl.dialects.builtin import (
            AffineMapAttr, Float32Type, MemRefType, ModuleOp,
        )
        from xdsl.dialects.linalg import IteratorTypeAttr
        from xdsl.ir import Block, Region
        from xdsl.ir.affine import AffineMap

        f32 = Float32Type()
        memref_type = MemRefType(f32, [64])
        identity = AffineMap.identity(1)
        maps = [AffineMapAttr(identity)] * 3
        iters = [IteratorTypeAttr.parallel()]

        # Body with mulf instead of addf
        body_block = Block(arg_types=[f32, f32, f32])
        mul = arith.MulfOp(body_block.args[0], body_block.args[1])
        yield_op = linalg.YieldOp(mul.result)
        body_block.add_ops([mul, yield_op])

        # Build func with this generic
        func_block = Block(arg_types=[memref_type, memref_type, memref_type])
        generic = linalg.GenericOp(
            inputs=[func_block.args[0], func_block.args[1]],
            outputs=[func_block.args[2]],
            body=Region([body_block]),
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[],
        )
        func_block.add_ops([generic, func.ReturnOp()])
        func_op = func.FuncOp(
            name="kernel",
            function_type=([memref_type, memref_type, memref_type], []),
            region=Region([func_block]),
        )
        module = ModuleOp([func_op])

        # Run only the NPU lowering pass (not the full pipeline)
        ctx = Context()
        LinalgToNpuPass().apply(ctx, module)
        module.verify()

        ir = str(module)
        # mulf generic should still be there, not matched
        assert "linalg.generic" in ir
        assert "arith.mulf" in ir
        assert "npu.fvadd" not in ir
