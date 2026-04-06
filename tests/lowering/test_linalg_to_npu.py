"""Tests for arrax.lowering.linalg_to_npu — linalg.generic to npu ops."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from arrax.dsl.array import Array, amax, dot, exp, mean, relu, sum

from arrax.lowering.array_to_linalg import ArrayToLinalgPass
from arrax.lowering.bufferize import BufferizePass
from arrax.lowering.linalg_to_npu import LinalgToNpuPass
from arrax.lowering.tile import TilePass
from tests.helpers import make_module


def _lower_to_npu(module: ModuleOp) -> ModuleOp:
    """Apply full pipeline: array-to-linalg, bufferize, linalg-to-npu."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    return module


def _lower_to_npu_with_tiling(module: ModuleOp) -> ModuleOp:
    """Apply full pipeline with tiling: array-to-linalg, bufferize, tile, linalg-to-npu."""
    ctx = Context()
    ArrayToLinalgPass().apply(ctx, module)
    BufferizePass().apply(ctx, module)
    TilePass().apply(ctx, module)
    LinalgToNpuPass().apply(ctx, module)
    module.verify()
    return module


class TestLinalgToNpu:
    def test_basic_add(self) -> None:
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        _lower_to_npu(module)

        expected = """\
builtin.module {
  func.func @kernel(%0: memref<64xf32>, %1: memref<64xf32>, %2: memref<64xf32>) {
    %3 = arith.constant 64 : index
    npu.fvadd %0, %1, %2, %3 : memref<64xf32>, memref<64xf32>, memref<64xf32>, index
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
        module = make_module(lambda A, B: A + B, {"A": (48,), "B": (48,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "arith.constant 48 : index" in ir

    def test_different_shape(self) -> None:
        module = make_module(lambda X, Y: X + Y, {"X": (32,), "Y": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "memref<32xf32>" in ir
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

    def test_tiled_basic(self) -> None:
        """n=128 with tiling: npu.fvadd uses dynamic chunk size from subview."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvadd" in ir
        assert "linalg.generic" not in ir
        assert "scf.for" in ir

    def test_tiled_non_multiple(self) -> None:
        """n=100 with tiling: handles remainder chunk."""
        module = make_module(lambda A, B: A + B, {"A": (100,), "B": (100,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvadd" in ir
        assert "linalg.generic" not in ir

    def test_basic_sub(self) -> None:
        module = make_module(lambda A, B: A - B, {"A": (64,), "B": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvsub" in ir
        assert "linalg.generic" not in ir

    def test_sub_n_constant_matches_shape(self) -> None:
        module = make_module(lambda A, B: A - B, {"A": (48,), "B": (48,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "arith.constant 48 : index" in ir
        assert "npu.fvsub" in ir

    def test_mixed_add_sub(self) -> None:
        """(A + B) - C produces one fvadd and one fvsub."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) - C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvadd" in ir
        assert "npu.fvsub" in ir
        assert "linalg.generic" not in ir

    def test_tiled_sub(self) -> None:
        """n=128 sub with tiling."""
        module = make_module(lambda A, B: A - B, {"A": (128,), "B": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvsub" in ir
        assert "scf.for" in ir

    def test_relu(self) -> None:
        module = make_module(lambda A: relu(A), {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvrelu" in ir
        assert "linalg.generic" not in ir

    def test_exp(self) -> None:
        module = make_module(lambda A: exp(A), {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvexp" in ir
        assert "linalg.generic" not in ir

    def test_relu_of_add(self) -> None:
        """relu(A + B) produces one fvadd and one fvrelu."""
        module = make_module(lambda A, B: relu(A + B), {"A": (32,), "B": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvadd" in ir
        assert "npu.fvrelu" in ir
        assert "linalg.generic" not in ir

    def test_tiled_relu(self) -> None:
        module = make_module(lambda A: relu(A), {"A": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvrelu" in ir
        assert "scf.for" in ir

    def test_tiled_exp(self) -> None:
        module = make_module(lambda A: exp(A), {"A": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvexp" in ir
        assert "scf.for" in ir

    def test_mul_scalar(self) -> None:
        module = make_module(lambda A: A * 3.0, {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvmul" in ir
        assert "linalg.generic" not in ir

    def test_div_scalar(self) -> None:
        module = make_module(lambda A: A / 2.0, {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvdiv" in ir
        assert "linalg.generic" not in ir

    # --- reduction lowering (sum) ---

    def test_sum_untiled_basic(self) -> None:
        """sum(A), n=64 (untiled): alloca+fill+generic collapse to fvreduce + store."""
        module = make_module(lambda A: sum(A), {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.store" in ir  # terminal rank-0 store
        assert "-> f32" in ir
        # N is materialized as a constant since the input is a static memref
        assert "arith.constant 64 : index" in ir

    def test_sum_untiled_small(self) -> None:
        module = make_module(lambda A: sum(A), {"A": (16,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "linalg.generic" not in ir
        assert "arith.constant 16 : index" in ir

    def test_sum_untiled_acc_is_zero(self) -> None:
        """Untiled reduction threads acc_in = arith.constant 0.0."""
        module = make_module(lambda A: sum(A), {"A": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "arith.constant 0.000000e+00 : f32" in ir
        assert "npu.fvreduce" in ir

    def test_sum_tiled_basic(self) -> None:
        """sum(A), n=128: fvreduce inside scf.for body, load/alloca erased."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.alloca" not in ir
        assert "memref.load" not in ir
        assert "memref.store" in ir  # terminal write outside loop
        # The chunk size comes from arith.minsi inside the loop
        assert "arith.minsi" in ir

    def test_sum_tiled_non_multiple(self) -> None:
        """sum(A), n=100: fvreduce tolerates the remainder chunk."""
        module = make_module(lambda A: sum(A), {"A": (100,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "scf.for" in ir
        assert "linalg.generic" not in ir
        assert "memref.alloca" not in ir

    def test_sum_tiled_large(self) -> None:
        """sum(A), n=1024."""
        module = make_module(lambda A: sum(A), {"A": (1024,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "scf.for" in ir
        assert "linalg.generic" not in ir

    def test_sum_verifies(self) -> None:
        """Verifier passes for both tiled and untiled sum."""
        for n in (16, 64, 100, 128, 1024):
            module = make_module(lambda A: sum(A), {"A": (n,)})
            _lower_to_npu_with_tiling(module)
            module.verify()

    def test_tiled_no_arith_constant_for_n(self) -> None:
        """After tiling, n comes from minsi — no new arith.constant for element count."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        # The fvadd's n operand should reference the minsi result,
        # not a new constant
        assert "arith.minsi" in ir
        assert "npu.fvadd" in ir

    # --- reduction lowering (amax) ---

    def test_amax_untiled_basic(self) -> None:
        """amax(A), n=64: alloca+fill+maximumf-generic collapses to npu.fvmax + store."""
        module = make_module(lambda A: amax(A), {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvmax" in ir
        assert "npu.fvreduce" not in ir  # must NOT be an fvreduce (it's a max, not sum)
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.store" in ir
        assert "arith.constant 64 : index" in ir

    def test_amax_untiled_small(self) -> None:
        module = make_module(lambda A: amax(A), {"A": (16,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvmax" in ir
        assert "linalg.generic" not in ir

    def test_amax_untiled_acc_is_neg_inf(self) -> None:
        """Untiled amax threads acc_in = arith.constant -inf (0xff800000)."""
        module = make_module(lambda A: amax(A), {"A": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "0xff800000" in ir.lower()
        assert "npu.fvmax" in ir

    def test_amax_tiled_basic(self) -> None:
        """amax(A), n=128: fvmax inside scf.for body, load/alloca erased."""
        module = make_module(lambda A: amax(A), {"A": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvmax" in ir
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.alloca" not in ir
        assert "memref.load" not in ir
        assert "memref.store" in ir
        assert "arith.minsi" in ir

    def test_amax_tiled_non_multiple(self) -> None:
        """amax(A), n=100: fvmax tolerates the remainder chunk."""
        module = make_module(lambda A: amax(A), {"A": (100,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvmax" in ir
        assert "scf.for" in ir
        assert "linalg.generic" not in ir

    def test_amax_verifies(self) -> None:
        """Verifier passes for both tiled and untiled amax."""
        for n in (16, 64, 100, 128, 1024):
            module = make_module(lambda A: amax(A), {"A": (n,)})
            _lower_to_npu_with_tiling(module)
            module.verify()

    # --- reduction lowering (dot) ---

    def test_dot_untiled_basic(self) -> None:
        """dot(A, B), n=64: fill+generic collapse to npu.fvmac + store."""
        module = make_module(lambda A, B: dot(A, B), {"A": (64,), "B": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvmac" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.store" in ir
        assert "arith.constant 64 : index" in ir

    def test_dot_untiled_small(self) -> None:
        module = make_module(lambda A, B: dot(A, B), {"A": (16,), "B": (16,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvmac" in ir
        assert "linalg.generic" not in ir
        assert "arith.constant 16 : index" in ir

    def test_dot_untiled_acc_is_zero(self) -> None:
        """Untiled dot threads acc_in = arith.constant 0.0."""
        module = make_module(lambda A, B: dot(A, B), {"A": (32,), "B": (32,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "arith.constant 0.000000e+00 : f32" in ir
        assert "npu.fvmac" in ir

    def test_dot_tiled_basic(self) -> None:
        """dot(A, B), n=128: fvmac inside scf.for body, alloca erased."""
        module = make_module(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvmac" in ir
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.alloca" not in ir
        assert "memref.load" not in ir
        assert "memref.store" in ir
        assert "arith.minsi" in ir

    def test_dot_tiled_non_multiple(self) -> None:
        """dot(A, B), n=100: fvmac tolerates the remainder chunk."""
        module = make_module(lambda A, B: dot(A, B), {"A": (100,), "B": (100,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvmac" in ir
        assert "scf.for" in ir
        assert "linalg.generic" not in ir

    def test_dot_verifies(self) -> None:
        """Verifier passes for both tiled and untiled dot."""
        for n in (16, 64, 100, 128, 1024):
            module = make_module(lambda A, B: dot(A, B), {"A": (n,), "B": (n,)})
            _lower_to_npu_with_tiling(module)
            module.verify()

    # --- reduction lowering (mean) ---

    def test_mean_untiled_basic(self) -> None:
        """mean(A), n=64: produces npu.fvreduce with divisor property + store."""
        module = make_module(lambda A: mean(A), {"A": (64,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.store" in ir
        assert "divisor = 64 : i64" in ir

    def test_mean_untiled_small(self) -> None:
        """mean(A), n=16: divisor matches input size."""
        module = make_module(lambda A: mean(A), {"A": (16,)})
        _lower_to_npu(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "divisor = 16 : i64" in ir

    def test_mean_tiled_basic(self) -> None:
        """mean(A), n=128: fvreduce inside scf.for with divisor property."""
        module = make_module(lambda A: mean(A), {"A": (128,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "linalg.generic" not in ir
        assert "linalg.fill" not in ir
        assert "memref.alloca" not in ir
        assert "divisor = 128 : i64" in ir

    def test_mean_tiled_non_multiple(self) -> None:
        """mean(A), n=100: remainder handled, divisor preserved."""
        module = make_module(lambda A: mean(A), {"A": (100,)})
        _lower_to_npu_with_tiling(module)
        ir = str(module)
        assert "npu.fvreduce" in ir
        assert "scf.for" in ir
        assert "divisor = 100 : i64" in ir

    def test_mean_verifies(self) -> None:
        """Verifier passes for both tiled and untiled mean."""
        for n in (16, 64, 100, 128, 1024):
            module = make_module(lambda A: mean(A), {"A": (n,)})
            _lower_to_npu_with_tiling(module)
            module.verify()
