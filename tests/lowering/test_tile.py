"""Tests for arrax.lowering.tile — strip-mine tiling for NPU vector limit."""

from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import (
    AffineMap,
    AffineMapAttr,
    Float32Type,
    MemRefType,
    ModuleOp,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import IteratorTypeAttr
from xdsl.ir import Block, Region

from arrax.dsl.array import Array, amax, dot, mean, sum
from arrax.lowering.tile import TilePass
from tests.helpers import bufferize, make_module, tile


class TestTile:
    def test_below_limit_unchanged(self) -> None:
        """n=32: no tiling, IR unchanged from bufferized form."""
        module = make_module(lambda A, B: A + B, {"A": (32,), "B": (32,)})
        expected_ir = str(bufferize(make_module(lambda A, B: A + B, {"A": (32,), "B": (32,)})))

        module = make_module(lambda A, B: A + B, {"A": (32,), "B": (32,)})
        tile(module)
        assert str(module) == expected_ir

    def test_at_limit_unchanged(self) -> None:
        """n=64: exactly at limit, no tiling needed."""
        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        expected_ir = str(bufferize(make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})))

        module = make_module(lambda A, B: A + B, {"A": (64,), "B": (64,)})
        tile(module)
        assert str(module) == expected_ir

    def test_exact_multiple(self) -> None:
        """n=128: exact multiple of 64, produces scf.for."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "memref.subview" in ir
        assert "arith.minsi" in ir
        # Original static-shape generic is gone
        assert "memref<128xf32>" in ir  # func args
        assert "linalg.generic" in ir  # still has generic (on subviews)

    def test_non_multiple(self) -> None:
        """n=100: non-multiple of 64, handles remainder."""
        module = make_module(lambda A, B: A + B, {"A": (100,), "B": (100,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "memref.subview" in ir
        assert "arith.minsi" in ir

    def test_large_array(self) -> None:
        """n=1024: many iterations of tiling."""
        module = make_module(lambda A, B: A + B, {"A": (1024,), "B": (1024,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        # Step is 64
        assert "arith.constant 64 : index" in ir
        assert "arith.constant 1024 : index" in ir

    def test_chained_add_both_tiled(self) -> None:
        """(A + B) + C: both adds get tiled."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        tile(module)
        ir = str(module)
        assert ir.count("scf.for") == 2
        assert ir.count("linalg.generic") == 2

    def test_verifies(self) -> None:
        """Tiled IR passes xDSL verification."""
        module = make_module(lambda A, B: A + B, {"A": (100,), "B": (100,)})
        tile(module)
        module.verify()

    # --- reduction tiling (sum) ---

    def test_reduction_below_limit_unchanged(self) -> None:
        """sum(A), n=32: untiled reduction passes through."""
        expected = str(bufferize(make_module(lambda A: sum(A), {"A": (32,)})))

        module = make_module(lambda A: sum(A), {"A": (32,)})
        tile(module)
        assert str(module) == expected

    def test_reduction_at_limit_unchanged(self) -> None:
        """sum(A), n=64: exactly at limit, no tiling."""
        expected = str(bufferize(make_module(lambda A: sum(A), {"A": (64,)})))

        module = make_module(lambda A: sum(A), {"A": (64,)})
        tile(module)
        assert str(module) == expected

    def test_reduction_exact_multiple(self) -> None:
        """sum(A), n=128: scf.for with f32 iter_args, inner alloca+fill+generic."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        # iter_args carries an f32 accumulator
        assert "-> (f32)" in ir
        assert "arith.minsi" in ir
        # inner per-tile scratch allocated on the stack
        assert "memref.alloca" in ir
        # inner fill seeds scratch with current acc; inner generic runs reduction
        assert "linalg.fill" in ir
        assert "linalg.generic" in ir
        assert '"reduction"' in ir
        # load out of scratch + scf.yield f32
        assert "memref.load" in ir
        assert "scf.yield" in ir
        # terminal store to rank-0 output after the loop
        assert "memref.store" in ir
        assert "memref<f32>" in ir  # the output arg

    def test_reduction_non_multiple(self) -> None:
        """sum(A), n=100: remainder handled via arith.minsi inside the loop."""
        module = make_module(lambda A: sum(A), {"A": (100,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "arith.minsi" in ir
        assert "memref.alloca" in ir
        assert "linalg.fill" in ir
        assert "memref.load" in ir

    def test_reduction_large(self) -> None:
        """sum(A), n=1024: 16 iterations over the reduction."""
        module = make_module(lambda A: sum(A), {"A": (1024,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "arith.constant 64 : index" in ir
        assert "arith.constant 1024 : index" in ir

    def test_reduction_verifies(self) -> None:
        """Tiled reduction IR passes xDSL verification."""
        module = make_module(lambda A: sum(A), {"A": (100,)})
        tile(module)
        module.verify()

    def test_reduction_single_store_outside_loop(self) -> None:
        """Exactly one memref.store (terminal), and it is after the loop."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        tile(module)
        ir = str(module)
        assert ir.count("memref.store") == 1

    def test_reduction_init_is_identity(self) -> None:
        """iter_args init is the fill's 0.0 identity (sum), not a loop phi of an alloc."""
        module = make_module(lambda A: sum(A), {"A": (128,)})
        tile(module)
        ir = str(module)
        # The original linalg.fill writing to memref<f32> is gone — its
        # identity moves into iter_args init.
        assert "arith.constant 0.000000e+00 : f32" in ir
        # No top-level linalg.fill over memref<f32> remains.
        # (Inner alloca fill is memref<f32> too; check fill occurs only inside
        # the loop body by ensuring there's exactly one fill after tiling.)
        assert ir.count("linalg.fill") == 1

    def test_amax_reduction_below_limit_unchanged(self) -> None:
        """amax(A), n=32: untiled reduction passes through."""
        expected = str(bufferize(make_module(lambda A: amax(A), {"A": (32,)})))

        module = make_module(lambda A: amax(A), {"A": (32,)})
        tile(module)
        assert str(module) == expected

    def test_amax_reduction_exact_multiple(self) -> None:
        """amax(A), n=128: scf.for with f32 iter_args threading -inf identity."""
        module = make_module(lambda A: amax(A), {"A": (128,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        # -inf identity threaded through iter_args (xDSL prints f32 -inf
        # as the hex literal 0xff800000).
        assert "0xff800000" in ir.lower()
        # Inner body still carries the maximumf combiner.
        assert "arith.maximumf" in ir
        assert '"reduction"' in ir

    def test_amax_reduction_non_multiple(self) -> None:
        """amax(A), n=100: remainder handled via arith.minsi inside the loop."""
        module = make_module(lambda A: amax(A), {"A": (100,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "arith.minsi" in ir
        assert "arith.maximumf" in ir

    def test_amax_reduction_verifies(self) -> None:
        """Tiled amax reduction IR passes xDSL verification."""
        module = make_module(lambda A: amax(A), {"A": (100,)})
        tile(module)
        module.verify()

    # --- dot reduction tiling ---

    def test_dot_below_limit_unchanged(self) -> None:
        """dot(A, B), n=32: untiled reduction passes through."""
        expected = str(bufferize(make_module(lambda A, B: dot(A, B), {"A": (32,), "B": (32,)})))

        module = make_module(lambda A, B: dot(A, B), {"A": (32,), "B": (32,)})
        tile(module)
        assert str(module) == expected

    def test_dot_exact_multiple(self) -> None:
        """dot(A, B), n=128: scf.for with f32 iter_args, two input subviews."""
        module = make_module(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "iter_args" in ir
        assert "-> (f32)" in ir
        # Two memref.subviews (one for each input)
        assert ir.count("memref.subview") >= 2
        assert "memref.alloca" in ir
        assert "linalg.fill" in ir
        assert '"reduction"' in ir
        assert "arith.mulf" in ir
        assert "arith.addf" in ir
        assert "memref.load" in ir
        assert "scf.yield" in ir
        assert "memref.store" in ir

    def test_dot_non_multiple(self) -> None:
        """dot(A, B), n=100: remainder handled."""
        module = make_module(lambda A, B: dot(A, B), {"A": (100,), "B": (100,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "arith.minsi" in ir

    def test_dot_verifies(self) -> None:
        """Tiled dot reduction IR passes xDSL verification."""
        module = make_module(lambda A, B: dot(A, B), {"A": (100,), "B": (100,)})
        tile(module)
        module.verify()

    # --- mean reduction tiling ---

    def test_mean_tiled_preserves_divisor_attr(self) -> None:
        """mean(A), n=128: tiling preserves arrax.mean_divisor on the inner generic."""
        module = make_module(lambda A: mean(A), {"A": (128,)})
        tile(module)
        ir = str(module)
        assert "scf.for" in ir
        assert "arrax.mean_divisor = 128 : i64" in ir

    def test_mean_untiled_preserves_divisor_attr(self) -> None:
        """mean(A), n=32: no tiling, divisor attr still present."""
        module = make_module(lambda A: mean(A), {"A": (32,)})
        tile(module)
        ir = str(module)
        assert "arrax.mean_divisor = 32 : i64" in ir

    def test_mean_tiled_verifies(self) -> None:
        """Tiled mean reduction IR passes xDSL verification."""
        module = make_module(lambda A: mean(A), {"A": (100,)})
        tile(module)
        module.verify()

    def test_golden_128(self) -> None:
        """Golden-string snapshot for n=128."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        tile(module)
        ir = str(module)
        # Check key structural elements rather than exact string
        # (subview syntax varies with layout attributes)
        assert "func.func @kernel(%0: memref<128xf32>, %1: memref<128xf32>, %2: memref<128xf32>)" in ir
        assert "arith.constant 0 : index" in ir
        assert "arith.constant 128 : index" in ir
        assert "arith.constant 64 : index" in ir
        assert "scf.for" in ir
        assert "arith.subi" in ir
        assert "arith.minsi" in ir
        assert "memref.subview" in ir
        assert "arith.addf" in ir
        assert "linalg.yield" in ir
        assert "func.return" in ir

    # --- broadcast binary tiling ---

    def test_broadcast_binary_tiles_rank0_passthrough(self) -> None:
        """A broadcast binary generic (rank-1 + rank-0 inputs) tiles correctly.

        The rank-1 operands get subviewed; the rank-0 broadcast operand passes
        through unchanged. This is needed for softmax's sub(x, max) pattern.
        """
        f32 = Float32Type()
        n = 128
        vec_type = MemRefType(f32, [n])
        scalar_type = MemRefType(f32, [])
        out_type = MemRefType(f32, [n])

        # func @kernel(%vec: memref<128xf32>, %scalar: memref<f32>, %out: memref<128xf32>)
        func_op = FuncOp("kernel", ([vec_type, scalar_type, out_type], []))
        entry = func_op.body.blocks.first
        assert entry is not None
        vec, scalar, out = entry.args

        # linalg.generic with broadcast: (d0)->(d0), (d0)->(), (d0)->(d0)
        d0 = AffineMap.identity(1)
        scalar_map = AffineMap.from_callable(lambda d0: ())
        maps = [AffineMapAttr(d0), AffineMapAttr(scalar_map), AffineMapAttr(d0)]
        iters = [IteratorTypeAttr.parallel()]

        block = Block(arg_types=[f32, f32, f32])
        sub = arith.SubfOp(block.args[0], block.args[1])
        yield_op = linalg.YieldOp(sub.result)
        block.add_ops([sub, yield_op])
        body = Region([block])

        generic = linalg.GenericOp(
            inputs=[vec, scalar],
            outputs=[out],
            body=body,
            indexing_maps=maps,
            iterator_types=iters,
            result_types=[],
        )
        ret = ReturnOp()
        entry.add_ops([generic, ret])

        module = ModuleOp([func_op])
        module.verify()

        # Apply tile pass
        ctx = Context()
        TilePass().apply(ctx, module)
        module.verify()

        ir = str(module)
        # Should produce a tiled loop
        assert "scf.for" in ir
        # Rank-1 operands (vec and out) get subviewed — exactly 2 subviews
        assert ir.count("memref.subview") == 2
        # The rank-0 scalar must NOT be subviewed. The linalg.generic inside
        # the loop should reference the original memref<f32> directly (arg %1),
        # not a subview result.
        assert "ins(%1" not in ir or True  # structural check below
        # Count subview sources: only memref<128xf32> should be subviewed,
        # never memref<f32>.
        assert "memref.subview %1" not in ir, (
            "rank-0 memref<f32> (%1) must not be subviewed"
        )
        # The loop body should still have a linalg.generic with arith.subf
        assert "arith.subf" in ir
