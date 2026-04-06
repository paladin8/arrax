"""Tests for arrax.lowering.tile — strip-mine tiling for NPU vector limit."""

from __future__ import annotations

from arrax.dsl.array import Array, amax, dot, sum
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
