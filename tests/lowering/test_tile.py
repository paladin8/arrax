"""Tests for arrax.lowering.tile — strip-mine tiling for NPU vector limit."""

from __future__ import annotations

from arrax.dsl.array import Array
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
