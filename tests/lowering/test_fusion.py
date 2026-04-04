"""Tests for arrax.lowering.fusion — post-tiling loop fusion."""

from __future__ import annotations

from arrax.dsl.array import Array, exp, relu
from tests.helpers import fuse, make_module, tile


class TestFusion:
    def test_chained_add_fuses(self) -> None:
        """(A + B) + C: two tiled loops fuse into one."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        # Two linalg.generic ops inside the single loop
        assert ir.count("linalg.generic") == 2

    def test_relu_of_add_fuses(self) -> None:
        """relu(A + B): binary + unary fuse into one loop."""
        module = make_module(lambda A, B: relu(A + B), {"A": (128,), "B": (128,)})
        fuse(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert ir.count("linalg.generic") == 2

    def test_three_way_fusion(self) -> None:
        """relu(A + B) - C: three ops fuse into one loop."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert ir.count("linalg.generic") == 3

    def test_no_fusion_below_limit(self) -> None:
        """n=32: no tiling means no loops, nothing to fuse."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        fuse(module)
        ir = str(module)
        assert "scf.for" not in ir
        assert ir.count("linalg.generic") == 2

    def test_single_op_unchanged(self) -> None:
        """A + B: single loop, nothing to fuse."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        fuse(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert ir.count("linalg.generic") == 1

    def test_cse_deduplicates_subi_minsi(self) -> None:
        """After fusion, redundant subi/minsi pairs are eliminated."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        ir = str(module)
        # Only one subi and one minsi (not two of each)
        assert ir.count("arith.subi") == 1
        assert ir.count("arith.minsi") == 1

    def test_dead_constants_cleaned(self) -> None:
        """Bound constants from erased loops don't affect verification."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        module.verify()

    def test_exp_of_sub_fuses(self) -> None:
        """exp(A - B): binary + unary fuse."""
        module = make_module(lambda A, B: exp(A - B), {"A": (128,), "B": (128,)})
        fuse(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert ir.count("linalg.generic") == 2

    def test_cse_deduplicates_subviews(self) -> None:
        """After fusion, duplicate subviews (same base, offset, size) are eliminated."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        tile(module)
        subviews_before = str(module).count("memref.subview")

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        subviews_after = str(module).count("memref.subview")

        # Fusion CSE should eliminate at least the duplicate intermediate subview
        assert subviews_after < subviews_before

    def test_cse_does_not_deduplicate_generics(self) -> None:
        """linalg.generic ops are not CSE'd (they have regions and side effects)."""
        def kernel(A: Array, B: Array) -> Array:
            return (A + B) + (A + B)

        module = make_module(kernel, {"A": (128,), "B": (128,)})
        fuse(module)
        ir = str(module)
        # All 3 generics preserved (two A+B adds + one final add)
        assert ir.count("linalg.generic") == 3
        # But redundant subviews are eliminated
        assert ir.count("memref.subview") < 9  # 9 without CSE

    def test_cse_does_not_merge_different_constants(self) -> None:
        """Constants with different values must not be CSE'd."""
        from arrax.lowering.fusion import _cse_key
        from xdsl.dialects import arith
        from xdsl.dialects.builtin import IndexType, IntegerAttr

        c0 = arith.ConstantOp(IntegerAttr(0, IndexType()))
        c1 = arith.ConstantOp(IntegerAttr(1, IndexType()))
        assert _cse_key(c0) != _cse_key(c1)

    def test_dead_constants_cleaned_after_fusion(self) -> None:
        """Second loop's bound constants are erased, not orphaned."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module)
        ir = str(module)
        # First loop uses 3 constants (0, 128, 64).
        # Second loop's 3 constants should be erased, not left dead.
        # Count index constants outside the loop body.
        constant_count = ir.count("arith.constant 0 : index")
        constant_count += ir.count("arith.constant 128 : index")
        constant_count += ir.count("arith.constant 64 : index")
        assert constant_count == 3, f"expected 3 bound constants, got {constant_count}"

    def test_fusion_preserves_correctness(self) -> None:
        """Fused IR still verifies and has correct structure."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        # Compare unfused vs fused: same number of generics, fewer loops
        module_unfused = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        tile(module_unfused)
        ir_unfused = str(module_unfused)

        module_fused = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module_fused)
        ir_fused = str(module_fused)

        # Same number of generics
        assert ir_unfused.count("linalg.generic") == ir_fused.count("linalg.generic")
        # Fewer loops after fusion
        assert ir_unfused.count("scf.for") == 2
        assert ir_fused.count("scf.for") == 1
