"""Tests for arrax.lowering.buffer_optimize — buffer shrinking and reuse."""

from __future__ import annotations

from arrax.dsl.array import Array, amax, relu
from arrax.dsl.array import sum as arr_sum
from tests.helpers import fuse, make_module, optimize_buffers


class TestBufferShrink:
    def test_intermediate_shrinks(self) -> None:
        """(A + B) + C: intermediate shrinks from 128 to 64 elements."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        assert "memref.alloc() : memref<64xf32>" in ir
        # No 128-element alloc remains
        assert "memref.alloc() : memref<128xf32>" not in ir

    def test_no_shrink_below_limit(self) -> None:
        """n=32 (no tiling): intermediate stays at 32 elements."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (32,), "B": (32,), "C": (32,)})
        optimize_buffers(module)
        ir = str(module)
        assert "memref<32xf32>" in ir

    def test_subview_offset_becomes_zero(self) -> None:
        """After shrinking, subviews into the intermediate use offset 0."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        # The intermediate's subview should use constant 0, not the loop IV
        # Look for subview of a 64-element memref with a constant offset
        assert "memref.subview" in ir
        # All subviews of the shrunken buffer use offset from c0
        for line in ir.split("\n"):
            if "memref<64xf32>" in line and "memref.subview" in line:
                # Should reference a constant 0 offset, not the IV
                assert "[%0]" not in line or "64xf32" not in line

    def test_single_op_no_alloc(self) -> None:
        """A + B: no intermediate, nothing to shrink."""
        module = make_module(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        optimize_buffers(module)
        ir = str(module)
        assert "memref.alloc" not in ir

    def test_verifies_after_shrink(self) -> None:
        """Shrunken IR passes verification."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        module.verify()


class TestBufferReuse:
    def test_two_intermediates_reused(self) -> None:
        """relu(A + B) - C: two non-overlapping intermediates merged into one."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        alloc_count = ir.count("memref.alloc")
        assert alloc_count == 1, f"expected 1 alloc, got {alloc_count}"

    def test_reuse_alloc_is_tile_sized(self) -> None:
        """After shrink + reuse, the single alloc is 64 elements."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        assert "memref.alloc() : memref<64xf32>" in ir

    def test_no_reuse_single_intermediate(self) -> None:
        """(A + B) + C: only one intermediate, nothing to reuse."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        assert ir.count("memref.alloc") == 1

    def test_fewer_allocs_than_unfused(self) -> None:
        """Buffer optimization reduces alloc count vs fuse-only."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        module_fused = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        fuse(module_fused)
        fused_allocs = str(module_fused).count("memref.alloc")

        module_opt = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module_opt)
        opt_allocs = str(module_opt).count("memref.alloc")

        assert opt_allocs < fused_allocs

    def test_four_intermediates_two_slots(self) -> None:
        """relu(A+B) + relu(C-D): 4 intermediates collapse to 2 allocs.

        Intervals: tmp1=[5,7], tmp2=[7,15], tmp3=[11,13], tmp4=[13,15].
        tmp1+tmp2 share one slot, tmp3+tmp4 share another.
        A single-slot algorithm would leave 3 allocs.
        """
        def kernel(A: Array, B: Array, C: Array, D: Array) -> Array:
            return relu(A + B) + relu(C - D)

        module = make_module(
            kernel,
            {"A": (128,), "B": (128,), "C": (128,), "D": (128,)},
        )
        optimize_buffers(module)
        ir = str(module)
        alloc_count = ir.count("memref.alloc")
        assert alloc_count == 2, f"expected 2 allocs, got {alloc_count}"

    def test_verifies_after_reuse(self) -> None:
        """Reused-buffer IR passes verification."""
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        module.verify()

    def test_binary_op_aliasing_no_reuse(self) -> None:
        """sum((A + B) - C): add and sub intermediates must NOT share a buffer.

        The NPU binary sub copies ins[1] (C) into dst before computing
        dst = ins[0] - dst. If ins[0] and dst alias (same buffer), the
        copy destroys ins[0]. Two intermediate allocs must survive.
        """
        def kernel(A: Array, B: Array, C: Array) -> Array:
            return arr_sum((A + B) - C)

        module = make_module(kernel, {"A": (128,), "B": (128,), "C": (128,)})
        optimize_buffers(module)
        ir = str(module)
        # Two tile-sized allocs (add output + sub output), not merged
        assert ir.count("memref.alloc() : memref<64xf32>") == 2
        module.verify()


class TestBufferOptimizeFusedReduction:
    def test_sum_of_add_intermediate_shrinks(self) -> None:
        """sum(A + B): after fusion, intermediate buffer shrinks to 64."""
        module = make_module(
            lambda A, B: arr_sum(A + B), {"A": (128,), "B": (128,)}
        )
        optimize_buffers(module)
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert "memref.alloc() : memref<64xf32>" in ir
        assert "memref.alloc() : memref<128xf32>" not in ir

    def test_amax_of_sub_intermediate_shrinks(self) -> None:
        """amax(A - B): after fusion, intermediate shrinks."""
        module = make_module(
            lambda A, B: amax(A - B), {"A": (128,), "B": (128,)}
        )
        optimize_buffers(module)
        module.verify()
        ir = str(module)
        assert ir.count("scf.for") == 1
        assert "memref.alloc() : memref<128xf32>" not in ir

    def test_fused_reduction_verifies(self) -> None:
        """Post-fusion buffer-optimized reduction IR passes verification."""
        module = make_module(
            lambda A, B: arr_sum(A + B), {"A": (128,), "B": (128,)}
        )
        optimize_buffers(module)
        module.verify()
