"""Golden tests for the LLVM IR emitter.

Verifies that known Python expressions produce LLVM IR containing the
expected intrinsic calls, loop structure, and scalar operations. These
tests do NOT require a patched llc — they only check the .ll text output.
"""

from __future__ import annotations

import pytest

from arrax import amax, dot, exp, relu, rmsnorm, softmax, sum as asum
from arrax.pipeline import compile_to_asm


def _ll(fn, shapes):
    """Compile to LLVM IR text via the llvm backend."""
    text, _ = compile_to_asm(fn, shapes, backend="llvm")
    return text


class TestLlvmEmitterBasic:
    """Untiled operations (N <= 64)."""

    def test_add_small(self):
        ll = _ll(lambda A, B: A + B, {"A": (4,), "B": (4,)})
        assert "llvm.riscv.npu.fvadd" in ll
        assert "define void" in ll

    def test_sub_small(self):
        ll = _ll(lambda A, B: A - B, {"A": (4,), "B": (4,)})
        assert "llvm.riscv.npu.fvsub" in ll

    def test_relu_small(self):
        ll = _ll(lambda A: relu(A), {"A": (4,)})
        assert "llvm.riscv.npu.fvrelu" in ll

    def test_exp_small(self):
        ll = _ll(lambda A: exp(A), {"A": (4,)})
        assert "llvm.riscv.npu.fvexp" in ll

    def test_mul_scalar_small(self):
        ll = _ll(lambda A: A * 0.5, {"A": (4,)})
        assert "llvm.riscv.npu.fvmul" in ll
        assert "llvm.riscv.npu.fmacc" in ll  # facc load sequence
        assert "llvm.riscv.npu.frstacc" in ll

    def test_div_scalar_small(self):
        ll = _ll(lambda A: A / 2.0, {"A": (4,)})
        assert "llvm.riscv.npu.fvdiv" in ll

    def test_sum_small(self):
        ll = _ll(lambda A: asum(A), {"A": (4,)})
        assert "llvm.riscv.npu.fvreduce" in ll
        assert "fadd float" in ll  # accumulator combine

    def test_amax_small(self):
        ll = _ll(lambda A: amax(A), {"A": (4,)})
        assert "llvm.riscv.npu.fvmax" in ll
        assert "llvm.maxnum.f32" in ll  # NaN-safe combine

    def test_dot_small(self):
        ll = _ll(lambda A, B: dot(A, B), {"A": (4,), "B": (4,)})
        assert "llvm.riscv.npu.fvmac" in ll
        assert "llvm.riscv.npu.frstacc" in ll  # FRSTACC bracket
        assert "for.header" not in ll  # untiled: no loop


class TestLlvmEmitterTiled:
    """Tiled operations (N > 64)."""

    def test_add_tiled(self):
        ll = _ll(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        assert "llvm.riscv.npu.fvadd" in ll
        assert "for.header" in ll  # loop structure
        assert "phi" in ll  # IV phi node

    def test_dot_tiled(self):
        ll = _ll(lambda A, B: dot(A, B), {"A": (128,), "B": (128,)})
        assert "llvm.riscv.npu.fvmac" in ll
        assert "llvm.riscv.npu.frstacc" in ll
        assert "for.header" in ll

    def test_sum_tiled(self):
        ll = _ll(lambda A: asum(A), {"A": (128,)})
        assert "llvm.riscv.npu.fvreduce" in ll
        assert "for.header" in ll
        assert "phi" in ll  # accumulator phi


class TestLlvmEmitterComposite:
    """Composite operations (softmax)."""

    def test_softmax(self):
        ll = _ll(lambda A: softmax(A), {"A": (128,)})
        # softmax = amax + sub_scalar + exp + sum + div
        assert "llvm.riscv.npu.fvmax" in ll
        assert "llvm.riscv.npu.fvsub.scalar" in ll
        assert "llvm.riscv.npu.fvexp" in ll
        assert "llvm.riscv.npu.fvreduce" in ll
        assert "llvm.riscv.npu.fvdiv" in ll

    def test_rmsnorm(self):
        ll = _ll(lambda A: rmsnorm(A), {"A": (128,)})
        # rmsnorm = dot(x,x) + divf + addf + frsqrt + broadcast-mul
        assert "llvm.riscv.npu.fvmac" in ll
        assert "llvm.riscv.npu.frsqrt" in ll
        assert "llvm.riscv.npu.fvmul" in ll
        assert "fdiv float" in ll
        assert "fadd float" in ll

    def test_fused_expression(self):
        """A * 0.5 + B — fused scalar-mul then add."""
        ll = _ll(lambda A, B: A * 0.5 + B, {"A": (128,), "B": (128,)})
        assert "llvm.riscv.npu.fvmul" in ll
        assert "llvm.riscv.npu.fvadd" in ll


class TestLlvmEmitterDefensive:
    """Test handlers for ops that don't appear in the current pipeline but
    could appear if lowering changes (alloca, load)."""

    def test_alloca_and_load(self):
        """Build IR with memref.alloca + memref.load, verify emission."""
        from xdsl.dialects import func as func_d, memref as memref_d
        from xdsl.dialects.builtin import (
            Float32Type,
            FunctionType,
            MemRefType,
            ModuleOp,
        )
        from xdsl.ir import Block, Region

        f32 = Float32Type()
        memref_0d = MemRefType(f32, [])

        # func.func @test(%out: memref<f32>) {
        #   %scratch = memref.alloca() : memref<f32>
        #   %val = memref.load %scratch[] : memref<f32>
        #   memref.store %val, %out[] : memref<f32>
        #   return
        # }
        body = Block(arg_types=[memref_0d])
        out_arg = body.args[0]
        alloca = memref_d.AllocaOp.get(f32, shape=[])
        load = memref_d.LoadOp.get(alloca.memref, [])
        store = memref_d.StoreOp.get(load.res, out_arg, [])
        ret = func_d.ReturnOp()
        body.add_ops([alloca, load, store, ret])

        func_op = func_d.FuncOp(
            "test",
            FunctionType.from_lists([memref_0d], []),
            Region([body]),
        )
        module = ModuleOp([func_op])

        from arrax.codegen.llvm_emitter import emit_llvm_ir
        ll = emit_llvm_ir(module)
        assert "alloca float" in ll
        assert "load float" in ll
        assert "store float" in ll


class TestLlvmEmitterStructure:
    """Verify structural properties of the emitted IR."""

    def test_target_triple(self):
        ll = _ll(lambda A, B: A + B, {"A": (4,), "B": (4,)})
        assert 'target triple = "riscv32-unknown-none-elf"' in ll

    def test_function_signature(self):
        ll = _ll(lambda A, B: A + B, {"A": (4,), "B": (4,)})
        assert "define void @" in ll
        assert "ptr" in ll

    def test_memcpy_for_inplace(self):
        """Binary ops (fvadd) need memcpy when src2 != dst."""
        ll = _ll(lambda A, B: A + B, {"A": (4,), "B": (4,)})
        assert "llvm.memcpy" in ll

    def test_no_register_allocation(self):
        """LLVM IR should NOT contain s-registers or t-registers."""
        ll = _ll(lambda A, B: A + B, {"A": (128,), "B": (128,)})
        assert " s0" not in ll
        assert " t0" not in ll
        assert " a0" not in ll

    def test_asm_backend_unchanged(self):
        """The asm backend should still work and not be affected."""
        asm, _ = compile_to_asm(
            lambda A, B: A + B, {"A": (4,), "B": (4,)}, backend="asm"
        )
        assert ".insn r 0x2B" in asm
