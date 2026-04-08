"""Tests for arrax.dialects.npu_dialect — NPU hardware operations."""

from __future__ import annotations

import pytest

from xdsl.dialects import arith
from xdsl.dialects.builtin import Float32Type, Float64Type, IndexType, IntegerAttr, MemRefType, ModuleOp
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from arrax.dialects.npu_dialect import (
    FVAddOp,
    FVDivOp,
    FVExpOp,
    FVMacOp,
    FVMaxOp,
    FVMulOp,
    FVReduceOp,
    FVReluOp,
    FRsqrtOp,
    FVSubOp,
    FVSubScalarOp,
    NPUDialect,
)


class TestNPUDialect:
    def test_dialect_name(self) -> None:
        assert NPUDialect.name == "npu"

    def test_dialect_contains_fvadd(self) -> None:
        assert FVAddOp in NPUDialect._operations

    def test_dialect_contains_fvsub(self) -> None:
        assert FVSubOp in NPUDialect._operations

    def test_dialect_contains_fvrelu(self) -> None:
        assert FVReluOp in NPUDialect._operations

    def test_dialect_contains_fvexp(self) -> None:
        assert FVExpOp in NPUDialect._operations


class TestFVAddOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVAddOp(src1, src2, dst, n)
        assert op.src1 == src1
        assert op.src2 == src2
        assert op.dst == dst
        assert op.n == n
        assert len(op.results) == 0

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVAddOp(src1, src2, dst, n)
        op.verify()

    def test_verify_in_module(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVAddOp(src1, src2, dst, n)
        module = ModuleOp([src1.owner, src2.owner, dst.owner, n.owner, op])
        module.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVAddOp(src1, src2, dst, n)
        module = ModuleOp([src1.owner, src2.owner, dst.owner, n.owner, op])

        ir = str(module)
        assert "npu.fvadd" in ir
        assert "memref<1024xf32>" in ir

    def test_no_results(self) -> None:
        """FVAddOp has no results — it writes to dst in-place."""
        memref_type = MemRefType(Float32Type(), [8])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVAddOp(src1, src2, dst, n)
        assert len(op.results) == 0

    def test_different_shapes(self) -> None:
        """FVAddOp works with different memref shapes."""
        for shape in [[1], [64], [4096]]:
            memref_type = MemRefType(Float32Type(), shape)
            index_type = IndexType()
            src1 = create_ssa_value(memref_type)
            src2 = create_ssa_value(memref_type)
            dst = create_ssa_value(memref_type)
            n = create_ssa_value(index_type)

            op = FVAddOp(src1, src2, dst, n)
            op.verify()

    def test_verify_mismatched_shapes_fails(self) -> None:
        type_a = MemRefType(Float32Type(), [1024])
        type_b = MemRefType(Float32Type(), [512])
        src1 = create_ssa_value(type_a)
        src2 = create_ssa_value(type_b)
        dst = create_ssa_value(type_a)
        n = create_ssa_value(IndexType())

        op = FVAddOp(src1, src2, dst, n)
        with pytest.raises(VerifyException, match="all memref operands must have the same type"):
            op.verify()

    def test_verify_wrong_element_type_fails(self) -> None:
        f64_memref = MemRefType(Float64Type(), [1024])
        src1 = create_ssa_value(f64_memref)
        src2 = create_ssa_value(f64_memref)
        dst = create_ssa_value(f64_memref)
        n = create_ssa_value(IndexType())

        op = FVAddOp(src1, src2, dst, n)
        with pytest.raises(VerifyException, match="expected f32 element type"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        """n > 64 (NPU vector limit) with a known constant triggers verify."""
        memref_type = MemRefType(Float32Type(), [128])
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))

        op = FVAddOp(src1, src2, dst, n_const.result)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_verify_n_at_limit_ok(self) -> None:
        """n = 64 (exactly at limit) passes verification."""
        memref_type = MemRefType(Float32Type(), [64])
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))

        op = FVAddOp(src1, src2, dst, n_const.result)
        op.verify()

    def test_ir_round_trips(self) -> None:
        """Verify the assembly_format parses back correctly."""
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func

        ir_text = """\
builtin.module {
  func.func @test(%s1: memref<1024xf32>, %s2: memref<1024xf32>, %d: memref<1024xf32>, %n: index) {
    npu.fvadd %s1, %s2, %d, %n : memref<1024xf32>, memref<1024xf32>, memref<1024xf32>, index
    func.return
  }
}"""
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(NPUDialect)
        ctx.load_dialect(Func)
        module = Parser(ctx, ir_text).parse_module()
        module.verify()


class TestFVSubOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVSubOp(src1, src2, dst, n)
        assert op.src1 == src1
        assert op.src2 == src2
        assert op.dst == dst
        assert op.n == n
        assert len(op.results) == 0

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVSubOp(src1, src2, dst, n)
        op.verify()

    def test_verify_mismatched_shapes_fails(self) -> None:
        type_a = MemRefType(Float32Type(), [1024])
        type_b = MemRefType(Float32Type(), [512])
        src1 = create_ssa_value(type_a)
        src2 = create_ssa_value(type_b)
        dst = create_ssa_value(type_a)
        n = create_ssa_value(IndexType())

        op = FVSubOp(src1, src2, dst, n)
        with pytest.raises(VerifyException, match="all memref operands must have the same type"):
            op.verify()

    def test_verify_wrong_element_type_fails(self) -> None:
        f64_memref = MemRefType(Float64Type(), [1024])
        src1 = create_ssa_value(f64_memref)
        src2 = create_ssa_value(f64_memref)
        dst = create_ssa_value(f64_memref)
        n = create_ssa_value(IndexType())

        op = FVSubOp(src1, src2, dst, n)
        with pytest.raises(VerifyException, match="expected f32 element type"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))

        op = FVSubOp(src1, src2, dst, n_const.result)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        index_type = IndexType()
        src1 = create_ssa_value(memref_type)
        src2 = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(index_type)

        op = FVSubOp(src1, src2, dst, n)
        module = ModuleOp([src1.owner, src2.owner, dst.owner, n.owner, op])

        ir = str(module)
        assert "npu.fvsub" in ir
        assert "memref<1024xf32>" in ir


class TestFVReluOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVReluOp(src, dst, n)
        assert op.src == src
        assert op.dst == dst
        assert op.n == n
        assert len(op.results) == 0

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVReluOp(src, dst, n)
        op.verify()

    def test_verify_mismatched_types_fails(self) -> None:
        type_a = MemRefType(Float32Type(), [1024])
        type_b = MemRefType(Float32Type(), [512])
        src = create_ssa_value(type_a)
        dst = create_ssa_value(type_b)
        n = create_ssa_value(IndexType())

        op = FVReluOp(src, dst, n)
        with pytest.raises(VerifyException, match="src and dst must have the same type"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))

        op = FVReluOp(src, dst, n_const.result)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVReluOp(src, dst, n)
        module = ModuleOp([src.owner, dst.owner, n.owner, op])

        ir = str(module)
        assert "npu.fvrelu" in ir


class TestFVReduceOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n, acc_in)
        assert op.src == src
        assert op.n == n
        assert op.acc_in == acc_in
        assert op.result.type == Float32Type()

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n, acc_in)
        op.verify()

    def test_verify_wrong_element_type_fails(self) -> None:
        f64_memref = MemRefType(Float64Type(), [64])
        src = create_ssa_value(f64_memref)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n, acc_in)
        with pytest.raises(VerifyException, match="expected f32 element type"):
            op.verify()

    def test_verify_rank0_src_fails(self) -> None:
        """FVReduceOp requires a rank-1 src memref."""
        src = create_ssa_value(MemRefType(Float32Type(), []))
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n, acc_in)
        with pytest.raises(VerifyException, match="rank-1"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n_const.result, acc_in)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_verify_n_at_limit_ok(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n_const.result, acc_in)
        op.verify()


    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVReduceOp(src, n, acc_in)
        module = ModuleOp([src.owner, n.owner, acc_in.owner, op])
        ir = str(module)
        assert "npu.fvreduce" in ir

    def test_ir_round_trips(self) -> None:
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.arith import Arith

        ir_text = """\
builtin.module {
  func.func @test(%src: memref<64xf32>) -> f32 {
    %n = arith.constant 64 : index
    %acc = arith.constant 0.0 : f32
    %r = npu.fvreduce %src, %n, %acc : memref<64xf32>, index, f32 -> f32
    func.return %r : f32
  }
}"""
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(NPUDialect)
        ctx.load_dialect(Func)
        ctx.load_dialect(Arith)
        module = Parser(ctx, ir_text).parse_module()
        module.verify()


class TestFVMaxOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n, acc_in)
        assert op.src == src
        assert op.n == n
        assert op.acc_in == acc_in
        assert op.result.type == Float32Type()

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n, acc_in)
        op.verify()

    def test_verify_wrong_element_type_fails(self) -> None:
        f64_memref = MemRefType(Float64Type(), [64])
        src = create_ssa_value(f64_memref)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n, acc_in)
        with pytest.raises(VerifyException, match="expected f32 element type"):
            op.verify()

    def test_verify_rank0_src_fails(self) -> None:
        src = create_ssa_value(MemRefType(Float32Type(), []))
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n, acc_in)
        with pytest.raises(VerifyException, match="rank-1"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n_const.result, acc_in)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_verify_n_at_limit_ok(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n_const.result, acc_in)
        op.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMaxOp(src, n, acc_in)
        module = ModuleOp([src.owner, n.owner, acc_in.owner, op])
        ir = str(module)
        assert "npu.fvmax" in ir

    def test_ir_round_trips(self) -> None:
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.arith import Arith

        ir_text = """\
builtin.module {
  func.func @test(%src: memref<64xf32>) -> f32 {
    %n = arith.constant 64 : index
    %acc = arith.constant 0xff800000 : f32
    %r = npu.fvmax %src, %n, %acc : memref<64xf32>, index, f32 -> f32
    func.return %r : f32
  }
}"""
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(NPUDialect)
        ctx.load_dialect(Func)
        ctx.load_dialect(Arith)
        module = Parser(ctx, ir_text).parse_module()
        module.verify()


class TestFVExpOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVExpOp(src, dst, n)
        assert op.src == src
        assert op.dst == dst
        assert op.n == n
        assert len(op.results) == 0

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVExpOp(src, dst, n)
        op.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [1024])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())

        op = FVExpOp(src, dst, n)
        module = ModuleOp([src.owner, dst.owner, n.owner, op])

        ir = str(module)
        assert "npu.fvexp" in ir


class TestFVMacOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        lhs = create_ssa_value(memref_type)
        rhs = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        assert op.lhs == lhs
        assert op.rhs == rhs
        assert op.n == n
        assert op.acc_in == acc_in
        assert op.result.type == Float32Type()

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        lhs = create_ssa_value(memref_type)
        rhs = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        op.verify()

    def test_verify_wrong_element_type_fails(self) -> None:
        f64_memref = MemRefType(Float64Type(), [64])
        lhs = create_ssa_value(f64_memref)
        rhs = create_ssa_value(f64_memref)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        with pytest.raises(VerifyException, match="expected f32 element type"):
            op.verify()

    def test_verify_rank0_src_fails(self) -> None:
        lhs = create_ssa_value(MemRefType(Float32Type(), []))
        rhs = create_ssa_value(MemRefType(Float32Type(), []))
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        with pytest.raises(VerifyException, match="rank-1"):
            op.verify()

    def test_verify_shape_mismatch_fails(self) -> None:
        lhs = create_ssa_value(MemRefType(Float32Type(), [32]))
        rhs = create_ssa_value(MemRefType(Float32Type(), [64]))
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        with pytest.raises(VerifyException, match="matching"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        lhs = create_ssa_value(memref_type)
        rhs = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n_const.result, acc_in)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_verify_n_at_limit_ok(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        lhs = create_ssa_value(memref_type)
        rhs = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(64, IndexType()))
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n_const.result, acc_in)
        op.verify()

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        lhs = create_ssa_value(memref_type)
        rhs = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        acc_in = create_ssa_value(Float32Type())

        op = FVMacOp(lhs, rhs, n, acc_in)
        module = ModuleOp([lhs.owner, rhs.owner, n.owner, acc_in.owner, op])
        ir = str(module)
        assert "npu.fvmac" in ir

    def test_ir_round_trips(self) -> None:
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.arith import Arith

        ir_text = """\
builtin.module {
  func.func @test(%lhs: memref<64xf32>, %rhs: memref<64xf32>) -> f32 {
    %n = arith.constant 64 : index
    %acc = arith.constant 0.000000e+00 : f32
    %r = npu.fvmac %lhs, %rhs, %n, %acc : memref<64xf32>, memref<64xf32>, index, f32 -> f32
    func.return %r : f32
  }
}"""
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(NPUDialect)
        ctx.load_dialect(Func)
        ctx.load_dialect(Arith)
        module = Parser(ctx, ir_text).parse_module()
        module.verify()


class TestFVMulOp:
    """Tests for the refactored FVMulOp (SSA scalar operand)."""

    def test_construction_with_ssa_scalar(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVMulOp(src, dst, n, scalar)
        assert op.src == src
        assert op.dst == dst
        assert op.n == n
        assert op.scalar == scalar

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVMulOp(src, dst, n, scalar)
        op.verify()

    def test_verify_mismatched_types_fails(self) -> None:
        type_a = MemRefType(Float32Type(), [64])
        type_b = MemRefType(Float32Type(), [32])
        src = create_ssa_value(type_a)
        dst = create_ssa_value(type_b)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVMulOp(src, dst, n, scalar)
        with pytest.raises(VerifyException, match="src and dst must have the same type"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))
        scalar = create_ssa_value(Float32Type())

        op = FVMulOp(src, dst, n_const.result, scalar)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()


class TestFVDivOp:
    """Tests for the refactored FVDivOp (SSA scalar operand)."""

    def test_construction_with_ssa_scalar(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVDivOp(src, dst, n, scalar)
        assert op.scalar == scalar

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVDivOp(src, dst, n, scalar)
        op.verify()


class TestFVSubScalarOp:
    def test_construction(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVSubScalarOp(src, dst, n, scalar)
        assert op.src == src
        assert op.dst == dst
        assert op.n == n
        assert op.scalar == scalar

    def test_verify(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVSubScalarOp(src, dst, n, scalar)
        op.verify()

    def test_verify_mismatched_types_fails(self) -> None:
        type_a = MemRefType(Float32Type(), [64])
        type_b = MemRefType(Float32Type(), [32])
        src = create_ssa_value(type_a)
        dst = create_ssa_value(type_b)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVSubScalarOp(src, dst, n, scalar)
        with pytest.raises(VerifyException, match="src and dst must have the same type"):
            op.verify()

    def test_verify_n_exceeds_limit_fails(self) -> None:
        memref_type = MemRefType(Float32Type(), [128])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n_const = arith.ConstantOp(IntegerAttr(128, IndexType()))
        scalar = create_ssa_value(Float32Type())

        op = FVSubScalarOp(src, dst, n_const.result, scalar)
        with pytest.raises(VerifyException, match="exceeds NPU vector length limit"):
            op.verify()

    def test_dialect_contains_op(self) -> None:
        assert FVSubScalarOp in NPUDialect._operations

    def test_ir_prints_correctly(self) -> None:
        memref_type = MemRefType(Float32Type(), [64])
        src = create_ssa_value(memref_type)
        dst = create_ssa_value(memref_type)
        n = create_ssa_value(IndexType())
        scalar = create_ssa_value(Float32Type())

        op = FVSubScalarOp(src, dst, n, scalar)
        module = ModuleOp([src.owner, dst.owner, n.owner, scalar.owner, op])
        ir = str(module)
        assert "npu.fvsub_scalar" in ir


class TestFRsqrtOp:
    def test_construction(self) -> None:
        src = create_ssa_value(Float32Type())

        op = FRsqrtOp(src)
        assert op.src == src
        assert op.result.type == Float32Type()

    def test_verify(self) -> None:
        src = create_ssa_value(Float32Type())

        op = FRsqrtOp(src)
        op.verify()

    def test_dialect_contains_op(self) -> None:
        assert FRsqrtOp in NPUDialect._operations

    def test_ir_prints_correctly(self) -> None:
        src = create_ssa_value(Float32Type())

        op = FRsqrtOp(src)
        module = ModuleOp([src.owner, op])
        ir = str(module)
        assert "npu.frsqrt" in ir
