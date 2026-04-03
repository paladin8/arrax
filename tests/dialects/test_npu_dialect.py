"""Tests for arrax.dialects.npu_dialect — NPU hardware operations."""

from __future__ import annotations

import pytest

from xdsl.dialects import arith
from xdsl.dialects.builtin import Float32Type, Float64Type, IndexType, IntegerAttr, MemRefType, ModuleOp
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from arrax.dialects.npu_dialect import FVAddOp, FVSubOp, NPUDialect


class TestNPUDialect:
    def test_dialect_name(self) -> None:
        assert NPUDialect.name == "npu"

    def test_dialect_contains_fvadd(self) -> None:
        assert FVAddOp in NPUDialect._operations

    def test_dialect_contains_fvsub(self) -> None:
        assert FVSubOp in NPUDialect._operations


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
