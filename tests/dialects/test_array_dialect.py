"""Tests for arrax.dialects.array_dialect — array dialect definition."""

from __future__ import annotations

import pytest

from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from arrax.dialects.array_dialect import AddOp, ArrayDialect


class TestArrayDialect:
    def test_dialect_name(self) -> None:
        assert ArrayDialect.name == "array"

    def test_dialect_contains_add(self) -> None:
        assert AddOp in ArrayDialect._operations


class TestAddOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)

        assert op.lhs == lhs
        assert op.rhs == rhs
        assert op.result.type == tensor_type

    def test_result_type_matches_inputs(self) -> None:
        tensor_type = TensorType(Float32Type(), [512])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)

        assert op.result.type == tensor_type

    def test_verify_matching_types(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)
        op.verify()

    def test_verify_mismatched_shapes_fails(self) -> None:
        type_a = TensorType(Float32Type(), [1024])
        type_b = TensorType(Float32Type(), [512])
        lhs = create_ssa_value(type_a)
        rhs = create_ssa_value(type_b)
        op = AddOp(lhs, rhs)

        with pytest.raises(VerifyException, match="operand types must match"):
            op.verify()

    def test_verify_in_module(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)
        module = ModuleOp([lhs.owner, rhs.owner, op])
        module.verify()

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)
        module = ModuleOp([lhs.owner, rhs.owner, op])

        ir_text = str(module)
        assert "array.add" in ir_text
        assert "tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>" in ir_text

    def test_different_shapes(self) -> None:
        """AddOp works with different valid shapes."""
        for shape in [[1], [64], [4096]]:
            tensor_type = TensorType(Float32Type(), shape)
            lhs = create_ssa_value(tensor_type)
            rhs = create_ssa_value(tensor_type)
            op = AddOp(lhs, rhs)
            op.verify()
            assert op.result.type == tensor_type

    def test_2d_tensor(self) -> None:
        """AddOp works with 2D tensors (for future matmul support)."""
        tensor_type = TensorType(Float32Type(), [32, 64])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = AddOp(lhs, rhs)
        op.verify()
        assert op.result.type == tensor_type

    def test_ir_round_trips(self) -> None:
        """Verify the assembly_format parses back correctly."""
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func

        ir_text = """\
builtin.module {
  func.func @test(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = array.add %a, %b : tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>
    func.return %0 : tensor<1024xf32>
  }
}"""
        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(ArrayDialect)
        ctx.load_dialect(Func)
        module = Parser(ctx, ir_text).parse_module()
        module.verify()
