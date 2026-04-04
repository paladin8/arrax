"""Tests for arrax.dialects.array_dialect — array dialect definition."""

from __future__ import annotations

import pytest

from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value

from arrax.dialects.array_dialect import (
    AddOp,
    ArrayDialect,
    DivScalarOp,
    ExpOp,
    MulScalarOp,
    ReluOp,
    SubOp,
)


class TestArrayDialect:
    def test_dialect_name(self) -> None:
        assert ArrayDialect.name == "array"

    def test_dialect_contains_add(self) -> None:
        assert AddOp in ArrayDialect._operations

    def test_dialect_contains_sub(self) -> None:
        assert SubOp in ArrayDialect._operations

    def test_dialect_contains_relu(self) -> None:
        assert ReluOp in ArrayDialect._operations

    def test_dialect_contains_exp(self) -> None:
        assert ExpOp in ArrayDialect._operations


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


class TestSubOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = SubOp(lhs, rhs)

        assert op.lhs == lhs
        assert op.rhs == rhs
        assert op.result.type == tensor_type

    def test_verify_matching_types(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = SubOp(lhs, rhs)
        op.verify()

    def test_verify_mismatched_shapes_fails(self) -> None:
        type_a = TensorType(Float32Type(), [1024])
        type_b = TensorType(Float32Type(), [512])
        lhs = create_ssa_value(type_a)
        rhs = create_ssa_value(type_b)
        op = SubOp(lhs, rhs)

        with pytest.raises(VerifyException, match="operand types must match"):
            op.verify()

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        lhs = create_ssa_value(tensor_type)
        rhs = create_ssa_value(tensor_type)
        op = SubOp(lhs, rhs)
        module = ModuleOp([lhs.owner, rhs.owner, op])

        ir_text = str(module)
        assert "array.sub" in ir_text
        assert "tensor<1024xf32>, tensor<1024xf32> -> tensor<1024xf32>" in ir_text


class TestReluOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        input_val = create_ssa_value(tensor_type)
        op = ReluOp(input_val)

        assert op.input == input_val
        assert op.result.type == tensor_type

    def test_verify(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = ReluOp(input_val)
        op.verify()

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        input_val = create_ssa_value(tensor_type)
        op = ReluOp(input_val)
        module = ModuleOp([input_val.owner, op])

        ir_text = str(module)
        assert "array.relu" in ir_text


class TestExpOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        input_val = create_ssa_value(tensor_type)
        op = ExpOp(input_val)

        assert op.input == input_val
        assert op.result.type == tensor_type

    def test_verify(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = ExpOp(input_val)
        op.verify()

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [1024])
        input_val = create_ssa_value(tensor_type)
        op = ExpOp(input_val)
        module = ModuleOp([input_val.owner, op])

        ir_text = str(module)
        assert "array.exp" in ir_text


class TestMulScalarOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = MulScalarOp(input_val, 3.0)

        assert op.input == input_val
        assert op.result.type == tensor_type

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = MulScalarOp(input_val, 3.0)
        module = ModuleOp([input_val.owner, op])

        ir_text = str(module)
        assert "array.mul_scalar" in ir_text
        assert "3.0" in ir_text


class TestDivScalarOp:
    def test_construction(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = DivScalarOp(input_val, 2.0)

        assert op.input == input_val
        assert op.result.type == tensor_type

    def test_ir_prints_correctly(self) -> None:
        tensor_type = TensorType(Float32Type(), [64])
        input_val = create_ssa_value(tensor_type)
        op = DivScalarOp(input_val, 2.0)
        module = ModuleOp([input_val.owner, op])

        ir_text = str(module)
        assert "array.div_scalar" in ir_text
        assert "2.0" in ir_text
