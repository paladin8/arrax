"""The array dialect: high-level value-semantic array operations (IRDL)."""

from __future__ import annotations

from xdsl.ir import Dialect, SSAValue, Operation
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.dialects.builtin import Float32Type, FloatAttr, TensorType
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class AddOp(IRDLOperation):
    """Elementwise addition of two tensors."""

    name = "array.add"

    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ) -> None:
        lhs_val = SSAValue.get(lhs)
        super().__init__(operands=[lhs, rhs], result_types=[lhs_val.type])

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        if lhs_type != rhs_type:
            raise VerifyException(
                f"operand types must match, got {lhs_type} and {rhs_type}"
            )


@irdl_op_definition
class SubOp(IRDLOperation):
    """Elementwise subtraction of two tensors."""

    name = "array.sub"

    lhs = operand_def(TensorType)
    rhs = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = (
        "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)"
    )

    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
    ) -> None:
        lhs_val = SSAValue.get(lhs)
        super().__init__(operands=[lhs, rhs], result_types=[lhs_val.type])

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        if lhs_type != rhs_type:
            raise VerifyException(
                f"operand types must match, got {lhs_type} and {rhs_type}"
            )


@irdl_op_definition
class ReluOp(IRDLOperation):
    """Elementwise ReLU: max(input, 0)."""

    name = "array.relu"

    input = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation) -> None:
        input_val = SSAValue.get(input)
        super().__init__(operands=[input], result_types=[input_val.type])


@irdl_op_definition
class ExpOp(IRDLOperation):
    """Elementwise exponential."""

    name = "array.exp"

    input = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation) -> None:
        input_val = SSAValue.get(input)
        super().__init__(operands=[input], result_types=[input_val.type])


@irdl_op_definition
class MulScalarOp(IRDLOperation):
    """Elementwise multiply by scalar: result[i] = input[i] * scalar."""

    name = "array.mul_scalar"

    input = operand_def(TensorType)
    result = result_def(TensorType)
    scalar = prop_def(FloatAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$input attr-dict `:` type($input) `->` type($result)"
    )

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation, scalar: float) -> None:
        input_val = SSAValue.get(input)
        super().__init__(
            operands=[input],
            result_types=[input_val.type],
            properties={"scalar": FloatAttr(scalar, Float32Type())},
        )


@irdl_op_definition
class DivScalarOp(IRDLOperation):
    """Elementwise divide by scalar: result[i] = input[i] / scalar."""

    name = "array.div_scalar"

    input = operand_def(TensorType)
    result = result_def(TensorType)
    scalar = prop_def(FloatAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$input attr-dict `:` type($input) `->` type($result)"
    )

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation, scalar: float) -> None:
        input_val = SSAValue.get(input)
        super().__init__(
            operands=[input],
            result_types=[input_val.type],
            properties={"scalar": FloatAttr(scalar, Float32Type())},
        )


@irdl_op_definition
class SumOp(IRDLOperation):
    """Sum reduction: result = sum(input[i] for i in range(n)).

    Takes a rank-1 f32 tensor and produces a rank-0 f32 tensor.
    """

    name = "array.sum"

    input = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation) -> None:
        input_val = SSAValue.get(input)
        result_type = TensorType(Float32Type(), [])
        super().__init__(operands=[input], result_types=[result_type])

    def verify_(self) -> None:
        input_type = self.input.type
        result_type = self.result.type
        assert isinstance(input_type, TensorType)
        assert isinstance(result_type, TensorType)
        if len(input_type.get_shape()) != 1:
            raise VerifyException(
                f"array.sum: input must be rank-1, got shape {input_type.get_shape()}"
            )
        if not isinstance(input_type.element_type, Float32Type):
            raise VerifyException(
                f"array.sum: expected f32 element type, got {input_type.element_type}"
            )
        if len(result_type.get_shape()) != 0:
            raise VerifyException(
                f"array.sum: result must be rank-0, got shape {result_type.get_shape()}"
            )
        if not isinstance(result_type.element_type, Float32Type):
            raise VerifyException(
                f"array.sum: expected f32 result element type, got {result_type.element_type}"
            )


@irdl_op_definition
class AmaxOp(IRDLOperation):
    """Max reduction: result = max(input[i] for i in range(n)).

    Takes a rank-1 f32 tensor and produces a rank-0 f32 tensor.
    Named after NumPy's ``amax`` to avoid shadowing Python's ``max`` builtin.
    """

    name = "array.amax"

    input = operand_def(TensorType)
    result = result_def(TensorType)

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue | Operation) -> None:
        input_val = SSAValue.get(input)
        result_type = TensorType(Float32Type(), [])
        super().__init__(operands=[input], result_types=[result_type])

    def verify_(self) -> None:
        input_type = self.input.type
        result_type = self.result.type
        assert isinstance(input_type, TensorType)
        assert isinstance(result_type, TensorType)
        if len(input_type.get_shape()) != 1:
            raise VerifyException(
                f"array.amax: input must be rank-1, got shape {input_type.get_shape()}"
            )
        if not isinstance(input_type.element_type, Float32Type):
            raise VerifyException(
                f"array.amax: expected f32 element type, got {input_type.element_type}"
            )
        if len(result_type.get_shape()) != 0:
            raise VerifyException(
                f"array.amax: result must be rank-0, got shape {result_type.get_shape()}"
            )
        if not isinstance(result_type.element_type, Float32Type):
            raise VerifyException(
                f"array.amax: expected f32 result element type, got {result_type.element_type}"
            )


ArrayDialect = Dialect(
    "array",
    [AddOp, SubOp, ReluOp, ExpOp, MulScalarOp, DivScalarOp, SumOp, AmaxOp],
    [],
)
