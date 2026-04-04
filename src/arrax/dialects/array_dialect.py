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


ArrayDialect = Dialect(
    "array", [AddOp, SubOp, ReluOp, ExpOp, MulScalarOp, DivScalarOp], []
)
