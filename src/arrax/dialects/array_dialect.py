"""The array dialect: high-level value-semantic array operations (IRDL)."""

from __future__ import annotations

from xdsl.ir import Dialect, SSAValue, Operation
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.dialects.builtin import TensorType
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


ArrayDialect = Dialect("array", [AddOp, SubOp], [])
