"""The npu dialect: hardware-specific NPU operations (IRDL).

Each operation maps 1:1 to an NPU instruction. Operations use a 3-address
form (src1, src2, dst, n) at the dialect level; the assembly emitter handles
the hardware's in-place semantics (copy + insn when dst != src2).
"""

from __future__ import annotations

from xdsl.dialects import arith
from xdsl.dialects.builtin import Float32Type, IndexType, IntegerAttr, MemRefType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
)
from xdsl.utils.exceptions import VerifyException

NPU_MAX_VEC_LEN = 64


@irdl_op_definition
class FVAddOp(IRDLOperation):
    """Elementwise vector addition: dst[i] = src1[i] + src2[i].

    Maps to NPU.FVADD (opcode=0x2B, funct7=0x07).
    """

    name = "npu.fvadd"

    src1 = operand_def(MemRefType)
    src2 = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)

    assembly_format = (
        "$src1 `,` $src2 `,` $dst `,` $n attr-dict"
        " `:` type($src1) `,` type($src2) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src1: SSAValue | Operation,
        src2: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
    ) -> None:
        super().__init__(operands=[src1, src2, dst, n], result_types=[])

    def verify_(self) -> None:
        src1_type = self.src1.type
        src2_type = self.src2.type
        dst_type = self.dst.type
        if src1_type != src2_type or src1_type != dst_type:
            raise VerifyException(
                f"npu.fvadd: all memref operands must have the same type, "
                f"got {src1_type}, {src2_type}, {dst_type}"
            )
        assert isinstance(src1_type, MemRefType)
        if not isinstance(src1_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvadd: expected f32 element type, got {src1_type.element_type}"
            )
        # When n is a known constant, enforce the hardware vector length limit
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvadd: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVSubOp(IRDLOperation):
    """Elementwise vector subtraction: dst[i] = src1[i] - src2[i].

    Maps to NPU.FVSUB (opcode=0x2B, funct7=0x08).
    """

    name = "npu.fvsub"

    src1 = operand_def(MemRefType)
    src2 = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)

    assembly_format = (
        "$src1 `,` $src2 `,` $dst `,` $n attr-dict"
        " `:` type($src1) `,` type($src2) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src1: SSAValue | Operation,
        src2: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
    ) -> None:
        super().__init__(operands=[src1, src2, dst, n], result_types=[])

    def verify_(self) -> None:
        src1_type = self.src1.type
        src2_type = self.src2.type
        dst_type = self.dst.type
        if src1_type != src2_type or src1_type != dst_type:
            raise VerifyException(
                f"npu.fvsub: all memref operands must have the same type, "
                f"got {src1_type}, {src2_type}, {dst_type}"
            )
        assert isinstance(src1_type, MemRefType)
        if not isinstance(src1_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvsub: expected f32 element type, got {src1_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvsub: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


NPUDialect = Dialect("npu", [FVAddOp, FVSubOp], [])
