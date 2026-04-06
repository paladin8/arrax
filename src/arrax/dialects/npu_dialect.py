"""The npu dialect: hardware-specific NPU operations (IRDL).

Each operation maps 1:1 to an NPU instruction. Operations use a 3-address
form (src1, src2, dst, n) at the dialect level; the assembly emitter handles
the hardware's in-place semantics (copy + insn when dst != src2).
"""

from __future__ import annotations

from xdsl.dialects import arith
from xdsl.dialects.builtin import Float32Type, FloatAttr, IndexType, IntegerAttr, MemRefType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
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


@irdl_op_definition
class FVReluOp(IRDLOperation):
    """Elementwise vector ReLU: dst[i] = max(src[i], 0.0).

    Maps to NPU.FVRELU (opcode=0x2B, funct7=0x09).
    """

    name = "npu.fvrelu"

    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)

    assembly_format = (
        "$src `,` $dst `,` $n attr-dict"
        " `:` type($src) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
    ) -> None:
        super().__init__(operands=[src, dst, n], result_types=[])

    def verify_(self) -> None:
        src_type = self.src.type
        dst_type = self.dst.type
        if src_type != dst_type:
            raise VerifyException(
                f"npu.fvrelu: src and dst must have the same type, "
                f"got {src_type} and {dst_type}"
            )
        assert isinstance(src_type, MemRefType)
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvrelu: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvrelu: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVExpOp(IRDLOperation):
    """Elementwise vector exponential: dst[i] = exp(src[i]).

    Maps to NPU.FVEXP (opcode=0x2B, funct7=0x02).
    """

    name = "npu.fvexp"

    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)

    assembly_format = (
        "$src `,` $dst `,` $n attr-dict"
        " `:` type($src) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
    ) -> None:
        super().__init__(operands=[src, dst, n], result_types=[])

    def verify_(self) -> None:
        src_type = self.src.type
        dst_type = self.dst.type
        if src_type != dst_type:
            raise VerifyException(
                f"npu.fvexp: src and dst must have the same type, "
                f"got {src_type} and {dst_type}"
            )
        assert isinstance(src_type, MemRefType)
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvexp: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvexp: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVMulOp(IRDLOperation):
    """Vector multiply by scalar: dst[i] = src[i] * scalar.

    Maps to NPU.FVMUL (opcode=0x2B, funct7=0x04).
    The scalar is loaded into facc before execution.
    """

    name = "npu.fvmul"

    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)
    scalar = prop_def(FloatAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$src `,` $dst `,` $n attr-dict"
        " `:` type($src) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
        scalar: float,
    ) -> None:
        super().__init__(
            operands=[src, dst, n],
            result_types=[],
            properties={"scalar": FloatAttr(scalar, Float32Type())},
        )

    def verify_(self) -> None:
        src_type = self.src.type
        dst_type = self.dst.type
        if src_type != dst_type:
            raise VerifyException(
                f"npu.fvmul: src and dst must have the same type, "
                f"got {src_type} and {dst_type}"
            )
        assert isinstance(src_type, MemRefType)
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvmul: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvmul: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVDivOp(IRDLOperation):
    """Vector divide by scalar: dst[i] = src[i] / scalar.

    Maps to NPU.FVDIV (opcode=0x2B, funct7=0x0B).
    The scalar is loaded into facc before execution.
    """

    name = "npu.fvdiv"

    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    n = operand_def(IndexType)
    scalar = prop_def(FloatAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$src `,` $dst `,` $n attr-dict"
        " `:` type($src) `,` type($dst) `,` type($n)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        dst: SSAValue | Operation,
        n: SSAValue | Operation,
        scalar: float,
    ) -> None:
        super().__init__(
            operands=[src, dst, n],
            result_types=[],
            properties={"scalar": FloatAttr(scalar, Float32Type())},
        )

    def verify_(self) -> None:
        src_type = self.src.type
        dst_type = self.dst.type
        if src_type != dst_type:
            raise VerifyException(
                f"npu.fvdiv: src and dst must have the same type, "
                f"got {src_type} and {dst_type}"
            )
        assert isinstance(src_type, MemRefType)
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvdiv: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvdiv: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVReduceOp(IRDLOperation):
    """Chunk sum reduction: result = acc_in + sum(src[0..n]).

    Maps to NPU.FVREDUCE (opcode=0x2B, funct7=0x05) followed by fadd.s
    with the prior accumulator.

    The scalar accumulator is threaded through SSA (not a rank-0 memref) so
    it stays in an FP register across loop iterations.

    An optional `divisor` property encodes mean semantics: when set, the asm
    emitter emits a trailing `fdiv.s result, result, divisor` after the
    reduction loop closes.
    """

    name = "npu.fvreduce"

    src = operand_def(MemRefType)
    n = operand_def(IndexType)
    acc_in = operand_def(Float32Type)
    result = result_def(Float32Type)

    divisor = opt_prop_def(IntegerAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = (
        "$src `,` $n `,` $acc_in attr-dict"
        " `:` type($src) `,` type($n) `,` type($acc_in) `->` type($result)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        n: SSAValue | Operation,
        acc_in: SSAValue | Operation,
        divisor: int | None = None,
    ) -> None:
        properties: dict[str, object] = {}
        if divisor is not None:
            properties["divisor"] = IntegerAttr(divisor, 64)
        super().__init__(
            operands=[src, n, acc_in],
            result_types=[Float32Type()],
            properties=properties,
        )

    def verify_(self) -> None:
        src_type = self.src.type
        assert isinstance(src_type, MemRefType)
        if len(src_type.get_shape()) != 1:
            raise VerifyException(
                f"npu.fvreduce: src must be a rank-1 memref, got shape "
                f"{src_type.get_shape()}"
            )
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvreduce: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvreduce: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVMaxOp(IRDLOperation):
    """Chunk max reduction: result = max(acc_in, max(src[0..n])).

    Maps to NPU.FVMAX (opcode=0x2B, funct7=0x06) followed by an
    ``fmax.s`` combine with the prior accumulator.

    The scalar accumulator is threaded through SSA (not a rank-0 memref)
    so it stays in an FP register across loop iterations. NPU.FVMAX itself
    propagates NaN from any input element into its result register, but
    RISC-V ``fmax.s`` is NaN-suppressing when exactly one operand is NaN.
    The asm emitter therefore follows the ``fmax.s`` combine with a
    NaN-check that forces ``result := ft0`` whenever the FVMAX partial was
    NaN, preserving ``arith.maximumf`` / ``np.amax`` semantics end-to-end.
    """

    name = "npu.fvmax"

    src = operand_def(MemRefType)
    n = operand_def(IndexType)
    acc_in = operand_def(Float32Type)
    result = result_def(Float32Type)

    assembly_format = (
        "$src `,` $n `,` $acc_in attr-dict"
        " `:` type($src) `,` type($n) `,` type($acc_in) `->` type($result)"
    )

    def __init__(
        self,
        src: SSAValue | Operation,
        n: SSAValue | Operation,
        acc_in: SSAValue | Operation,
    ) -> None:
        super().__init__(
            operands=[src, n, acc_in],
            result_types=[Float32Type()],
        )

    def verify_(self) -> None:
        src_type = self.src.type
        assert isinstance(src_type, MemRefType)
        if len(src_type.get_shape()) != 1:
            raise VerifyException(
                f"npu.fvmax: src must be a rank-1 memref, got shape "
                f"{src_type.get_shape()}"
            )
        if not isinstance(src_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvmax: expected f32 element type, got {src_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvmax: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


@irdl_op_definition
class FVMacOp(IRDLOperation):
    """Chunk dot product: result = acc_in + dot(lhs[0..n], rhs[0..n]).

    Maps to NPU.FVMAC (opcode=0x2B, funct7=0x01) which accumulates into
    the hardware ``facc`` register across calls. The ``acc_in`` / ``result``
    SSA thread is cosmetic at the IR level — the real accumulation state
    lives in ``facc``. The asm emitter brackets the loop with FRSTACC
    (zero before, read after) and ignores the SSA thread during the body.
    """

    name = "npu.fvmac"

    lhs = operand_def(MemRefType)
    rhs = operand_def(MemRefType)
    n = operand_def(IndexType)
    acc_in = operand_def(Float32Type)
    result = result_def(Float32Type)

    assembly_format = (
        "$lhs `,` $rhs `,` $n `,` $acc_in attr-dict"
        " `:` type($lhs) `,` type($rhs) `,` type($n) `,` type($acc_in)"
        " `->` type($result)"
    )

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        n: SSAValue | Operation,
        acc_in: SSAValue | Operation,
    ) -> None:
        super().__init__(
            operands=[lhs, rhs, n, acc_in],
            result_types=[Float32Type()],
        )

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type
        if not isinstance(lhs_type, MemRefType):
            raise VerifyException(
                f"npu.fvmac: lhs must be a memref, got {lhs_type}"
            )
        if not isinstance(rhs_type, MemRefType):
            raise VerifyException(
                f"npu.fvmac: rhs must be a memref, got {rhs_type}"
            )
        if len(lhs_type.get_shape()) != 1:
            raise VerifyException(
                f"npu.fvmac: lhs must be rank-1 memref, got shape "
                f"{lhs_type.get_shape()}"
            )
        if len(rhs_type.get_shape()) != 1:
            raise VerifyException(
                f"npu.fvmac: rhs must be rank-1 memref, got shape "
                f"{rhs_type.get_shape()}"
            )
        if lhs_type.get_shape() != rhs_type.get_shape():
            raise VerifyException(
                f"npu.fvmac: lhs and rhs must have matching shapes, got "
                f"{lhs_type.get_shape()} and {rhs_type.get_shape()}"
            )
        if not isinstance(lhs_type.element_type, Float32Type):
            raise VerifyException(
                f"npu.fvmac: expected f32 element type, got {lhs_type.element_type}"
            )
        if isinstance(self.n.owner, arith.ConstantOp):
            n_attr = self.n.owner.value
            if isinstance(n_attr, IntegerAttr):
                n_val = n_attr.value.data
                if n_val > NPU_MAX_VEC_LEN:
                    raise VerifyException(
                        f"npu.fvmac: n={n_val} exceeds NPU vector length "
                        f"limit ({NPU_MAX_VEC_LEN})"
                    )


NPUDialect = Dialect(
    "npu",
    [FVAddOp, FVSubOp, FVReluOp, FVExpOp, FVMulOp, FVDivOp, FVReduceOp, FVMaxOp, FVMacOp],
    [],
)
