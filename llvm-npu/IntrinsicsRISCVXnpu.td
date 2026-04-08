//===-- IntrinsicsRISCVXnpu.td - NPU FP Intrinsics --------*- tablegen -*-===//
//
// Intrinsic definitions for the Xnpu RISC-V vendor extension.
// Each intrinsic maps 1:1 to an NPU floating-point coprocessor instruction.
//
// All instructions use opcode 0x2B (CUSTOM_1) with R-type encoding.
// Vector ops use funct3=0 with varying funct7.
// Scalar ops (FRELU, FGELU, FRSTACC) use varying funct3.
//
// Include this file from llvm/include/llvm/IR/IntrinsicsRISCV.td:
//   include "IntrinsicsRISCVXnpu.td"
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//===----------------------------------------------------------------------===//
// Vector ops: all GPR operands, no register output.
// Encoding: .insn r 0x2B, 0, funct7, GPR(n), GPR(rs1), GPR(rs2)
//
// All share the signature (ptr, ptr, i32) -> void. The three arguments
// map to rs1, rs2, rd in the instruction encoding. Semantic meaning
// varies by instruction (see per-op comments).
//===----------------------------------------------------------------------===//

// FVADD: dst[i] += src1[i] (in-place on rs2)
// args: (ptr src1, ptr src2_dst, i32 n)
def int_riscv_npu_fvadd
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrArgMemOnly, IntrWillReturn]>;

// FVSUB: dst[i] = src1[i] - dst[i] (in-place on rs2)
// args: (ptr src1, ptr src2_dst, i32 n)
def int_riscv_npu_fvsub
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrArgMemOnly, IntrWillReturn]>;

// FVEXP: dst[i] = exp(src[i])
// args: (ptr src, ptr dst, i32 n)
def int_riscv_npu_fvexp
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrArgMemOnly, IntrWillReturn]>;

// FVRELU: dst[i] = max(src[i], 0)
// args: (ptr src, ptr dst, i32 n)
def int_riscv_npu_fvrelu
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrArgMemOnly, IntrWillReturn]>;

// FVGELU: dst[i] = gelu(src[i])
// args: (ptr src, ptr dst, i32 n)
def int_riscv_npu_fvgelu
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrArgMemOnly, IntrWillReturn]>;

// FVMUL: dst[i] = src[i] * facc
// args: (ptr src, ptr dst, i32 n)
// Caller must load facc before calling (via FRSTACC + FMACC).
def int_riscv_npu_fvmul
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn]>;

// FVDIV: dst[i] = src[i] / facc
// args: (ptr src, ptr dst, i32 n)
// Caller must load facc before calling.
def int_riscv_npu_fvdiv
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn]>;

// FVSUB_SCALAR: dst[i] = src[i] - facc
// args: (ptr src, ptr dst, i32 n)
// Caller must load facc before calling.
def int_riscv_npu_fvsub_scalar
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn]>;

// FVMAC: facc += dot(lhs[0..n], rhs[0..n])
// args: (ptr lhs, ptr rhs, i32 n)
// Accumulates into implicit facc register across calls.
def int_riscv_npu_fvmac
    : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
                 [IntrInaccessibleMemOrArgMemOnly, IntrWillReturn]>;

//===----------------------------------------------------------------------===//
// Reduction ops: FPR output, GPR inputs.
// Encoding: .insn r 0x2B, 0, funct7, FPR(result), GPR(src), GPR(n)
//
// Note the asymmetry with other vector ops: rd=FPR (result), rs2=GPR (n).
//===----------------------------------------------------------------------===//

// FVREDUCE: result = sum(src[0..n])
def int_riscv_npu_fvreduce
    : Intrinsic<[llvm_float_ty], [llvm_ptr_ty, llvm_i32_ty],
                 [IntrReadMem, IntrArgMemOnly, IntrWillReturn]>;

// FVMAX: result = max(src[0..n])
def int_riscv_npu_fvmax
    : Intrinsic<[llvm_float_ty], [llvm_ptr_ty, llvm_i32_ty],
                 [IntrReadMem, IntrArgMemOnly, IntrWillReturn]>;

//===----------------------------------------------------------------------===//
// Scalar FP ops
//===----------------------------------------------------------------------===//

// FMACC: facc += f[rs1] * f[rs2] (double-precision accumulator)
// Encoding: .insn r 0x2B, 0, 0x00, f0, FPR(a), FPR(b)
def int_riscv_npu_fmacc
    : Intrinsic<[], [llvm_float_ty, llvm_float_ty],
                 [IntrInaccessibleMemOnly, IntrWillReturn]>;

// FRSQRT: rd = 1/sqrt(rs1)
// Encoding: .insn r 0x2B, 0, 0x03, FPR(rd), FPR(rs1), f0
def int_riscv_npu_frsqrt
    : Intrinsic<[llvm_float_ty], [llvm_float_ty],
                 [IntrNoMem, IntrWillReturn]>;

// FRSTACC: rd = (float32)facc; facc = 0.0
// Encoding: .insn r 0x2B, 5, 0x00, FPR(rd), f0, f0
def int_riscv_npu_frstacc
    : Intrinsic<[llvm_float_ty], [],
                 [IntrInaccessibleMemOnly, IntrWillReturn]>;

// FRELU: rd = max(rs1, +0.0)
// Encoding: .insn r 0x2B, 1, 0x00, FPR(rd), FPR(rs1), f0
def int_riscv_npu_frelu
    : Intrinsic<[llvm_float_ty], [llvm_float_ty],
                 [IntrNoMem, IntrWillReturn]>;

// FGELU: rd = gelu(rs1)
// Encoding: .insn r 0x2B, 4, 0x00, FPR(rd), FPR(rs1), f0
def int_riscv_npu_fgelu
    : Intrinsic<[llvm_float_ty], [llvm_float_ty],
                 [IntrNoMem, IntrWillReturn]>;

} // TargetPrefix = "riscv"
