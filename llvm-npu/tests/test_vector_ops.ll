; RUN: llc -mtriple=riscv32 -mattr=+f,+xnpu -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Test vector binary, unary, scalar-vector, and dot product instructions.
; All use the XNpuVecGPR encoding: opcode=0x2B, funct3=0, all GPR operands.

;--- Binary in-place ops ---

declare void @llvm.riscv.npu.fvadd(ptr, ptr, i32)
declare void @llvm.riscv.npu.fvsub(ptr, ptr, i32)

; CHECK-LABEL: test_fvadd:
; CHECK:       npu.fvadd
define void @test_fvadd(ptr %src1, ptr %src2_dst, i32 %n) {
  call void @llvm.riscv.npu.fvadd(ptr %src1, ptr %src2_dst, i32 %n)
  ret void
}

; CHECK-LABEL: test_fvsub:
; CHECK:       npu.fvsub
define void @test_fvsub(ptr %src1, ptr %src2_dst, i32 %n) {
  call void @llvm.riscv.npu.fvsub(ptr %src1, ptr %src2_dst, i32 %n)
  ret void
}

;--- Unary ops ---

declare void @llvm.riscv.npu.fvexp(ptr, ptr, i32)
declare void @llvm.riscv.npu.fvrelu(ptr, ptr, i32)
declare void @llvm.riscv.npu.fvgelu(ptr, ptr, i32)

; CHECK-LABEL: test_fvexp:
; CHECK:       npu.fvexp
define void @test_fvexp(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvexp(ptr %src, ptr %dst, i32 %n)
  ret void
}

; CHECK-LABEL: test_fvrelu:
; CHECK:       npu.fvrelu
define void @test_fvrelu(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvrelu(ptr %src, ptr %dst, i32 %n)
  ret void
}

; CHECK-LABEL: test_fvgelu:
; CHECK:       npu.fvgelu
define void @test_fvgelu(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvgelu(ptr %src, ptr %dst, i32 %n)
  ret void
}

;--- Scalar-vector ops (facc pre-loaded) ---

declare void @llvm.riscv.npu.fvmul(ptr, ptr, i32)
declare void @llvm.riscv.npu.fvdiv(ptr, ptr, i32)
declare void @llvm.riscv.npu.fvsub.scalar(ptr, ptr, i32)

; CHECK-LABEL: test_fvmul:
; CHECK:       npu.fvmul
define void @test_fvmul(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvmul(ptr %src, ptr %dst, i32 %n)
  ret void
}

; CHECK-LABEL: test_fvdiv:
; CHECK:       npu.fvdiv
define void @test_fvdiv(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvdiv(ptr %src, ptr %dst, i32 %n)
  ret void
}

; CHECK-LABEL: test_fvsub_scalar:
; CHECK:       npu.fvsub.sc
define void @test_fvsub_scalar(ptr %src, ptr %dst, i32 %n) {
  call void @llvm.riscv.npu.fvsub.scalar(ptr %src, ptr %dst, i32 %n)
  ret void
}

;--- Dot product ---

declare void @llvm.riscv.npu.fvmac(ptr, ptr, i32)

; CHECK-LABEL: test_fvmac:
; CHECK:       npu.fvmac
define void @test_fvmac(ptr %lhs, ptr %rhs, i32 %n) {
  call void @llvm.riscv.npu.fvmac(ptr %lhs, ptr %rhs, i32 %n)
  ret void
}
