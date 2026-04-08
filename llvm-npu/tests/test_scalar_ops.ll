; RUN: llc -mtriple=riscv32 -mattr=+f,+xnpu -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Test scalar FPR instructions: FMACC, FRSQRT, FRSTACC, FRELU, FGELU.

declare void @llvm.riscv.npu.fmacc(float, float)
declare float @llvm.riscv.npu.frsqrt(float)
declare float @llvm.riscv.npu.frstacc()
declare float @llvm.riscv.npu.frelu(float)
declare float @llvm.riscv.npu.fgelu(float)

; CHECK-LABEL: test_fmacc:
; CHECK:       npu.fmacc
define void @test_fmacc(float %a, float %b) {
  call void @llvm.riscv.npu.fmacc(float %a, float %b)
  ret void
}

; CHECK-LABEL: test_frsqrt:
; CHECK:       npu.frsqrt
define float @test_frsqrt(float %x) {
  %result = call float @llvm.riscv.npu.frsqrt(float %x)
  ret float %result
}

; CHECK-LABEL: test_frstacc:
; CHECK:       npu.frstacc
define float @test_frstacc() {
  %result = call float @llvm.riscv.npu.frstacc()
  ret float %result
}

; CHECK-LABEL: test_frelu:
; CHECK:       npu.frelu
define float @test_frelu(float %x) {
  %result = call float @llvm.riscv.npu.frelu(float %x)
  ret float %result
}

; CHECK-LABEL: test_fgelu:
; CHECK:       npu.fgelu
define float @test_fgelu(float %x) {
  %result = call float @llvm.riscv.npu.fgelu(float %x)
  ret float %result
}

; Test facc sequence: zero facc, load scalar via FMACC, read via FRSTACC
; CHECK-LABEL: test_facc_load_sequence:
; CHECK:       npu.frstacc
; CHECK:       npu.fmacc
; CHECK:       npu.frstacc
define float @test_facc_load_sequence(float %scalar) {
  ; Zero facc (discard result)
  %discard = call float @llvm.riscv.npu.frstacc()
  ; facc += scalar * 1.0
  call void @llvm.riscv.npu.fmacc(float %scalar, float 1.0)
  ; Read facc
  %result = call float @llvm.riscv.npu.frstacc()
  ret float %result
}

; Test FRSQRT used in a computation chain
; CHECK-LABEL: test_frsqrt_chain:
; CHECK:       npu.frsqrt
; CHECK:       fmul.s
define float @test_frsqrt_chain(float %x, float %y) {
  %rsqrt = call float @llvm.riscv.npu.frsqrt(float %x)
  %result = fmul float %y, %rsqrt
  ret float %result
}
