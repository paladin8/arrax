; RUN: llc -mtriple=riscv32 -mattr=+f,+xnpu -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Test reduction instructions: FVREDUCE and FVMAX.
; These use the XNpuReduce encoding: FPR output, GPR inputs.
; Encoding: .insn r 0x2B, 0, funct7, FPR(result), GPR(src), GPR(n)

declare float @llvm.riscv.npu.fvreduce(ptr, i32)
declare float @llvm.riscv.npu.fvmax(ptr, i32)

; CHECK-LABEL: test_fvreduce:
; CHECK:       npu.fvreduce
define float @test_fvreduce(ptr %src, i32 %n) {
  %result = call float @llvm.riscv.npu.fvreduce(ptr %src, i32 %n)
  ret float %result
}

; CHECK-LABEL: test_fvmax:
; CHECK:       npu.fvmax
define float @test_fvmax(ptr %src, i32 %n) {
  %result = call float @llvm.riscv.npu.fvmax(ptr %src, i32 %n)
  ret float %result
}

; Test reduction used in a computation (result feeds into fadd.s)
; CHECK-LABEL: test_fvreduce_accumulate:
; CHECK:       npu.fvreduce
; CHECK:       fadd.s
define float @test_fvreduce_accumulate(ptr %src, i32 %n, float %acc) {
  %partial = call float @llvm.riscv.npu.fvreduce(ptr %src, i32 %n)
  %result = fadd float %acc, %partial
  ret float %result
}

; Test reduction used with fmax.s
; CHECK-LABEL: test_fvmax_accumulate:
; CHECK:       npu.fvmax
define float @test_fvmax_accumulate(ptr %src, i32 %n, float %acc) {
  %partial = call float @llvm.riscv.npu.fvmax(ptr %src, i32 %n)
  %cmp = fcmp ogt float %partial, %acc
  %result = select i1 %cmp, float %partial, float %acc
  ret float %result
}
