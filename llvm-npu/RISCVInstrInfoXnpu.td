//===-- RISCVInstrInfoXnpu.td - NPU FP Instructions -------*- tablegen -*-===//
//
// Instruction definitions for the Xnpu RISC-V vendor extension.
// All instructions use R-type encoding with opcode 0x2B (CUSTOM_1).
//
// Copy this file to llvm/lib/Target/RISCV/RISCVInstrInfoXnpu.td
// and add to RISCVInstrInfo.td:
//   include "RISCVInstrInfoXnpu.td"
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format classes
//===----------------------------------------------------------------------===//

// Vector op with 3 GPR operands, no register output.
// Used for elementwise, scalar-vector, and dot product instructions.
// Encoding: opcode=0x2B, funct3=0, rd=GPR(n), rs1=GPR(addr), rs2=GPR(addr)
class XNpuVecGPR<bits<7> funct7, string opcodestr>
    : RVInst<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
             opcodestr, "$rd, $rs1, $rs2", [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7}  = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let hasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 1;
}

// Reduction op: FPR output, GPR inputs.
// Encoding: opcode=0x2B, funct3=0, rd=FPR(result), rs1=GPR(src), rs2=GPR(n)
class XNpuReduce<bits<7> funct7, string opcodestr>
    : RVInst<(outs FPR32:$rd), (ins GPR:$rs1, GPR:$rs2),
             opcodestr, "$rd, $rs1, $rs2", [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;
  let Inst{11-7}  = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let hasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 0;
}

// Scalar FPR unary: FPR output, FPR input, rs2 hardcoded to 0.
// Parameterized by funct7 and funct3 to cover both groups.
class XNpuScalarFF<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInst<(outs FPR32:$rd), (ins FPR32:$rs1),
             opcodestr, "$rd, $rs1", [], InstFormatR> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-25} = funct7;
  let Inst{24-20} = 0b00000; // rs2 = f0 (unused)
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
}

//===----------------------------------------------------------------------===//
// Instruction definitions
//===----------------------------------------------------------------------===//

let Predicates = [HasVendorXnpu] in {

// --- Vector ops (funct3=0, all GPR, no register output) ---

// Binary in-place: dst[i] = src1[i] op dst[i]
def XNPU_FVADD         : XNpuVecGPR<0b0000111, "npu.fvadd">;     // funct7=0x07
def XNPU_FVSUB         : XNpuVecGPR<0b0001000, "npu.fvsub">;     // funct7=0x08

// Unary: dst[i] = f(src[i])
def XNPU_FVEXP         : XNpuVecGPR<0b0000010, "npu.fvexp">;     // funct7=0x02
def XNPU_FVRELU        : XNpuVecGPR<0b0001001, "npu.fvrelu">;    // funct7=0x09
def XNPU_FVGELU        : XNpuVecGPR<0b0001010, "npu.fvgelu">;    // funct7=0x0A

// Scalar-vector: dst[i] = f(src[i], facc)  (facc loaded separately)
def XNPU_FVMUL         : XNpuVecGPR<0b0000100, "npu.fvmul">;     // funct7=0x04
def XNPU_FVDIV         : XNpuVecGPR<0b0001011, "npu.fvdiv">;     // funct7=0x0B
def XNPU_FVSUB_SCALAR  : XNpuVecGPR<0b0001100, "npu.fvsub.sc">;  // funct7=0x0C

// Dot product: facc += dot(lhs[0..n], rhs[0..n])
def XNPU_FVMAC         : XNpuVecGPR<0b0000001, "npu.fvmac">;     // funct7=0x01

// --- Reductions (funct3=0, FPR result, GPR inputs) ---

def XNPU_FVREDUCE      : XNpuReduce<0b0000101, "npu.fvreduce">;  // funct7=0x05
def XNPU_FVMAX         : XNpuReduce<0b0000110, "npu.fvmax">;     // funct7=0x06

// --- Scalar FPR ops ---

// FRSQRT: rd = 1/sqrt(rs1)  (funct3=0, funct7=0x03)
// Hardware takes FPR rs1 (scalar f32 value), returns FPR rd.
def XNPU_FRSQRT : XNpuScalarFF<0b0000011, 0b000, "npu.frsqrt">;

// FRELU: rd = max(rs1, +0.0)  (funct3=1, funct7=0x00)
def XNPU_FRELU : XNpuScalarFF<0b0000000, 0b001, "npu.frelu">;

// FGELU: rd = gelu(rs1)  (funct3=4, funct7=0x00)
def XNPU_FGELU : XNpuScalarFF<0b0000000, 0b100, "npu.fgelu">;

// FMACC: facc += f[rs1] * f[rs2]  (funct3=0, funct7=0x00)
// No register output (result goes to implicit facc). rd hardcoded to 0.
def XNPU_FMACC
    : RVInst<(outs), (ins FPR32:$rs1, FPR32:$rs2),
             "npu.fmacc", "$rs1, $rs2", [], InstFormatR> {
  bits<5> rs2;
  bits<5> rs1;

  let Inst{31-25} = 0b0000000; // funct7=0x00
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b000;     // funct3=0
  let Inst{11-7}  = 0b00000;   // rd = f0 (unused)
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let hasSideEffects = 1; // writes implicit facc (inaccessible mem)
  let mayLoad = 1;
  let mayStore = 1;
}

// FRSTACC: rd = (float32)facc; facc = 0.0  (funct3=5, funct7=0x00)
// No register input. rs1 and rs2 hardcoded to 0.
def XNPU_FRSTACC
    : RVInst<(outs FPR32:$rd), (ins),
             "npu.frstacc", "$rd", [], InstFormatR> {
  bits<5> rd;

  let Inst{31-25} = 0b0000000; // funct7=0x00
  let Inst{24-20} = 0b00000;   // rs2 = 0
  let Inst{19-15} = 0b00000;   // rs1 = 0
  let Inst{14-12} = 0b101;     // funct3=5
  let Inst{11-7}  = rd;
  let Inst{6-0} = OPC_CUSTOM_1.Value;

  let hasSideEffects = 1; // reads/clears implicit facc (inaccessible mem)
  let mayLoad = 1;
  let mayStore = 1;
}

//===----------------------------------------------------------------------===//
// Instruction selection patterns (intrinsic -> instruction, 1:1)
//===----------------------------------------------------------------------===//

// Vector ops: (ptr rs1, ptr rs2, i32 rd) -> void
def : Pat<(int_riscv_npu_fvadd GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVADD GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvsub GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVSUB GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvexp GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVEXP GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvrelu GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVRELU GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvgelu GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVGELU GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvmul GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVMUL GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvdiv GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVDIV GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvsub_scalar GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVSUB_SCALAR GPR:$rs1, GPR:$rs2, GPR:$rd)>;
def : Pat<(int_riscv_npu_fvmac GPR:$rs1, GPR:$rs2, GPR:$rd),
          (XNPU_FVMAC GPR:$rs1, GPR:$rs2, GPR:$rd)>;

// Reductions: (ptr rs1, i32 rs2) -> float rd
def : Pat<(int_riscv_npu_fvreduce GPR:$rs1, GPR:$rs2),
          (XNPU_FVREDUCE GPR:$rs1, GPR:$rs2)>;
def : Pat<(int_riscv_npu_fvmax GPR:$rs1, GPR:$rs2),
          (XNPU_FVMAX GPR:$rs1, GPR:$rs2)>;

// Scalar FPR ops
def : Pat<(int_riscv_npu_frsqrt FPR32:$rs1),
          (XNPU_FRSQRT FPR32:$rs1)>;
def : Pat<(int_riscv_npu_frelu FPR32:$rs1),
          (XNPU_FRELU FPR32:$rs1)>;
def : Pat<(int_riscv_npu_fgelu FPR32:$rs1),
          (XNPU_FGELU FPR32:$rs1)>;
def : Pat<(int_riscv_npu_fmacc FPR32:$rs1, FPR32:$rs2),
          (XNPU_FMACC FPR32:$rs1, FPR32:$rs2)>;
def : Pat<(int_riscv_npu_frstacc),
          (XNPU_FRSTACC)>;

} // Predicates = [HasVendorXnpu]
