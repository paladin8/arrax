"""arrax: MLIR-based compiler for fused array expressions targeting riscv-npu."""

from arrax.dsl.array import Array, exp, relu
from arrax.pipeline import compile_to_asm

__all__ = ["Array", "compile_to_asm", "exp", "relu"]
