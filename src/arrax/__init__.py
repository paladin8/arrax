"""arrax: MLIR-based compiler for fused array expressions targeting riscv-npu."""

from arrax.dsl.array import Array, amax, dot, exp, relu, sum
from arrax.pipeline import compile_to_asm

__all__ = ["Array", "amax", "compile_to_asm", "dot", "exp", "relu", "sum"]
