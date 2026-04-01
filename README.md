# arrax

An MLIR-based compiler that fuses array expressions in Python and emits optimized RISC-V assembly targeting a custom neural processing unit ([riscv-npu](https://github.com/paladin8/riscv-npu)).

## What it does

Write array math in Python. arrax traces the expression graph, fuses operations to eliminate intermediate allocations, and compiles directly to NPU instructions — producing firmware that runs on the riscv-npu emulator.

```python
from arrax import Array, compile

def my_kernel(A: Array, B: Array, C: Array) -> Array:
    return (A + B) * C

binary = compile(my_kernel, shapes={"A": (1024,), "B": (1024,), "C": (1024,)})
```

NumPy would execute `A + B` (allocate temp), then `temp * C` (allocate another temp). arrax fuses this into a single pass — one NPU instruction sequence, zero intermediate buffers.

## Compiler pipeline

```
Python DSL (tracing)
  │
  ▼
array dialect (fused expression DAGs on abstract arrays)
  │
  ▼
linalg dialect (structured ops with indexing maps)
  │  ── tiling pass (fit memory budget)
  │  ── fusion pass (merge adjacent elementwise ops)
  │
  ▼
linalg on memrefs (after bufferization)
  │
  ▼
npu dialect (1:1 NPU instructions: fvmac, fvexp, ...)
  │
  ▼
RISC-V assembly (.S with .insn directives)
  │
  ▼
riscv32 toolchain → ELF → riscv-npu emulator
```

Built on [xDSL](https://github.com/xdslproject/xdsl), a Python-native MLIR framework.

## Supported operations

| Category           | Operations                    | Notes                                  |
|--------------------|-------------------------------|----------------------------------------|
| Elementwise binary | add, sub, mul, div            | Array-array and scalar-array           |
| Elementwise unary  | neg, relu, gelu, exp          | Maps to NPU activation instructions    |
| Reductions         | sum, max                      | Full-array reductions                  |
| Dot product        | dot(A, B)                     | 1D, maps to FVMAC + FRSTACC           |
| Matrix multiply    | A @ B                         | 2D, via tiled dot products             |
| Composite          | softmax(A), rmsnorm(A, gamma) | Multi-instruction lowering patterns    |

All shapes are static (known at compile time).

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/paladin8/arrax.git
cd arrax
uv sync
```

This installs runtime dependencies and the `dev` dependency group (pytest) by default.

### RISC-V toolchain

End-to-end compilation requires the RISC-V cross-compiler:

```bash
# Ubuntu/Debian (the 64-bit toolchain targets rv32 with -march=rv32imf)
sudo apt install gcc-riscv64-unknown-elf

# Or build riscv-gnu-toolchain from source for rv32imf
```

### riscv-npu emulator

To run compiled firmware, install the [riscv-npu](https://github.com/paladin8/riscv-npu) emulator:

```bash
cd ../riscv-npu
uv sync
```

## Usage

```python
from arrax import Array, compile

# Define a kernel
def fused_relu(A: Array, B: Array) -> Array:
    return (A + B).relu()

# Compile to firmware
binary = compile(fused_relu, shapes={"A": (1024,), "B": (1024,)})

# Run on emulator
binary.run()
```

## Development

```bash
# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_dsl.py -v

# Stop on first failure
uv run pytest -x
```

## Project structure

```
src/arrax/
├── dsl/            Python tracing DSL (Array class, DAG capture)
├── dialects/       xDSL dialect definitions (array, npu)
├── lowering/       Compiler passes (array→linalg, tiling, fusion, bufferize, linalg→npu, npu→asm)
├── codegen/        Assembly emission, firmware harness, toolchain invocation
└── pipeline.py     Full pass pipeline orchestration
tests/              pytest test suite
firmware/harness/   C template + linker script for riscv-npu firmware
examples/           Usage examples
```

## Design

See [.ai/OVERALL_DESIGN.md](.ai/OVERALL_DESIGN.md) for the full design document covering:
- Dialect definitions and lowering strategies
- Tiling and fusion pass design
- NPU instruction mapping
- Firmware integration
- LLVM backend extension (Phase 2)

## Key design decisions

- **xDSL over C++ MLIR**: Faster iteration, Python-native, 1:1 compatible with MLIR IR.
- **No vector dialect**: The NPU's memory-to-memory model (address + length) maps to linalg's destination-passing style, not to SIMD register files.
- **Composite ops retained**: Softmax and RMSNorm stay as single ops in the array dialect — multi-instruction lowering benefits from recognizing them as a unit rather than decomposing early.
- **Assembly first**: `.insn` directives are production-quality and serve as reference implementation. LLVM backend is Phase 2.
