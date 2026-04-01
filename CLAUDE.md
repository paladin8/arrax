# arrax

MLIR-based compiler that fuses array expressions in Python and emits optimized RISC-V assembly targeting riscv-npu.

## Commands
- uv run pytest                          — run all tests
- uv run pytest tests/test_dsl.py -v     — DSL tests only
- uv run pytest tests/test_lowering.py -v — lowering pass tests only
- uv run pytest -x                       — stop on first failure
- uv run pytest -k "test_name"           — run specific test
- uv sync                                — install/update dependencies
- uv add <pkg>                           — add new dependency

## Session start
1. Read this file
2. Read .ai/memory.md
3. Read the active phase spec in .ai/plans/

## Session end
1. uv run pytest — confirm passing
2. Update .ai/memory.md: what you did, what works, what's blocked
3. Compact .ai/memory.md: drop stale details, keep only what future sessions need (status, patterns, blockers). Aim for <30 lines.
4. Commit working changes including .ai/memory.md (atomic, descriptive messages)

## Architecture
- src/arrax/dsl/             — Python tracing DSL (Array class, DAG capture)
- src/arrax/dialects/        — xDSL dialect definitions (array, npu)
- src/arrax/lowering/        — compiler passes (array->linalg, tiling, fusion, bufferize, linalg->npu, npu->asm)
- src/arrax/codegen/         — assembly emission, firmware harness, toolchain invocation
- src/arrax/pipeline.py      — full pass pipeline orchestration
- firmware/harness/           — C template, linker script, Makefile for riscv-npu firmware
- examples/                   — usage examples

## Compiler pipeline
```
Python DSL (tracing) -> array dialect -> linalg dialect -> tiling -> fusion
-> bufferize (tensor->memref) -> npu dialect -> RISC-V assembly (.S)
-> riscv32 toolchain -> ELF -> riscv-npu emulator
```

## Conventions
- Python 3.10+, type hints on all signatures
- Built on xDSL (Python MLIR framework) — all dialects defined via IRDL
- snake_case everywhere
- Docstrings on all public functions
- One test file per module, pytest, descriptive names
- No external deps without logging rationale in .ai/memory.md. Add via uv add <pkg>.
- Markdown tables must be ASCII-aligned: pad every cell so all rows have identical column widths

## Key design decisions
- xDSL over C++ MLIR: faster iteration, Python-native, 1:1 MLIR IR compatible
- No vector dialect: NPU is memory-to-memory (addr+len), maps to linalg destination-passing, not SIMD registers
- Composite ops (softmax, rmsnorm) retained in array dialect: multi-instruction lowering benefits from unit recognition
- Assembly emission via .insn directives first; LLVM backend is Phase 2
- All shapes are static (known at compile time)

## Testing rules
- Tests verify against NumPy float32 reference with tolerance for FP ordering
- Run uv run pytest after every few functions, not just end of session
- Never modify tests to make them pass — fix the implementation
- Never skip or disable tests
- End-to-end tests: Python expression -> assembly -> emulator -> compare with NumPy

## Git rules
- main is always passing
- No branches, commit directly to main
- Atomic commits after each working milestone
- Always include .ai/memory.md updates in the same commit as the code changes they describe
- Descriptive messages: "Implement array.add lowering to linalg.generic" not "update files"

## Target hardware
The FP32 instruction set of the NPU coprocessor in riscv-npu. Memory-to-memory vector ops (address + length from registers), double-precision accumulator for dot products, scalar activation functions. Cross-compile with: -march=rv32imf -mabi=ilp32f
