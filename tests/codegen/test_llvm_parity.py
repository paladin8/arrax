"""Output parity tests: asm backend vs LLVM backend through the emulator.

Each test compiles the same Python expression through both backends,
runs the resulting ELF on the riscv-npu emulator, and compares the
numerical outputs. This verifies that the LLVM path produces
functionally identical code to the assembly path.

Requires: patched llc (built from llvm-npu/) and riscv-gcc.
Skipped automatically when llc is not available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from riscv_npu import Emulator

from arrax import dot, rmsnorm, softmax, sum as arr_sum
from arrax.codegen.build import _LLC_PATH, build_elf, build_elf_from_ll
from arrax.pipeline import compile_to_asm

_has_llc = shutil.which(_LLC_PATH) is not None or Path(_LLC_PATH).is_file()
requires_llc = pytest.mark.skipif(
    not _has_llc, reason="patched llc not available"
)


def _run_asm(fn, shapes, inputs, n_out, tmp_path):
    """Compile via asm backend, run on emulator, return output array."""
    asm, params = compile_to_asm(fn, shapes, backend="asm")
    out_dir = tmp_path / "asm"
    out_dir.mkdir(exist_ok=True)
    elf = build_elf(asm, params, shapes, output_dir=out_dir)
    emu = Emulator()
    emu.load_elf(str(elf))
    for name, data in inputs.items():
        emu.write_f32(name, data)
    result = emu.run()
    assert result.exit_code == 0
    return emu.read_f32("out", n_out)


def _run_llvm(fn, shapes, inputs, n_out, tmp_path):
    """Compile via LLVM backend, run on emulator, return output array."""
    ll, params = compile_to_asm(fn, shapes, backend="llvm")
    out_dir = tmp_path / "llvm"
    out_dir.mkdir(exist_ok=True)
    elf = build_elf_from_ll(ll, params, shapes, output_dir=out_dir)
    emu = Emulator()
    emu.load_elf(str(elf))
    for name, data in inputs.items():
        emu.write_f32(name, data)
    result = emu.run()
    assert result.exit_code == 0
    return emu.read_f32("out", n_out)


@requires_llc
class TestLlvmParity:
    """Same expression through both backends → same numerical output."""

    def test_add_small(self, tmp_path: Path) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        fn = lambda A, B: A + B
        shapes = {"A": (4,), "B": (4,)}
        inputs = {"A": a, "B": b}
        asm_out = _run_asm(fn, shapes, inputs, 4, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 4, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_add_tiled(self, tmp_path: Path) -> None:
        np.random.seed(1)
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        fn = lambda A, B: A + B
        shapes = {"A": (128,), "B": (128,)}
        inputs = {"A": a, "B": b}
        asm_out = _run_asm(fn, shapes, inputs, 128, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 128, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_sum(self, tmp_path: Path) -> None:
        np.random.seed(2)
        a = np.random.randn(128).astype(np.float32)
        fn = lambda A: arr_sum(A)
        shapes = {"A": (128,)}
        inputs = {"A": a}
        asm_out = _run_asm(fn, shapes, inputs, 1, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 1, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_dot(self, tmp_path: Path) -> None:
        np.random.seed(3)
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        fn = lambda A, B: dot(A, B)
        shapes = {"A": (128,), "B": (128,)}
        inputs = {"A": a, "B": b}
        asm_out = _run_asm(fn, shapes, inputs, 1, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 1, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_softmax(self, tmp_path: Path) -> None:
        np.random.seed(4)
        a = np.random.randn(128).astype(np.float32)
        fn = lambda A: softmax(A)
        shapes = {"A": (128,)}
        inputs = {"A": a}
        asm_out = _run_asm(fn, shapes, inputs, 128, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 128, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_rmsnorm(self, tmp_path: Path) -> None:
        np.random.seed(6)
        a = np.random.randn(128).astype(np.float32)
        fn = lambda A: rmsnorm(A)
        shapes = {"A": (128,)}
        inputs = {"A": a}
        asm_out = _run_asm(fn, shapes, inputs, 128, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 128, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)

    def test_fused_expression(self, tmp_path: Path) -> None:
        np.random.seed(5)
        a = np.random.randn(128).astype(np.float32)
        b = np.random.randn(128).astype(np.float32)
        fn = lambda A, B: A * 0.5 + B
        shapes = {"A": (128,), "B": (128,)}
        inputs = {"A": a, "B": b}
        asm_out = _run_asm(fn, shapes, inputs, 128, tmp_path)
        llvm_out = _run_llvm(fn, shapes, inputs, 128, tmp_path)
        np.testing.assert_allclose(llvm_out, asm_out, rtol=1e-5)
