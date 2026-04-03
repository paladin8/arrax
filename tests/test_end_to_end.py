"""End-to-end tests: Python DSL → assembly → emulator → verify against NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from riscv_npu import Emulator

from arrax.codegen.build import build_elf
from arrax.dsl.array import Array
from arrax.pipeline import compile_to_asm


def _compile_and_run(
    fn,
    shapes: dict[str, tuple[int, ...]],
    inputs: dict[str, np.ndarray],
    tmp_path: Path,
) -> tuple[np.ndarray, int]:
    """Compile fn, build ELF, run on emulator, return (output, cycles)."""
    asm, param_names = compile_to_asm(fn, shapes)
    elf_path = build_elf(asm, param_names, shapes, output_dir=tmp_path)

    emu = Emulator()
    emu.load_elf(str(elf_path))
    for name, data in inputs.items():
        emu.write_f32(name, data)
    result = emu.run()
    assert result.exit_code == 0

    n = shapes[param_names[0]][0]
    actual = emu.read_f32("out", n)
    return actual, result.cycles


class TestEndToEnd:
    def test_add(self, tmp_path: Path) -> None:
        """A + B: basic elementwise addition with tiling (N > 64)."""
        N = 128
        A = np.arange(N, dtype=np.float32)
        B = np.arange(N, dtype=np.float32) * 2
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_small(self, tmp_path: Path) -> None:
        """A + B with N=1."""
        A = np.array([3.0], dtype=np.float32)
        B = np.array([7.0], dtype=np.float32)
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (1,), "B": (1,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_negative_values(self, tmp_path: Path) -> None:
        """A + B with negative floats."""
        N = 16
        A = np.linspace(-10.0, 10.0, N, dtype=np.float32)
        B = np.linspace(5.0, -5.0, N, dtype=np.float32)
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_zeros(self, tmp_path: Path) -> None:
        """A + B where B is all zeros."""
        N = 32
        A = np.arange(N, dtype=np.float32)
        B = np.zeros(N, dtype=np.float32)
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_non_multiple(self, tmp_path: Path) -> None:
        """A + B with N=100: exercises tiling remainder path."""
        N = 100
        A = np.arange(N, dtype=np.float32)
        B = np.arange(N, dtype=np.float32) * 0.5
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_large(self, tmp_path: Path) -> None:
        """A + B with N=1024: many tiling iterations."""
        N = 1024
        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 3.14
        expected = A + B

        actual, _ = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_chained_add(self, tmp_path: Path) -> None:
        """(A + B) + C: two fused additions."""
        N = 32

        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) + C

        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 10
        C = np.ones(N, dtype=np.float32) * 100
        expected = (A + B) + C

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,)},
            {"A": A, "B": B, "C": C},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_reports_cycles(self, tmp_path: Path) -> None:
        """Emulator reports nonzero cycle count."""
        N = 16
        A = np.ones(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32)

        _, cycles = _compile_and_run(
            lambda A, B: A + B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        assert cycles > 0
