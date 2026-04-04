"""End-to-end tests: Python DSL → assembly → emulator → verify against NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from riscv_npu import Emulator

from arrax.codegen.build import build_elf
from arrax.dsl.array import Array, exp, relu
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

    def test_sub(self, tmp_path: Path) -> None:
        """A - B: basic elementwise subtraction."""
        N = 128
        A = np.arange(N, dtype=np.float32) * 3
        B = np.arange(N, dtype=np.float32)
        expected = A - B

        actual, _ = _compile_and_run(
            lambda A, B: A - B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_sub_negative_result(self, tmp_path: Path) -> None:
        """A - B where result has negative values."""
        N = 64
        A = np.ones(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 5
        expected = A - B

        actual, _ = _compile_and_run(
            lambda A, B: A - B,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_then_sub(self, tmp_path: Path) -> None:
        """(A + B) - C: mixed add and sub."""
        N = 100

        def kernel(A: Array, B: Array, C: Array) -> Array:
            return (A + B) - C

        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 10
        C = np.ones(N, dtype=np.float32) * 3
        expected = (A + B) - C

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,)},
            {"A": A, "B": B, "C": C},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_relu(self, tmp_path: Path) -> None:
        """relu(A): clamps negative values to zero."""
        N = 128
        A = np.linspace(-5.0, 5.0, N, dtype=np.float32)
        expected = np.maximum(A, 0.0)

        actual, _ = _compile_and_run(
            lambda A: relu(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_relu_all_positive(self, tmp_path: Path) -> None:
        """relu on all-positive input is identity."""
        N = 64
        A = np.arange(1, N + 1, dtype=np.float32)
        expected = A.copy()

        actual, _ = _compile_and_run(
            lambda A: relu(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_exp(self, tmp_path: Path) -> None:
        """exp(A): elementwise exponential."""
        N = 64
        A = np.linspace(-2.0, 2.0, N, dtype=np.float32)
        expected = np.exp(A)

        actual, _ = _compile_and_run(
            lambda A: exp(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_relu_of_add(self, tmp_path: Path) -> None:
        """relu(A + B): chained binary + unary."""
        N = 100

        def kernel(A: Array, B: Array) -> Array:
            return relu(A + B)

        A = np.linspace(-10.0, 10.0, N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * (-3.0)
        expected = np.maximum(A + B, 0.0)

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_exp_large(self, tmp_path: Path) -> None:
        """exp(A) with N=256: exercises tiling."""
        N = 256
        A = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        expected = np.exp(A)

        actual, _ = _compile_and_run(
            lambda A: exp(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_relu_sub_chain(self, tmp_path: Path) -> None:
        """relu(A + B) - C: 3-way fusion exercised end-to-end."""
        N = 128

        def kernel(A: Array, B: Array, C: Array) -> Array:
            return relu(A + B) - C

        A = np.linspace(-10.0, 10.0, N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 2.0
        C = np.ones(N, dtype=np.float32) * 1.0
        expected = np.maximum(A + B, 0.0) - C

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,)},
            {"A": A, "B": B, "C": C},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_exp_of_sub(self, tmp_path: Path) -> None:
        """exp(A - B): 2-way fusion."""
        N = 64

        actual, _ = _compile_and_run(
            lambda A, B: exp(A - B),
            {"A": (N,), "B": (N,)},
            {
                "A": np.ones(N, dtype=np.float32),
                "B": np.ones(N, dtype=np.float32) * 0.5,
            },
            tmp_path,
        )
        expected = np.exp(np.ones(N, dtype=np.float32) * 0.5)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_relu_add_relu_sub(self, tmp_path: Path) -> None:
        """relu(A+B) + relu(C-D): 4 intermediates, buffer reuse exercised."""
        N = 128

        def kernel(A: Array, B: Array, C: Array, D: Array) -> Array:
            return relu(A + B) + relu(C - D)

        A = np.linspace(-5.0, 5.0, N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 2.0
        C = np.ones(N, dtype=np.float32) * 10.0
        D = np.linspace(0.0, 20.0, N, dtype=np.float32)
        expected = np.maximum(A + B, 0.0) + np.maximum(C - D, 0.0)

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,), "D": (N,)},
            {"A": A, "B": B, "C": C, "D": D},
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
