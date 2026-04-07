"""End-to-end tests: Python DSL → assembly → emulator → verify against NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from riscv_npu import Emulator

from arrax.codegen.build import build_elf
from arrax.dsl.array import Array, amax, dot, exp, mean, relu, softmax
from arrax.dsl.array import sum as arr_sum
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


def _compile_and_run_scalar(
    fn,
    shapes: dict[str, tuple[int, ...]],
    inputs: dict[str, np.ndarray],
    tmp_path: Path,
) -> tuple[float, int]:
    """Compile fn, build ELF, run on emulator, return a single f32 output."""
    asm, param_names = compile_to_asm(fn, shapes)
    elf_path = build_elf(asm, param_names, shapes, output_dir=tmp_path)

    emu = Emulator()
    emu.load_elf(str(elf_path))
    for name, data in inputs.items():
        emu.write_f32(name, data)
    result = emu.run()
    assert result.exit_code == 0

    actual = emu.read_f32("out", 1)
    return float(actual[0]), result.cycles


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

    def test_mul_scalar(self, tmp_path: Path) -> None:
        """A * 3.0: scalar-vector multiply."""
        N = 128
        A = np.arange(N, dtype=np.float32)
        expected = A * 3.0

        actual, _ = _compile_and_run(
            lambda A: A * 3.0,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_rmul_scalar(self, tmp_path: Path) -> None:
        """2.5 * A: reverse multiply."""
        N = 64
        A = np.linspace(-5.0, 5.0, N, dtype=np.float32)
        expected = 2.5 * A

        actual, _ = _compile_and_run(
            lambda A: 2.5 * A,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_div_scalar(self, tmp_path: Path) -> None:
        """A / 2.0: scalar-vector divide."""
        N = 128
        A = np.arange(N, dtype=np.float32) + 1.0
        expected = A / 2.0

        actual, _ = _compile_and_run(
            lambda A: A / 2.0,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_add_then_mul_scalar(self, tmp_path: Path) -> None:
        """(A + B) * 0.5: chained binary + scalar-vector."""
        N = 100

        def kernel(A: Array, B: Array) -> Array:
            return (A + B) * 0.5

        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 10
        expected = (A + B) * 0.5

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_mul_negative_scalar(self, tmp_path: Path) -> None:
        """A * -2.0: negative scalar."""
        N = 64
        A = np.arange(N, dtype=np.float32)
        expected = A * -2.0

        actual, _ = _compile_and_run(
            lambda A: A * -2.0,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_chained_scalar_ops(self, tmp_path: Path) -> None:
        """(A * 3.0) / 2.0: two scalar ops with different facc values."""
        N = 128

        def kernel(A: Array) -> Array:
            return (A * 3.0) / 2.0

        A = np.arange(N, dtype=np.float32) + 1.0
        expected = (A * 3.0) / 2.0

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_relu_of_mul_scalar(self, tmp_path: Path) -> None:
        """relu(A * 2.0): scalar + unary fusion."""
        N = 100

        def kernel(A: Array) -> Array:
            return relu(A * 2.0)

        A = np.linspace(-5.0, 5.0, N, dtype=np.float32)
        expected = np.maximum(A * 2.0, 0.0)

        actual, _ = _compile_and_run(
            kernel,
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

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

    def test_sum_small(self, tmp_path: Path) -> None:
        """sum(A) with N=16: untiled rank-0 reduction."""
        N = 16
        A = np.arange(N, dtype=np.float32)
        expected = float(A.sum())

        actual, _ = _compile_and_run_scalar(
            lambda A: arr_sum(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_sum_exact_tile(self, tmp_path: Path) -> None:
        """sum(A) with N=64: exact one-tile untiled reduction."""
        N = 64
        A = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        expected = float(A.sum())

        actual, _ = _compile_and_run_scalar(
            lambda A: arr_sum(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_sum_non_multiple(self, tmp_path: Path) -> None:
        """sum(A) with N=100: tiled reduction with remainder chunk."""
        N = 100
        A = np.arange(N, dtype=np.float32) * 0.1
        expected = float(A.sum())

        actual, _ = _compile_and_run_scalar(
            lambda A: arr_sum(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_sum_of_add(self, tmp_path: Path) -> None:
        """sum(A + B) at N=128: exercises a parallel elementwise loop
        followed by a reduction loop in the same kernel. Phase 1 must NOT
        fuse these (fusion safety guard), and the end-to-end result must
        still match NumPy.
        """
        N = 128
        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 0.5
        expected = float((A + B).sum())

        actual, _ = _compile_and_run_scalar(
            lambda A, B: arr_sum(A + B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_sum_large(self, tmp_path: Path) -> None:
        """sum(A) with N=1024: many tiled iterations threading fs0 iter_arg."""
        N = 1024
        A = np.ones(N, dtype=np.float32) * 0.5
        expected = float(A.sum())

        actual, _ = _compile_and_run_scalar(
            lambda A: arr_sum(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_amax_small(self, tmp_path: Path) -> None:
        """amax(A) with N=16: untiled rank-0 max reduction."""
        N = 16
        A = np.array(
            [-3.0, 1.0, 7.5, -0.5, 2.0, -10.0, 4.25, 0.0,
             3.5, -1.5, 6.25, -7.0, 5.0, -2.0, 7.5, 1.25],
            dtype=np.float32,
        )
        expected = float(A.max())

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_amax_exact_tile(self, tmp_path: Path) -> None:
        """amax(A) with N=64: exact one-tile untiled reduction."""
        N = 64
        A = np.linspace(-2.0, 3.0, N, dtype=np.float32)
        expected = float(A.max())

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_amax_non_multiple(self, tmp_path: Path) -> None:
        """amax(A) with N=100: tiled reduction with remainder chunk."""
        N = 100
        rng = np.random.default_rng(42)
        A = rng.standard_normal(N).astype(np.float32) * 5.0
        expected = float(A.max())

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_amax_large(self, tmp_path: Path) -> None:
        """amax(A) with N=1024: many tiled iterations threading -inf through fs0."""
        N = 1024
        A = np.arange(N, dtype=np.float32) - 500.0  # crosses zero, positive max
        expected = float(A.max())

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_amax_all_negative(self, tmp_path: Path) -> None:
        """amax over all-negative values: result is the least-negative element.

        Exercises the -inf identity thread: if the seed ever got clobbered to
        0.0 (like sum), the result would wrongly be 0.0 instead of the true
        maximum.
        """
        N = 128
        A = -np.arange(1, N + 1, dtype=np.float32)  # -1, -2, ..., -128
        expected = float(A.max())  # -1.0
        assert expected == -1.0

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_amax_nan_untiled(self, tmp_path: Path) -> None:
        """amax propagates NaN like np.amax (untiled path).

        The tile-level NPU.FVMAX already propagates NaN into ft0, but the
        scalar ``fmax.s`` combine with the iter-arg accumulator suppresses
        NaN from one operand per RISC-V spec. The emitter must add an
        explicit NaN-check so a NaN in the input survives into the final
        result.
        """
        N = 16
        A = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.5, 0.0,
             -1.0, -2.0, 6.0, -7.0, 0.5, 2.5, 3.5, 1.5],
            dtype=np.float32,
        )
        expected = float(np.amax(A))
        assert np.isnan(expected)

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        assert np.isnan(actual), f"expected NaN, got {actual!r}"

    def test_amax_nan_tiled(self, tmp_path: Path) -> None:
        """amax propagates NaN across tile boundaries.

        Places a NaN in one tile and a large finite value (99.0) in a
        *different* tile to catch the subtle bug where the NaN-carrying
        tile is silently dropped and a later tile's max becomes the
        result.
        """
        N = 128
        A = np.ones(N, dtype=np.float32)
        A[70] = np.nan  # NaN in chunk 1 (tile [64, 128))
        A[100] = 99.0   # large finite in the same chunk
        expected = float(np.amax(A))
        assert np.isnan(expected)

        actual, _ = _compile_and_run_scalar(
            lambda A: amax(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        assert np.isnan(actual), f"expected NaN, got {actual!r}"

    def test_dot_small(self, tmp_path: Path) -> None:
        """dot(A, B) with N=16: untiled dot product."""
        N = 16
        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 2.0
        expected = float(np.dot(A, B))

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_dot_exact_tile(self, tmp_path: Path) -> None:
        """dot(A, B) with N=64: exact one-tile untiled dot product."""
        N = 64
        A = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        B = np.linspace(0.0, 2.0, N, dtype=np.float32)
        expected = float(np.dot(A, B))

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_dot_non_multiple(self, tmp_path: Path) -> None:
        """dot(A, B) with N=100: tiled dot with remainder chunk."""
        N = 100
        A = np.arange(N, dtype=np.float32) * 0.1
        B = np.ones(N, dtype=np.float32)
        expected = float(np.dot(A, B))

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_dot_large(self, tmp_path: Path) -> None:
        """dot(A, B) with N=1024: many tiled iterations."""
        N = 1024
        A = np.ones(N, dtype=np.float32) * 0.5
        B = np.ones(N, dtype=np.float32) * 2.0
        expected = float(np.dot(A, B))  # 1024.0

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_dot_orthogonal(self, tmp_path: Path) -> None:
        """dot of orthogonal vectors is zero."""
        N = 64
        A = np.zeros(N, dtype=np.float32)
        A[0] = 1.0
        B = np.zeros(N, dtype=np.float32)
        B[1] = 1.0
        expected = 0.0

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_mean_small(self, tmp_path: Path) -> None:
        """mean(A) with N=16: untiled mean reduction."""
        N = 16
        A = np.arange(N, dtype=np.float32)
        expected = float(A.mean())

        actual, _ = _compile_and_run_scalar(
            lambda A: mean(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_mean_exact_tile(self, tmp_path: Path) -> None:
        """mean(A) with N=64: exact one-tile untiled mean."""
        N = 64
        A = np.linspace(-1.0, 1.0, N, dtype=np.float32)
        expected = float(A.mean())

        actual, _ = _compile_and_run_scalar(
            lambda A: mean(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_mean_non_multiple(self, tmp_path: Path) -> None:
        """mean(A) with N=100: tiled mean with remainder chunk."""
        N = 100
        A = np.arange(N, dtype=np.float32) * 0.1
        expected = float(A.mean())

        actual, _ = _compile_and_run_scalar(
            lambda A: mean(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_mean_large(self, tmp_path: Path) -> None:
        """mean(A) with N=1024: many tiled iterations."""
        N = 1024
        A = np.ones(N, dtype=np.float32) * 3.0
        expected = float(A.mean())  # 3.0

        actual, _ = _compile_and_run_scalar(
            lambda A: mean(A),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    # --- fused reduction + elementwise ---

    def test_sum_of_add(self, tmp_path: Path) -> None:
        """sum(A + B): fused parallel + reduction."""
        N = 128
        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 0.5
        expected = float((A + B).sum())

        actual, _ = _compile_and_run_scalar(
            lambda A, B: arr_sum(A + B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_sum_of_relu(self, tmp_path: Path) -> None:
        """sum(relu(A)): fused unary + reduction."""
        N = 128
        A = np.linspace(-5.0, 5.0, N, dtype=np.float32)
        expected = float(np.maximum(A, 0.0).sum())

        actual, _ = _compile_and_run_scalar(
            lambda A: arr_sum(relu(A)),
            {"A": (N,)},
            {"A": A},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_amax_of_sub(self, tmp_path: Path) -> None:
        """amax(A - B): fused binary + max reduction."""
        N = 128
        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 50.0
        expected = float((A - B).max())

        actual, _ = _compile_and_run_scalar(
            lambda A, B: amax(A - B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_dot_of_add(self, tmp_path: Path) -> None:
        """dot(A + B, C): fused parallel + dot reduction."""
        N = 128

        def kernel(A: Array, B: Array, C: Array) -> Array:
            return dot(A + B, C)

        A = np.arange(N, dtype=np.float32) * 0.1
        B = np.ones(N, dtype=np.float32)
        C = np.ones(N, dtype=np.float32) * 2.0
        expected = float(np.dot(A + B, C))

        actual, _ = _compile_and_run_scalar(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,)},
            {"A": A, "B": B, "C": C},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_mean_of_chain(self, tmp_path: Path) -> None:
        """mean((A + B) - C): fused chain + mean reduction."""
        N = 128

        def kernel(A: Array, B: Array, C: Array) -> Array:
            return mean((A + B) - C)

        A = np.arange(N, dtype=np.float32)
        B = np.ones(N, dtype=np.float32) * 10.0
        C = np.ones(N, dtype=np.float32) * 3.0
        expected = float(((A + B) - C).mean())

        actual, _ = _compile_and_run_scalar(
            kernel,
            {"A": (N,), "B": (N,), "C": (N,)},
            {"A": A, "B": B, "C": C},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_dot_scalar_vec_no_fuse_correct(self, tmp_path: Path) -> None:
        """dot(A * 2.0, B): facc conflict prevents fusion, but result is correct."""
        N = 128
        A = np.arange(N, dtype=np.float32) * 0.1
        B = np.ones(N, dtype=np.float32)
        expected = float(np.dot(A * 2.0, B))

        actual, _ = _compile_and_run_scalar(
            lambda A, B: dot(A * 2.0, B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4)

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

    # --- softmax ---

    def test_softmax_basic(self, tmp_path: Path) -> None:
        """softmax(A) with N=128 (tiled)."""
        N = 128
        A = np.random.default_rng(42).standard_normal(N).astype(np.float32)
        shifted = A - A.max()
        e = np.exp(shifted)
        expected = e / e.sum()

        actual, _ = _compile_and_run(
            lambda A: softmax(A), {"A": (N,)}, {"A": A}, tmp_path
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_small(self, tmp_path: Path) -> None:
        """softmax(A) with N=32 (untiled)."""
        N = 32
        A = np.random.default_rng(7).standard_normal(N).astype(np.float32)
        shifted = A - A.max()
        e = np.exp(shifted)
        expected = e / e.sum()

        actual, _ = _compile_and_run(
            lambda A: softmax(A), {"A": (N,)}, {"A": A}, tmp_path
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_exact_tile(self, tmp_path: Path) -> None:
        """softmax(A) with N=64 (exactly at tile boundary)."""
        N = 64
        A = np.random.default_rng(13).standard_normal(N).astype(np.float32)
        shifted = A - A.max()
        e = np.exp(shifted)
        expected = e / e.sum()

        actual, _ = _compile_and_run(
            lambda A: softmax(A), {"A": (N,)}, {"A": A}, tmp_path
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_non_multiple(self, tmp_path: Path) -> None:
        """softmax(A) with N=135 (non-multiple of 64)."""
        N = 135
        A = np.random.default_rng(99).standard_normal(N).astype(np.float32)
        shifted = A - A.max()
        e = np.exp(shifted)
        expected = e / e.sum()

        actual, _ = _compile_and_run(
            lambda A: softmax(A), {"A": (N,)}, {"A": A}, tmp_path
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_uniform(self, tmp_path: Path) -> None:
        """softmax of identical values should produce uniform 1/N."""
        N = 128
        A = np.full(N, 5.0, dtype=np.float32)
        expected = np.full(N, 1.0 / N, dtype=np.float32)

        actual, _ = _compile_and_run(
            lambda A: softmax(A), {"A": (N,)}, {"A": A}, tmp_path
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_of_add(self, tmp_path: Path) -> None:
        """softmax(A + B): elementwise producer fused before softmax."""
        N = 128
        A = np.random.default_rng(42).standard_normal(N).astype(np.float32)
        B = np.random.default_rng(43).standard_normal(N).astype(np.float32)
        x = A + B
        shifted = x - x.max()
        e = np.exp(shifted)
        expected = e / e.sum()

        actual, _ = _compile_and_run(
            lambda A, B: softmax(A + B),
            {"A": (N,), "B": (N,)},
            {"A": A, "B": B},
            tmp_path,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
