"""Toolchain invocation: assemble and link kernel into ELF firmware.

Wraps the kernel assembly with a main function and data declarations,
then invokes riscv64-unknown-elf-gcc to produce an ELF binary.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


# riscv-npu common firmware directory
RISCV_NPU_DIR = Path(
    os.environ.get("RISCV_NPU_DIR", Path(__file__).parents[4] / "riscv-npu")
)
COMMON_DIR = RISCV_NPU_DIR / "firmware" / "common"


def build_elf(
    kernel_asm: str,
    param_names: list[str],
    shapes: dict[str, tuple[int, ...]],
    output_dir: Path | None = None,
) -> Path:
    """Build a complete ELF firmware from kernel assembly.

    Wraps the kernel function with:
    - .comm declarations for input/output arrays (named symbols)
    - A main() that loads array addresses into registers and calls kernel
    - Linked with start.o (provides _start) and linker.ld from riscv-npu

    Returns the path to the built ELF file.
    """
    full_asm = _generate_firmware_asm(kernel_asm, param_names, shapes)
    return _assemble_and_link(full_asm, output_dir)


def _generate_firmware_asm(
    kernel_asm: str,
    param_names: list[str],
    shapes: dict[str, tuple[int, ...]],
) -> str:
    """Generate complete .S file with kernel, main wrapper, and data."""
    lines: list[str] = []

    # Kernel function
    lines.append(kernel_asm.rstrip())
    lines.append("")

    # Main wrapper: load array addresses into a-registers, call kernel
    # Argument order: input params first (in signature order), then output
    # The output param is named "out"
    all_names = list(param_names) + ["out"]
    if len(all_names) > 8:
        raise ValueError(
            f"too many kernel arguments ({len(all_names)}); "
            f"RISC-V only has a0-a7 (8 argument registers)"
        )

    lines.append("    .text")
    lines.append("    .globl main")
    lines.append("    .type main, @function")
    lines.append("main:")
    lines.append("    addi sp, sp, -4")
    lines.append("    sw ra, 0(sp)")
    for i, name in enumerate(all_names):
        lines.append(f"    la a{i}, {name}")
    lines.append("    call kernel")
    lines.append("    lw ra, 0(sp)")
    lines.append("    addi sp, sp, 4")
    lines.append("    li a0, 0")
    lines.append("    ret")
    lines.append("")

    # Data declarations in .bss (global symbols so emulator can find them)
    lines.append("    .section .bss")
    for name in param_names:
        size = 1
        for dim in shapes[name]:
            size *= dim
        size *= 4  # f32 = 4 bytes
        lines.append(f"    .globl {name}")
        lines.append(f"    .comm {name}, {size}, 4")

    # Output buffer — same shape as the function result (first input's shape for add)
    first_shape = shapes[param_names[0]]
    out_size = 1
    for dim in first_shape:
        out_size *= dim
    out_size *= 4
    lines.append("    .globl out")
    lines.append(f"    .comm out, {out_size}, 4")
    lines.append("")

    return "\n".join(lines) + "\n"


def _assemble_and_link(full_asm: str, output_dir: Path | None = None) -> Path:
    """Assemble and link the .S file into an ELF binary."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="arrax_"))
    asm_path = output_dir / "firmware.S"
    elf_path = output_dir / "firmware.elf"

    asm_path.write_text(full_asm)

    start_o = COMMON_DIR / "start.o"
    linker_ld = COMMON_DIR / "linker.ld"

    if not start_o.exists():
        raise FileNotFoundError(f"riscv-npu start.o not found at {start_o}")
    if not linker_ld.exists():
        raise FileNotFoundError(f"riscv-npu linker.ld not found at {linker_ld}")

    result = subprocess.run(
        [
            "riscv64-unknown-elf-gcc",
            "-march=rv32imf",
            "-mabi=ilp32f",
            "-nostdlib",
            "-ffreestanding",
            f"-T{linker_ld}",
            "-o",
            str(elf_path),
            str(asm_path),
            str(start_o),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"riscv64-unknown-elf-gcc failed:\n{result.stderr}"
        )

    return elf_path
