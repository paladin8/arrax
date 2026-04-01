/* Linker script for riscv-npu firmware.
 * Matches the memory map expected by the emulator.
 */

MEMORY {
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 1M
}

SECTIONS {
    .text : {
        *(.text.init)
        *(.text*)
    } > RAM

    .rodata : {
        *(.rodata*)
    } > RAM

    .data : {
        *(.data*)
    } > RAM

    .bss : {
        *(.bss*)
        *(COMMON)
    } > RAM

    . = ALIGN(16);
    _stack_top = ORIGIN(RAM) + LENGTH(RAM) - 16;
}

ENTRY(_start)
