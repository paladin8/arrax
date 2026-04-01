# Makefile for building arrax compiler output into riscv-npu firmware.
#
# Usage: make KERNEL=kernel.S HARNESS=main.c
# Produces: firmware.elf

CROSS   ?= riscv64-unknown-elf-
CC       = $(CROSS)gcc
AS       = $(CROSS)as
OBJDUMP  = $(CROSS)objdump

ARCH     = -march=rv32imf -mabi=ilp32f
CFLAGS   = $(ARCH) -O2 -nostdlib -ffreestanding
ASFLAGS  = $(ARCH)
LDFLAGS  = -T linker.ld -nostdlib

KERNEL  ?= kernel.S
HARNESS ?= main.c
OUTPUT  ?= firmware.elf

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(HARNESS) $(KERNEL) linker.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(HARNESS) $(KERNEL)

clean:
	rm -f $(OUTPUT)
