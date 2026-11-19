"""Execute pinned CP region-scale selection and getters on explicit RAM fixtures."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import pack, unpack, unpack_from

from unicorn import UC_ARCH_ARM, UC_MODE_ARM, UC_PROT_EXEC, UC_PROT_READ, UC_PROT_WRITE, Uc
from unicorn.arm_const import (UC_ARM_REG_C1_C0_2, UC_ARM_REG_FPEXC, UC_ARM_REG_LR,
                               UC_ARM_REG_PC, UC_ARM_REG_R0)


BASE = 0x108000
RETURN = 0x30000000
FIELDS = (2, 4, 6, 8, 0xa, 0x5e, 0x60, 0x62)
REGION_SCALE_RTTI = b"N6Camera2FC8Keypoint22TuningParamRegionScaleE\0"


@dataclass(frozen=True)
class Firmware:
    model: str
    select_offset: int
    return_offset: int
    mode_va: int
    pointer_va: int
    table_va: int
    vtable_offset: int


FIRMWARE = {
    "3269bf62426339791fcfbe8529698cd4accbe6c3d2ea61cd7c368a90ab7ee7c8":
        Firmware("A7R V", 0x1c0a98, 0x1c0b0c, 0x0145029c, 0x01718e54, 0x00c9b860, 0xc3020c),
    "96b41427958c1b412b3e15c5bd9b51e34657ae1e1246344438b8368031ca4859":
        Firmware("A1 II", 0x1f7d28, 0x1f7d9c, 0x0143eda4, 0x01726444, 0x00d74cb0, 0xd20034),
}


class CpSlices:
    def __init__(self, image):
        self.firmware = FIRMWARE.get(sha256(image).hexdigest())
        if self.firmware is None:
            raise ValueError("expected one of the pinned A7R V / A1 II CP images")
        # ldr r1,[pc,#4]; add r1,r1,#4; mov pc,r1; linked literal.
        if unpack_from("<4I", image) != (0xe59f1004, 0xe2811004, 0xe1a0f001, BASE + 0xc):
            raise ValueError("unexpected CP startup base")
        self.image = image
        self.machine = m = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        m.mem_map(BASE, (len(image) + 0xfff) & ~0xfff, UC_PROT_READ | UC_PROT_EXEC)
        m.mem_write(BASE, image)
        self.ram_pages = set()
        self.map_ram(RETURN)
        m.reg_write(UC_ARM_REG_C1_C0_2, 0xf00000)
        m.reg_write(UC_ARM_REG_FPEXC, 1 << 30)

    def map_ram(self, va):
        page = va & ~0xfff
        if page not in self.ram_pages:
            self.machine.mem_map(page, 0x1000, UC_PROT_READ | UC_PROT_WRITE)
            self.ram_pages.add(page)

    def execute(self, start, stop, *, limit=256):
        self.machine.emu_start(start | 1, stop, count=limit)
        if self.machine.reg_read(UC_ARM_REG_PC) != stop:
            raise ValueError(f"did not reach CP slice boundary {stop:#x}")

    def virtual_methods(self, offset, name, count):
        _, rtti = unpack_from("<2I", self.image, offset - 8)
        symbol, = unpack_from("<I", self.image, rtti - BASE + 4)
        if self.image[symbol - BASE:symbol - BASE + len(name)] != name:
            raise ValueError("unexpected CP RTTI name")
        methods = unpack_from(f"<{count}I", self.image, offset)
        if not all(addr & 1 and BASE <= addr < BASE + len(self.image) for addr in methods):
            raise ValueError("unexpected Thumb getter address")
        return methods


class RegionScale(CpSlices):
    def __init__(self, image):
        super().__init__(image)
        f = self.firmware
        self.table = unpack_from("<6I", image, f.table_va - BASE)
        self.getters = self.virtual_methods(f.vtable_offset, REGION_SCALE_RTTI, 10)[2:]
        for va in (f.mode_va, f.pointer_va):
            self.map_ram(va)

    def select(self, mode, target):
        if not 0 <= mode <= 255 or target not in range(1, 7):
            raise ValueError("mode must be a byte and detection target must be 1..6")
        f = self.firmware
        self.machine.mem_write(f.mode_va, bytes([mode]))
        self.machine.reg_write(UC_ARM_REG_R0, target)
        self.execute(BASE + f.select_offset, BASE + f.return_offset)
        pointer, = unpack("<I", self.machine.mem_read(f.pointer_va, 4))
        if pointer not in self.table:
            raise ValueError("region-scale selector returned an unexpected pointer")
        return pointer

    def evaluate_fixture(self, mode, target, fields):
        if len(fields) != len(FIELDS) or not all(0 <= value <= 0xffff for value in fields):
            raise ValueError("expected eight explicit u16 fixture fields")
        pointer = self.select(mode, target)
        for offset, value in zip(FIELDS, fields):
            self.map_ram(pointer + offset)
            self.machine.mem_write(pointer + offset, pack("<H", value))
        result = []
        for getter in self.getters:
            self.machine.reg_write(UC_ARM_REG_LR, RETURN | 1)
            self.execute(getter, RETURN)
            result.append(self.machine.reg_read(UC_ARM_REG_R0))
        return pointer, tuple(result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("--mode", type=int, default=0)
    cli.add_argument("--target", type=int, default=1)
    cli.add_argument("--fixture-fields", type=int, nargs=8, default=(0,) * 8)
    args = cli.parse_args()
    control = RegionScale(args.cp.read_bytes())
    pointer, bits = control.evaluate_fixture(args.mode, args.target, args.fixture_fields)
    print(json.dumps({
        "model": control.firmware.model, "mode_fixture": args.mode, "target_fixture": args.target,
        "region_scale_table": [hex(va) for va in control.table],
        "selected_runtime_pointer": hex(pointer),
        "region_scale_fixtures": [
            {"field_offset": hex(off), "raw_fixture": raw, "returned_fp32": unpack("<f", pack("<I", b))[0]}
            for off, raw, b in zip(FIELDS, args.fixture_fields, bits)
        ],
    }, indent=2))
