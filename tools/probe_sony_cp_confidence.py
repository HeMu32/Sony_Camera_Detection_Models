"""Execute original Sony CP confidence getters with explicit tuning fixtures."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from struct import pack, unpack, unpack_from

from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_READ
from unicorn.arm_const import (UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1,
                               UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_SP)

from probe_sony_cp_region_scale import BASE, RETURN, CpSlices


RTTI = b"N6Camera2FC8Keypoint21TuningParamPostResultE\0"
FACE_INDEX = (2, 0, 1, 2, 3, 4, 2)
STACK = RETURN + 0xe00
FIXTURE_BYTES = 0x220
GETTER_NAMES = ("FaceConfidenceMid", "FaceConfidenceHigh", "XhdConfidenceMid", "XhdConfidenceHigh")


@dataclass(frozen=True)
class Layout:
    vtable: int
    targets: int
    xhd_types: int
    xhd_tables: tuple[int, int]
    logger_offset: int | None = None


LAYOUTS = {
    "A7R V": Layout(0xc2fb90, 0x00c9c290, 0x00f389fd, (0x00d37d70, 0x00d37fb0)),
    "A1 II": Layout(0xd1f9a8, 0x00d762b0, 0x0105729c, (0x00e27b98, 0x00e27dd8), 0x17a508),
}


class ConfidenceProbe(CpSlices):
    def __init__(self, image):
        super().__init__(image)
        layout = self.layout = LAYOUTS[self.firmware.model]
        self.getters = self.virtual_methods(layout.vtable, RTTI, 6)[2:]
        self.targets = unpack_from("<9I", image, layout.targets - BASE)
        self.records = set(self.targets)
        self.read_fields = []
        self.diagnostic_names = []
        for ptr in self.records:
            self.map_ram(ptr)
            self.map_ram(ptr + FIXTURE_BYTES - 1)
        self.machine.hook_add(UC_HOOK_MEM_READ, self.observe_field)
        if layout.logger_offset is not None:
            logger = BASE + layout.logger_offset
            self.machine.hook_add(UC_HOOK_CODE, self.diagnostic_return, begin=logger, end=logger)

    def observe_field(self, _machine, _access, address, size, _value, _user_data):
        for base in self.records:
            if base <= address < base + FIXTURE_BYTES:
                self.read_fields.append((base, address - base, size))
                break

    def diagnostic_return(self, machine, _address, _size, _user_data):
        fmt = machine.reg_read(UC_ARM_REG_R2) - BASE
        prefix = b"virtual f32_t Camera::FC::Keypoint::TuningParamPostResult::get"
        if not 0 <= fmt < len(self.image) or not self.image.startswith(prefix, fmt):
            raise ValueError("unexpected diagnostic branch")
        name = self.image[fmt + len(prefix):fmt + len(prefix) + 64].split(b"(", 1)[0].decode("ascii")
        if name not in tuple(n + "Threshold" for n in GETTER_NAMES):
            raise ValueError("unexpected confidence diagnostic")
        self.diagnostic_names.append(name)
        machine.reg_write(UC_ARM_REG_PC, machine.reg_read(UC_ARM_REG_LR))

    def field_offsets(self, kind, sensitivity, xhd_type, part):
        if kind not in range(4) or sensitivity not in range(8):
            raise ValueError("expected a supported getter and sensitivity fixture 0..7")
        if kind < 2:
            if sensitivity == 7:
                return ()
            base = (0xac, 0xc0)[kind] + 4 * FACE_INDEX[sensitivity]
        else:
            if xhd_type not in range(9) or part not in range(4):
                raise ValueError("expected a validated XHD type index 0..8 and part index 0..3")
            route = self.image[self.layout.xhd_types - BASE + xhd_type]
            if route > 2:
                raise ValueError("XHD type does not select a proven parameter group")
            table = self.layout.xhd_tables[kind - 2] - BASE
            effective = sensitivity if sensitivity < 6 else 3
            return unpack_from("<2I", self.image, table + 8 * (24 * route + 6 * part + effective))
        return base, base + 2

    def evaluate_fixture(self, name, target, sensitivity, *, base_raw=0, adjustment_raw=0, xhd_type=0, part=0):
        if name not in GETTER_NAMES or target not in range(len(self.targets)):
            raise ValueError("unsupported getter or detected-target index")
        if not all(0 <= v <= 0xffff for v in (base_raw, adjustment_raw)):
            raise ValueError("fixture fields must fit u16")
        kind = GETTER_NAMES.index(name)
        fields = self.field_offsets(kind, sensitivity, xhd_type, part)
        ptr = self.targets[target]
        fixture = bytearray(FIXTURE_BYTES)
        for offset, value in zip(fields, (base_raw, adjustment_raw)):
            fixture[offset:offset + 2] = pack("<H", value)
        self.machine.mem_write(ptr, bytes(fixture))
        self.machine.mem_write(STACK, pack("<I", sensitivity))
        third = sensitivity if kind < 2 else xhd_type
        for register, value in zip((UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
                                    UC_ARM_REG_SP, UC_ARM_REG_LR),
                                   (0, target, third, part, STACK, RETURN | 1)):
            self.machine.reg_write(register, value)
        self.read_fields.clear()
        self.diagnostic_names.clear()
        self.execute(self.getters[kind], RETURN)
        if sorted(self.read_fields) != sorted((ptr, field, 2) for field in fields):
            raise ValueError(f"original getter read unexpected fields: {self.read_fields}")
        bits = self.machine.reg_read(UC_ARM_REG_R0)
        return {"configuration_va": hex(ptr), "fixture_field_offsets": [hex(f) for f in fields],
                "threshold_bits": f"0x{bits:08x}", "threshold": unpack("<f", pack("<I", bits))[0],
                "suppressed_diagnostic_names": list(self.diagnostic_names)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("--target", type=int, default=0, help="DetectedTarget fixture, not the NAB slot")
    cli.add_argument("--sensitivity", type=int, default=3)
    cli.add_argument("--xhd-type", type=int, default=0)
    cli.add_argument("--part", type=int, default=0)
    cli.add_argument("--fixture-fields", type=int, nargs=2, default=(0, 0))
    args = cli.parse_args()
    probe = ConfidenceProbe(args.cp.read_bytes())
    print(json.dumps({"model": probe.firmware.model, "target_fixture": args.target,
                      "sensitivity_fixture": args.sensitivity, "raw_fixture_fields": args.fixture_fields,
                      "getters": {name: probe.evaluate_fixture(
                          name, args.target, args.sensitivity, xhd_type=args.xhd_type, part=args.part,
                          base_raw=args.fixture_fields[0], adjustment_raw=args.fixture_fields[1])
                          for name in GETTER_NAMES}}, indent=2))
