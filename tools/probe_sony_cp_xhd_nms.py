"""Replay the native CP XHD/NMS parameter and command builders without submitting jobs."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from struct import Struct, pack, unpack_from

from unicorn.arm_const import (UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1,
                               UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_SP)

from probe_sony_cp_region_scale import BASE, RETURN, CpSlices


RTTI = b"N6Camera2FC8Keypoint13AdamXhdDriverE\0"
LAYOUT = {
    "A7R V": (0xc2f508, 0x1b81d4, 0x01723c7c, (8, 9, 10, 11, 12)),
    "A1 II": (0xd1f2b4, 0x1ef450, 0x01738b40, (8, 9, 10, 11, 16, 17, 18)),
}
TUNING_BINDINGS = {
    # Constructor file offset; target, destination and ID table virtual addresses.
    "A7R V": (0x1bf4a0, 0xc9b820, 0x10fd70c, 0x10fd758),
    "A1 II": (0x1f6318, 0xd74c70, 0x11e90c0, 0x11e910c),
}
DRIVER, BLOCK, SETTINGS, WORK = (RETURN + 0x1000 * i for i in range(1, 5))
PARAMETER_SIZE = 0x520
RESULT_SIZE = 0x260
NMS_OFFSET = 0x4bc
PROC = Struct("<HHI")
INPUT_POINTERS = (0x65000000, 0x65019000, 0x65032000)


@dataclass(frozen=True)
class TuningFixture:
    threshold_pairs: tuple[tuple[int, int], ...]
    global_threshold: int
    foreground_threshold: int
    skip_max_decode: int

    def as_block(self):
        if len(self.threshold_pairs) != 10 or any(len(pair) != 2 for pair in self.threshold_pairs):
            raise ValueError("expected ten paired NMS fixture bytes")
        block = bytearray(0x18b)
        block[0x171:0x185] = bytes(value for pair in self.threshold_pairs for value in pair)
        block[0x185:0x187] = bytes((self.global_threshold, self.foreground_threshold))
        block[0x18a] = self.skip_max_decode
        return bytes(block)


class NmsProbe(CpSlices):
    def __init__(self, image):
        super().__init__(image)
        table, init, debug_flags, self.driver_ids = LAYOUT[self.firmware.model]
        methods = self.virtual_methods(table, RTTI, 7)
        self.init = BASE + init
        self.get_procs, self.set_parameters = methods[2:4]
        self.parameter_view, self.result_view = methods[5:]
        for va in (DRIVER, BLOCK, SETTINGS, WORK, debug_flags):
            self.map_ram(va)
        self.machine.mem_write(debug_flags + 0x35, b"\0")
        self.machine.mem_write(debug_flags + 0x49, b"\0")

    def call(self, pc, *args):
        m = self.machine
        for reg, value in zip((UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3), args):
            m.reg_write(reg, value)
        m.reg_write(UC_ARM_REG_LR, RETURN | 1)
        m.reg_write(UC_ARM_REG_SP, RETURN + 0xd00)
        self.execute(pc, RETURN, limit=16000)
        return m.reg_read(UC_ARM_REG_R0)

    def tuning_binding(self, target):
        """Resolve a target's RAM pointer and paired configuration ID, not its values."""
        if target not in range(1, 7):
            raise ValueError("expected a detection target in 1..6")
        ctor, target_table, destinations, identifiers = TUNING_BINDINGS[self.firmware.model]
        expected, = unpack_from("<I", self.image, target_table - BASE + (target - 1) * 4)
        m = self.machine
        m.mem_write(WORK, b"\xa5" * 8)
        if self.call(BASE + ctor, WORK, target) != WORK:
            raise ValueError("unexpected TuningParamIP constructor return value")
        pointer, = unpack_from("<I", m.mem_read(WORK + 4, 4))
        if pointer != expected:
            raise ValueError("TuningParamIP selection disagrees with the firmware table")
        dest = unpack_from("<19I", self.image, destinations - BASE)
        ids = unpack_from("<19I", self.image, identifiers - BASE)
        if dest[-1] or ids[-1] or dest.count(pointer) != 1:
            raise ValueError("unexpected configuration destination or table terminator")
        return pointer, ids[dest.index(pointer)]

    def describe(self, driver_id, tuning, *, inputs=INPUT_POINTERS, prefill=0xcd):
        if driver_id not in self.driver_ids or len(inputs) != 3:
            raise ValueError("unexpected XHD driver fixture")
        if not all(0 <= addr <= 0xffffffff for addr in inputs) or not 0 <= prefill <= 255:
            raise ValueError("invalid address or fill byte")
        m = self.machine
        m.mem_write(DRIVER, bytes(0x110))
        m.mem_write(BLOCK, bytes([prefill]) * (PARAMETER_SIZE + RESULT_SIZE))
        m.mem_write(SETTINGS, tuning.as_block())
        m.mem_write(WORK, bytes(0x40))
        for off, val in ((4, driver_id), (0xc, 0x60000000), (0x10, 0x61234000),
                         (0x14, inputs[0]), (0x18, inputs[1]), (0xc4, BLOCK), (0xcc, SETTINGS)):
            m.mem_write(DRIVER + off, pack("<I", val))
        self.call(self.init, DRIVER, BLOCK)
        self.call(self.set_parameters, DRIVER, WORK, inputs[2], 0)  # No ROI list in this fixture.
        raw = m.mem_read(BLOCK + NMS_OFFSET, PARAMETER_SIZE - NMS_OFFSET)
        inputs_and_id = unpack_from("<4I", raw)
        output, = unpack_from("<I", raw, 0x54)
        lists = []
        for mode in (0, 1):
            count = self.call(self.get_procs, DRIVER, BLOCK, mode, WORK)
            if count != (1, 3)[mode]:
                raise ValueError("unexpected native XHD procedure count")
            lists.append([(op, engine, va - BLOCK)
                          for op, engine, va in PROC.iter_unpack(m.mem_read(WORK, count * PROC.size))])
        spans = []
        for getter, expected in ((self.parameter_view, (0, PARAMETER_SIZE)),
                                 (self.result_view, (PARAMETER_SIZE, RESULT_SIZE))):
            self.call(getter, DRIVER, WORK)
            address, size = unpack_from("<2I", m.mem_read(WORK, 8))
            if (address - BLOCK, size) != expected:
                raise ValueError("unexpected native XHD buffer view")
            spans.append({"block_offset": address - BLOCK, "size": size})
        return {
            "driver_id": driver_id, "inputs": inputs_and_id[:3], "nms_id": inputs_and_id[3],
            "accelerator_output_address": output, "skip_max_decode": raw[0x10],
            "threshold_rows": [list(raw[0x11 + i:0x15 + i]) for i in range(0, 32, 4)],
            "threshold_per_rows": [list(raw[0x31 + i:0x35 + i]) for i in range(0, 32, 4)],
            "threshold_global": raw[0x51], "threshold_background_foreground": raw[0x58],
            "short_schedule": lists[0], "long_schedule": lists[1],
            "parameter_view": spans[0], "result_view": spans[1],
            "fixture_prefill": prefill,
        }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("--driver-id", type=int, default=8)
    cli.add_argument("--tuning-bindings", action="store_true")
    args = cli.parse_args()
    source = NmsProbe(args.cp.read_bytes())
    if args.tuning_bindings:
        bindings = []
        for target in range(1, 7):
            address, ident = source.tuning_binding(target)
            bindings.append({"target": target, "address": hex(address), "configuration_id": hex(ident)})
        result = {"tuning_bindings": bindings}
    else:
        sample = TuningFixture(tuple((33 + 2 * i, 34 + 2 * i) for i in range(10)), 91, 92, 7)
        result = {"explicit_tuning_fixture": True, **source.describe(args.driver_id, sample)}
    print(json.dumps({"model": source.firmware.model, **result}, indent=2))
