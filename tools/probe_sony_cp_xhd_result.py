"""Replay the original CP XHD result converter, not the NBG detector."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from struct import Struct, pack_into

from unicorn.arm_const import UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_SP

from parsers.xhd_cp_result import COUNT, Geometry, MAX_RECORDS, RAW, RAW_START, XhdResult, raw_count
from probe_sony_cp_region_scale import BASE, RETURN, CpSlices


LAYOUT = {
    "A7R V": (0x1b8eb4, 0x1b90ec, 0x1b8f3e, 0xf8),
    "A1 II": (0x1f01b8, 0x1f03f0, 0x1f0242, 0xfc),
}
CP_RECORD = Struct("<f4H2B2x")
RAW_PAGE, CONTEXT, OUTPUT, SETTINGS = (RETURN + 0x1000 * i for i in range(1, 5))


class XhdResultProbe(CpSlices):
    def __init__(self, image):
        super().__init__(image)
        for va in (RAW_PAGE, CONTEXT, OUTPUT, SETTINGS):
            self.map_ram(va)

    def convert_fixture(self, payload, geometry):
        geometry.validate()
        count = raw_count(payload)
        invalid = count > MAX_RECORDS
        required = RAW_START if invalid else RAW_START + RAW.size * count
        if len(payload) < required:
            raise ValueError("truncated XHD records for CP replay")
        entry, done, reject, frame_offset = LAYOUT[self.firmware.model]
        if invalid:
            done = reject

        context = bytearray(0x100)
        pack_into("<I4xI", context, 0xc4, RAW_PAGE, SETTINGS)
        pack_into("<5H", context, 0xdc, *geometry.center, *geometry.search_size, geometry.angle)
        pack_into("<2H", context, frame_offset, *geometry.frame_size)
        m = self.machine
        m.mem_write(CONTEXT, bytes(context))
        m.mem_write(RAW_PAGE, payload[:required])
        m.mem_write(SETTINGS + 0x187, bytes([geometry.min_side_raw]))
        output_size = 4 + CP_RECORD.size * (0 if invalid else count)
        m.mem_write(OUTPUT, b"\xff" * output_size)
        for reg, val in ((UC_ARM_REG_R0, CONTEXT), (UC_ARM_REG_R1, OUTPUT),
                         (UC_ARM_REG_SP, RETURN + 0xfa0), (UC_ARM_REG_LR, RETURN | 1)):
            m.reg_write(reg, val)
        self.execute(BASE + entry, BASE + done, limit=32000)
        if invalid:
            if COUNT.unpack_from(m.mem_read(OUTPUT, 4))[0] != 0xffffffff:
                raise ValueError("unexpected output write before invalid-count diagnostic")
            raise ValueError("original CP took invalid XHD count branch")
        output = m.mem_read(OUTPUT, output_size)
        if COUNT.unpack_from(output)[0] != count:
            raise ValueError("original converter produced an unexpected count")
        return [XhdResult((x0, y0, x1, y1), score, kind, aux)
                for score, x0, y0, x1, y1, kind, aux in CP_RECORD.iter_unpack(output[4:])]

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("raw_result", type=Path)
    cli.add_argument("--frame", type=int, nargs=2, required=True)
    cli.add_argument("--search", type=int, nargs=2, required=True)
    cli.add_argument("--center", type=int, nargs=2, required=True)
    cli.add_argument("--angle", type=int, default=0)
    cli.add_argument("--min-side-raw", type=int, default=0)
    args = cli.parse_args()
    fixture = Geometry(tuple(args.frame), tuple(args.search), tuple(args.center),
                       args.angle, args.min_side_raw)
    probe = XhdResultProbe(args.cp.read_bytes())
    print(json.dumps([asdict(item) for item in probe.convert_fixture(args.raw_result.read_bytes(), fixture)],
                     indent=2))
