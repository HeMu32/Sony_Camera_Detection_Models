"""Replay original AED/AED2/BED CP requests up to the queue boundary."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from struct import Struct

from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_R4, UC_ARM_REG_SP

from probe_a7iv_fdn_cp_request import REGS, U32, execute
from probe_a7iv_fdn_cp_result import MANAGER, RECORD, cp_machine


ENTRY = 0xb01cc4
QUEUE_BOUNDARY = 0xb02610
INPUT_ROW = Struct("<QQBB6x")
REQUEST_HEADER = Struct("<4B3I")
REQUEST_ROW = Struct("<8B6I")


@dataclass(frozen=True)
class Driver:
    vptr_slice: tuple[int, int]
    initializer: int
    max_rows: int


DRIVERS = {
    "aed": Driver((0x25f8bc, 0x25f8ce), 0x25f9aa, 5),
    "aed2": Driver((0x25ff6c, 0x25ff7a), 0x26007c, 5),
    "bed": Driver((0x2607e4, 0x2607f6), 0x2608d2, 5),
}


def probe_request(blob, family, rows, *, driver_flag=False):
    """Use explicit memory/address fixtures, never start the SA or queue."""
    driver = DRIVERS[family]
    if len(rows) > driver.max_rows:
        raise ValueError(f"expected at most {driver.max_rows} records for {family}")
    pages = {0x26b000, 0x26d000, 0x26e000, 0x26f000, 0x566000, 0x6fa000, 0xb01000,
             0xb88000, 0xb89000, 0xb9b000, 0xc16000,
             driver.vptr_slice[0] & ~0xfff, driver.initializer & ~0xfff}
    machine = cp_machine(blob, pages)
    for base in (0x0123e000, 0x01243000, 0x014b6000):
        machine.mem_map(base, 0x1000)

    machine.mem_write(MANAGER + 0xac, U32.pack(int(driver_flag)))
    machine.mem_write(RECORD, b"".join(INPUT_ROW.pack(*row) for row in rows))
    execute(machine, *driver.vptr_slice, ((UC_ARM_REG_R4, MANAGER),))
    execute(machine, driver.initializer, driver.initializer + 4)
    job, = U32.unpack(machine.mem_read(MANAGER + 0x50, 4))
    vtable, = U32.unpack(machine.mem_read(MANAGER, 4))
    rtti, = U32.unpack(machine.mem_read(vtable - 4, 4))
    type_name_va, = U32.unpack(machine.mem_read(rtti + 4, 4))
    type_name = bytes(machine.mem_read(type_name_va, 48)).split(b"\0", 1)[0].decode("ascii")
    target, = U32.unpack(machine.mem_read(vtable + 0x68, 4))
    execute(machine, target, ENTRY, zip(REGS, (MANAGER, RECORD, len(rows))))
    pointer = machine.reg_read(UC_ARM_REG_R0)
    if pointer != machine.reg_read(UC_ARM_REG_SP) + 0x48:
        raise ValueError("unexpected keypoint request pointer")

    execute(machine, ENTRY, QUEUE_BOUNDARY)
    queue, payload, length = (machine.reg_read(r) for r in REGS)
    if (queue, payload, length) != (0x930101, pointer,
                                    REQUEST_HEADER.size + len(rows) * REQUEST_ROW.size):
        raise ValueError("unexpected keypoint queue arguments")
    raw = bytes(machine.mem_read(payload, length))
    return {"job": job, "vtable": vtable, "type_name": type_name,
            "request_builder": target & ~1, "queue": queue, "length": length,
            "header": REQUEST_HEADER.unpack_from(raw),
            "rows": tuple(REQUEST_ROW.iter_unpack(raw[REQUEST_HEADER.size:]))}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("family", choices=DRIVERS)
    args = cli.parse_args()
    fixture = ((0x80001000, 0x500002000, 0x19, 0x37),)
    print(json.dumps(probe_request(args.cp.read_bytes(), args.family, fixture), indent=2))
