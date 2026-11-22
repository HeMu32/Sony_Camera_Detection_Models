"""Observe the original BED resource requests without calling the allocator."""

import argparse
import json
from pathlib import Path
from struct import pack, unpack_from

from unicorn.arm_const import (
    UC_ARM_REG_R3, UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6,
    UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_SP,
)

from probe_a7iv_fdn_cp_request import REGS, execute
from probe_a7iv_fdn_cp_result import MANAGER, SP, cp_machine


def trace_resources(blob, physical_fixtures=(0x500100000, 0x500280000)):
    pages = (0x260000, 0x275000, 0x276000, 0x478000, 0x479000, 0x706000, 0x713000,
             0xb9b000, 0xc16000, 0xc18000)
    m = cp_machine(blob, pages)
    m.mem_map(0x10c6000, 0x1000)
    m.reg_write(UC_ARM_REG_SP, SP)
    execute(m, 0x2607bc, 0x2607e4, ((UC_ARM_REG_R4, MANAGER),))
    execute(m, 0x2607e4, 0x2607f6, ((UC_ARM_REG_R4, MANAGER),))
    execute(m, 0x260856, 0x260920, ((UC_ARM_REG_R4, MANAGER),))
    vptr, = unpack_from("<I", m.mem_read(MANAGER, 4))
    resources = unpack_from("<3I", m.mem_read(MANAGER + 0xb8, 12))
    definitions = unpack_from("<6I", m.mem_read(MANAGER + 4, 24))

    requests = []
    for resource, launch, capacity in zip(resources[1:], (0x7063ca, 0x7063e4), definitions[3::2], strict=True):
        # Original resource vptr plus a zero status reply; no allocation call runs.
        m.mem_write(resource, pack("<I", 0xc18760))
        m.mem_write(resource + 0x5f, b"\1")
        execute(m, launch, 0x478368, ((UC_ARM_REG_R4, MANAGER), (UC_ARM_REG_SP, SP)))
        register_values = tuple(m.reg_read(r) for r in (*REGS, UC_ARM_REG_R3))
        identity, mode, flags, _ = register_values
        message = MANAGER + 0x700
        m.mem_write(SP + 0x20, pack("<I", message))
        execute(m, 0x4783c0, 0x4783e4, (
            (UC_ARM_REG_SP, SP), (UC_ARM_REG_R4, identity), (UC_ARM_REG_R5, mode), (UC_ARM_REG_R6, flags),
        ))
        raw = bytes(m.mem_read(message, 0x18))
        requests.append({"resource": resource, "capacity_check": capacity,
                         "request_hex": raw.hex(), "id": unpack_from("<I", raw, 4)[0]})

    get_binary, = unpack_from("<I", m.mem_read(vptr + 0x54, 4))
    for i, (resource, req, physical) in enumerate(zip(resources[1:], requests, physical_fixtures, strict=True)):
        handle = 0x13572400 + i
        reply = MANAGER + 0x780
        m.mem_write(SP + 0x10, pack("<H2xI8xQI", 1, handle, physical, req["capacity_check"]))
        m.reg_write(UC_ARM_REG_SP, SP)
        execute(m, 0x479c82, 0x479cd2, (
            (UC_ARM_REG_R6, req["id"]), (UC_ARM_REG_R8, SP + 8), (UC_ARM_REG_R9, resource + 8),
        ))
        execute(m, 0x2765d4, 0x2766c6, ((REGS[0], resource),))
        lo, hi = (m.reg_read(r) for r in REGS[:2])
        if lo | hi << 32 != physical:
            raise ValueError("BED resource accessor did not recover the fixture's physical base")
        m.mem_write(reply, pack("<Q", physical))
        execute(m, 0x2759a8, 0x275a08, ((REGS[0], reply),))
        word = m.reg_read(REGS[0])
        m.mem_write(SP + 0x10, pack("<I", MANAGER + 0xc0))
        execute(m, 0x706cd8, get_binary & ~1, ((UC_ARM_REG_SP, SP),
                (UC_ARM_REG_R8, MANAGER), (UC_ARM_REG_R5, i)))
        owner, passed, name, offset = (m.reg_read(r) for r in (*REGS, UC_ARM_REG_R3))
        if (owner, passed) != (MANAGER, handle):
            raise ValueError("original BED loader selected an unexpected resource")
        req.update(filename=bytes(m.mem_read(name, 24)).split(b"\0", 1)[0].decode("ascii"),
                   handle_fixture=passed, offset=offset, physical_fixture=physical, sa_word_fixture=word)
    return requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cp", type=Path)
    args = parser.parse_args()
    print(json.dumps(trace_resources(args.cp.read_bytes()), indent=2))
