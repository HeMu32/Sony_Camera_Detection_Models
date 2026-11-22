"""Compare controller inputs to the shared AED/AED2/BED image program."""

import argparse
from collections import deque
import json
from pathlib import Path
from struct import pack_into

from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import startup_rows
from probe_sa_bed_image import (
    PARAMETER_ALIAS, WORK_BYTES, DmaTraceMachine, make_candidate, pc_literal, probe_keypoint,
)


def trace_entry(cp, bundle, job, *, center=(280, 192), roll=0):
    """Use a common geometry fixture, not a claim of native CP dispatch for each job."""
    if job not in (0x3e, 0x43, 0x44):
        raise ValueError("expected AED, AED2 or BED")
    member = members(bundle)[job]
    data = bundle[member.offset:member.offset + member.size]
    regions = tuple((target, data[src:src + (a >> 16) + 1])
                    for src, target, a, b in startup_rows(data)
                    if a & 0xffff == 0x4000 and b == a & 0xffff0000)
    origin, image = regions[0]
    record = image[0x1c:0xc0]
    params = bytearray(probe_keypoint(cp, bytes(0x100), make_candidate(center, (64, 64), 0, roll))["parameter"])
    if job != 0x44:
        # AED and AED2 validate the leading field as 1..3, unlike BED's 4.
        pack_into("<H", params, 0, 3)
    machine = DmaTraceMachine(image, scratch_base=0x9000, scratch_size=0xf000)
    machine.pc = origin
    machine.regions = (*regions, (0x49e00, bytes(0x160)))
    machine.stack_high = pc_literal(lambda a, n: image[a - origin:a - origin + n], origin, 15)
    machine.stack_low = machine.stack_high - 0x800
    machine.stack = bytearray(0x800)
    machine.fixture_reads = {a: deque(v) for a, v in
                             ((0x48004, (0,)), (0x48024, (0, 0)), (0x49c80, (0,) * 8))}
    machine.extra_scratch = ((0, 0x100), (0x49002, 0x49003),
                             (PARAMETER_ALIAS, PARAMETER_ALIAS + len(params)), (0xe0004040, 0xe00040d8))
    machine.scratch[0x49002] = 0
    machine.scratch.update(enumerate(params, PARAMETER_ALIAS))
    machine.capture_dma = True
    waits = []
    for _ in range(12):
        prior = len(machine.dma_submissions)
        stop = machine.run(8192)
        if stop["reason"] != "wfi" or len(machine.dma_submissions) != prior + 1:
            break
        waits.append(stop["pc"])
        # The only script-provided DMA payload is the explicit zero work buffer.
        if prior == 0:
            (request,) = machine.dma_submissions[-1]["rows"]
            src, dst, a, b = request
            start = (dst & 0x7fffffff) - machine.stack_low
            if (src, a, b) != (0x20004040, (WORK_BYTES - 1) << 16, (WORK_BYTES - 1) << 16):
                raise ValueError(f"unexpected shared image work DMA: {request}")
            if dst >> 31 != 1 or not 0 <= start <= len(machine.stack) - WORK_BYTES:
                raise ValueError(f"unexpected shared image work destination: {request}")
            machine.stack[start:start + WORK_BYTES] = bytes(WORK_BYTES)
        machine.pc += 2
    else:
        raise ValueError("shared image entry exceeded the wait limit")
    pointer = machine.reg[4]
    return {"waits": waits, "dma": machine.dma_submissions, "stop": stop,
            "arguments": tuple(machine.reg[4:8]), "return": machine.link,
            "uses_first_record": (stop.get("opcode") == 0xa and
                                  image[pointer - origin:pointer - origin + len(record)] == record),
            "compute_setup": {w["address"]: w["value"] for w in machine.mmio_writes
                              if 0x40000 <= w["address"] < 0x40100}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("--roll", type=int, default=0)
    args = cli.parse_args()
    cp, bundle = args.cp.read_bytes(), args.bundle.read_bytes()
    print(json.dumps({hex(job): trace_entry(cp, bundle, job, roll=args.roll)
                      for job in (0x3e, 0x43, 0x44)}, indent=2))
