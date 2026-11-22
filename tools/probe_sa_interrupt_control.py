"""Replay the pinned SA interrupt test through its first wake, with no hardware."""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path

from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import boot_programs
from probe_sa_fdn_boot import BootMachine
from probe_sa_fdn_dispatch import ROW, word


TEST_SHA = "da619919ed0fd0348f1ece8e64b7cf599d422eb4bb714e7bed480d081ad0625f"
TIMER_WAIT = {"pc": 0x178, "reason": "wfi"}
STOP_BEFORE_TIME_LOG = {"pc": 0x1ae, "reason": "slice boundary"}
PRINT_PORT = 0x48c18
TIMER_BIT = 1 << 26


def extract_interrupt_test(bundle):
    item = members(bundle)[0x60]
    data = bundle[item.offset:item.offset + item.size]
    if sha256(data).hexdigest() != TEST_SHA:
        raise ValueError("expected pinned SA_TEST1SA07 member")
    return data


def probe(data, *, interrupt_status, cancel_value):
    if sha256(data).hexdigest() != TEST_SHA:
        raise ValueError("expected pinned SA_TEST1SA07 member")
    if not all(0 <= v <= 0xffffffff for v in (interrupt_status, cancel_value)):
        raise ValueError("interrupt fixture words must fit u32")
    stage, table = (0x100 + word(data, at) for at in (0x110, 0x120))
    relative, aperture, flags, other = ROW.unpack_from(data, table)
    size = (flags >> 16) + 1
    if (relative, aperture, size, flags & 0xffff, other) != (0, 0x08980000, 0x8a0, 0x4000, 0):
        raise ValueError("unexpected SA interrupt-test load request")
    image = data[stage:stage + size]
    if len(image) != size or boot_programs(data):
        raise ValueError("unexpected SA interrupt-test load data")

    machine = BootMachine(image)
    # The load aperture is not the execution PC: the first stub jumps to 0xc8.
    machine.pc = 0
    machine.regions = ((0, image), (0x48c00, b"\0"), (0x49002, b"\0"))
    machine.stack_low, machine.stack_high = 0x2000, word(image, 0x10)
    machine.stack = bytearray(machine.stack_high - machine.stack_low)
    words = dict.fromkeys((0x48004, 0x48024, 0x481d0, 0x481d4, 0x481d8, 0x481dc), 0)
    words.update({0x48200: interrupt_status, 0x481cc: cancel_value})
    machine.fixture_reads = {addr: deque((value, value)) for addr, value in words.items()}
    wait = machine.run(8192)
    if wait != TIMER_WAIT:
        raise ValueError(f"interrupt test did not reach timer wait: {wait}")
    boundary = len(machine.mmio_writes)
    machine.pc += 2  # Synthetic wake with explicit status; no interrupt delivered.
    stop = machine.run(8192, stop_at=STOP_BEFORE_TIME_LOG["pc"])
    if stop not in (TIMER_WAIT, STOP_BEFORE_TIME_LOG):
        raise ValueError(f"unexpected interrupt-test continuation: {stop}")
    writes = machine.mmio_writes[boundary:]

    def log(events):
        return bytes(w["value"] for w in events if w["address"] == PRINT_PORT).decode("ascii")

    return {
        "load_aperture": aperture, "load_member_offset": stage, "load_size": size,
        "wait": wait, "before_wait_log": log(machine.mmio_writes[:boundary]),
        "before_wait_registers": [w for w in machine.mmio_writes[:boundary] if w["address"] != PRINT_PORT],
        "after_wake_stop": stop, "after_wake_log": log(writes),
        "after_wake_registers": [w for w in writes if w["address"] != PRINT_PORT],
        "event_reads": [w for w in machine.fixture_trace if w["address"] in (0x48200, 0x481cc)],
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mp_bundle", type=Path)
    cli.add_argument("--interrupt-status", type=lambda v: int(v, 0), default=TIMER_BIT)
    cli.add_argument("--cancel-value", type=lambda v: int(v, 0), default=0)
    args = cli.parse_args()
    print(json.dumps(probe(extract_interrupt_test(args.mp_bundle.read_bytes()),
                           interrupt_status=args.interrupt_status, cancel_value=args.cancel_value), indent=2))
