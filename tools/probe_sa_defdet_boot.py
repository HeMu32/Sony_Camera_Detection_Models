"""Trace original DEFDET startup and controller requests without executing DMA."""

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from struct import iter_unpack, pack, pack_into, unpack

from compare_sa_payload_records import MARKER, spans
from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import (
    PACKED_TARGET_BITS, header_word, join_record, packed_boot_programs, startup_rows, summarize,
)
from probe_sa_fdn_boot import (
    BootMachine, DMA_POINTER, DMA_SUBMIT, DmaTraceMachine, FIXTURE_BASE, FIXTURE_READS,
)
from run_sa_fdn_dispatch import DispatchMachine


DEFDET_SHA = "407b0932ab2bbd972702830ab1cab816786d1d8375f7d261c34c72b9c20d68a4"
BLOCKS = (0x100, 0x6700)
SCRATCH = 0x17800
WORKER_BASES = (0x4800, 0x5800, 0x6000, 0x8800, 0x9000)
TARGET_MASK = (1 << PACKED_TARGET_BITS) - 1
CONTROLLER_WRAP = 0x95c


def controller_request(member, *, register=0, field18=240, field24=240, field28=0,
                       reply_words=None):
    """Replay the local A controller; optional reply words are fixture data, not DMA."""
    if sha256(member).hexdigest() != DEFDET_SHA:
        raise ValueError("expected pinned A7 IV DEFDET member")
    offset, _, length = regular_transfers(member, BLOCKS[0])[0]
    image = member[offset:offset + length]
    parameters = bytearray(0x80)
    pack_into("<4I8H", parameters, 0, 0x40002000, 0x40008000, 0x40010000, 0x40018000,
              640, 480, 640, 640, field18, 480, 2, 640)
    pack_into("<H", parameters, 0x24, field24)
    pack_into("<H", parameters, 0x28, field28)
    machine = DmaTraceMachine(b"", scratch_base=0x9800, scratch_size=0xe800)
    machine.regions = ((0, image), (0xc0000000, parameters),
                       (0x48170, b"\0"), (0x49e00, bytes(0x160)))
    machine.pc = 0
    machine.stack_high = header_word(image, 0, 0x10)
    machine.stack_low = machine.stack_high - 0x800
    machine.stack = bytearray(0x800)
    machine.extra_scratch = ((length - 16, length),)
    machine.fixture_reads = {addr: deque([0] * 32) for addr in
                             (0x48024, 0x48004, 0x48174, 0x49c00, 0x48100, 0x49804)}
    machine.fixture_reads[0x49c80] = deque([register] * 3)
    machine.capture_dma = True
    limit = 100000
    stop = machine.run(limit)
    trace = {"first_stop": stop, "submissions": machine.dma_submissions, "r0": machine.reg[0],
             "parameter_pointer_writes": [r for r in machine.scratch_trace
                                          if r["address"] == 0x23a4],
             "gate_writes": [r for r in machine.scratch_trace if r["address"] in
                             (0x11840, 0x1184c) and r["size"] == 2]}
    if reply_words is None or stop != {"pc": 0x1230, "reason": "wfi"}:
        return trace
    # Preserve scalar writes and seed only the requested words. Zero for other
    # unmapped scratch bytes is also a fixture, including any future outputs.
    for address in range(machine.scratch_base, machine.scratch_end):
        machine.scratch.setdefault(address, 0)
    for address, value in zip((0x13028, 0x13828), reply_words, strict=True):
        machine.scratch.update(enumerate(pack("<I", value), address))
    start = len(machine.scratch_trace)
    machine.pc += 2
    stop = machine.run(limit)
    trace.update(reply_fixture=list(reply_words), second_stop=stop,
                 reply_field_writes=machine.scratch_trace[start:])
    if stop != {"pc": 0x126c, "reason": "wfi"}:
        return trace
    machine.pc += 2
    stop = machine.run(limit, stop_at=0x127a)
    if stop != {"pc": 0x127a, "reason": "slice boundary"}:
        trace["continuation_stop"] = stop
        return trace
    compare = machine.reg[3]
    machine.step()
    trace["controller_branch"] = {"r3": compare, "target": machine.pc}
    waits = []
    for _ in range(8):
        stop = machine.run(limit)
        if stop["reason"] != "wfi":
            break
        waits.append(stop["pc"])
        if stop["pc"] == 0xa00:
            break
        machine.pc += 2
    trace.update(continuation_stop=stop, continuation_wait_pcs=waits)
    if stop == {"pc": CONTROLLER_WRAP, "reason": "unsupported opcode", "opcode": 0x000a}:
        programs = packed_boot_programs(member, BLOCKS[0])
        program = programs[0x2710000]
        trace["compute_entry"] = {
            "return_pc": machine.link, "r4_r7": machine.reg[4:8],
            "record": command_span(image, machine.reg[4], program, offset),
        }
    return trace


def regular_transfers(member, block):
    rows = [(offset, target, ((flags >> 16) & 0x3fff) + 1)
            for offset, target, flags, tag in startup_rows(member, block)
            if tag == 0 and target != 0x40000000]
    if not rows or any(size != 8 * ((target >> PACKED_TARGET_BITS) + 1)
                       or offset + size > len(member) for offset, target, size in rows):
        raise ValueError("unexpected DEFDET regular row")
    return rows


def first_local_slice(member, rows):
    offset, target, size = rows[0]
    if target & TARGET_MASK:
        raise ValueError("expected zero low target bits for the first regular row")
    image = member[offset:offset + size]
    stack = header_word(image, 0, 0x10)
    if not size <= stack <= 0x10000:
        raise ValueError("invalid local slice stack")

    # Inject the slice independently, not as the completion of either DMA wait.
    machine = DispatchMachine({
        "stage_dsp_address": "0", "first_transfer_destination": "0",
        "stack": hex(stack), "mailbox_selector": hex(SCRATCH),
    }, image)
    return {"rows": len(rows), "offset": offset, "bytes": size,
            "raw_target": target, "fixture_base": 0, "stop": machine.run(1024),
            "register_writes": machine.mmio_writes}


def command_span(image, start, program, file_start):
    if not 0 <= start < len(image) or start % 4:
        raise ValueError("command pointer outside local image")
    end = image.find(MARKER, start)
    if end < 0 or end % 4:
        raise ValueError("command stream has no aligned trailing marker")
    raw = image[start:end + len(MARKER)]
    indices = {row["index"] for row in join_record(program, raw)}
    return {"member_offset": file_start + start, "bytes": len(raw),
            "sha256": sha256(raw).hexdigest(), "max_index": max(indices),
            "distinct_indices": len(indices)}


def command_differences(member, lhs, rhs):
    def words(record):
        offset, size = record["member_offset"], record["bytes"]
        return (value for value, in iter_unpack("<I", member[offset:offset + size]))

    return [{"position": i, "controller_word": a, "worker_word": b}
            for i, (a, b) in enumerate(zip(words(lhs), words(rhs), strict=True)) if a != b]


def controller_calls(image, file_start, program):
    """Replay the literal-to-indirect-call slices, not the enclosing branches."""
    starts = {0x1c + pos for pos, _ in spans(image[0x1c:CONTROLLER_WRAP])}
    calls = []
    for pc in range(header_word(image, 0, 0x14), 0xd50, 2):
        insn = int.from_bytes(image[pc:pc + 2], "little")
        literal = (pc & ~3) + 4 + 4 * ((insn >> 4) & 255)
        if insn >> 12 != 5 or header_word(image, 0, literal) != CONTROLLER_WRAP:
            continue
        prev = int.from_bytes(image[pc - 2:pc], "little")
        start_pc = pc - 2 if (prev & 0xf00f) == 0x5004 else pc
        machine = DispatchMachine({
            "stage_dsp_address": "0", "first_transfer_destination": "0",
            "stack": hex(header_word(image, 0, 0x10)), "mailbox_selector": hex(SCRATCH),
        }, image)
        machine.pc = start_pc
        machine.reg[2] = 0x1234
        machine.reg[5:8] = [0xffff7777, 0x6000, 0x7000]
        stop = machine.run(12)
        if stop != {"pc": CONTROLLER_WRAP, "reason": "unsupported opcode", "opcode": 0x000a}:
            raise ValueError(f"unexpected controller call: {stop}")
        start = machine.reg[4]
        if start not in starts:
            raise ValueError(f"controller did not select an original record: {start:#x}")
        calls.append({"slice_pc": start_pc, "call_pc": machine.link - 2, "r4": start,
                      "record": command_span(image, start, program, file_start)})
    return calls


def worker_run(regions, base, selector_addr, selector, wakes):
    machine = BootMachine(b"", scratch_base=0x11000, scratch_size=0x1000)
    machine.extra_scratch = ((0x13000, 0x14000),)
    machine.regions, machine.pc = regions, base
    machine.stack_high = 0x5800 if base == 0x4800 else base + 0x800
    machine.stack_low = machine.stack_high - 0x100
    machine.stack = bytearray(0x100)
    stops = []
    for _ in range(wakes):
        stop = machine.run(8192)
        if stop["reason"] != "wfi":
            raise ValueError(f"worker did not reach its wait: {stop}")
        stops.append(stop["pc"])
        if len(stops) == 1:
            mailbox = bytearray(0x1000)
            pack_into("<H", mailbox, selector_addr - 0x11000, selector)
            machine.scratch.update(enumerate(mailbox, 0x11000))
            machine.scratch.update(enumerate(bytes(0x1000), 0x13000))
        machine.pc += 2  # Synthetic delivery only. No DMA or device-side effects.
    end = machine.run(8192)
    if end["reason"] != "wfi" and end.get("opcode") != 0x000a:
        raise ValueError(f"worker stopped before the wait or compute entry: {end}")
    return {"selector_fixture": selector, "wait_pcs": stops, "stop": end,
            "read_addresses": [event["address"] for event in machine.scratch_reads],
            "r4": machine.reg[4]}


def local_compute_calls(member, block, rows, programs, boot_writes):
    """Cross-check the worker addresses against the original boot's writes."""
    images = {((target & TARGET_MASK) << 2): (offset, member[offset:offset + size])
              for offset, target, size in rows}
    mapped = tuple(sorted((base, image) for base, (_, image) in images.items()))
    if len(images) != len(rows) or any(a + len(data) > b
                                      for (a, data), (b, _) in zip(mapped, mapped[1:])):
        raise ValueError("regular transfers overlap")
    banks = tuple(r["value"] for r in boot_writes)
    if banks != WORKER_BASES or any(base not in images for base in banks):
        raise ValueError("worker destinations do not match observed boot registers")
    by_raw_low = {target & TARGET_MASK: prog for target, prog in programs.items()}
    if set(by_raw_low) != {0x10000 + (i << 10) for i in range(6)}:
        raise ValueError("unexpected program windows")
    file_start, image = images[0]
    calls = controller_calls(image, file_start, by_raw_low[0x10000])
    controller_records = {call["r4"]: call["record"] for call in calls}
    selector1, selector2 = ((0x118ea, 0x118f6) if block == BLOCKS[0] else
                            (0x11996, 0x119a2))
    workers = []
    for bank, base in enumerate(banks, 1):
        file_start, image = images[base]
        program = by_raw_low[0x10000 + (bank << 10)]
        runs = []
        for selector in range(4) if bank == 1 else (0,):
            run = worker_run(mapped, base, selector1 if bank == 1 else selector2,
                             selector, wakes=1 if bank == 1 else 2)
            if run["stop"].get("opcode") == 0x000a:
                run["record"] = command_span(image, run["r4"] - base, program, file_start)
                if bank == 1:
                    control = controller_records[run["r4"] - base - 0x18]
                    run["diff_from_controller"] = command_differences(member, control, run["record"])
            runs.append(run)
        workers.append({"base": base, "program_offset": program.offset,
                        "program_slots": program.slots, "runs": runs})
    return {"controller_call_slices": calls, "worker_base_writes": boot_writes, "workers": workers}


def probe(member, block):
    if sha256(member).hexdigest() != DEFDET_SHA or block not in BLOCKS:
        raise ValueError("expected a pinned A7 IV DEFDET block")
    source = header_word(member, block, 0x10)
    table = header_word(member, block, 0x20)
    size = header_word(member, block, 0x24)
    programs = packed_boot_programs(member, block)

    machine = BootMachine(member[block:block + source], scratch_base=SCRATCH)
    first = machine.run(1024)
    if first != {"pc": FIXTURE_BASE + 0x94, "reason": "unmodeled read",
                 "address": 0x48004, "size": 4}:
        raise ValueError(f"unexpected stop without register fixture: {first}")
    machine.fixture_reads = {addr: deque(values) for addr, values in FIXTURE_READS.items()}
    wait = machine.run(1024)
    if wait["reason"] != "wfi":
        raise ValueError(f"expected a wait after the table request: {wait}")
    descriptor = unpack("<6I", bytes(machine.scratch[SCRATCH + i] for i in range(0x18)))
    if descriptor != (FIXTURE_BASE + table, 0x3ff85e06, (size - 1) << 16, 0, 1, 0x40000000):
        raise ValueError("descriptor does not request the original startup table")
    writes = [r for r in machine.mmio_writes if r["address"] in
              (DMA_POINTER, DMA_SUBMIT, 0x48238)]
    if [(r["address"], r["value"]) for r in writes] != [
            (DMA_POINTER, SCRATCH), (DMA_SUBMIT, 1), (0x48238, FIXTURE_BASE + source)]:
        raise ValueError("unexpected table submission or transfer base")
    machine.pc += 2  # Synthetic wake only; no DMA or program execution.
    after_wake = machine.run(1024)
    rows = regular_transfers(member, block)
    banks = [r for r in machine.mmio_writes if 0x48104 <= r["address"] <= 0x48114]
    return {
        "block": block, "fixture_base": FIXTURE_BASE, "table_offset": block + table,
        "table_bytes": size, "first_stop": first, "first_wait": wait,
        "staging_words": descriptor, "staging_register_writes": writes,
        "after_synthetic_wake": after_wake,
        "isolated_first_local_slice": first_local_slice(member, rows),
        "isolated_compute_calls": local_compute_calls(member, block, rows, programs, banks),
        "programs": {hex(target): summarize(member, image) for target, image in programs.items()},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sp_bundle", type=Path)
    cli.add_argument("--controller-request", action="store_true")
    args = cli.parse_args()
    data = args.sp_bundle.read_bytes()
    record = members(data)[2]
    member = data[record.offset:record.offset + record.size]
    result = ([controller_request(member, field28=mode, reply_words=(0x2e, 0xa5a5003d))
               for mode in (0, 1, 2)]
              if args.controller_request else [probe(member, block) for block in BLOCKS])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
