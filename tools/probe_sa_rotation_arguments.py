"""Trace native rotation sizing and wrapper setup without evaluating unknown opcodes."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from struct import pack_into, unpack

from compare_sa_payload_records import MARKER, ROT_SHA, rotation_image, spans
from probe_sa_fdn_boot import BootMachine
from probe_sa_fdn_dispatch import word


PARAMS = 0x20000
VALIDATOR_RETURN = 0x22000
DMA_RECORD = 0x5000


def transfer_words(machine):
    return list(unpack("<6I", bytes(machine.scratch[DMA_RECORD + i] for i in range(24))))


def trace_output_request(machine):
    # Start separate copies at the recorded return address. No compute or
    # pixel writeback is supplied, so only report the firmware's DMA request.
    variants = []
    for return_value in (0, 0xdeadbeef):
        post = deepcopy(machine)
        post.pc = post.link
        post.reg[0] = return_value
        first_write = len(post.scratch_trace)
        waits = []
        for _ in range(2):
            stop = post.run(8192)
            if stop["reason"] == "wfi":
                waits.append(stop["pc"])
            if any(e["address"] == DMA_RECORD for e in post.scratch_trace[first_write:]):
                break
            if stop["reason"] != "wfi":
                raise ValueError(f"unexpected rotation output stop: {stop}")
            post.pc += 2
        if stop not in ({"pc": 0x6d4, "reason": "wfi"},
                        {"pc": 0x19c, "reason": "unsupported opcode", "opcode": 0x000a}):
            raise ValueError("output record was not submitted in the fixture")
        variants.append({"wait_pcs": waits, "stop": stop, "descriptor_words": transfer_words(post)})
    if variants[0] != variants[1]:
        raise ValueError("chosen return register values changed the output request")
    return {**variants[0], "fixture_r0_values": [0, 0xdeadbeef]}


def probe(blob, width, height, mode, *, src_stride=160, dst_stride=192):
    if sha256(blob).hexdigest() != ROT_SHA:
        raise ValueError("expected the pinned A7 IV rotation member")
    stage, image = rotation_image(blob)
    params = bytearray(0x48)
    pack_into("<IIHH", params, 0x18, width, height, src_stride, dst_stride)
    pack_into("<II", params, 0x30, mode, 0x40010000)
    pack_into("<I", params, 0x44, 0x40020000)
    machine = BootMachine(image, scratch_base=DMA_RECORD, scratch_size=0x800)
    machine.regions = ((0, image), (PARAMS, params), (0x49e00, bytes(0x160)))
    stack = word(image, 0x10)
    machine.stack_low, machine.stack_high = len(image), stack
    machine.stack = bytearray(stack - len(image))
    machine.reg[4], machine.reg[15] = PARAMS, stack
    machine.pc, machine.link = 0x278, VALIDATOR_RETURN
    validation = machine.run(256, stop_at=VALIDATOR_RETURN)
    if validation["reason"] != "slice boundary":
        raise ValueError(f"unexpected rotation validation stop: {validation}")
    result = {
        "fixture": {"width": width, "height": height, "mode": mode,
                    "source_stride": src_stride, "destination_stride": dst_stride},
        "validator_return": machine.reg[0],
    }
    if result["validator_return"] != 0:
        result["validator_writes"] = machine.mmio_writes
        return result

    machine.pc = 0x350
    machine.reg[4] = PARAMS
    first = machine.run(8192)
    if first not in ({"pc": 0x434, "reason": "wfi"}, {"pc": 0x68c, "reason": "wfi"}):
        raise ValueError(f"rotation setup did not reach the transfer wait: {first}")
    descriptor = transfer_words(machine)
    machine.pc += 2  # Synthetic wake only; no pixel bytes are moved.
    first_unknown = machine.run(8192)
    if first_unknown != {"pc": 0x19c, "reason": "unsupported opcode", "opcode": 0x000a}:
        raise ValueError(f"unexpected rotation wrapper stop: {first_unknown}")
    record = machine.reg[4]
    records = dict(spans(image[0x1c:0x19c]))
    data = records[record - 0x1c]
    call_pc = machine.link - 2
    at_000a = {f"r{i}": machine.reg[i] for i in range(4, 11)}
    output = trace_output_request(machine)

    # This independent slice starts after the unknown halfword with unchanged
    # fixture registers. It does not simulate or assume any 0x000a side effects.
    machine.pc = 0x19e
    second = machine.run(32)
    if second != {"pc": 0x1b6, "reason": "unsupported opcode", "opcode": 0x0009}:
        raise ValueError(f"unexpected second rotation wrapper stop: {second}")
    result.update({
        "image_wait": first, "image_descriptor_words": descriptor,
        "call_pc": call_pc,
        "record": {"file_offset": stage + record, "dsp_address": record,
                   "body_words": list(unpack("<20I", data[:-len(MARKER)]))},
        "first_unknown": first_unknown, "registers_at_000a": at_000a,
        "isolated_post_call_slice": output,
        "isolated_wrapper_body": {"stop": second,
                                  "registers_at_0009": {f"r{i}": machine.reg[i]
                                                        for i in (0, *range(5, 11))}},
    })
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    cli.add_argument("width", type=int)
    cli.add_argument("height", type=int)
    cli.add_argument("mode", type=int, choices=(0, 8))
    cli.add_argument("--source-stride", type=int, default=160)
    cli.add_argument("--destination-stride", type=int, default=192)
    args = cli.parse_args()
    print(json.dumps(probe(args.member.read_bytes(), args.width, args.height, args.mode,
                           src_stride=args.source_stride, dst_stride=args.destination_stride),
                     indent=2))
