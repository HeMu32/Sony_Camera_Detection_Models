"""Run the original SHA and NV12 worker setup, without simulating 000a/0009."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from math import isqrt
from pathlib import Path
from struct import iter_unpack, pack

from compare_sa_payload_records import MARKER
from probe_sa_fdn_boot import BootMachine
from probe_sa_fdn_dispatch import ROW, pc_literal, word


SHA = "35b7b8fd8ca2efa89b3d23c19d77a08b97c077cf89169ecb960307d3d8b7d573"
NV12 = "c7bf410ecf7cc004ccfda8356ddaa62af97d78e0e269743e4a3a95f06f58d3c3"
SHA_WORKERS = (0x2000, 0x2800, 0x3000, 0x8800, 0x9000)
SHA_MAILBOXES = (0x8000, 0x17000, 0x17000, 0x17800, 0x17800)
SHA_FIXTURE = tuple(i * 0x11000 for i in range(1, 9))
NV12_FIXTURE = (0x10000, 0x20000, 0x30000, 0x440, 5)


def sha256_words():
    """Derive the standard SHA-256 IV and K table using exact integer roots."""
    primes = []
    n = 2
    while len(primes) < 64:
        if all(n % p for p in primes if p <= isqrt(n)):
            primes.append(n)
        n += 1
    constants = []
    for p in primes:
        target, lo, hi = p << 96, 0, 1 << 36
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if mid ** 3 <= target:
                lo = mid
            else:
                hi = mid
        constants.append(lo & 0xffffffff)
    iv = [isqrt(p << 64) & 0xffffffff for p in primes[:8]]
    return iv, constants


def regions(blob, expected):
    if sha256(blob).hexdigest() != expected:
        raise ValueError("expected the pinned A7 IV SA member")
    stage, table = (0x100 + word(blob, pos) for pos in (0x110, 0x120))
    size = word(blob, 0x124)
    loaded = {}
    for src, dst, a, b in ROW.iter_unpack(blob[table:table + size]):
        # Only build byte-for-byte RAM copies, not the spatial/strided entries.
        if a & 0xffff == 0x4000 and b == a & 0xffff0000:
            data = blob[stage + src:stage + src + (a >> 16) + 1]
            if len(data) != (a >> 16) + 1 or dst in loaded:
                raise ValueError("invalid linear transfer")
            loaded[dst] = (stage + src, data)
    return loaded


def worker(loaded, base, mailbox, words):
    _, image = loaded[base]
    data = pack(f"<{len(words)}I", *words)
    machine = BootMachine(image, scratch_base=mailbox, scratch_size=len(data))
    machine.regions = tuple((address, region) for address, (_, region) in loaded.items())
    machine.pc = base
    machine.stack_high = pc_literal(
        lambda address, size: image[address - base:address - base + size], base + 0x10, 15)
    machine.stack_low = machine.stack_high - 0x200
    machine.stack = bytearray(0x200)
    first = machine.run(1024)
    if first["reason"] != "wfi":
        raise ValueError(f"expected worker wait: {first}")
    startup = list(machine.mmio_writes)
    machine.scratch.update(enumerate(data, mailbox))
    machine.pc += 2  # Synthetic wake: no transfers or compute are emulated.
    pending = machine.run(1024)
    return machine, first, startup, pending


def describe_call(machine, stop, file_base, base):
    if stop.get("opcode") != 0x000a:
        raise ValueError(f"expected first unknown halfword: {stop}")
    pointer = machine.reg[4]
    if not base <= pointer < stop["pc"]:
        raise ValueError("record pointer outside worker payload")
    body = deepcopy(machine)
    body.pc += 2  # Independent known-instruction slice, not an opcode model.
    next_stop = body.run(32)
    if next_stop.get("opcode") != 0x0009 or body.reg[0] != pointer:
        raise ValueError(f"expected wrapper to put the record in r0: {next_stop}")
    return {
        "stop": stop, "call_pc": machine.link - 2, "record_dsp": pointer,
        "record_member": file_base + pointer - base,
        "mailbox_reads": list(machine.scratch_reads),
        "isolated_wrapper_body": {"stop": next_stop, "r0_to_r10": body.reg[:11]},
    }


def sha_control(blob, *, mailbox_words=SHA_FIXTURE):
    if len(mailbox_words) != 8:
        raise ValueError("expected eight SHA worker mailbox words")
    loaded = regions(blob, SHA)
    iv, k = sha256_words()
    if blob[0x11e0:0x1200] != pack("<8I", *iv) or blob[0x1724:0x1824] != pack("<64I", *k):
        raise ValueError("SHA-256 standard tables do not match the original member")
    workers = []
    for base, mailbox in zip(SHA_WORKERS, SHA_MAILBOXES):
        machine, first, startup, stop = worker(loaded, base, mailbox, mailbox_words)
        preload = [(w["address"], w["value"]) for w in startup if 0x40000 <= w["address"] < 0x40100]
        if preload not in ([], list(zip(range(0x40000, 0x40100, 4), k))):
            raise ValueError("worker wrote an unexpected SHA table")
        file_base, image = loaded[base]
        call = describe_call(machine, stop, file_base, base)
        data = image[call["record_dsp"] - base:stop["pc"] - base]
        if not data.endswith(MARKER):
            raise ValueError("SHA record is not followed by the observed marker")
        workers.append({"base": base, "wait": first, "fixture_mailbox": mailbox,
                        "sha_round_words_written": len(preload),
                        "record_size": len(data), "record_sha256": sha256(data).hexdigest(), **call})
    file_base, image = loaded[0]
    outer = BootMachine(image, scratch_base=0x8000, scratch_size=0xc)
    outer.regions = ((0, image),)
    outer.stack_high = word(image, 0x10)
    outer.stack_low, outer.reg[15] = outer.stack_high - 0x200, outer.stack_high - 0x80
    outer.stack = bytearray(0x200)
    outer.scratch.update(enumerate(pack("<3I", 0x10000, 0x20000, 0x30000), 0x8000))
    # Isolate the original outer call setup; do not infer the startup system.
    outer.pc = 0x9e8
    stop = outer.run(32)
    call = describe_call(outer, stop, file_base, 0)
    short_record = image[call["record_dsp"]:stop["pc"]]
    long_record = loaded[0x2000][1][0x34:0x11c]
    if sha256(short_record).hexdigest() != workers[1]["record_sha256"]:
        raise ValueError("expected the outer record to match the two smaller workers")
    signals = observe_sha_record_words(short_record, long_record)
    return {"iv_member_offset": 0x11e0, "k_member_offset": 0x1724, "workers": workers,
            "isolated_outer_call": call, "record_word_observations": signals}


def observe_sha_record_words(short_record, long_record):
    """Locate literal patterns; these are not decoded operands or opcodes."""
    short, long = ([w for w, in iter_unpack("<I", record)]
                   for record in (short_record, long_record))
    pairs = [(w >> 16 & 0xff, w >> 8 & 0xff) for w in long]
    trio = [(11, 21), (6, 26), (25, 7)]
    return {
        "short_low24_sample": [w & 0xffffff for w in short[19:23]],
        "long_complement_triplet_indices": [i for i in range(len(pairs) - 2)
                                            if pairs[i:i + 3] == trio],
        "long_bit31_words": [w for w in long if w >> 31],
        "long_first_seen_7bit_indices": list(dict.fromkeys(
            w >> 24 & 0x7f for w in long if w >> 24 & 0x7f > 1)),
    }


def nv12_control(blob, followup, *, mailbox_words=NV12_FIXTURE):
    if len(mailbox_words) != 5:
        raise ValueError("expected five NV12 mailbox words before the continuation flag")
    loaded = regions(blob, NV12)
    base, mailbox = 0x2000, 0x1800
    values = (*mailbox_words, int(followup))
    machine, first, startup, stop = worker(loaded, base, mailbox, values)
    file_base, image = loaded[base]
    call = describe_call(machine, stop, file_base, base)
    writes = machine.mmio_writes[len(startup):]
    # Begin the caller continuation with the saved return address only.
    machine.pc = machine.link
    reads = len(machine.scratch_reads)
    continuation = machine.run(1024)
    machine.scratch_reads = machine.scratch_reads[reads:]
    second = describe_call(machine, continuation, file_base, base) if followup else None
    if not followup and continuation != first:
        raise ValueError(f"zero fixture did not skip the second call: {continuation}")
    if second:
        payload = image[second["record_dsp"] - base:continuation["pc"] - base]
        if not payload.endswith(MARKER):
            raise ValueError("second NV12 payload is not followed by the observed marker")
        second["leading_words"] = [w for w, in iter_unpack("<I", payload[:8])]
    return {"wait": first, "fixture_mailbox": mailbox, "fixture_words": values,
            "first_call": call, "setup_writes": writes, "isolated_continuation": continuation,
            "second_call": second}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sha", type=Path)
    cli.add_argument("nv12", type=Path)
    args = cli.parse_args()
    nv12 = args.nv12.read_bytes()
    print(json.dumps({"sha256": sha_control(args.sha.read_bytes()),
                      "nv12": [nv12_control(nv12, followup) for followup in (False, True)]}, indent=2))
