"""Compare FDN and rotation spans delimited by the same four stored words."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from struct import pack

from extract_sa_fdn_payloads import payloads
from probe_sa_fdn_dispatch import MEMBER_SHA, ROW, word
from run_sa_fdn_dispatch import DispatchMachine, bank_entry


ROT_SHA = "e84fbc41a6dbb0f6178514fb94cd49f74d9a9361fd35faed8d9f413aa39e8ec9"
MARKER = pack("<4I", 0, 0x01000000, 0x01000000, 0x01000000)


def spans(blob):
    """Yield each body with its observed marker, without assigning it a role."""
    start = 0
    for offset in range(0, len(blob) - len(MARKER) + 1, 4):
        if offset >= start and blob[offset:offset + len(MARKER)] == MARKER:
            end = offset + len(MARKER)
            yield start, blob[start:end]
            start = end
    if not start or start != len(blob):
        raise ValueError("expected a nonempty word sequence ending at a marker")


def rotation_image(blob):
    block = 0x100
    stage = block + word(blob, block + 0x10)
    table = block + word(blob, block + 0x20)
    if word(blob, block + 0x24) != 3 * ROW.size:
        raise ValueError("unexpected rotation descriptor count")
    src, dest, w2, w3 = ROW.unpack_from(blob, table)
    if src != 0 or dest != 0 or w2 & 0xffff != 0x4000 or w3 != w2 & 0xffff0000:
        raise ValueError("unexpected first rotation transfer descriptor")
    end = stage + (w2 >> 16) + 1
    if table + 3 * ROW.size > stage or end > len(blob):
        raise ValueError("rotation image outside member")
    return stage, blob[stage:end]


def rotation(blob):
    stage, image = rotation_image(blob)
    addresses = {"stage_dsp_address": "0x0", "first_transfer_destination": "0x0",
                 "mailbox_selector": "0x17800", "stack": hex(word(image, 0x10))}
    calls = {}
    for call_slice in (0x480, 0x5d8, 0x69c, 0x728):
        # Execute only the straight-line entry setup; no fabricated device reads.
        machine = DispatchMachine(addresses, image)
        machine.pc = call_slice
        stop = machine.run(16)
        if (stop != {"pc": 0x19c, "reason": "unsupported opcode", "opcode": 0x000a}
                or machine.mmio_writes or machine.mailbox_reads):
            raise ValueError(f"unexpected rotation call stop: {stop}")
        calls[machine.reg[4]] = machine.link - 2
    start, wrapper = min(calls), 0x19c
    data = image[start:wrapper]
    if set(calls) != {start + offset for offset, _ in spans(data)}:
        raise ValueError("rotation calls and observed span starts do not agree")
    return stage, start, wrapper, calls, data


def compare(fdn, rot):
    expected = ((fdn, MEMBER_SHA, "FDN"), (rot, ROT_SHA, "rotation"))
    for blob, digest, label in expected:
        if sha256(blob).hexdigest() != digest:
            raise ValueError(f"expected pinned {label} member")

    stage, start, wrapper, calls, data = rotation(rot)
    ranges = {"rotation": (stage + start, data)}
    a1 = None
    for block in "AB":
        for raw, meta in payloads(fdn, block):
            label = f"{block}/{meta['selector']}"
            ranges[label] = (meta["copies"][0]["file_offset"], raw)
            if label == "A/1":
                a1 = (raw, meta)
    records = {}
    duplicates = defaultdict(list)
    for label, (base, raw) in ranges.items():
        records[label] = []
        for index, (offset, record) in enumerate(spans(raw)):
            digest = sha256(record).hexdigest()
            entry = {"member_offset": base + offset,
                     "body_words": (len(record) - len(MARKER)) // 4,
                     "sha256_including_marker": digest}
            if label == "rotation":
                entry["dsp_call_pc"] = calls[start + offset]
            records[label].append(entry)
            duplicates[record].append(f"{label}:{index}")

    raw, meta = a1
    boundaries = dict(spans(raw))
    a4 = []
    for bank, copy in enumerate(meta["copies"]):
        run = bank_entry(fdn, "A", bank, 4)
        stop = run["after_fixture"]
        if stop != {"pc": copy["dispatch_call_pc"], "reason": "unsupported opcode", "opcode": 0x000a}:
            raise ValueError(f"A/4 did not reach the wrapper in bank {bank}: {stop}")
        dsp = run["registers_at_stop"]["r4"]
        relative = dsp - copy["local_r4"]
        if relative not in boundaries:
            raise ValueError("A/4 points inside A/1 but not at an observed span start")
        a4.append({"bank": bank, "dsp_r4": dsp, "a1_byte_offset": relative,
                   "member_offset": copy["file_offset"] + relative,
                   "mailbox_offsets": [event["address"] - copy["selector_read"]
                                       for event in run["mailbox_reads"]]})
    return {"records": records, "rotation_wrapper_dsp": wrapper,
            "fdn_a4": a4, "identical_records": [v for v in duplicates.values() if len(v) > 1]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fdn", type=Path, help="Exact SA_FDN__SA07 member (SP job 0x40)")
    cli.add_argument("rotation", type=Path, help="Exact SA_ROT900x02 member (SP job 0x07)")
    args = cli.parse_args()
    print(json.dumps(compare(args.fdn.read_bytes(), args.rotation.read_bytes()), indent=2))
