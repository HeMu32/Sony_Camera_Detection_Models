"""Compare candidate port-control bits against the order of original SA records."""

import argparse
import json
from math import sqrt
from pathlib import Path
from struct import pack

from .sony_sa_program_planes import HEADER_SLOTS, PROGRAM_BASE, boot_programs, join_record, overlays
from .sony_sa_slot_fields import SlotFields


MARKER = pack("<4I", 0, 0x01000000, 0x01000000, 0x01000000)
DEFAULT_SUFFIX = SlotFields.decode(HEADER_SLOTS[1]).suffix50


def marked_records(data, image, *, first_only=False):
    start = 0
    records = []
    for offset in range(0, len(data) - len(MARKER) + 1, 4):
        if offset < start or data[offset:offset + len(MARKER)] != MARKER:
            continue
        end = offset + len(MARKER)
        records.append([SlotFields.decode(row["plane_words"])
                        for row in join_record(image, data[start:end])])
        if first_only:
            return records
        start = end
    if start != len(data) or not records:
        raise ValueError("record range does not end at an aligned marker")
    return records


def timing_counts(records, selector, bit, offset):
    both = selected = changed = total = 0
    mask = 1 << bit
    for record in records:
        for i, slot in enumerate(record):
            other = i + offset
            if not 0 <= other < len(record):
                continue
            a = selector in slot.selectors4
            b = bool((record[other].suffix50 ^ DEFAULT_SUFFIX) & mask)
            both += a and b
            selected += a
            changed += b
            total += 1
    denominator = sqrt(selected * changed * (total - selected) * (total - changed))
    return {"both": both, "selected": selected, "changed": changed, "compared": total,
            "correlation": (both * total - selected * changed) / denominator if denominator else None}


def rank_bits(controls, selector):
    results = []
    for bit in range(50):
        for offset in range(-4, 3):
            correlations = [timing_counts(records, selector, bit, offset)["correlation"]
                            for records in controls.values()]
            correlations = [s for s in correlations if s is not None]
            # Exclude candidates with no observable variation in most controls.
            if len(correlations) >= 3:
                results.append({"suffix_bit": bit, "stored_word_offset": offset,
                                "correlation": sum(correlations) / len(correlations)})
    return sorted(results, key=lambda entry: -entry["correlation"])


def analyze(parts):
    controls, held_out = {}, {}
    for name, label, bank, lo, hi in (
            ("SA_ROT900x02", "rotation", 0, 0x35c, 0x4dc),
            ("SA_NV12_0x02", "nv12", 1, 0xe54, 0xf60),
            ("SA_SHA250x02", "sha_short", 0, 0x4bc, 0x564),
            ("SA_SHA250x02", "sha_long", 1, 0x14e4, 0x15cc)):
        data = (parts / f"{name}.bin").read_bytes()
        image = boot_programs(data)[PROGRAM_BASE + bank * 0x1000]
        controls[label] = marked_records(data[lo:hi], image)
    fdn = (parts / "SA_FDN__SA07.bin").read_bytes()
    _, a = overlays(fdn)
    _, b = overlays(fdn, 0x1a800)
    for label, image, lo, hi in (
            ("a96", a[0][2], 0x968, 0xde4),
            ("a87", a[0][3], 0xb00, 0xde4),
            ("b87", b[0][1], 0x1ae4c, 0x1b360),
            ("b62_head", b[0][3], 0x1b320, 0x1b360)):
        held_out[label] = marked_records(fdn[lo:hi], image, first_only=True)
    output = {}
    for selector in (6, 7):
        rankings = rank_bits(controls, selector)
        bit, offset = (rankings[0][k] for k in ("suffix_bit", "stored_word_offset"))
        output[str(selector)] = {
            "highest_ranked": rankings[:5],
            "controls": {name: timing_counts(records, selector, bit, offset)
                         for name, records in controls.items()},
            "held_out_fdn": {name: timing_counts(records, selector, bit, offset)
                             for name, records in held_out.items()},
        }
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parts", type=Path)
    args = cli.parse_args()
    print(json.dumps(analyze(args.parts), indent=2))
