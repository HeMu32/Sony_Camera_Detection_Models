"""Compare stored SA operands with known shift constants and scalar setup writes."""

import argparse
import json
from pathlib import Path

from .fdn_geometry import APERTURE
from .sony_sa_pipeline import MARKER
from .sony_sa_program_planes import PROGRAM_BASE, boot_programs, header_word, join_record, overlays
from .sony_sa_slot_fields import SlotFields


def argument_bytes(word):
    """Return the three little-endian argument bytes in high-to-low display order."""
    return tuple(word >> bit & 0xff for bit in (16, 8, 0))


def commands(data, program, extent):
    lo, hi = extent
    record = data[lo:hi]
    if len(record) != hi - lo or not record.endswith(MARKER):
        raise ValueError(f"unrecognized command record at {lo:#x}")
    return join_record(program, record[:-len(MARKER)])


def complementary_triplets(record, shifts):
    """Locate triples of stored byte pairs; do not infer executed instructions."""
    target = [(s, 32 - s) for s in shifts]
    high_middle = [argument_bytes(entry["record_word"])[:2] for entry in record]
    return [i for i in range(len(high_middle) - len(target) + 1)
            if high_middle[i:i + len(target)] == target]


def compare_rows(record, old_indices, new_indices):
    result = []
    for old, new in zip(old_indices, new_indices, strict=True):
        old_row, new_row = record[old], record[new]
        a, b = (argument_bytes(row["record_word"]) for row in (old_row, new_row))
        first, second = (SlotFields.decode(row["plane_words"]) for row in (old_row, new_row))
        result.append({
            "old_word_index": old, "new_word_index": new,
            "old_bytes": a, "new_bytes": b,
            "old_high_reappears_as_new_low": a[0] == b[2],
            "middle_unchanged": a[1] == b[1],
            "changed_fields13": [i for i, (x, y) in enumerate(zip(first.fields13, second.fields13)) if x != y],
            "changed_selectors": [i for i, (x, y) in enumerate(zip(first.selectors4, second.selectors4)) if x != y],
            "suffix_xor": first.suffix50 ^ second.suffix50,
            "top4_xor": first.top4 ^ second.top4,
        })
    return result


def sha_operands(data):
    programs = boot_programs(data)
    short = commands(data, programs[PROGRAM_BASE], (0x4bc, 0x564))
    long = commands(data, programs[PROGRAM_BASE + 0x1000], (0x14e4, 0x15cc))
    return {
        "large_sigma_1_triplets_high_middle": complementary_triplets(long, (11, 6, 25)),
        "large_sigma_0_triplets_high_middle": complementary_triplets(long, (2, 13, 22)),
        "large_sigma_0_observed_high_middle": [argument_bytes(long[i]["record_word"])[:2]
                                               for i in range(17, 20)],
        "long_tail_matching_pairs": compare_rows(long, range(16, 20), range(38, 42)),
        "short_schedule_bytes": [argument_bytes(short[i]["record_word"]) for i in range(19, 23)],
        "short_repeat_bytes": [argument_bytes(short[i]["record_word"]) for i in range(26, 30)],
    }


def fdn_operands(data):
    _, a = overlays(data)
    _, b = overlays(data, 0x1a800)
    result = {}
    for name, prog, extent, words_written in (
            ("a_first", a[0][0], (0x51c, 0x5c8), 20),
            ("b_compact", b[0][3], (0x1b320, 0x1b360), 2)):
        record = commands(data, prog, extent)
        arguments = sorted({entry["low24"] for entry in record})
        repeated_slots = {}
        for item in record:
            repeated_slots.setdefault(tuple(item["plane_words"]), set()).add(item["low24"])
        result[name] = {
            "member_extent": extent, "stored_words": len(record),
            "argument_values": arguments,
            "outside_written_word_indices": [v for v in arguments if v >= words_written],
            "not_aligned_byte_offsets": [v for v in arguments if v & 3],
            "different_arguments_for_identical_slots": sorted(
                sorted(values) for values in repeated_slots.values() if len(values) > 1),
        }
    return result


def aperture_candidates(record, setup):
    """Compare each nonzero low byte to the last scalar setup, not actual reads."""
    candidates = []
    for position, row in enumerate(record):
        index = row["low24"] & 0xff
        if index == 0:
            continue
        address = APERTURE + 4 * index
        candidates.append({"position": position, "word": row["record_word"], "address": address,
                           "last_written_value": setup.get(address)})
    return candidates


def fdn_aperture_candidates(data, workload):
    """Use only the setup and selected command known at each original call boundary."""
    _, catalog = overlays(data)
    banks = {image.slots: image for image in catalog[0] if image is not None}
    source = 0x100 + header_word(data, 0x100, 0x30)
    limit = 0x100 + header_word(data, 0x100, 0x10)
    boot = workload["boot_control_fixture"]
    second = boot["isolated_after_first_compute"]

    def snapshot(writes):
        return {w["address"]: w["value"] for w in writes if w["size"] == 4}

    calls = [
        (boot["first_registers_r4_to_r7"][0], 108, snapshot(workload["register_writes"])),
        (second["registers_r4_to_r7"][0], 108, snapshot(second["aperture_writes"])),
        *((call["registers_r4_to_r7"][0], call["program_slots"],
           {int(k, 16): v for k, v in call["register_snapshot"].items()})
          for call in second.get("later_control", {}).get("calls", [])),
    ]
    report = []
    for pointer, count, setup in calls:
        offset = source + pointer - 0x50000000
        if not source <= offset < limit:
            raise ValueError(f"record pointer outside the external firmware image at {offset:#x}")
        stop = data.find(MARKER, offset, limit)
        if stop <= offset or (stop - offset) % 4:
            raise ValueError(f"expected an aligned firmware command record at {offset:#x}")
        native = commands(data, banks[count], (offset, stop + len(MARKER)))
        report.append({"record_offset": offset, "slots": count,
                       "operands": aperture_candidates(native, setup)})
    return report


def compared_kernel(fdn, pdn):
    def signatures(program, data, lo, hi):
        return [(entry["record_word"] & 0x80ffffff, entry["plane_words"])
                for entry in join_record(program, data[lo:hi])]

    _, fdn_banks = overlays(fdn)
    _, pdn_banks = overlays(pdn)
    refs = signatures(fdn_banks[0][0], fdn, 0x5c8, 0x708)
    pdn_extent = (0x538, 0x678)
    prime, alternate = pdn_banks[0][:2]
    native = signatures(prime, pdn, *pdn_extent)
    other = signatures(alternate, pdn, *pdn_extent)
    native_record = pdn[slice(*pdn_extent)]
    pdn_copies = []
    offset = pdn.find(native_record)
    while offset != -1:
        pdn_copies.append(offset)
        offset = pdn.find(native_record, offset + 1)
    return {
        "fdn_extent": (0x5c8, 0x708), "pdn_extent": pdn_extent, "words": len(refs),
        "fdn_program_offset": fdn_banks[0][0].offset,
        "pdn_121_offset": prime.offset, "pdn_122_offset": alternate.offset,
        "matching_121": sum(x == y for x, y in zip(refs, native, strict=True)),
        "matching_122": sum(x == y for x, y in zip(refs, other, strict=True)),
        "same_121_table_all_banks": all(bank[0] == prime for bank in pdn_banks),
        "pdn_record_copies": pdn_copies,
    }


def compared_first_stage(fdn, pdn):
    _, f = overlays(fdn)
    _, p = overlays(pdn)
    short, = (image for image in p[0] if image is not None and image.slots == 39)
    extent = (0xfb0, 0x105c)
    a = commands(fdn, f[0][0], (0x51c, 0x5c8))
    b = commands(pdn, short, extent)
    pairs = list(zip(a, b, strict=True))
    return {
        "words": len(pairs), "pdn_extent": extent, "pdn_table": short.offset,
        "identical_plane_positions": [i for i, (x, y) in enumerate(pairs)
                                      if x["plane_words"] == y["plane_words"]],
        "changed_slot_positions": [i for i, (x, y) in enumerate(pairs) if x["index"] != y["index"]],
        "changed_arguments": [(i, x["low24"], y["low24"]) for i, (x, y) in enumerate(pairs)
                              if x["low24"] != y["low24"]],
        "same_bit31": all(x["bit31"] == y["bit31"] for x, y in pairs),
    }


def analyze(parts):
    fdn = (parts / "SA_FDN__SA07.bin").read_bytes()
    pdn = (parts / "SA_PDN__SA07.bin").read_bytes()
    return {
        "sha": sha_operands((parts / "SA_SHA250x02.bin").read_bytes()),
        "fdn": fdn_operands(fdn),
        "fdn_pdn_first_stage": compared_first_stage(fdn, pdn),
        "fdn_pdn_second_kernel": compared_kernel(fdn, pdn),
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parts", type=Path, help="Directory with original extracted SA members")
    cli.add_argument("--fdn-trace", type=Path, help="FDN trace from --boot-control --trace-later-stages")
    args = cli.parse_args()
    result = analyze(args.parts)
    if args.fdn_trace:
        trace = json.loads(args.fdn_trace.read_text(encoding="utf-8-sig"))
        fdn = (args.parts / "SA_FDN__SA07.bin").read_bytes()
        result["aperture_candidates"] = fdn_aperture_candidates(fdn, trace["first_workload_fixture"])
    print(json.dumps(result, indent=2))
