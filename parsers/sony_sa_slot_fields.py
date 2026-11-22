"""Lossless SA slot split and tests of candidate datapath field boundaries."""

import argparse
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path

from .sony_sa_program_planes import HEADER_SLOTS, boot_programs, overlays


FIELDS = 6
FIELD_BITS = 13
IDLE_FIELD = 0x1f7
SELECTORS = 15
SELECTOR_BITS = 4
MASK64 = (1 << 64) - 1


def bit_fields(word, count, width):
    return tuple(word >> (width * i) & ((1 << width) - 1) for i in range(count))


def pack_fields(values, count, width):
    if len(values) != count or not all(0 <= value < 1 << width for value in values):
        raise ValueError(f"expected {count} unsigned {width}-bit fields")
    return sum(value << (width * i) for i, value in enumerate(values))


@dataclass(frozen=True)
class SlotFields:
    fields13: tuple
    suffix50: int
    selectors4: tuple
    top4: int

    @classmethod
    def decode(cls, words):
        a, b, c = words
        if not all(0 <= word <= MASK64 for word in words):
            raise ValueError("expected three unsigned 64-bit plane words")
        packed = a | b << 64
        return cls(bit_fields(packed, FIELDS, FIELD_BITS), packed >> (FIELDS * FIELD_BITS),
                   bit_fields(c, SELECTORS, SELECTOR_BITS), c >> (SELECTORS * SELECTOR_BITS))

    def encode(self):
        if not 0 <= self.suffix50 < 1 << 50 or not 0 <= self.top4 < 16:
            raise ValueError("invalid SA slot suffix")
        ab = pack_fields(self.fields13, FIELDS, FIELD_BITS) | self.suffix50 << (FIELDS * FIELD_BITS)
        c = pack_fields(self.selectors4, SELECTORS, SELECTOR_BITS) | self.top4 << 60
        return ab & MASK64, ab >> 64, c

    @property
    def field_parts(self):
        # These are observable bit groups, not a named opcode or register file.
        return tuple((value >> 8, value >> 4 & 15, value & 15) for value in self.fields13)


def slot_population(members):
    """Deduplicate slots by all three raw words, not by a candidate field split."""
    population = set()
    for member, blocks, has_overlays in members:
        for block in blocks:
            programs = list(boot_programs(member, block).values())
            if has_overlays:
                _, catalog = overlays(member, block)
                programs.extend(image for row in catalog for image in row if image is not None)
            for image in programs:
                population.update(zip(*image.planes))
    return population


def selector_association(population):
    """Count candidate channel/selector associations, preserving counterexamples."""
    matrix = [[0] * FIELDS for _ in range(FIELDS)]
    counts = [0] * FIELDS
    mismatches = []
    for words in sorted(population):
        slot = SlotFields.decode(words)
        for code in sorted(set(slot.selectors4) & set(range(FIELDS))):
            counts[code] += 1
            for field, word in enumerate(slot.fields13):
                matrix[code][field] += word != IDLE_FIELD
            if slot.fields13[code] == IDLE_FIELD:
                mismatches.append({"words": words, "selector_code": code})
    return {"unique_slots": len(population), "occurrences_by_code": counts,
            "nondefault_by_code_and_field": matrix, "counterexamples": mismatches}


def rank_boundaries(population):
    """Compare nearby competing splits by mean binary correlation per channel."""
    inputs = [(a | b << 64, set(bit_fields(c, SELECTORS, SELECTOR_BITS)))
              for a, b, c in population]
    total = len(inputs)
    idle_a, idle_b, _ = HEADER_SLOTS[1]
    idle = idle_a | idle_b << 64
    selected = [sum(j in route for _, route in inputs) for j in range(FIELDS)]
    results = []
    for width in range(8, 18):
        mask = (1 << width) - 1
        for start in range(13):
            scores = []
            for j in range(FIELDS):
                shift = start + j * width
                default = idle >> shift & mask
                active = both = 0
                for bits, selector in inputs:
                    if bits >> shift & mask != default:
                        active += 1
                        both += j in selector
                n = selected[j]
                denominator = sqrt(n * active * (total - n) * (total - active))
                scores.append((both * total - n * active) / denominator if denominator else 0)
            results.append({"width": width, "start_bit": start, "score": sum(scores) / FIELDS})
    return sorted(results, key=lambda result: -result["score"])


def analyze(parts):
    def member(name, blocks=(0x100,), has_overlays=False):
        return (parts / f"{name}.bin").read_bytes(), blocks, has_overlays

    cohorts = {
        "controls": [member(name) for name in ("SA_SHA250x02", "SA_ROT900x02", "SA_NV12_0x02")],
        "fdn": [member("SA_FDN__SA07", (0x100, 0x1a800), has_overlays=True)],
        "pdn": [member("SA_PDN__SA07", has_overlays=True)],
    }
    result = {}
    for name, sources in cohorts.items():
        population = slot_population(sources)
        result[name] = {**selector_association(population),
                        "highest_ranked_boundaries": rank_boundaries(population)[:5]}
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parts", type=Path, help="Directory of extracted A7 IV members")
    args = cli.parse_args()
    print(json.dumps(analyze(args.parts), indent=2))
