"""Read the three program-word planes from Sony SA startup and overlay tables."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct


U32 = Struct("<I")
U64 = Struct("<Q")
ROW = Struct("<4I")
PAIR = Struct("<2I")
PROGRAM_BASE = 0x04000000
PROGRAM_BANK_BYTES = 0x1000
BANK_COUNT = 6
PLANES = 3
OVERLAY_ROW_BYTES = 0x40
HEADER_SLOTS = (
    (0x0000080000000000, 0xf1c78f000701c000, 0x8fffffffffffffff),
    (0x1f70fb87dc3ee1f7, 0xf1c78f000701c3ee, 0xbfffffffffffffff),
)
PACKED_PROGRAM_FLAGS = 0x04000200
PACKED_TARGET_BITS = 19


@dataclass(frozen=True)
class Program:
    offset: int
    planes: tuple

    @property
    def slots(self):
        return len(self.planes[0])

    def at(self, index):
        if not 0 <= index < self.slots:
            raise ValueError(f"slot {index} outside program of {self.slots} slots")
        return tuple(plane[index] for plane in self.planes)


def program(member, offset, slots):
    if not 2 <= slots <= 128 or not 0 <= offset <= len(member) - PLANES * slots * U64.size:
        raise ValueError("program plane extent outside member")
    plane_bytes = slots * U64.size
    planes = tuple(tuple(w for w, in U64.iter_unpack(
        member[offset + i * plane_bytes:offset + (i + 1) * plane_bytes])) for i in range(PLANES))
    result = Program(offset, planes)
    if tuple(result.at(i) for i in range(len(HEADER_SLOTS))) != HEADER_SLOTS:
        raise ValueError(f"unrecognized SA program plane prefix at {offset:#x}")
    return result


def descriptor_slots(destination, flags2, flags3):
    if (destination < PROGRAM_BASE or destination >= PROGRAM_BASE + BANK_COUNT * PROGRAM_BANK_BYTES
            or destination % PROGRAM_BANK_BYTES):
        return None
    return plane_slots(flags2, flags3)


def plane_slots(flags2, flags3):
    """Read the split-plane length independently of the destination window."""
    if (flags2 >> 30, flags3 >> 30, flags2 & 0xffff, flags3 & 0xffff) not in (
            (2, 1, 0x4000, 0x200), (2, 1, 0, 0x200)):
        return None
    source_size, plane_size = (((f >> 16) & 0x3fff) + 1 for f in (flags2, flags3))
    if source_size != PLANES * plane_size or plane_size % U64.size:
        raise ValueError("program descriptor plane size mismatch")
    return plane_size // U64.size


def header_word(member, block, offset):
    if not 0 <= block <= len(member) - offset - U32.size:
        raise ValueError("startup header outside member")
    return U32.unpack_from(member, block + offset)[0]


def startup_rows(member, block=0x100):
    source = block + header_word(member, block, 0x10)
    table = block + header_word(member, block, 0x20)
    end = table + header_word(member, block, 0x24)
    if not block <= table <= end <= source <= len(member) or (end - table) % ROW.size:
        raise ValueError("invalid startup table")
    yield from ((source + offset, target, flags2, flags3)
                for offset, target, flags2, flags3 in ROW.iter_unpack(member[table:end]))


def boot_programs(member, block=0x100):
    results = {}
    for offset, target, flags2, flags3 in startup_rows(member, block):
        slots = descriptor_slots(target, flags2, flags3)
        if slots is not None:
            if target in results:
                raise ValueError(f"duplicate program window {target:#x}")
            results[target] = program(member, offset, slots)
    return results


def packed_boot_programs(member, block=0x100):
    """Decode rows that pack a slot count into the raw destination field."""
    results = {}
    for offset, target, flags2, flags3 in startup_rows(member, block):
        if flags3 != PACKED_PROGRAM_FLAGS:
            continue
        slots = (target >> PACKED_TARGET_BITS) + 1
        size = ((flags2 >> 16) & 0x3fff) + 1
        if flags2 & 0xffff != 0x4000 or size != PLANES * slots * U64.size:
            raise ValueError(f"packed program size mismatch at {offset:#x}")
        if target in results:
            raise ValueError(f"duplicate packed target {target:#x}")
        results[target] = program(member, offset, slots)
    return results


def overlays(member, block=0x100):
    """Decode six bank rows of eight (relative offset, slot count) pairs."""
    table = block + header_word(member, block, 0x0c)
    min_start = table + BANK_COUNT * OVERLAY_ROW_BYTES
    if not block < table < min_start <= len(member):
        raise ValueError("invalid overlay directory")
    result = []
    for bank in range(BANK_COUNT):
        start = table + bank * OVERLAY_ROW_BYTES
        images = []
        for relative, slots in PAIR.iter_unpack(member[start:start + OVERLAY_ROW_BYTES]):
            if (relative, slots) == (0, 0xffffffff):
                images.append(None)
                continue
            if table + relative < min_start:
                raise ValueError("overlay extent overlaps directory")
            images.append(program(member, table + relative, slots))
        result.append(tuple(images))
    return table, tuple(result)


def join_record(image, record):
    """Attach a candidate indexed slot to each raw 32-bit argument word."""
    return [
        {"record_word": word, "index": index, "bit31": word >> 31,
         "low24": word & 0xffffff, "plane_words": image.at(index)}
        for word, in U32.iter_unpack(record)
        for index in ((word >> 24) & 0x7f,)
    ]


def summarize(member, image):
    size = image.slots * PLANES * U64.size
    return {"offset": image.offset, "slots": image.slots, "bytes": size,
            "sha256": sha256(member[image.offset:image.offset + size]).hexdigest()}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    cli.add_argument("--block", type=lambda s: int(s, 0), default=0x100)
    cli.add_argument("--overlays", action="store_true")
    args = cli.parse_args()
    data = args.member.read_bytes()
    output = {"boot": {hex(dst): summarize(data, image)
                       for dst, image in boot_programs(data, args.block).items()}}
    packed = packed_boot_programs(data, args.block)
    if packed:
        output["packed_boot"] = {
            hex(dst): summarize(data, image) | {"raw_low19": dst & ((1 << PACKED_TARGET_BITS) - 1)}
            for dst, image in packed.items()
        }
    if args.overlays:
        offset, banks = overlays(data, args.block)
        output |= {"overlay_table": offset,
                   "banks": [[summarize(data, image) if image else None for image in bank]
                             for bank in banks]}
    print(json.dumps(output, indent=2))
