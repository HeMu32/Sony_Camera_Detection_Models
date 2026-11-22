"""Map Sony's raw little-endian 64-bit WBI2 warm-boot image.

Layout: Sony kernel include/linux/warmboot.h and kernel/power/wbi_raw.c.
Compressed images and 32-bit descriptors are intentionally unsupported.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import mmap
from pathlib import Path
from struct import Struct


HEADER = Struct("<4sI10Q")
RECORD = Struct("<8Q")
SECTOR = 512
ZERO = 1
SEED = 0x0F0F0F0F


def round_sector(size):
    return (size + SECTOR - 1) // SECTOR * SECTOR


@dataclass(frozen=True)
class Section:
    index: int
    descriptor_offset: int
    file_offset: int | None
    stored_size: int
    physical: int
    virtual: int
    memory_size: int
    flags: int


@dataclass
class Image:
    header: dict
    sections: list[Section]
    table_start: int
    table_end: int
    trailer_start: int
    file_size: int

    def locate(self, address, space="file"):
        if space not in ("file", "physical", "virtual"):
            raise ValueError(f"unknown address space: {space}")
        for section in self.sections:
            if space == "file":
                start, size = section.file_offset, section.stored_size
            else:
                start, size = getattr(section, space), section.memory_size
            if start is not None and start <= address < start + size:
                offset = address - start
                return {
                    "section": section.index,
                    "offset_in_section": offset,
                    "file": None if section.file_offset is None else section.file_offset + offset,
                    "physical": section.physical + offset,
                    "virtual": section.virtual + offset,
                }
        return None


def parse(data):
    if len(data) < SECTOR:
        raise ValueError("missing header sector")
    (magic, padding, count, flags, resume, version, sector, raw_size,
     kernel_start, kernel_size, kernel_checksum, memory_size) = HEADER.unpack_from(data)
    if magic != b"WBI2" or padding != 0 or sector != SECTOR:
        raise ValueError("expected WBI2 with a 64-bit header and 512-byte sectors")
    if flags & 0x09:
        raise ValueError("compressed WBI2 is not supported")
    if not count or raw_size % SECTOR:
        raise ValueError("invalid section count or raw payload size")
    table_start = SECTOR + raw_size
    table_end = table_start + count * RECORD.size
    trailer_start = round_sector(table_end)
    if trailer_start > len(data):
        raise ValueError("section table truncated")

    cursor = SECTOR
    sections = []
    for index in range(count):
        descriptor = table_start + index * RECORD.size
        pa, stored, checksum, attr, size, va, meta_checksum, padding = RECORD.unpack_from(
            data, descriptor
        )
        if meta_checksum != SEED ^ pa ^ stored ^ checksum ^ attr ^ size ^ va:
            raise ValueError(f"section {index}: invalid descriptor checksum")
        if not size or attr & ~ZERO or padding or stored != (0 if attr & ZERO else size):
            raise ValueError(f"section {index}: invalid raw section")
        sections.append(Section(
            index, descriptor, cursor if stored else None, stored, pa, va, size, attr
        ))
        cursor += round_sector(stored)
        if cursor > table_start:
            raise ValueError(f"section {index}: payload overlaps the table")
    if cursor != table_start or sum(s.memory_size for s in sections) != memory_size:
        raise ValueError("section lengths do not match the header")
    for space in ("physical", "virtual"):
        starts = sorted((getattr(s, space), s.memory_size) for s in sections)
        if any(start + size > next_start for (start, size), (next_start, _) in
               zip(starts, starts[1:])):
            raise ValueError(f"overlapping {space} sections")

    return Image({
        "version": version, "sector_size": sector, "section_count": count, "flags": flags,
        "resume_vector": resume, "stored_size": raw_size, "memory_size": memory_size,
        "kernel_start": kernel_start, "kernel_size": kernel_size,
        "kernel_checksum": kernel_checksum,
    }, sections, table_start, table_end, trailer_start, len(data))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("image", type=Path)
    cli.add_argument("--sections", action="store_true")
    cli.add_argument("--find-ascii", action="append", default=[])
    cli.add_argument("--find-hex", action="append", default=[])
    cli.add_argument("--file-offset", action="append", type=lambda s: int(s, 0), default=[])
    cli.add_argument("--physical", action="append", type=lambda s: int(s, 0), default=[])
    cli.add_argument("--virtual", action="append", type=lambda s: int(s, 0), default=[])
    args = cli.parse_args()
    with args.image.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        image = parse(data)
        report = {
            "header": image.header,
            "payload": [SECTOR, image.table_start],
            "section_table": [image.table_start, image.table_end],
            "trailer": [image.trailer_start, image.file_size],
            "zero_sections": sum(s.file_offset is None for s in image.sections),
            "resume_mapping": image.locate(image.header["resume_vector"], "physical"),
            "lookups": [
                {"space": space, "address": address, "mapping": image.locate(address, space)}
                for space, addresses in (("file", args.file_offset), ("physical", args.physical),
                                         ("virtual", args.virtual))
                for address in addresses
            ],
            "hits": {},
        }
        if args.sections:
            report["sections"] = [asdict(s) for s in image.sections]
        needles = [(s, s.encode("ascii")) for s in args.find_ascii]
        needles.extend((s, bytes.fromhex(s)) for s in args.find_hex)
        for label, needle in needles:
            if not needle:
                cli.error("empty search pattern")
            hits = report["hits"][label] = []
            start = 0
            while (found := data.find(needle, start)) != -1:
                mapping = image.locate(found)
                if mapping and mapping["offset_in_section"] + len(needle) > (
                    image.sections[mapping["section"]].stored_size
                ):
                    mapping = None
                hits.append({"file": found, "mapping": mapping})
                start = found + 1
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
