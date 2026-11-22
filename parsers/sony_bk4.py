"""Read contiguous Sony BK4 backup namespace and item tables."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct
import tarfile


COUNTS = Struct("<II")
DATA_RANGE = Struct("<II")
ENTRY = Struct("<HI")
U16 = Struct("<H")
GROUPS_OFFSET = 0x100


class Backup:
    def __init__(self, data):
        if len(data) < GROUPS_OFFSET or data[0x0c:0x10] != b"BK4\0":
            raise ValueError("expected a BK4 backup")
        self.data = data
        groups, items = COUNTS.unpack_from(data, 0x10)
        self.items_offset = GROUPS_OFFSET + groups * ENTRY.size
        self.end = self.items_offset + items * ENTRY.size
        if self.end > len(data):
            raise ValueError("BK4 index exceeds backup length")
        limit = 0
        self.groups = tuple(ENTRY.iter_unpack(data[GROUPS_OFFSET:self.items_offset]))
        for count, first in self.groups:
            if first != limit:
                raise ValueError("noncontiguous BK4 namespace index")
            limit += count
        if limit != items:
            raise ValueError("BK4 item count disagrees with namespace counts")

    def lookup(self, key):
        if not 0 <= key < 1 << 32:
            raise ValueError("expected a 32-bit backup key")
        namespace, item = key >> 16, key & 0xffff
        if namespace >= len(self.groups):
            return None
        count, first = self.groups[namespace]
        if item >= count:
            return None
        index = self.items_offset + (first + item) * ENTRY.size
        _, ref = ENTRY.unpack_from(self.data, index)
        if ref == 0xffffffff:
            return None
        kind, offset = ref >> 24, ref & 0xffffff
        if not self.end <= offset <= len(self.data):
            raise ValueError(f"BK4 value outside data range for {key:#x}")
        if kind in (0, 0xff):
            prefix = 4 if kind == 0 else 2
            if offset + prefix > len(self.data):
                raise ValueError(f"truncated BK4 size prefix for {key:#x}")
            size, = U16.unpack_from(self.data, offset)
            offset += prefix
        else:
            size = kind
        if offset + size > len(self.data):
            raise ValueError(f"BK4 value outside data range for {key:#x}")
        return offset, self.data[offset:offset + size]


def published_pair(data):
    """Split a two-image Sony 0110_backup member at its aligned image boundary."""
    def image_end(start):
        image = Backup(data[start:])
        offset, length = DATA_RANGE.unpack_from(data, start + 0x18)
        end = start + offset + length
        if not (image.end <= offset and start + offset < end <= len(data)):
            raise ValueError("invalid published BK4 data range")
        return end

    end = image_end(0)
    second = (end + 0xfff) & ~0xfff
    if second >= len(data) or any(data[end:second]) or image_end(second) != len(data):
        raise ValueError("invalid published BK4 pair extent")
    return (0, data[:end]), (second, data[second:])


def report(data, keys):
    parsed = Backup(data)
    values = {}
    for key in keys:
        found = parsed.lookup(key)
        namespace = key >> 16
        count = parsed.groups[namespace][0] if namespace < len(parsed.groups) else 0
        item = {"namespace_size": count, "found": found is not None}
        if found is not None:
            offset, raw = found
            item.update(offset=hex(offset), length=len(raw), sha256=sha256(raw).hexdigest(),
                        byte_at_0x40=hex(raw[0x40]) if len(raw) >= 0x41 else None)
        values[hex(key)] = item
    return values


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path, help="BK4 file, or --firmware-tar backup archive")
    cli.add_argument("--key", required=True, action="append", type=lambda s: int(s, 0))
    cli.add_argument("--firmware-tar", action="store_true")
    cli.add_argument("--segment", choices=("first", "second"), default="first")
    args = cli.parse_args()
    if not args.firmware_tar:
        data = args.source.read_bytes()
        if args.segment == "second":
            _, data = published_pair(data)[1]
        print(json.dumps(report(data, args.key), indent=2))
        return
    results = Counter()
    examples = {}
    with tarfile.open(args.source, "r:") as tar:
        for member in tar:
            if not (member.isfile() and member.name.startswith("0110_backup/")
                    and member.name.endswith(".bin")):
                continue
            with tar.extractfile(member) as file:
                data = file.read()
            offset, image = published_pair(data)[1] if args.segment == "second" else (0, data)
            profile = json.dumps(report(image, args.key), sort_keys=True)
            results[profile] += 1
            if profile not in examples:
                examples[profile] = {"name": member.name, "archive_offset": hex(member.offset_data),
                                     "sha256": sha256(data).hexdigest(),
                                     "bk4_member_offset": hex(offset)}
    if not results:
        cli.error("no BK4 members under 0110_backup")
    print(json.dumps([{"member_count": count, "example": examples[profile],
                       "keys": json.loads(profile)} for profile, count in results.items()], indent=2))


if __name__ == "__main__":
    main()
