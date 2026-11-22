"""Enumerate and extract exact members of a Sony SA SP bundle."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct


SP_HEADER = Struct("<2sHHHII")
RECORD_PREFIX = Struct("<IHH")
WORD = Struct("<I")
RECORD_MAGIC = 0x91130AAA


@dataclass(frozen=True)
class Member:
    job_id: int
    offset: int
    size: int
    header_job_id: int
    header_size: int
    name: str
    sha256: str


def members(data):
    if len(data) < SP_HEADER.size:
        raise ValueError("truncated SP header")
    magic, version, first, count, offsets, sizes = SP_HEADER.unpack_from(data)
    data_min = sizes + count * WORD.size
    if (magic != b"SP" or version != 1 or not count or
            offsets != SP_HEADER.size or sizes != offsets + count * WORD.size or
            data_min > len(data)):
        raise ValueError("unsupported SP index")
    records = {}
    for index in range(count):
        offset, = WORD.unpack_from(data, offsets + index * WORD.size)
        length, = WORD.unpack_from(data, sizes + index * WORD.size)
        if not offset and not length:
            continue
        if offset < data_min or length < 32 or offset + length > len(data):
            raise ValueError(f"invalid extent for job {first + index:#x}")
        payload = data[offset:offset + length]
        magic, record_version, header_size = RECORD_PREFIX.unpack_from(payload)
        if magic != RECORD_MAGIC or record_version != 7 or not 32 <= header_size <= length:
            raise ValueError(f"unsupported record at {offset:#x}")
        name = payload[20:32].split(b"\0", 1)[0].decode("ascii")
        records[first + index] = Member(
            first + index, offset, length, payload[9], header_size, name,
            sha256(payload).hexdigest(),
        )
    extents = sorted({(r.offset, r.offset + r.size) for r in records.values()})
    if any(end > next_start for (_, end), (next_start, _) in zip(extents, extents[1:])):
        raise ValueError("overlapping SA records")
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("--job", type=lambda value: int(value, 0))
    cli.add_argument("--extract", type=Path)
    args = cli.parse_args()
    if args.extract is not None and args.job is None:
        cli.error("--extract requires --job")
    data = args.bundle.read_bytes()
    records = members(data)
    if args.job is not None:
        if args.job not in records:
            cli.error(f"job {args.job:#x} not present")
        records = {args.job: records[args.job]}
    if args.extract is not None:
        member = records[args.job]
        with args.extract.open("xb") as file:
            file.write(data[member.offset:member.offset + member.size])
    print(json.dumps({
        "bundle": str(args.bundle),
        "sha256": sha256(data).hexdigest(),
        "members": [asdict(member) for member in records.values()],
    }, indent=2))


if __name__ == "__main__":
    main()
