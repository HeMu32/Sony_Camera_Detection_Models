"""Extract Sony NAB graphs by their on-disk slot IDs; do not scan for magic."""

import argparse
import hashlib
import json
from pathlib import Path
from struct import Struct, unpack_from

HEADER = Struct("<12I")
RECORD = Struct("<8I")


def extract(data):
    if len(data) < HEADER.size or data[:4] != b"NA\0\0":
        raise ValueError("not a Sony NAB header")
    _, _, version, count, _, _, directory, stride, table_end, _, pos, _ = (
        HEADER.unpack_from(data)
    )
    if (version != 1 or not count or stride != RECORD.size or
            not HEADER.size <= directory < table_end <= pos <= len(data) or
            directory + count * stride != table_end):
        raise ValueError("invalid NAB directory bounds")
    seen = set()
    for i in range(count):
        record_offset = directory + i * stride
        slot, input_bytes, output_bytes, size, prefix, *_ = RECORD.unpack_from(
            data, record_offset
        )
        start = pos + prefix
        end = start + size
        if (not slot or slot in seen or prefix != stride or size < 76 or
                end > len(data) or data[pos:start] != data[record_offset:record_offset + stride]
                or data[start:start + 4] != b"VPMN"):
            raise ValueError(f"invalid NAB graph record {i} at {pos:#x}")
        graph = data[start:end]
        version, target = unpack_from("<2I", graph, 4)
        yield {
            "slot": slot,
            "name": graph[12:76].split(b"\0", 1)[0].decode("ascii"),
            "start": start,
            "end": end,
            "size": size,
            "version": f"{version:#010x}",
            "target": f"{target:#x}",
            "input_allocation_bytes": input_bytes,
            "output_allocation_bytes": output_bytes,
            "sha256": hashlib.sha256(graph).hexdigest(),
        }, graph
        pos = end
        seen.add(slot)
    if pos != len(data):
        raise ValueError(f"unaccounted trailing data at {pos:#x}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("nab", type=Path)
    cli.add_argument("output_dir", type=Path)
    args = cli.parse_args()
    data = args.nab.read_bytes()
    graphs = list(extract(data))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for metadata, graph in graphs:
        filename = f"slot{metadata['slot']:02}.nb"
        (args.output_dir / filename).write_bytes(graph)
        rows.append(dict(metadata, filename=filename))
    manifest = {"source": str(args.nab), "sha256": hashlib.sha256(data).hexdigest(),
                "graphs": rows}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    print(f"Extracted {len(rows)} graphs into {args.output_dir}")


if __name__ == "__main__":
    main()
