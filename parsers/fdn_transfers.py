"""Map original FDN dictionary DMA requests to byte spans, not model layers."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .fdn_sections import parse


def dictionary_spans(data):
    layout = parse(data)
    if layout["remaining_bytes"]:
        raise ValueError("unparsed FDN dictionary bytes")
    for group in ("blocks", "dense_candidates_144x1", "extra_blocks",
                  "extra_dense_candidates_144x1"):
        for index, block in enumerate(layout[group]):
            header, coefficients, bias = (block[field] for field in
                                          ("offset", "coefficient_offset", "bias_offset"))
            end = bias + 4 * block.get("bias_i32_count", 1)
            yield group, index, "header", header, coefficients
            yield group, index, "coefficients", coefficients, bias
            yield group, index, "bias", bias, end


def project(data, requests):
    """Resolve requests, and reconstruct disjoint destinations as an offline fixture."""
    spans = list(dictionary_spans(data))
    coverage = Counter()
    mapped = []
    payload = bytearray()
    for index, request in enumerate(requests):
        offset, destination, size = (request[key] for key in ("offset", "destination", "size"))
        if size <= 0 or destination < 0:
            raise ValueError(f"invalid extent in request {index}")
        matches = [(group, block, field, start, end) for group, block, field, start, end in spans
                   if start <= offset and offset + size <= end]
        if len(matches) != 1:
            raise ValueError(f"request {index} does not fit exactly one dictionary span")
        group, block, field, start, _ = matches[0]
        coverage.update(range(offset, offset + size))
        payload += data[offset:offset + size]
        mapped.append({**request, "index": index, "group": group, "block": block,
                       "field": field, "field_offset": offset - start})

    regions = []
    for row in sorted(mapped, key=lambda item: item["destination"]):
        addr, size, offset = (row[key] for key in ("destination", "size", "offset"))
        if not regions or addr > regions[-1][0] + len(regions[-1][1]):
            regions.append((addr, bytearray()))
        base, contents = regions[-1]
        if addr != base + len(contents):
            raise ValueError(f"destination overlap at {addr:#x}; order-dependent writes")
        contents.extend(data[offset:offset + size])

    report_blocks = defaultdict(dict)
    for group, block, field, start, end in spans:
        hits = sum(i in coverage for i in range(start, end))
        report_blocks[group, block][field] = {"covered": hits, "size": end - start}
    return {
        "requests": mapped,
        "covered_source_bytes": len(coverage),
        "repeated_source_bytes": sum(count - 1 for count in coverage.values()),
        "sha256_in_request_order": sha256(payload).hexdigest(),
        "blocks": [{"group": group, "block": block, **fields}
                   for (group, block), fields in report_blocks.items()],
        "destination_regions": [{"address": start, "size": len(bytes_),
                                 "sha256": sha256(bytes_).hexdigest()}
                                for start, bytes_ in regions],
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path, help="JSON from probe_sa_fdn_parameters.py --dict-dir")
    cli.add_argument("dict_dir", type=Path)
    args = cli.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8-sig"))["dictionary_fixture"]
    dictionary = trace["dictionary"]
    if dictionary not in ("front", "side"):
        cli.error("unexpected dictionary name")
    data = (args.dict_dir / f"fdn_{dictionary}_dict.bin").read_bytes()
    result = project(data, trace["requests"])
    if result["sha256_in_request_order"] != trace["sha256_in_request_order"]:
        cli.error("dictionary bytes do not match the recorded requests")
    print(json.dumps({"dictionary": dictionary, **result}, indent=2))
