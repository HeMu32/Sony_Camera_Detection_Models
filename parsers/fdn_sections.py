"""Identify padded 16-bit FDN blocks without assigning unproven layer semantics."""

import argparse
import hashlib
import json
from pathlib import Path
from struct import Struct

HEADER = Struct("<13I")
CLASSIFIER = Struct("<2I")


def read_coefficients(data, pos):
    blocks = []
    while pos + HEADER.size <= len(data):
        fields = HEADER.unpack_from(data, pos)
        w, h, group, kernel, padded = fields[:5]
        if (not 0 < w <= 256 or not 0 < h <= 256 or group not in (1, 6, 12) or
                w % group or (kernel, padded) not in ((1, 1), (3, 4), (5, 6)) or
                fields[9:] != (h, w // group, kernel, padded)):
            break
        start = pos + HEADER.size
        coeff_bytes = (w // group) * h * kernel * padded * 2
        bias_start = start + coeff_bytes
        end = bias_start + h * 4
        if end > len(data):
            raise ValueError(f"invalid FDN payload size at {pos:#x}")
        lanes = Struct("<" + "h" * padded)
        if any(any(row[kernel:]) for row in lanes.iter_unpack(data[start:bias_start])):
            raise ValueError(f"nonzero padding lane in block at {pos:#x}")
        blocks.append({
            "offset": pos, "fields": fields,
            "coefficient_shape_padded": [h, w // group, kernel, padded],
            "coefficient_offset": start, "coefficient_bytes": coeff_bytes,
            "bias_offset": bias_start, "bias_i32_count": h,
            "sha256": hashlib.sha256(data[start:end]).hexdigest(),
        })
        pos = end
    return pos, blocks


def read_dense(data, pos):
    heads = []
    while pos + CLASSIFIER.size <= len(data):
        inputs, lanes = CLASSIFIER.unpack_from(data, pos)
        if (inputs, lanes) != (144, 1):
            break
        start = pos + CLASSIFIER.size
        bias_start = start + inputs * 2
        end = bias_start + 4
        if end > len(data):
            raise ValueError(f"truncated dense block at {pos:#x}")
        heads.append({"offset": pos, "coefficient_offset": start,
                      "coefficient_i16_count": inputs, "bias_offset": bias_start})
        pos = end
    return pos, heads


def parse(data):
    pos, blocks = read_coefficients(data, 0)
    pos, heads = read_dense(data, pos)
    extra_blocks, extra_heads = [], []
    separator = Struct("<12I")
    separator_values = None
    if pos + separator.size <= len(data) and separator.unpack_from(data, pos) == (2,) * 12:
        separator_values = {"offset": pos, "values": separator.unpack_from(data, pos)}
        pos, extra_blocks = read_coefficients(data, pos + separator.size)
        pos, extra_heads = read_dense(data, pos)
    return {
        "blocks": blocks,
        "dense_candidates_144x1": heads,
        "separator": separator_values,
        "extra_blocks": extra_blocks,
        "extra_dense_candidates_144x1": extra_heads,
        "remaining_offset": pos,
        "remaining_bytes": len(data) - pos,
        "remaining_sha256": hashlib.sha256(data[pos:]).hexdigest(),
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("files", nargs="+", type=Path)
    args = cli.parse_args()
    print(json.dumps({str(path): parse(path.read_bytes()) for path in args.files}, indent=2))
