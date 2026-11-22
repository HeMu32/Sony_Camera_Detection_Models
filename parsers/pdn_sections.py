"""Verify observed PDN dictionary packing; do not infer an execution graph."""

import argparse
import hashlib
import json
from pathlib import Path
from struct import Struct


ROW3 = Struct("<4h")
ROW5 = Struct("<8h")
BOUNDS = Struct("<ii")
PADDED = (
    (0x0000, 0x0090, 3),
    (0x00A0, 0x0400, 3),
    (0x0410, 0x0AD0, 3),
    (0x0AE8, 0x0F68, 3),
    (0x11F0, 0x1670, 3),
    (0x18F8, 0x1D78, 3),
    (0x2708, 0x11708, 5),
)
BOUNDS_START = 0x11A20
BOUNDS_END = 0x11A88
DICTIONARY_LENGTH = 0x11D1C
SCALAR_HEAD_OFFSET = 0x11958
SCALAR_HEAD = Struct("<96h")


def segment(data, start, end, **fields):
    return {
        "start": start, "end_exclusive": end, "size": end - start,
        "sha256": hashlib.sha256(data[start:end]).hexdigest(), **fields,
    }


def scalar_head_dot(data, values):
    """Reproduce the original scalar PDN dot product, before bias/score conversion."""
    if len(data) != DICTIONARY_LENGTH or len(values) != 96 or any(not -32768 <= v <= 32767 for v in values):
        raise ValueError("expected a PDN dictionary and 96 signed halfwords")
    weights = SCALAR_HEAD.unpack_from(data, SCALAR_HEAD_OFFSET)
    bits = sum(w * v for w, v in zip(weights, values, strict=True)) & 0xffffffff
    return (bits ^ 0x80000000) - 0x80000000


def scalar_head_quantized(data, values):
    """Reproduce the scalar confidence head's integer input to FP32 conversion."""
    dot = scalar_head_dot(data, values)
    bias, = Struct("<h").unpack_from(data, 0x11a18)
    upper, lower = BOUNDS.unpack_from(data, 0x11a70)
    left, = Struct("<b").unpack_from(data, 0x11c4a)
    shift = data[0x11c4f]
    if not -31 <= left <= 0 or not 0 <= shift <= 31 or lower > upper:
        raise ValueError("invalid original PDN scalar quantization fields")
    bits = (dot + (bias << -left)) & 0xffffffff
    biased = (bits ^ 0x80000000) - 0x80000000
    low16 = (min(max(biased, lower), upper) >> shift) & 0xffff
    return (low16 ^ 0x8000) - 0x8000


def parse(data):
    if len(data) != DICTIONARY_LENGTH:
        raise ValueError(f"expected {DICTIONARY_LENGTH} bytes, got {len(data)}")
    blocks, pos = [], 0
    for start, end, used in PADDED:
        if start > pos:
            blocks.append(segment(data, pos, start, kind="packed_unknown"))
        fmt = ROW3 if used == 3 else ROW5
        rows = list(fmt.iter_unpack(data[start:end]))
        if any(any(row[used:]) for row in rows):
            raise ValueError(f"nonzero padding at {start:#x}")
        blocks.append(segment(data, start, end, kind="padded_signed_i16",
                              lanes=fmt.size // 2, used=used, rows=len(rows)))
        pos = end
    blocks.append(segment(data, pos, BOUNDS_START, kind="packed_unknown"))
    bounds = []
    for offset in range(BOUNDS_START, BOUNDS_END, BOUNDS.size):
        upper, lower = BOUNDS.unpack_from(data, offset)
        s = (-lower).bit_length() - 16
        if not 0 <= s <= 16 or upper != 32767 << s or lower != -32768 << s:
            raise ValueError(f"not a shifted i16 bound pair at {offset:#x}")
        bounds.append({"offset": offset, "upper": upper, "lower": lower,
                       "implied_shift": s})
    blocks.append(segment(data, BOUNDS_START, BOUNDS_END, kind="shifted_i16_bounds"))
    blocks.append(segment(data, BOUNDS_END, len(data), kind="footer_unknown"))
    if sum(block["size"] for block in blocks) != len(data):
        raise AssertionError("PDN ranges do not cover the file")
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "blocks": blocks, "bounds": bounds}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("files", type=Path, nargs="+")
    args = cli.parse_args()
    print(json.dumps({str(file): parse(file.read_bytes()) for file in args.files}, indent=2))
