"""Bind the original FDN A image-fetch request, without guessing pixel lanes."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from struct import Struct


IMAGE_PARAMS = Struct("<I12x7H")
IMAGE_WAIT = {"pc": 0x50002524, "reason": "wfi"}
ROW_PCS = (0x500024e0, 0x500024f2, 0x50002502, 0x50002512)
LOCAL_ADDRESS = 0x11000
LOCAL_PITCH_OPERAND = 112


@dataclass(frozen=True)
class ImageRead:
    window_xywh: tuple[int, int, int, int]
    source_xy: tuple[int, int]
    source_offset: int
    source_pitch_bytes: int
    source_row_bytes: int
    source_rows: int
    local_address: int
    local_pitch_operand: int
    descriptor_words: tuple[int, int, int, int]

    @property
    def compute_local_address(self) -> int:
        """First original computation's r5, distinct from the aligned DMA target."""
        return self.local_address + (self.window_xywh[0] - self.source_xy[0])

    def gather(self, frame: bytes) -> bytes:
        """Pack requested source rows tightly, preserving both bytes of every pixel."""
        start, pitch, width = self.source_offset, self.source_pitch_bytes, self.source_row_bytes
        last = start + (self.source_rows - 1) * pitch + width
        if len(frame) < last:
            raise ValueError("truncated packed image at requested source row")
        return b"".join(frame[start + y * pitch:start + y * pitch + width] for y in range(self.source_rows))


def bind(trace) -> ImageRead:
    """Validate the four emitted words against the normalized CP request."""
    if trace.get("setup_stop") != IMAGE_WAIT:
        raise ValueError("FDN A did not reach the image request wait")
    params = bytes.fromhex(trace["normalized_parameter_hex"])
    if len(params) != 0x28:
        raise ValueError("expected an original 40-byte FDN parameter block")
    base, width, height, pitch, x, y, window_w, window_h = IMAGE_PARAMS.unpack_from(params)
    if (base >> 30 != 2 or base & 31 or not 60 <= width <= 640 or width % 2 or
            not 48 <= height <= 480 or not width <= pitch <= 65532 or pitch % 4 or
            window_w != window_h or not 48 <= window_w <= 95 or
            x + window_w > width or y + window_h > height):
        raise ValueError("unsupported or out-of-frame FDN A image request")
    aligned_x = x & ~15
    fetch_pixels = min((x + window_w + 15) & ~15, pitch) - aligned_x
    source_pitch = pitch * 2
    source_offset = y * source_pitch + aligned_x * 2
    source_start = base + source_offset
    if source_start >> 30 != 2:
        raise ValueError("FDN A image request crosses the supported source aperture")

    # The first branch writes these words at +0x10..+0x1c. These are
    # encodings of scalar operands, not a claim about the DMA pixel transform.
    expected = (
        source_start & 0x3fffffff,
        ((LOCAL_PITCH_OPERAND >> 1) << 16) | LOCAL_ADDRESS,
        0xc0000000 | ((fetch_pixels * 2 - 1) << 16) | (source_pitch >> 3),
        0x40000000 | ((fetch_pixels // 2 - 1) << 16) | ((window_h - 1) << 8) | (window_h - 1),
    )
    writes = [e for e in trace["setup_scratch_writes"] if 0x10 <= e["address"] < 0x20]
    if ([(e["pc"], e["address"], e["size"]) for e in writes] !=
            [(pc, 0x10 + i * 4, 4) for i, pc in enumerate(ROW_PCS)] or
            tuple(e["value"] for e in writes) != expected):
        raise ValueError("image descriptor does not match the original aligned fetch")
    return ImageRead((x, y, window_w, window_h), (aligned_x, y), source_offset,
                     source_pitch, fetch_pixels * 2, window_h, LOCAL_ADDRESS, LOCAL_PITCH_OPERAND, expected)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path, help="JSON from probe_sa_fdn_parameters.py --trace-setup")
    cli.add_argument("--packed-frame", type=Path, help="Input rows in their existing two-byte pixel packing")
    cli.add_argument("--gathered-rows", type=Path, help="Write the original packed bytes, not the local post-DMA tensor")
    args = cli.parse_args()
    if (args.packed_frame is None) != (args.gathered_rows is None):
        cli.error("--packed-frame and --gathered-rows must be specified together")
    request = bind(json.loads(args.trace.read_text(encoding="utf-8-sig")))
    if args.packed_frame is not None:
        gathered = request.gather(args.packed_frame.read_bytes())
        with args.gathered_rows.open("xb") as stream:
            stream.write(gathered)
    print(json.dumps(asdict(request), indent=2))
