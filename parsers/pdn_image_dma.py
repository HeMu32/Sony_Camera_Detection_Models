"""Validate original PDN image reads and replay its local edge-copy plan."""

from dataclasses import dataclass


LOCAL_BASE = 0x16000
LOCAL_PITCH = 64
IMAGE_SUBMIT = 0x500017e4
LEFT_SUBMIT, RIGHT_SUBMIT = 0x500011f0, 0x5000102e
TOP_SUBMIT, BOTTOM_SUBMIT = 0x50001158, 0x500010c4


@dataclass(frozen=True)
class EdgeCopy:
    source: int
    target: int
    count: int
    stride: int


@dataclass(frozen=True)
class ImageRead:
    window_xywh: tuple[int, int, int, int]
    source_offset: int
    source_pitch_bytes: int
    source_row_bytes: int
    source_rows: int
    local_origin: int
    compute_address: int
    edge_copies: tuple[EdgeCopy, ...]

    def gather(self, frame: bytes) -> bytes:
        """Read exact source rows without selecting or rearranging byte lanes."""
        start, pitch, width = self.source_offset, self.source_pitch_bytes, self.source_row_bytes
        if len(frame) < start + (self.source_rows - 1) * pitch + width:
            raise ValueError("truncated packed source frame")
        return b"".join(frame[start + i * pitch:start + i * pitch + width]
                        for i in range(self.source_rows))

    def crop_from_local_rows(self, data: bytes) -> bytes:
        """Apply edge copies to supplied post-transfer bytes; do not model lane selection."""
        width = self.source_row_bytes // 2
        if len(data) != width * self.source_rows:
            raise ValueError("expected tightly packed one-byte post-transfer rows")
        scratch = {}
        for row in range(self.source_rows):
            base = self.local_origin + row * LOCAL_PITCH
            scratch.update(enumerate(data[row * width:(row + 1) * width], base))
        try:
            for copy in self.edge_copies:
                values = [scratch[copy.source + copy.stride * i] for i in range(copy.count)]
                scratch.update((copy.target + copy.stride * i, value) for i, value in enumerate(values))
            _, _, width, height = self.window_xywh
            return bytes(scratch[self.compute_address + row * LOCAL_PITCH + col]
                         for row in range(height) for col in range(width))
        except KeyError as exc:
            raise ValueError(f"PDN local crop reads unknown byte {exc.args[0]:#x}") from exc


def bind(geometry) -> ImageRead:
    """Compare original DMA rows with an independently derived clipped crop plan."""
    base, frame_w, frame_h, pitch = geometry["frame"]
    x, y, width, height = geometry["window_xywh"]
    if (base >> 30 != 2 or base & 31 or not 60 <= frame_w <= 640 or frame_w % 2 or
            not 48 <= frame_h <= 480 or not 2 * frame_w <= pitch <= 131064 or pitch & 7 or
            not 22 <= width <= 44 or width != height or
            x >= frame_w or y >= frame_h or x + width <= 0 or y + height <= 0):
        raise ValueError("unsupported PDN packed image geometry")

    sx, sy = max(x, 0) & ~15, max(y, 0)
    end = min((x + width + 15) & ~15, pitch // 2)
    pixels = end - sx
    rows = min(frame_h, y + height) - sy
    top, left = max(-y, 0), (max(-x, 0) + 7) & ~7
    origin = LOCAL_BASE + top * LOCAL_PITCH + left
    compute_x = x + left if x < 0 else x - sx
    compute = LOCAL_BASE + compute_x
    source_offset = sy * pitch + 2 * sx
    source_start = base + source_offset
    if (source_start >> 30 != 2 or not 0 < pixels <= LOCAL_PITCH - left or
            not 0 <= compute_x <= LOCAL_PITCH - width):
        raise ValueError("PDN image read falls outside its modeled row")

    repeated_rows = (rows - 1) * 0x101
    expected_image = [
        [0, 0x4c000000, 0, 0],
        [source_start & 0x3fffffff, ((LOCAL_PITCH // 2) << 16) | origin,
         0xc0000000 | ((2 * pixels - 1) << 16) | (pitch >> 3),
         0x40000000 | ((pixels // 2 - 1) << 16) | repeated_rows],
    ]
    image_request = geometry["image_request"]
    if image_request["submit_pc"] != IMAGE_SUBMIT or image_request["rows"] != expected_image:
        raise ValueError("PDN image DMA descriptor does not match the clipped source")
    if geometry["compute"]["registers_r4_to_r7"][1:3] != [compute, LOCAL_PITCH]:
        raise ValueError("PDN local crop origin does not match the original compute arguments")

    copies, expected = [], []

    def columns(submit, source, targets):
        for target in targets:
            copies.append(EdgeCopy(source, target, rows, LOCAL_PITCH))
            expected.append((submit, [[target, 0xc0400000 | source, LOCAL_PITCH >> 3, repeated_rows]]))

    def lines(submit, source_row, targets):
        source, size = compute + source_row * LOCAL_PITCH, (width - 1) << 16
        for row in targets:
            target = compute + row * LOCAL_PITCH
            copies.append(EdgeCopy(source, target, width, 1))
            expected.append((submit, [[target, 0xc0000000 | source, size, size]]))

    local_row = LOCAL_BASE + top * LOCAL_PITCH
    if left:
        columns(LEFT_SUBMIT, origin, range(local_row, origin))
    if x + width > frame_w:
        first = local_row + compute_x + frame_w - x
        columns(RIGHT_SUBMIT, first - 1, range(first, local_row + compute_x + width))
    lines(TOP_SUBMIT, top, range(top))
    lines(BOTTOM_SUBMIT, top + rows - 1, range(top + rows, height))
    actual = [(request["submit_pc"], request["rows"]) for request in geometry["edge_requests"]]
    if actual != expected:
        raise ValueError("PDN edge DMA rows differ from the modeled copy plan")

    return ImageRead((x, y, width, height), source_offset, pitch, 2 * pixels, rows,
                     origin, compute, tuple(copies))
