"""Convert XHD result records at the native CP result-buffer boundary."""

from dataclasses import dataclass
from struct import Struct


COUNT = Struct("<I")
RAW = Struct("<4hBBH")
COUNT_OFFSET = 0x520
RAW_START = COUNT_OFFSET + COUNT.size
MAX_RECORDS = 50
ROTATION = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class Geometry:
    frame_size: tuple[int, int]
    search_size: tuple[int, int]
    center: tuple[int, int]
    angle: int = 0
    min_side_raw: int = 0

    def validate(self):
        if (len(self.frame_size) != 2 or len(self.search_size) != 2 or len(self.center) != 2
                or not all(1 <= n <= 8192 for n in (*self.frame_size, *self.search_size))
                or not all(0 <= n <= 8192 for n in self.center)
                or self.angle not in (0, 90, 180, 270) or not 0 <= self.min_side_raw <= 255):
            raise ValueError("unsupported XHD result geometry")


@dataclass(frozen=True)
class XhdResult:
    box: tuple[int, int, int, int]
    confidence: float
    type_code: int
    aux_code: int


def raw_count(payload):
    if len(payload) < RAW_START:
        raise ValueError("truncated XHD result header")
    return COUNT.unpack_from(payload, COUNT_OFFSET)[0]


def decode(payload, geometry):
    geometry.validate()
    count = raw_count(payload)
    if count > MAX_RECORDS:
        raise ValueError("XHD record count exceeds firmware limit")
    end = RAW_START + RAW.size * count
    if len(payload) < end:
        raise ValueError("truncated XHD results")
    width, height = geometry.search_size
    cx, cy = geometry.center
    cos, sin = ROTATION[(-geometry.angle % 360) // 90]

    def point(x, y):
        x = ((x * width + 512) >> 14) - width // 2
        y = ((y * height + 512) >> 14) - height // 2
        return cos * x - sin * y + cx, sin * x + cos * y + cy

    def clamp(value, frame_extent):
        return min(max(value, 0), frame_extent)

    items = []
    fw, fh = geometry.frame_size
    min_side = width * geometry.min_side_raw // 320
    for x0, y0, x1, y1, kind, aux, score in RAW.iter_unpack(payload[RAW_START:end]):
        ax, ay = point(x0, y0)
        bx, by = point(x1, y1)
        x0, x1 = (clamp(x, fw) for x in sorted((ax, bx)))
        y0, y1 = (clamp(y, fh) for y in sorted((ay, by)))
        confidence = score / 32768 if min(x1 - x0, y1 - y0) >= min_side else 0.0
        items.append(XhdResult((x0, y0, x1, y1), confidence, min(kind, 8), aux))
    return items
