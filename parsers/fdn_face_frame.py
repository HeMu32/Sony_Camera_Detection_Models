"""Recover FC face rectangle operations, separately from model inference."""


SAME_FACE_Q16 = 0x8ccd
LOW_SOURCE_WEIGHT_Q8 = 0xb4


def check_geometry(xywh, frame, transform, minimum):
    x, y, w, h = xywh
    width, height = frame
    if (any(not 16 <= n <= 4096 for n in (*frame, *transform)) or
            not 0 <= minimum <= min(255, width - 1, height - 1) or
            not 0 <= x < x + w <= width or not 0 <= y < y + h <= height):
        raise ValueError("expected a positive in-frame FC candidate and valid projection geometry")


def project(xywh, frame, *, transform=None, square=True, minimum=0):
    """Return the original display rectangle as pixel-edge, half-open xyxy."""
    transform = frame if transform is None else transform
    check_geometry(xywh, frame, transform, minimum)

    def endpoints(origin, length, extent, calibration_extent):
        upper = origin + length - 1
        first, second = ((upper, origin) if origin + length // 2 < extent // 2 else (origin, upper))
        forward = (calibration_extent << 16) // extent
        inverse = (extent << 16) // calibration_extent

        def convert(coordinate):
            mapped = (coordinate * forward + 0x8000) >> 16
            if mapped >= calibration_extent:
                raise ValueError("FC calibration input would reject the scaled coordinate")
            pixel = (mapped * inverse + 0x8000) >> 16
            return min(pixel, extent - 1)

        return convert(first), convert(second)

    x, y, w, h = xywh
    xa, xb = endpoints(x, w, frame[0], transform[0])
    ya, yb = endpoints(y, h, frame[1], transform[1])
    w, h = abs(xa - xb) + 1, abs(ya - yb) + 1
    if square:
        w = h = min(w, h)
    w, h = max(w, minimum), max(h, minimum)
    x = (xb - w + 1 if xa < xb else xb) & 0xffff
    y = (yb - h + 1 if ya < yb else yb) & 0xffff
    if x + w >= frame[0]:
        x = frame[0] - 1 - w
    if y + h >= frame[1]:
        y = frame[1] - 1 - h
    if x < 0 or y < 0:
        raise ValueError("FC rectangle would wrap when clamped to the frame")
    return x, y, x + w, y + h


def merge_pair(first, second, *, weights, sources):
    """Apply the released same-face gate and fuse one qualifying pair (xywh)."""
    for rect in (first, second):
        check_geometry(rect, (4096, 4096), (4096, 4096), 0)
    if (len(weights) != 2 or len(sources) != 2 or
            any(not 0 <= v <= 0xffff for v in weights) or any(not 0 <= v <= 255 for v in sources)):
        raise ValueError("expected two unsigned FC weights and source bytes")
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - overlap
    mask = 0xffffffff
    if not overlap or ((overlap << 16) & mask) < ((union * SAME_FACE_Q16) & mask):
        return None
    a, b = (v * LOW_SOURCE_WEIGHT_Q8 >> 8 if source <= 2 else v
            for v, source in zip(weights, sources, strict=True))
    if a == b == 0:
        a = b = 1
    return tuple((x * a + y * b) // (a + b) for x, y in zip(first, second, strict=True))
