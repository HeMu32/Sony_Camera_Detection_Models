"""Recover the fixed-point setup of the first FDN A workload, before sampling."""

from struct import Struct

from .sony_sa_program_planes import startup_rows


U32 = Struct("<I")
DESCRIPTOR = Struct("<4I")
SIGNED_WORD = Struct("<i")
BLOCK = 0x100
LOCAL_TABLE = 0x17d00
APERTURE = 0x40000
SETUP_WORDS = 20
NODE_INDEX = (8, 6, 10, 8, 10, 6, 9, 7, 11, 8, 10, 6, 9, 7, 11)
OPTION_INDEX_DELTA = (0, -6, 6)


def table_for_fdn_a(member):
    def field(offset):
        return U32.unpack_from(member, BLOCK + offset)[0]

    offset = BLOCK + field(0x20) + 10 * DESCRIPTOR.size
    src, dst, w2, w3 = DESCRIPTOR.unpack_from(member, offset)
    size = (w2 >> 16) + 1
    if (dst, size, w2 & 0xffff, w3) != (LOCAL_TABLE, 0x120, 0x4000, w2 & 0xffff0000):
        raise ValueError("expected the original FDN A coefficient table descriptor")
    src += BLOCK + field(0x10)
    if src + size > len(member):
        raise ValueError("truncated FDN table")
    return member[src:src + size]


def table_for_pdn(member):
    src, dst, w2, w3 = next(startup_rows(member), (0, 0, 0, 0))
    if (dst, w2, w3) != (0x100, 0x011f4000, 0x011f0000) or src + 0x120 > len(member):
        raise ValueError("expected the original PDN coefficient table descriptor")
    return member[src:src + 0x120]


def setup_words(table, edge, node, option=0):
    """Return the 20 words for the FDN A square-window setup."""
    if (len(table) != 0x120 or not 48 <= edge <= 95 or
            node not in range(15) or option not in range(3)):
        raise ValueError("expected an A table, square edge 48..95, node 0..14, and option 0..2")

    # FDN dispatch remaps (option, node) before four loads at 0x50001d5e..0x50001d84.
    index = NODE_INDEX[node] + OPTION_INDEX_DELTA[option]
    factors = matrix_for_index(table, index)
    step_q12 = ((edge << 12) + 24) // 48
    matrix_q24 = tuple((value * step_q12) & 0xffffffff for value in factors)
    center_q24, limit_q24 = edge << 23, ((edge - 1) << 24) - 1
    return (8, 48, center_q24, center_q24, 24, 24, *matrix_q24,
            limit_q24, limit_q24, 24, edge - 1, edge - 1, 12, 0x1000, 24, 0, 0x1200)


def matrix_for_index(table, index):
    if len(table) != 0x120 or not 0 <= index < 18:
        raise ValueError("expected an FDN table and an index 0..17")
    return tuple(SIGNED_WORD.unpack_from(table, i * 0x48 + 4 * index)[0] for i in range(4))


def pdn_geometry_words(table, matrix_index, extents, *, divisors=(22, 22)):
    """Reproduce the first 19 integer aperture writes of the related PDN stage."""
    x, y = extents
    width, height = divisors
    if any(not 1 <= v <= 128 for v in (x, y, width, height)):
        raise ValueError("expected positive PDN geometry no greater than 128")
    step_x = ((x << 12) + width // 2) // width
    step_y = ((y << 12) + height // 2) // height
    a, b, c, d = matrix_for_index(table, matrix_index)
    matrix = tuple(v & 0xffffffff for v in (a * step_x, b * step_x, c * step_y, d * step_y))
    limit_x = (((x - 1) << 24) - 1) & 0xffffffff
    limit_y = (((y - 1) << 24) - 1) & 0xffffffff
    return (4, width, x << 23, y << 23, width // 2, height // 2, *matrix, limit_x, limit_y,
            24, x - 1, y - 1, 12, 0x1000, 17, 0)


def i32(value):
    return ((value + 0x80000000) & 0xffffffff) - 0x80000000


def clamp(value, high):
    return min(max(value, 0), high)


def project_b_landmarks(table, frame, window, points, matrix_group, route_mode):
    """Apply the B result writer's integer coordinate projection, without scoring."""
    width, height = frame
    x0, y0, edge = window
    if (not 1 <= width <= 8192 or not 1 <= height <= 8192 or
            not 48 <= edge <= 95 or not 0 <= matrix_group < 9 or len(points) != 3 or
            not 0 <= x0 <= width - edge or not 0 <= y0 <= height - edge):
        raise ValueError("expected a valid FDN B square-window result geometry")
    mirror = route_mode in (3, 4)
    a, b, c, d = matrix_for_index(table, 2 * matrix_group + mirror)

    def project_axis(v, origin, size):
        relative = clamp(i32(v * edge + (edge << 23)), (edge - 1) << 24)
        return clamp((origin << 18) + (relative >> 6), (size - 1) << 18)

    out = []
    for x, y in points:
        u, v = i32(x - 0x100000) >> 9, i32(y - 0x100000) >> 9
        out.append((project_axis(a * u + b * v, x0, width),
                    project_axis(c * u + d * v, y0, height)))
    if mirror:
        out[0], out[1] = out[1], out[0]
    return tuple(out)
