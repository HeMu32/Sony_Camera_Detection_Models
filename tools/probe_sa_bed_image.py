"""Replay the original CP-to-BED path up to the first unknown SA opcode."""

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import iter_unpack, pack, pack_into, unpack_from
import tarfile

from parsers.pdn_image_dma import EdgeCopy
from parsers.sony_sa_post import members
from parsers.sony_sa_program_planes import boot_programs, join_record, plane_slots, program, startup_rows
from probe_a7iv_keypoint_cp_request import probe_request
from probe_a7iv_keypoint_cp_result import (
    BACKUP_MEMBER, CANDIDATE_BYTES, probe_keypoint, published_config,
)
from probe_sa_fdn_boot import DmaTraceMachine
from probe_sa_fdn_dispatch import pc_literal
from sa_local_dma import copy_local_rows


JOB = 0x44
BED_SHA = "e616eccded9f54397a3278a6c2d34414ce1ee77efcce18af54fc1744bc05968c"
BED_DICT1_SHA = "c4a796242e32d841ee7488b2209b9424a101c9c1bc3f65af542d4b2951bbf033"
BED_DICT2_SHA = "5d20069c26594375989d241d3bedb35eab1d989b55965fdf2548cd8e1750c3bb"
WORK_BYTES = 0x98
WORK_WAIT, IMAGE_WAIT = 0xd68, 0xf20
VALIDATOR_RETURN = 0xcd6
COMPUTE = {"pc": 0x8c0, "reason": "unsupported opcode", "opcode": 0xa}
EDGE_SUBMITS = (0x108e, 0x1124, 0x11b8, 0x1250)
STACK = 0x3800
PARAMETER_ALIAS = 0xc0000000
FRAME_BASE = 0x500030000
FRAMES = ((640, 480, 640), (320, 240, 320), (160, 120, 192), (80, 60, 128))
WORKER_BASES = (0x4000, 0x4800, 0x5000, 0x8000, 0x8800)
MAILBOX_START, MAILBOX_END = 0x17800, 0x1797c
BLOCK_VA = 0x50010000
COMMAND_MARKER = pack("<4I", 0, 0x01000000, 0x01000000, 0x01000000)


def make_candidate(center, sides, exec_id=0, roll=0):
    data = bytearray(CANDIDATE_BYTES)
    for i, geom in enumerate(FRAMES):
        pack_into("<Q3H", data, 8 + 0x10 * i, FRAME_BASE + 0x100000 * i, *geom)
    pack_into("<4HhHI", data, 0x48, *center, *sides, roll, 0, exec_id)
    pack_into("<I", data, 0x58, 4)
    pack_into("<I", data, 0x60, 1)
    return data


@dataclass(frozen=True)
class PackedRead:
    offset: int
    pitch: int
    row_bytes: int
    rows: int

    def gather(self, frame):
        """Gather both lanes in their original byte order from the supplied frame."""
        if len(frame) < self.offset + (self.rows - 1) * self.pitch + self.row_bytes:
            raise ValueError("packed BED source frame is truncated")
        return b"".join(frame[i:i + self.row_bytes] for i in
                        range(self.offset, self.offset + self.rows * self.pitch, self.pitch))


def bind_packed_read(parameter, request):
    base, = unpack_from("<I", parameter, 4)
    width, height, pitch = unpack_from("<3H", parameter, 0x14)
    prefix, (source, target, a, b) = request["rows"]
    offset = source - (0x20000000 | (base & 0x1fffffff))
    byte_pitch, row_bytes, rows = 2 * pitch, 1 + (a >> 16 & 0x3fff), 1 + (b & 0xff)
    if not byte_pitch:
        raise ValueError("BED image DMA requires a nonzero source pitch")
    y, x_bytes = divmod(offset, byte_pitch)
    if (request["submit_pc"] != 0xf18 or prefix != [0, 0x4c000000, 0, 0]
            or target >> 16 != 0x60 or (target & 0xffff) < 0x9000 or a >> 30 != 3 or b >> 30 != 1
            or byte_pitch != (a & 0xffff) * 8 or row_bytes % 4 or b >> 16 & 0x3fff != row_bytes // 4 - 1
            or (b >> 8 & 0xff) != rows - 1 or offset < 0 or x_bytes % 32
            or width > pitch or x_bytes + row_bytes > byte_pitch or y + rows > height):
        raise ValueError("BED image DMA row is outside the supported packed geometry")
    return PackedRead(offset, byte_pitch, row_bytes, rows)


@dataclass(frozen=True)
class BedLocalImage:
    window_xywh: tuple[int, int, int, int]
    packed_read: PackedRead
    origin: int
    address: int
    stride: int
    copies: tuple[EdgeCopy, ...]

    def from_local_rows(self, raster):
        """Replay edge DMA using explicit post-transfer bytes, not packed source lanes."""
        _, _, width, height = self.window_xywh
        present, rows = self.packed_read.row_bytes // 2, self.packed_read.rows
        if len(raster) != present * rows:
            raise ValueError("expected exact post-transfer bytes for BED local rows")
        data = {}
        for i in range(rows):
            data.update(enumerate(raster[i * present:(i + 1) * present], self.origin + i * self.stride))
        try:
            for copy in self.copies:
                values = [data[copy.source + i * copy.stride] for i in range(copy.count)]
                data.update((copy.target + i * copy.stride, value) for i, value in enumerate(values))
            return bytes(data[self.address + i * self.stride + j] for i in range(height) for j in range(width))
        except KeyError as exc:
            raise ValueError(f"BED local crop depends on unknown byte {exc.args[0]:#x}") from exc


def bind_bed_local_image(parameter, packed, submissions, address, stride):
    """Check the firmware geometry and every edge descriptor against a clipped window."""
    width, height, pitch, cx, cy = unpack_from("<5H", parameter, 0x14)
    size, base = 171, 0x9000
    x, y = cx - size // 2, cy - size // 2
    sx, sy = max(x, 0) & ~15, max(y, 0)
    end = min((x + size + 15) & ~15, pitch)
    rows = min(y + size, height) - sy
    top, left = max(-y, 0), (max(-x, 0) + 7) & ~7
    local_row = base + top * stride
    origin = local_row + left
    compute_x = x + left if x < 0 else x - sx
    fetch, *edges = submissions
    if (stride != 192 or address != base + compute_x or not 0 <= compute_x <= stride - size
            or not 0 < end - sx <= stride - left or not 0 < rows <= size
            or fetch["rows"][1][1] != (((stride // 2) << 16) | origin)
            or packed != PackedRead(2 * (sy * pitch + sx), 2 * pitch, 2 * (end - sx), rows)):
        raise ValueError("BED local crop disagrees with the original image descriptor")

    expected, copies = [], []

    def columns(submit, source, targets):
        for target in targets:
            expected.append((submit, [[target, 0xc0c00000 | source, stride >> 3, (rows - 1) * 0x101]]))
            copies.append(EdgeCopy(source, target, rows, stride))

    def lines(submit, source, targets):
        word = (size - 1) << 16
        for target in targets:
            expected.append((submit, [[target, 0xc0000000 | source, word, word]]))
            copies.append(EdgeCopy(source, target, size, 1))

    columns(0x1250, origin, range(local_row, origin))
    if x + size > width:
        first = local_row + compute_x + width - x
        columns(0x108e, first - 1, range(first, local_row + compute_x + size))
    lines(0x11b8, address + top * stride, range(address, address + top * stride, stride))
    stop = address + (top + rows) * stride
    lines(0x1124, stop - stride, range(stop, address + size * stride, stride))
    if [(d["submit_pc"], d["rows"]) for d in edges] != expected:
        raise ValueError("BED edge descriptors differ from the modeled local copies")
    return BedLocalImage((x, y, size, size), packed, origin, address, stride, tuple(copies))


def shared_image_entry(bundle, record):
    """Bind the initial BED record to the corresponding boot slots of three detectors."""
    members_by_job = members(bundle)
    reference = None
    result = {}
    for job in (JOB, 0x3e, 0x43):
        member = members_by_job[job]
        body = bundle[member.offset:member.offset + member.size]
        start, *_ = next(startup_rows(body))
        offset = start + 0x1c
        windows = boot_programs(body)
        main = windows[0x04000000]
        if (set(windows) != {0x04000000 + i * 0x1000 for i in range(6)}
                or any(image != main for image in windows.values())
                or body[offset:offset + len(record)] != record):
            raise ValueError(f"unexpected {member.name} entry program binding")
        slots = tuple(main.at(i) for i in range(37))
        if reference is None:
            reference = slots
        if slots != reference:
            raise ValueError(f"{member.name} does not share the BED entry slots")
        result[job] = {"record_offset": offset, "program_offset": main.offset, "slots": main.slots}
    tags = tuple(w >> 24 for w, in iter_unpack("<I", record))
    if set(tags) != set(range(37)):
        raise ValueError("BED entry record does not address exactly the initial 37 slots")
    return {"jobs": result, "tags": tags, "sha256": sha256(record).hexdigest()}


def first_stage_workers(controller, regions, record):
    """Wake each original worker using only the main controller's mailbox bytes."""
    jobs = []
    for base in WORKER_BASES:
        image = dict(regions)[base]
        worker = DmaTraceMachine(image, scratch_base=MAILBOX_START,
                                 scratch_size=MAILBOX_END - MAILBOX_START)
        worker.pc, worker.regions = base, regions
        worker.stack_high = pc_literal(
            lambda address, size: image[address - base:address - base + size], base + 0x10, 15)
        worker.stack_low = base + len(image)
        worker.stack = bytearray(worker.stack_high - worker.stack_low)
        worker.scratch = {a: v for a, v in controller.scratch.items() if MAILBOX_START <= a < MAILBOX_END}
        wait = worker.run(512)
        if wait != {"pc": base + 0x128, "reason": "wfi"}:
            raise ValueError(f"unexpected BED worker wait: {wait}")
        worker.pc += 2
        stop = worker.run(512)
        if stop != {"pc": base + 0xd8, "reason": "unsupported opcode", "opcode": 0xa}:
            raise ValueError(f"unexpected first BED worker compute: {stop}")
        pointer, *args = worker.reg[4:8]
        if image[pointer - base:worker.pc - base] != record:
            raise ValueError("BED worker command differs from the controller's")
        values = {w["address"]: w["value"] for w in worker.mmio_writes}
        jobs.append({
            "base": base, "wait": wait, "stop": stop, "record": pointer, "arguments": args,
            "partition": (values[0x40038], values[0x40000]),
            "mailbox_reads": tuple(r["address"] for r in worker.scratch_reads),
        })
    return jobs


def trace_handoff(controller, bed, dictionary=None, dictionary2=None):
    """Observe completion and the next block in isolated slices, not model output."""
    m = deepcopy(controller)
    m.pc = m.link  # Explicit synthetic return from the still unknown compute.
    begin = len(m.mmio_writes)
    wait = m.run(512)
    if wait != {"pc": 0x13fc, "reason": "wfi"}:
        raise ValueError(f"unexpected BED first-stage barrier: {wait}")
    writes = tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:])
    m.pc += 2
    resumed = m.run(512, stop_at=0xe06)
    if resumed != {"pc": 0xe06, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED completion dispatch: {resumed}")

    status_paths = {}
    for status in (0, 1, 2):
        test = deepcopy(m)
        test.fixture_reads[0x481cc] = deque((status,))
        begin = len(test.scratch_trace)
        stop = test.run(512, stop_at=0xe58)
        status_paths[status] = {"stop": stop, "work_writes": test.scratch_trace[begin:]}

    # Re-enter after the range sweep at 0xe20..0xe26; no feature values are supplied.
    directory = 0x100 + unpack_from("<I", bed, 0x11c)[0]
    offsets = unpack_from("<9I", bed, directory)
    member_offsets = tuple(0x100 + x for x in offsets)
    if tuple(bed[o + 9] for o in member_offsets) != tuple(range(len(offsets))):
        raise ValueError("BED block directory and header indices disagree")
    target = BLOCK_VA + offsets[1]
    m.pc = 0xe28
    m.regions += ((BLOCK_VA, bed[0x100:]),)
    m.fixture_reads[0x48160] = deque((BLOCK_VA,))
    routed = m.run(32, stop_at=target)
    if routed != {"pc": target, "reason": "slice boundary"} or m.link != 0xe36:
        raise ValueError(f"unexpected BED directory dispatch: {routed}")
    boot_boundary = m.run(128)
    return {"wait": wait, "mmio_writes": writes, "status_paths": status_paths,
            "directory": directory, "member_offsets": member_offsets,
            "branch_target": target, "boot_boundary": boot_boundary,
            "boot_fixture": next_block_boot(m, bed, member_offsets[1], dictionary, dictionary2)}


def next_block_boot(m, bed, block, dictionary, dictionary2):
    boot = boot_directory_block(m, bed, block)
    external_stop = boot["external_stop"]
    mapped = deepcopy(m)
    cp_word = mapped.read(PARAMETER_ALIAS + 8, 4)
    # Explicit table entry for the weight resource only. It is not a camera RAM dump.
    weight_bytes = (cp_word & 0x3fffffff) << 2
    row = (1, weight_bytes, weight_bytes + 0x70500, cp_word)
    mapped.regions = tuple((address, pack("<4I", *row) + image[16:] if address == 0x49e00 else image)
                           for address, image in mapped.regions)
    return {**boot, "request_fixture": trace_next_requests(m, external_stop, bed, dictionary, dictionary2),
            "mapped_weight_fixture": {"table_entry": row,
                                      **trace_next_requests(mapped, external_stop, bed)}}


def boot_directory_block(m, bed, block, *, trace_external_dma=False):
    m.capture_dma = False
    # Explicit zero control replies; no work/image payloads are supplied here.
    m.fixture_reads.update({addr: deque((0,) * 8) for addr in
                            (0x48004, 0x48024, 0x481c4, 0x48020, 0x48000,
                             0x481c0, 0x49d24, 0x49d2c)})
    begin = len(m.mmio_writes)
    waits = []
    for _ in range(2):
        stop = m.run(512)
        if stop["reason"] != "wfi":
            raise ValueError(f"BED next block did not reach its bootstrap wait: {stop}")
        waits.append(stop)
        m.pc += 2
    writes = {w["address"]: w["value"] for w in m.mmio_writes[begin:]}
    target, end = (writes[addr] for addr in (0x49d04, 0x49d08))
    size = end - target + 1
    src = block + unpack_from("<I", bed, block + 0x30)[0]
    local = block + unpack_from("<I", bed, block + 0x10)[0]
    if not block < src < local <= len(bed) or src + size != local:
        raise ValueError("BED external transfer does not reach the original local data boundary")
    jump = m.run(512, stop_at=target)
    if jump != {"pc": target, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED next-stage entry: {jump}")
    image = bed[src:local]
    m.regions += ((target, image),)
    m.stack_high = pc_literal(lambda a, n: image[a - target:a - target + n], target, 15)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    m.capture_dma = trace_external_dma
    m.dma_pointer = None
    return {"waits": waits, "linear_transfer": (writes[0x49d0c], target, size),
            "source_member_offset": src, "external_stop": m.run(8192),
            "catalog_fixture_va": writes[0x48168]}


def trace_next_requests(m, external_stop, bed, dictionary=None, dictionary2=None):
    """Observe requested copies without supplying either resource or feature data."""
    if external_stop != {"pc": 0x50800462, "reason": "unmodeled read", "address": 0x49000, "size": 1}:
        raise ValueError(f"unexpected BED block-one boundary: {external_stop}")
    control = next(w for w in reversed(m.mmio_writes) if w["address"] == 0x49000)
    if control["size"] != 1:
        raise ValueError("expected a preceding byte write to the block-one control")
    m.extra_scratch += ((0x49000, 0x49001), (0x3880, 0x3b74))
    m.scratch[control["address"]] = control["value"]
    m.capture_dma = True
    m.dma_pointer = None
    start = len(m.scratch_reads)
    requests = []
    for submit, wait, target, size in ((0x5080070c, 0x50800714, 0x4000, 76),
                                       (0x5080075a, 0x50800760, 0x404c, 200),
                                       (0x50800536, 0x5080053c, 0xc0015800, 8192)):
        begin = len(m.dma_submissions)
        stop = m.run(8192)
        if stop != {"pc": wait, "reason": "wfi"} or len(m.dma_submissions) != begin + 1:
            raise ValueError(f"unexpected BED block-one DMA boundary: {stop}")
        request = m.dma_submissions[-1]
        (_, dst, a, b), = request["rows"]
        if request["submit_pc"] != submit or dst != target or a != b or a != (size - 1) << 16:
            raise ValueError(f"unexpected BED block-one DMA request: {request}")
        requests.append({"wait": stop, "request": request, "bytes": size})
        m.pc += 2  # Resume control flow only; do not complete the transfer.
    stop = m.run(8192)
    scalar = deepcopy(m)
    scalar.extra_scratch += ((0x3800, 0x3880),)
    first_write = len(scalar.scratch_trace)
    scalar_stop = scalar.run(8192)
    scalar_writes = scalar.scratch_trace[first_write:]
    catalog = trace_next_program(scalar, scalar_stop, bed)
    reads = tuple((r["pc"], r["address"] - PARAMETER_ALIAS, r["value"])
                  for r in m.scratch_reads[start:] if PARAMETER_ALIAS <= r["address"] < PARAMETER_ALIAS + 0x80)
    result = {"prior_control_write": control, "requests": requests, "parameter_reads": reads,
              "stop": stop, "scalar_stop": scalar_stop, "scalar_writes": scalar_writes,
              "program_fixture": catalog}
    if dictionary is not None:
        source, *_ = requests[0]["request"]["rows"][0]
        weight_base = source - 0x75990
        bound = deepcopy(scalar) if dictionary2 is not None else None
        result["post_compute_fixture"] = trace_post_compute(scalar, bed, dictionary, weight_base)
        if bound is not None:
            layout = bind_second_dictionary(result, len(dictionary), dictionary2)
            for span in layout["early"]:
                dst, offset, size = (span[k] for k in ("destination", "offset", "bytes"))
                bound.extra_scratch += ((dst, dst + size),)
                bound.scratch.update(enumerate(dictionary2[offset:offset + size], dst))
            start = len(bound.scratch_reads)
            replay = trace_post_compute(bound, bed, dictionary, weight_base,
                                        second_binding=(layout, dictionary2))
            result["second_dictionary_candidate"] = {
                "layout": layout, "post_compute": replay,
                "config_reads": [r for r in bound.scratch_reads[start:]
                                 if 0x4000 <= r["address"] < 0x4114],
            }
    return result


def trace_post_compute(m, bed, dictionary, weight_base, *, second_binding=None):
    """Use a bounded dictionary slice; leave all activation memory untouched."""
    m.pc = m.link  # Isolated return from the still unimplemented opcode.
    ptr = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)
    m.fixture_reads.update({0x481cc: deque((0, 0)), 0x48168: deque((ptr,) * 2),
                            0x48230: deque((0,))})
    first = len(m.dma_submissions)
    for wait in (0x50800f08, 0x50800d9c, 0x50801054):
        stop = m.run(8192)
        if stop != {"pc": wait, "reason": "wfi"}:
            raise ValueError(f"unexpected BED post-compute wait: {stop}")
        m.pc += 2
    overlay, loads = m.dma_submissions[first:]
    (src, dst, a, b), = overlay["rows"]
    slots = plane_slots(a, b)
    if slots is None or dst != 0x0400f000:
        raise ValueError(f"unexpected BED post-compute program: {overlay}")
    planes = program(bed, 0x100 + src - BLOCK_VA, slots)
    spans = []
    for src, dst, a, b in loads["rows"]:
        size = 1 + (a >> 16)
        offset = src - weight_base
        if a != b or a & 0xffff or not 0 <= offset <= len(dictionary) - size:
            raise ValueError(f"unexpected BED dictionary row: {(src, dst, a, b)}")
        spans.append((offset, dst, size))
    wait = m.run(8192)
    if wait != {"pc": 0x508010cc, "reason": "wfi"}:
        raise ValueError(f"unexpected BED post-compute worker wait: {wait}")
    m.pc += 2
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x50800894, "reason": "unmodeled read", "address": 0x7830, "size": 2}:
        raise ValueError(f"unexpected BED dictionary dependency: {unseeded}")
    offset, dst, size = spans[0]
    if (dst, size) != (0x7830, 12):
        raise ValueError("expected the original 12-byte parameter request")
    data = dictionary[offset:offset + size]
    m.extra_scratch += ((dst, dst + size),)
    m.scratch.update(enumerate(data, dst))
    reads, writes = len(m.scratch_reads), len(m.mmio_writes)
    stop = m.run(8192)
    if stop != {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 0xa}:
        raise ValueError(f"unexpected BED dictionary continuation: {stop}")
    pointer, *args = m.reg[4:8]
    base, image = next((base, image) for base, image in m.regions if base <= pointer < stop["pc"] < base + len(image))
    begin = pointer - base
    end = begin + 8
    if image[end:end + len(COMMAND_MARKER)] != COMMAND_MARKER:
        raise ValueError("unexpected BED post-compute command extent")
    indices = tuple(row["index"] for row in join_record(planes, image[begin:end]))
    setup = {"stop": stop, "return": m.link, "record": pointer, "arguments": args,
             "slot_indices": indices, "parameter_reads": m.scratch_reads[reads:],
             "mmio_writes": m.mmio_writes[writes:]}
    return {"overlay": overlay, "program_offset": planes.offset, "slots": slots,
            "loads": loads, "weight_base_fixture": weight_base, "dictionary_spans": spans, "unseeded_stop": unseeded,
            "parameter_hex": data.hex(), **setup,
            "following_fixture": trace_following_compute(m, bed, weight_base, dictionary, second_binding)}


def trace_following_compute(m, bed, weight_base, dictionary, second_binding=None):
    """Trace two calls and bind their successor's first parameter, not their results."""
    m.pc = m.link
    stop = m.run(8192)
    if stop != {"pc": 0x508009b4, "reason": "wfi"}:
        raise ValueError(f"unexpected BED parameter compute wait: {stop}")
    m.pc += 2
    calls = []
    for _ in range(2):
        begin = len(m.mmio_writes)
        stop = m.run(8192)
        if stop != {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 0xa}:
            raise ValueError(f"unexpected BED paired compute: {stop}")
        calls.append({"arguments": tuple(m.reg[4:8]), "return": m.link,
                      "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:]
                                        if 0x40320 <= w["address"] <= 0x40330)})
        m.pc = m.link
    m.fixture_reads.update({0x481cc: deque((0,) * 8), 0x48230: deque((0,))})
    before = len(m.dma_submissions)
    waits = []
    for pc in (0x50801140, 0x508012c4, 0x508013a0):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED successor wait: {stop}")
        waits.append(stop)
        m.pc += 2
    (load,) = m.dma_submissions[before:]
    (src, dst, a, b), *coefficient_rows = load["rows"]
    offset = src - weight_base
    if (dst, a, b) != (0x16240, 0x170000, 0x170000) or not 0 <= offset <= len(dictionary) - 24:
        raise ValueError(f"unexpected BED successor parameter request: {load}")
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x50800b42, "reason": "unmodeled read", "address": dst, "size": 2}:
        raise ValueError(f"unexpected BED successor parameter dependency: {unseeded}")
    m.extra_scratch += ((dst, dst + 24),)
    m.scratch.update(enumerate(dictionary[offset:offset + 24], dst))
    begin = len(m.mmio_writes)
    stop = m.run(8192)
    if stop != {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 0xa}:
        raise ValueError(f"unexpected BED successor setup boundary: {stop}")
    params = tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:]
                   if 0x40140 <= w["address"] <= 0x40148)
    geometry = {w["address"]: w["value"] for w in m.mmio_writes[begin:]
                if 0x40190 <= w["address"] <= 0x401d0}
    control = next(w["value"] for w in m.mmio_writes[begin:] if w["address"] == 0x40218)
    return {"calls": calls, "waits": waits, "load": load, "parameter_offset": offset,
            "unseeded_stop": unseeded, "stop": stop, "arguments": tuple(m.reg[4:8]),
            "parameter_writes": params, "geometry": geometry, "aperture_control": control,
            "coefficient_fixture": trace_coefficient_calls(
                m, bed, weight_base, dictionary, coefficient_rows, second_binding)}


def padded_3x3(data):
    """Decode BED's stored coefficient blocks without assuming convolution semantics."""
    if not data or len(data) % 24:
        raise ValueError("expected complete BED 24-byte coefficient blocks")
    rows = tuple(iter_unpack("<4h", data))
    if any(row[3] for row in rows):
        raise ValueError("BED coefficient row padding must be zero")
    return tuple(tuple(row[:3] for row in rows[i:i + 3]) for i in range(0, len(rows), 3))


def locate_second_dictionary(dictionary, first_size, coefficient_spans, other_spans):
    """Find relative layouts compatible with all requested spans and 3x3 row padding."""
    if not coefficient_spans or any(not size or size % 24 for _, _, size in coefficient_spans):
        raise ValueError("expected complete second dictionary coefficient requests")
    spans = (*coefficient_spans, *other_spans)
    lo = max(first_size, max(src + size - len(dictionary) for src, _, size in spans))
    hi = min(src for src, _, _ in spans)
    return tuple(base for base in range((lo + 3) & ~3, hi + 1, 4)
                 if all(dictionary[i + 6:i + 8] == b"\0\0"
                        for src, _, size in coefficient_spans
                        for i in range(src - base, src - base + size, 8)))


def second_dictionary_layout(result, dictionary, dictionary2):
    fixture = result["handoff"]["boot_fixture"]["request_fixture"]
    return bind_second_dictionary(fixture, len(dictionary), dictionary2)


def bind_second_dictionary(fixture, first_size, dictionary2):
    if sha256(dictionary2).hexdigest() != BED_DICT2_SHA:
        raise ValueError("expected the pinned release bed_dic2.bin")
    first, second = fixture["requests"][:2]
    post = fixture["post_compute_fixture"]
    weight_base = post["weight_base_fixture"]
    later = post["following_fixture"]["coefficient_fixture"]["next_spans"]
    early = [(r["request"]["rows"][0][0] - weight_base, r["request"]["rows"][0][1], r["bytes"])
             for r in (first, second)]
    candidates = locate_second_dictionary(dictionary2, first_size, later[1:], [later[0], *early])
    if len(candidates) != 1:
        raise ValueError(f"expected one packed second dictionary layout, found {candidates}")
    (base,) = candidates
    def bind(span):
        src, dst, size = span
        offset = src - base
        data = dictionary2[offset:offset + size]
        return {"offset": offset, "destination": dst, "bytes": size, "sha256": sha256(data).hexdigest()}
    config_offset = early[1][0] - base
    return {"base_offset": base, "early": [bind(span) for span in early],
            "later": [bind(span) for span in later],
            "control_bytes": tuple(unpack_from("<b", dictionary2, config_offset + offset)[0]
                                   for offset in (0x78, 0x7f))}


def trace_coefficient_calls(m, bed, weight_base, dictionary, rows, second_binding=None):
    """Bind the scalar call addresses to stored blocks without executing the opcode."""
    spans = []
    for src, dst, a, b in rows:
        offset = src - weight_base
        if a != b or a != 0x1af0000 or not 0 <= offset <= len(dictionary) - 432:
            raise ValueError(f"unexpected BED 432-byte coefficient request: {(src, dst, a, b)}")
        spans.append({"offset": offset, "destination": dst,
                      "blocks": padded_3x3(dictionary[offset:offset + 432])})
    m.pc = m.link
    calls = []
    for _ in range(3):
        before = len(m.mmio_writes)
        wait = m.run(8192)
        if wait != {"pc": 0x50800c84, "reason": "wfi"}:
            raise ValueError(f"unexpected BED coefficient worker wait: {wait}")
        m.pc += 2
        stop = m.run(8192)
        if stop != {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 0xa}:
            raise ValueError(f"unexpected BED coefficient compute: {stop}")
        aperture = tuple((w["address"], w["value"]) for w in m.mmio_writes[before:]
                         if 0x40320 <= w["address"] <= 0x40330)
        calls.append({"arguments": tuple(m.reg[4:8]), "aperture": aperture, "return": m.link})
        m.pc = m.link
    stop = m.run(8192)
    if stop != {"pc": 0x5080141c, "reason": "wfi"}:
        raise ValueError(f"unexpected BED coefficient completion wait: {stop}")
    next_stage = deepcopy(m)
    next_stage.pc += 2
    next_stage.fixture_reads[0x48230] = deque((0,))
    before = len(next_stage.dma_submissions)
    wait = next_stage.run(8192)
    if wait != {"pc": 0x50801584, "reason": "wfi"}:
        raise ValueError(f"unexpected BED second dictionary DMA boundary: {wait}")
    (load,) = next_stage.dma_submissions[before:]
    next_spans = []
    for src, dst, a, b in load["rows"]:
        if a != b or a & 0xffff:
            raise ValueError(f"unexpected BED second dictionary DMA request: {load}")
        next_spans.append((src - weight_base, dst, 1 + (a >> 16)))
    result = {"spans": spans, "calls": calls, "stop": stop,
              "workers": trace_coefficient_workers(m, bed), "next_spans": next_spans}
    if second_binding is not None:
        layout, dictionary2 = second_binding
        parameter = layout["later"][0]
        offset = parameter["offset"]
        _, dst, size = next_spans[0]
        if (dst, size, parameter["bytes"]) != (0xafc0, 24, 24):
            raise ValueError("unexpected BED late parameter span")
        result["late_parameter_fixture"] = trace_late_parameter(
            next_stage, bed, dst, dictionary2[offset:offset + size], dictionary, weight_base, second_binding)
    return result


def trace_late_parameter(m, bed, dst, data, dictionary, weight_base, second_binding):
    """Use the published 24-byte parameter slice without populating coefficient or feature RAM."""
    m.pc += 2
    wait = m.run(8192)
    if wait != {"pc": 0x50801604, "reason": "wfi"}:
        raise ValueError(f"unexpected BED late worker wait: {wait}")
    m.pc += 2
    reads, writes = len(m.scratch_reads), len(m.mmio_writes)
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x50800894, "reason": "unmodeled read", "address": dst, "size": 2}:
        raise ValueError(f"unexpected BED late parameter dependency: {unseeded}")
    m.extra_scratch += ((dst, dst + len(data)),)
    m.scratch.update(enumerate(data, dst))
    opcode = {"pc": 0x50800374, "reason": "unsupported opcode", "opcode": 0xa}
    stop = m.run(8192)
    if stop != opcode:
        raise ValueError(f"unexpected BED late parameter setup: {stop}")
    setup = {"arguments": tuple(m.reg[4:8]), "return": m.link,
             "parameters": tuple((w["address"], w["value"]) for w in m.mmio_writes[writes:]
                                 if w["address"] in (0x40140, 0x40144, 0x40148, 0x40218)),
             "reads": m.scratch_reads[reads:]}
    calls = []
    for _ in range(3):
        m.pc = m.link
        wait = m.run(8192)
        if wait != {"pc": 0x508009b4, "reason": "wfi"}:
            raise ValueError(f"unexpected BED late compute wait: {wait}")
        m.pc += 2
        start = len(m.mmio_writes)
        stop = m.run(8192)
        if stop != opcode:
            raise ValueError(f"unexpected BED late compute: {stop}")
        calls.append({"arguments": tuple(m.reg[4:8]), "return": m.link,
                      "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[start:]
                                        if 0x40320 <= w["address"] <= 0x40330)})
    m.pc = m.link
    for pc in (0x50801680, 0x508016a0):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED late completion wait: {stop}")
        m.pc += 2
    stop = m.run(8192)
    return {"unseeded_stop": unseeded, "setup": setup, "calls": calls, "stop": stop,
            "handoff": trace_late_handoff(m, bed, stop, dictionary, weight_base, second_binding)}


def trace_late_handoff(m, bed, stop, dictionary, weight_base, second_binding):
    """Replay the control path only; never supply the scanned range or computed outputs."""
    if stop != {"pc": 0x50800568, "reason": "unmodeled read", "address": 0x48100, "size": 4}:
        raise ValueError(f"unexpected BED late handoff status read: {stop}")
    source = 0x20000000  # Explicit range-pointer fixture, not mapped feature memory.
    m.fixture_reads[0x48100] = deque((source,))
    flag_read = m.run(512)
    if flag_read != {"pc": 0x5080056a, "reason": "unmodeled read", "address": 0x48170, "size": 1}:
        raise ValueError(f"unexpected BED range control read: {flag_read}")
    m.extra_scratch += ((0x48170, 0x48171),)
    flagged = deepcopy(m)
    flagged.scratch[0x48170] = 1
    scan = flagged.run(512)
    if scan != {"pc": 0x50800580, "reason": "unmodeled read", "address": source, "size": 4}:
        raise ValueError(f"unexpected BED range scan dependency: {scan}")
    m.scratch[0x48170] = 0
    m.fixture_reads[0x48160] = deque((BLOCK_VA,))
    directory = 0x100 + unpack_from("<I", bed, 0x11c)[0]
    offset, = unpack_from("<I", bed, directory + 8)
    target = BLOCK_VA + offset
    branch = m.run(512, stop_at=target)
    if branch != {"pc": target, "reason": "slice boundary"} or m.link != 0x508005c2:
        raise ValueError(f"unexpected BED directory[2] handoff: {branch}")
    block = 0x100 + offset
    before = len(m.dma_submissions)
    boot = boot_directory_block(m, bed, block, trace_external_dma=True)
    if boot["external_stop"] != {"pc": 0x5100050c, "reason": "wfi"}:
        raise ValueError(f"unexpected BED directory[2] first DMA wait: {boot['external_stop']}")
    m.pc += 2
    wait = m.run(8192)
    if wait != {"pc": 0x51000558, "reason": "wfi"}:
        raise ValueError(f"unexpected BED directory[2] second DMA wait: {wait}")
    first, second = m.dma_submissions[before:]
    m.pc += 2
    stop = m.run(8192)
    return {"flag_read": flag_read, "scan": scan, "source_fixture": source, "branch": branch,
            "member_offset": block, "boot": boot, "submissions": (first, second),
            "second_wait": wait, "stop": stop,
            "compute_fixture": trace_third_compute(m, bed, stop, (first, second),
                                                   dictionary, weight_base, second_binding)}


def bind_linear_parameter(m, row, data, base, target, size):
    source, dst, a, b = row
    offset = source - base
    if dst != target or a != b or a != (size - 1) << 16 or not 0 <= offset <= len(data) - size:
        raise ValueError(f"unexpected BED parameter DMA row: {row}")
    m.extra_scratch += ((dst, dst + size),)
    m.scratch.update(enumerate(data[offset:offset + size], dst))
    return offset


def trace_third_compute(m, bed, stop, initial, dictionary, weight_base, second_binding):
    """Drive the third block with only the original dictionary parameter copies."""
    if stop != {"pc": 0x51000918, "reason": "unmodeled read", "address": 0x48230, "size": 4}:
        raise ValueError(f"unexpected BED third-block status read: {stop}")
    layout, dictionary2 = second_binding
    initial_offsets = tuple(bind_linear_parameter(m, r["rows"][0], dictionary2,
                                                  weight_base + layout["base_offset"], target, size)
                            for r, (target, size) in zip(initial, ((0x4000, 40), (0x4028, 124)), strict=True))
    m.extra_scratch += ((0x8500, 0x8560),)
    m.fixture_reads[0x48230] = deque((0, 0))
    groups = []
    for wait, worker_wait, dst, size in ((0x51000934, 0x510009ec, 0x8590, 12),
                                         (0x51000bfc, 0x51000c6c, 0x51b0, 24)):
        begin = len(m.dma_submissions)
        stop = m.run(8192)
        if stop != {"pc": wait, "reason": "wfi"}:
            raise ValueError(f"unexpected BED third-block parameter wait: {stop}")
        (request,) = m.dma_submissions[begin:]
        offset = bind_linear_parameter(m, request["rows"][0], dictionary, weight_base, dst, size)
        m.pc += 2
        stop = m.run(8192)
        if stop != {"pc": worker_wait, "reason": "wfi"}:
            raise ValueError(f"unexpected BED third-block worker wait: {stop}")
        m.pc += 2
        reads, writes = len(m.scratch_reads), len(m.mmio_writes)
        calls = []
        for record, return_pc in ((0x510001e8, 0x5100078c), (0x51000038, 0x510007aa)):
            stop = m.run(8192)
            if (stop, m.reg[4], m.link) != (
                    {"pc": 0x51000200, "reason": "unsupported opcode", "opcode": 0xa}, record, return_pc):
                raise ValueError(f"unexpected BED third-block compute: {stop}")
            calls.append({"arguments": tuple(m.reg[4:8]), "stop": stop, "return": m.link})
            m.pc = m.link
            if return_pc == 0x5100078c:
                barrier = m.run(8192)
                if barrier != {"pc": 0x51000798, "reason": "wfi"}:
                    raise ValueError(f"unexpected BED third-block setup wait: {barrier}")
                m.pc += 2
        groups.append({"request": request, "parameter_offset": offset, "calls": calls,
                       "reads": [r for r in m.scratch_reads[reads:]
                                 if 0x4000 <= r["address"] < 0x40a4 or dst <= r["address"] < dst + size],
                       "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[writes:]
                                         if w["address"] in (0x40140, 0x40144, 0x40148, 0x40218, 0x40328, 0x40330))})
        stop = m.run(8192)
        if stop != {"pc": 0x51000a60 if len(groups) == 1 else 0x51000ce0, "reason": "wfi"}:
            raise ValueError(f"unexpected BED third-block completion wait: {stop}")
        m.pc += 2
    stop = m.run(8192)
    return {"initial_offsets": initial_offsets, "groups": groups, "stop": stop,
            "completion": trace_third_completion(m, bed, stop, dictionary, weight_base, second_binding)}


def trace_third_completion(m, bed, stop, dictionary, weight_base, second_binding):
    """Observe state-dependent handoff and work writes without producing features."""
    if stop != {"pc": 0x51000cf6, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED third-block completion status: {stop}")
    directory = 0x100 + unpack_from("<I", bed, 0x11c)[0]
    offset, = unpack_from("<I", bed, directory + 12)
    target = BLOCK_VA + offset
    paths = {}
    for state in (0, 1, 2):
        path = deepcopy(m)
        start = len(path.scratch_trace)
        path.fixture_reads[0x481cc] = deque((state,) * 3)
        path.fixture_reads[0x48100] = deque((0x20000000,))
        if state == 1:
            status_stop = path.run(8192)
            if status_stop != {"pc": 0x51000e52, "reason": "unmodeled read", "address": 0x48230, "size": 4}:
                raise ValueError(f"unexpected BED state-1 DMA status: {status_stop}")
            path.fixture_reads[0x48230] = deque((0,))
            paths[state] = {"dma_status_stop": status_stop, "stop": path.run(8192)}
        else:
            wait = path.run(8192)
            if wait != {"pc": 0x51000d04, "reason": "wfi"}:
                raise ValueError(f"unexpected BED state-{state} completion wait: {wait}")
            path.pc += 2
            path.fixture_reads[0x48160] = deque((BLOCK_VA,))
            branch = path.run(8192, stop_at=target)
            if branch != {"pc": target, "reason": "slice boundary"} or path.link != 0x510003fe:
                raise ValueError(f"unexpected BED directory[3] handoff: {branch}")
            paths[state] = {"wait": wait, "branch": branch}
            if state == 0:
                next_block = path
        paths[state]["work_writes"] = [w for w in path.scratch_trace[start:]
                                       if 0xe0004040 <= w["address"] < 0xe00040d8]
    path = next_block
    path.fixture_reads[0x481cc] = deque((0,) * 3)
    boot = boot_directory_block(path, bed, 0x100 + offset, trace_external_dma=True)
    if boot["external_stop"] != {"pc": 0x518006e0, "reason": "wfi"}:
        raise ValueError(f"unexpected BED directory[3] first DMA: {boot['external_stop']}")
    first = path.dma_submissions[-1]
    path.pc += 2
    wait = path.run(8192)
    if wait != {"pc": 0x5180072c, "reason": "wfi"}:
        raise ValueError(f"unexpected BED directory[3] second DMA: {wait}")
    second = path.dma_submissions[-1]
    path.pc += 2
    stop = path.run(8192)
    return {"paths": paths, "boot": boot, "submissions": (first, second),
            "second_wait": wait, "stop": stop,
            "compute_fixture": trace_fourth_compute(path, bed, stop, boot["catalog_fixture_va"],
                                                    (first, second), dictionary, weight_base, second_binding)}


def trace_fourth_compute(m, bed, stop, catalog, initial, dictionary, weight_base, second_binding):
    """Bind fourth-block programs and parameters; capture but do not execute DMA of computed data."""
    if stop != {"pc": 0x51800ffc, "reason": "unmodeled read", "address": 0x48168, "size": 4}:
        raise ValueError(f"unexpected BED fourth-block directory read: {stop}")
    layout, dictionary2 = second_binding
    offsets = tuple(bind_linear_parameter(m, r["rows"][0], dictionary2, weight_base + layout["base_offset"],
                                          dst, size)
                    for r, (dst, size) in zip(initial, ((0x17e00, 52), (0x17e34, 140)), strict=True))
    m.fixture_reads.update({0x48168: deque((catalog,) * 4), 0x48230: deque((0,))})

    def wait_at(pc):
        wait = m.run(8192)
        if wait != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fourth-block wait: {wait}")
        m.pc += 2

    def overlay_at(pc):
        begin = len(m.dma_submissions)
        wait_at(pc)
        (request,) = m.dma_submissions[begin:]
        (src, dst, a, b), = request["rows"]
        slots = plane_slots(a, b)
        if slots is None or dst != 0x0400f000:
            raise ValueError(f"unexpected BED fourth-block program: {request}")
        bound = program(bed, src - BLOCK_VA + 0x100, slots)
        return {"request": request, "offset": bound.offset, "slots": bound.slots}

    def call_at(pc, record):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": "unsupported opcode", "opcode": 0xa} or m.reg[4] != record:
            raise ValueError(f"unexpected BED fourth-block compute: {stop}")
        call = {"stop": stop, "arguments": tuple(m.reg[4:8]), "return": m.link}
        m.pc = m.link
        return call

    first = overlay_at(0x51801054)
    wait_at(0x51801090)
    calls = [call_at(0x51801980, 0x518002e0)]
    wait_at(0x518010bc)
    second = overlay_at(0x51800f5c)
    begin = len(m.dma_submissions)
    wait_at(0x51801290)
    (load,) = m.dma_submissions[begin:]
    wait_at(0x51801310)
    reads, writes = len(m.scratch_reads), len(m.mmio_writes)
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x51800b34, "reason": "unmodeled read", "address": 0xdda0, "size": 2}:
        raise ValueError(f"unexpected BED fourth-block parameter dependency: {unseeded}")
    parameter = bind_linear_parameter(m, load["rows"][0], dictionary, weight_base, 0xdda0, 24)
    calls.append(call_at(0x518003d0, 0x518002c8))
    for _ in range(2):
        wait_at(0x51800bd8)
        calls.append(call_at(0x518003d0, 0x51800038))
    wait_at(0x518013e4)
    begin = len(m.dma_submissions)
    for _ in range(12):
        wait_at(0x51800920)
        wait_at(0x518009a4)
    data_requests = m.dma_submissions[begin:]
    stop = m.run(8192)
    return {"initial_offsets": offsets, "programs": (first, second), "parameter_request": load,
            "parameter_offset": parameter, "unseeded_stop": unseeded, "calls": calls,
            "reads": [r for r in m.scratch_reads[reads:] if 0x17e00 <= r["address"] < 0x17ec0
                      or 0xdda0 <= r["address"] < 0xddb8],
            "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[writes:]
                              if w["address"] in (0x40140, 0x40144, 0x40218)),
            "following_requests": data_requests, "stop": stop,
            "completion": trace_fourth_completion(m, bed, stop, dictionary, weight_base, second_binding)}


def trace_fourth_completion(m, bed, stop, dictionary, weight_base, second_binding):
    if stop != {"pc": 0x51801448, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED fourth-block status: {stop}")
    marked = deepcopy(m)
    marked.fixture_reads.update({0x481cc: deque((1,) * 5), 0x48100: deque((0x20000000,))})
    start = len(marked.scratch_trace)
    marked_status = marked.run(8192)
    if marked_status != {"pc": 0x518018da, "reason": "unmodeled read", "address": 0x48230, "size": 4}:
        raise ValueError(f"unexpected BED fourth-block state-one DMA status: {marked_status}")
    marked.fixture_reads[0x48230] = deque((0,) * 4)
    marked_stop = marked.run(8192)
    work_writes = [w for w in marked.scratch_trace[start:] if 0xe0004040 <= w["address"] < 0xe00040d8]

    m.fixture_reads.update({0x481cc: deque((0,) * 8), 0x48230: deque((0,) * 8),
                            0x48100: deque((0x20000000,)), 0x48160: deque((BLOCK_VA,))})
    first_write = m.run(8192)
    if first_write != {"pc": 0x518015ba, "reason": "unmodeled write", "address": 0x100, "size": 4, "value": 0}:
        raise ValueError(f"unexpected BED fourth-block scratch initialization: {first_write}")
    m.extra_scratch += ((0x100, 0x170),)
    start = len(m.dma_submissions)

    def wait_at(pc):
        wait = m.run(8192)
        if wait != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fourth-block completion wait: {wait}")
        m.pc += 2

    wait_at(0x51801620)
    (load,) = m.dma_submissions[start:]
    wait_at(0x51801710)
    reads = len(m.scratch_reads)
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x51800d00, "reason": "unmodeled read", "address": 0xca90, "size": 2}:
        raise ValueError(f"unexpected BED fourth-block final parameter dependency: {unseeded}")
    offset = bind_linear_parameter(m, load["rows"][0], dictionary, weight_base, 0xca90, 24)
    calls = []
    for record in (0x518002c8, 0x5180017c, 0x5180017c):
        start = len(m.mmio_writes)
        stop = m.run(8192)
        if stop != {"pc": 0x518003d0, "reason": "unsupported opcode", "opcode": 10} or m.reg[4] != record:
            raise ValueError(f"unexpected BED fourth-block final compute: {stop}")
        calls.append({"arguments": tuple(m.reg[4:8]), "aperture": tuple(
            (w["address"], w["value"]) for w in m.mmio_writes[start:]
            if w["address"] in (0x40140, 0x40144, 0x40218, 0x40320, 0x40328, 0x40330))})
        m.pc = m.link
        if len(calls) < 3:
            wait_at(0x51800e20)
    for pc in (0x51801788, 0x518017cc, 0x518017e8):
        wait_at(pc)
    directory = 0x100 + unpack_from("<I", bed, 0x11c)[0]
    offset5, = unpack_from("<I", bed, directory + 16)
    target = BLOCK_VA + offset5
    branch = m.run(8192, stop_at=target)
    if branch != {"pc": target, "reason": "slice boundary"} or m.link != 0x518005de:
        raise ValueError(f"unexpected BED directory[4] handoff: {branch}")
    return {"state_one_status": marked_status, "state_one_stop": marked_stop, "state_one_work_writes": work_writes,
            "first_write": first_write, "load": load, "parameter_offset": offset, "unseeded_stop": unseeded,
            "reads": [r for r in m.scratch_reads[reads:] if 0x17e00 <= r["address"] < 0x17ec0
                      or 0xca90 <= r["address"] < 0xcaa8],
            "calls": calls, "branch": branch, "next_member_offset": 0x100 + offset5,
            "fifth_fixture": trace_fifth_compute(m, bed, 0x100 + offset5, dictionary, weight_base, second_binding)}


def trace_fifth_compute(m, bed, block, dictionary, weight_base, second_binding):
    """Bind the fifth block's first three calls and stop at the mixed DMA wait."""
    layout, dictionary2 = second_binding
    start = len(m.dma_submissions)
    boot = boot_directory_block(m, bed, block, trace_external_dma=True)
    if boot["external_stop"] != {"pc": 0x52000984, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block first DMA: {boot['external_stop']}")

    def wait_at(pc):
        wait = m.run(8192)
        if wait != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fifth-block wait: {wait}")
        m.pc += 2

    m.pc += 2
    wait_at(0x520009d0)
    initial = m.dma_submissions[start:]
    offsets = tuple(bind_linear_parameter(m, r["rows"][0], dictionary2, weight_base + layout["base_offset"],
                                          dst, size)
                    for r, (dst, size) in zip(initial, ((0x17e00, 104), (0x17e68, 336)), strict=True))
    m.fixture_reads.update({0x48168: deque((boot["catalog_fixture_va"],) * 2), 0x48230: deque((0,) * 8)})
    start = len(m.dma_submissions)
    wait_at(0x52000aa0)
    (program_request,) = m.dma_submissions[start:]
    (src, dst, a, b), = program_request["rows"]
    slots = plane_slots(a, b)
    if slots is None or dst != 0x0400f000:
        raise ValueError(f"unexpected BED fifth-block program: {program_request}")
    overlay = program(bed, 0x100 + src - BLOCK_VA, slots)
    boundary = m.run(8192)
    if boundary != {"pc": 0x5200127e, "reason": "unmodeled write", "address": 0x400, "size": 4, "value": 0}:
        raise ValueError(f"unexpected BED fifth-block scratch initialization: {boundary}")
    m.extra_scratch += ((0x400, 0x800),)
    start = len(m.dma_submissions)
    wait_at(0x520014c4)
    copies = m.dma_submissions[start:]
    reads, writes = len(m.scratch_reads), len(m.mmio_writes)
    unseeded = m.run(8192)
    if unseeded != {"pc": 0x52000c82, "reason": "unmodeled read", "address": 0x92b8, "size": 2}:
        raise ValueError(f"unexpected BED fifth-block parameter dependency: {unseeded}")
    offset = bind_linear_parameter(m, copies[0]["rows"][0], dictionary, weight_base, 0x92b8, 24)
    calls = []
    for _ in range(3):
        begin = len(m.mmio_writes)
        stop = m.run(8192)
        if stop != {"pc": 0x5200060c, "reason": "unsupported opcode", "opcode": 10}:
            raise ValueError(f"unexpected BED fifth-block compute: {stop}")
        calls.append({"arguments": tuple(m.reg[4:8]), "return": m.link,
                      "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:]
                                        if w["address"] in (0x40320, 0x40328, 0x40330))})
        m.pc = m.link
    wait_at(0x52001538)
    start = len(m.dma_submissions)
    mixed_stop = m.run(8192)
    if mixed_stop != {"pc": 0x520014c4, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block mixed DMA: {mixed_stop}")
    (mixed,) = m.dma_submissions[start:]
    windows = decode_fifth_feature_rows(mixed)
    return {"boot": boot, "initial_offsets": offsets, "program_offset": overlay.offset, "slots": slots,
            "copies": copies, "unseeded_stop": unseeded, "parameter_offset": offset, "calls": calls,
            "control_writes": tuple((w["address"], w["value"]) for w in m.mmio_writes[writes:]
                                    if w["address"] in (0x40140, 0x40144, 0x40148, 0x40218)),
            "reads": [r for r in m.scratch_reads[reads:] if 0x17e00 <= r["address"] < 0x17fb8
                      or 0x92b8 <= r["address"] < 0x92d0],
            "mixed_request": mixed, "feature_windows": windows, "stop": mixed_stop,
            "second_pass": trace_fifth_second_pass(m, copies[1], dictionary, weight_base, second_binding)}


def decode_fifth_feature_rows(request):
    """Decode the observed row geometry, leaving raw destination aliasing unresolved."""
    if request["submit_pc"] != 0x520015ae or len(request["rows"]) != 17:
        raise ValueError("expected the fifth-block mixed DMA batch")
    first_src, first_dst, *_ = request["rows"][0]
    windows = []
    for i, (src, dst, a, b) in enumerate(request["rows"][:12]):
        pitch, row_bytes, rows = (a & 0xffff) * 8, (a >> 16) + 1, (b & 0xff) + 1
        if ((pitch, row_bytes, rows) != (64, 56, 14) or b != 0x370d0d
                or src != first_src + i * 0x780 or dst != first_dst + i * row_bytes * rows
                or src >> 29 != 1 or dst >> 16 != 0x8038):
            raise ValueError("unexpected fifth-block feature DMA format")
        windows.append({"source": src, "destination_raw": dst, "read": PackedRead(
            src - first_src, pitch, row_bytes, rows)})
    return windows


def trace_fifth_second_pass(m, duplicate, dictionary, weight_base, second_binding):
    """Supply the original duplicate parameter copy, leaving both inputs uncomputed."""
    offset = bind_linear_parameter(m, duplicate["rows"][0], dictionary, weight_base, 0x92a0, 24)
    m.pc += 2
    reads = len(m.scratch_reads)
    calls = []
    for _ in range(3):
        stop = m.run(8192)
        if stop != {"pc": 0x5200060c, "reason": "unsupported opcode", "opcode": 10}:
            raise ValueError(f"unexpected BED fifth-block second pass: {stop}")
        calls.append(tuple(m.reg[4:8]))
        m.pc = m.link
    m.fixture_reads.update({0x48230: deque((0,) * 6), 0x481cc: deque((0,) * 8)})
    before = len(m.dma_submissions)
    for pc in (0x52001538, 0x52001770, 0x52001800):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fifth-block second-pass wait: {stop}")
        m.pc += 2
    stop = m.run(8192)
    requests = m.dma_submissions[before:]
    return {"parameter_offset": offset, "calls": calls, "stop": stop,
            "parameter_reads": [r for r in m.scratch_reads[reads:] if 0x92a0 <= r["address"] < 0x92b8],
            "requests": requests, "next_compute": trace_fifth_next_compute(m, stop, requests[-1],
                                                                          dictionary, weight_base, second_binding)}


def trace_fifth_next_compute(m, stop, request, dictionary, weight_base, second_binding):
    """Bind the two parameter passes using their respective original DMA requests."""
    if stop != {"pc": 0x52000df8, "reason": "unmodeled read", "address": 0x15e40, "size": 2}:
        raise ValueError(f"unexpected BED fifth-block next dependency: {stop}")
    parameter, coefficients, *_ = request["rows"]
    offset = bind_linear_parameter(m, parameter, dictionary, weight_base, 0x15e40, 30)
    src, dst, a, b = coefficients
    coeff_offset = src - weight_base
    if (dst, a, b) != (0x159c0, 0x47f0000, 0x47f0000) or not 0 <= coeff_offset <= len(dictionary) - 1152:
        raise ValueError(f"unexpected BED fifth-block coefficient row: {coefficients}")
    blocks = padded_3x3(dictionary[coeff_offset:coeff_offset + 1152])
    first = trace_fifth_projection(m)
    m.pc += 2
    begin = len(m.dma_submissions)
    stop = m.run(8192)
    if stop != {"pc": 0x52001770, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block next parameter reload: {stop}")
    (reload,) = m.dma_submissions[begin:]
    return {"parameter_offset": offset, "coefficient_offset": coeff_offset, "coefficient_blocks": len(blocks),
            **first, "reload": reload, "stop": stop,
            "reloaded_compute": trace_fifth_reloaded(m, reload, dictionary, weight_base, second_binding)}


def trace_fifth_projection(m):
    start = len(m.scratch_reads)
    begin = len(m.mmio_writes)
    opcode = {"pc": 0x5200060c, "reason": "unsupported opcode", "opcode": 10}
    stop = m.run(8192)
    if stop != opcode or m.reg[4] != 0x520002c8:
        raise ValueError(f"unexpected BED fifth-block next setup: {stop}")
    setup = tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:]
                  if 0x40140 <= w["address"] <= 0x4014c or w["address"] == 0x40218)
    calls = []
    for i in range(4):
        m.pc = m.link
        if i < 3:
            wait = m.run(8192)
            if wait != {"pc": 0x52000f04, "reason": "wfi"}:
                raise ValueError(f"unexpected BED fifth-block worker wait: {wait}")
            m.pc += 2
        begin = len(m.mmio_writes)
        stop = m.run(8192)
        if stop != opcode or m.reg[4] != 0x5200017c:
            raise ValueError(f"unexpected BED fifth-block next compute: {stop}")
        calls.append({"arguments": tuple(m.reg[4:8]),
                      "aperture": tuple((w["address"], w["value"]) for w in m.mmio_writes[begin:]
                                        if w["address"] in (0x40320, 0x40328, 0x40330))})
    m.pc = m.link
    wait = m.run(8192)
    if wait != {"pc": 0x52001a28, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block next completion wait: {wait}")
    return {"reads": [r for r in m.scratch_reads[start:] if 0x15e40 <= r["address"] < 0x15e5e
                      or 0x17e00 <= r["address"] < 0x17fb8],
            "setup": setup, "calls": calls}


def trace_fifth_reloaded(m, request, dictionary, weight_base, second_binding):
    offset = bind_linear_parameter(m, request["rows"][0], dictionary, weight_base, 0x15e40, 30)
    m.pc += 2
    wait = m.run(8192)
    if wait != {"pc": 0x52001800, "reason": "wfi"}:
        raise ValueError(f"unexpected BED reloaded worker wait: {wait}")
    m.pc += 2
    result = trace_fifth_projection(m)
    m.pc += 2
    catalog_read = m.run(8192)
    if catalog_read != {"pc": 0x52000fc8, "reason": "unmodeled read", "address": 0x48168, "size": 4}:
        raise ValueError(f"unexpected BED fifth-block final directory read: {catalog_read}")
    pointer = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)
    m.fixture_reads[0x48168] = deque((pointer,) * 12)
    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    overlays = []
    for dst in range(0x04000000, 0x04004000, 0x1000):
        begin = len(m.dma_submissions)
        wait = m.run(8192)
        if wait != {"pc": 0x5200102c, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fifth-block final program wait: {wait}")
        (load,) = m.dma_submissions[begin:]
        (source, window, a, b), = load["rows"]
        slots = plane_slots(a, b)
        if window != dst or slots is None:
            raise ValueError(f"unexpected BED fifth-block final overlay: {load}")
        bound = program(image, source - BLOCK_VA, slots)
        overlays.append({"request": load, "member_offset": bound.offset + 0x100, "slots": bound.slots})
        m.pc += 2
    wait = m.run(8192)
    if wait != {"pc": 0x52001ae8, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block final worker wait: {wait}")
    m.pc += 2
    compute = m.run(8192)
    if compute != {"pc": 0x520023cc, "reason": "unsupported opcode", "opcode": 10}:
        raise ValueError(f"unexpected BED fifth-block final compute: {compute}")
    args = tuple(m.reg[4:8])
    m.pc = m.link
    wait = m.run(8192)
    if wait != {"pc": 0x52001b10, "reason": "wfi"}:
        raise ValueError(f"unexpected BED fifth-block output wait: {wait}")
    m.pc += 2
    stop = m.run(8192)
    copies = trace_fifth_halfword_copy(m, stop, dictionary, weight_base, second_binding)
    return {**result, "parameter_offset": offset, "catalog_read": catalog_read, "overlays": overlays,
            "compute": compute, "arguments": args, "unseeded_stop": stop, "copy_fixtures": copies}


def trace_fifth_halfword_copy(snapshot, stop, dictionary, weight_base, second_binding):
    if stop != {"pc": 0x52001b2c, "reason": "unmodeled read", "address": 0x10018, "size": 2}:
        raise ValueError(f"unexpected BED fifth-block copy input: {stop}")
    source = snapshot.reg[7]
    target, end = snapshot.reg[2:4]
    if (source, target, end) != (0x10018, 0xffd8, 0x10014):
        raise ValueError(f"unexpected BED fifth-block copy range: {(source, target, end)}")
    count = (end - target) // 2
    paths = []
    for seed in (0x7ff0, 0xfff0):
        m = deepcopy(snapshot)
        m.extra_scratch += ((source, source + 8 * count), (target, end))
        for i in range(count):
            m.scratch.update(enumerate(pack("<H", (seed + i) & 0xffff), source + i * 8))
        read_start, write_start = len(m.scratch_reads), len(m.scratch_trace)
        boundary = m.run(256, stop_at=0x52001b3a)
        if boundary != {"pc": 0x52001b3a, "reason": "slice boundary"}:
            raise ValueError(f"unexpected BED fifth-block copy exit: {boundary}")
        output = bytes(m.scratch[a] for a in range(target, end))
        paths.append({"seed": seed, "reads": m.scratch_reads[read_start:],
                      "writes": m.scratch_trace[write_start:],
                      "output": tuple(x for (x,) in iter_unpack("<h", output)),
                      "boundary": boundary,
                      "consumers": trace_fifth_copy_consumers(m, dictionary, weight_base, second_binding)})
    return paths


def trace_fifth_copy_consumers(m, dictionary, weight_base, second_binding):
    calls = []
    for wait_pc, target, size, return_pc in (
            (0x52001c4c, 0x10018, 640, 0x52001c74),
            (0x52001d78, 0x10050, 720, 0x52001d96)):
        begin = len(m.dma_submissions)
        wait = m.run(8192)
        if wait != {"pc": wait_pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED fifth-block consumer wait: {wait}")
        (request,) = m.dma_submissions[begin:]
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary, weight_base, target, size)
        m.pc += 2
        launch = m.run(8192)
        if launch != {"pc": 0x520023cc, "reason": "unsupported opcode", "opcode": 10} or m.link != return_pc:
            raise ValueError(f"unexpected BED fifth-block consumer launch: {launch}")
        wrapper = deepcopy(m)
        wrapper.pc += 2
        dispatch = wrapper.run(32)
        if dispatch != {"pc": 0x520023e2, "reason": "unsupported opcode", "opcode": 9}:
            raise ValueError(f"unexpected BED fifth-block consumer dispatch: {dispatch}")
        calls.append({"request": request, "dictionary_offset": offset, "arguments": tuple(m.reg[4:8]),
                      "dispatch": dispatch, "record_pointer": wrapper.reg[0], "return_pc": return_pc})
        # Resume only the scalar control path, without supplying any compute results.
        m.pc = m.link
    begin = len(m.scratch_reads)
    stop = m.run(8192)
    if stop != {"pc": 0x52001dc0, "reason": "unsupported opcode", "opcode": 0x3916}:
        raise ValueError(f"unexpected BED fifth-block scalar conversion: {stop}")
    operands = [r for r in m.scratch_reads[begin:] if r["pc"] in (0x52001dbc, 0x52001dbe)]
    return {"calls": calls, "stop": stop, "scalar_input_reads": operands,
            "sigmoid_fixture": trace_fifth_sigmoid_fixture(m, dictionary, weight_base, second_binding)}


def trace_fifth_sigmoid_fixture(snapshot, dictionary, weight_base, second_binding):
    """Isolate the scalar postprocessor from the unimplemented compute records."""
    m = deepcopy(snapshot)
    m.float32_enabled = True
    source, target, ceiling = m.reg[13], m.reg[0], m.reg[9]
    scale = tuple(m.read(m.reg[r], 1, sign=True) for r in (8, 4, 5))
    if (source, target, ceiling, scale) != (0xffd8, 0x10050, 65535, (25, 25, -16)):
        raise ValueError(f"unexpected BED sigmoid configuration: {(source, target, ceiling, scale)}")
    values = (-64, -20, -5, -1, 0, 1, 5, 20, 63) * 3 + (-2, 0, 2)
    quantized = tuple(v << scale[0] for v in values)
    m.extra_scratch += ((source, source + 4 * len(values)),)
    m.scratch.update(enumerate(pack("<30i", *quantized), source))
    m.reg[1] = m.read(source, 4)
    start = len(m.scratch_trace)
    boundary = m.run(4096, stop_at=0x52001e5a)
    if boundary != {"pc": 0x52001e5a, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED sigmoid exit: {boundary}")
    next_stop = m.run(16)
    return {"input_q25": quantized, "exponents": scale, "ceiling": ceiling, "boundary": boundary,
            "writes": m.scratch_trace[start:], "next_stop": next_stop,
            "state_fixtures": trace_fifth_sigmoid_states(m, next_stop, dictionary, weight_base, second_binding)}


def trace_fifth_sigmoid_states(snapshot, stop, dictionary, weight_base, second_binding):
    if stop != {"pc": 0x52001e5c, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED sigmoid status read: {stop}")
    m = deepcopy(snapshot)
    m.fixture_reads[0x481cc] = deque((1,))
    stopped = m.run(8192)

    m = deepcopy(snapshot)
    m.fixture_reads[0x481cc] = deque((0,))
    begin = len(m.dma_submissions)
    wait = m.run(8192)
    if wait != {"pc": 0x520010cc, "reason": "wfi"}:
        raise ValueError(f"unexpected BED sigmoid downstream program wait: {wait}")
    (request,) = m.dma_submissions[begin:]
    (source, target, a, b), = request["rows"]
    slots = plane_slots(a, b)
    if target != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED sigmoid downstream program: {request}")
    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    bound = program(image, source - BLOCK_VA, slots)
    m.pc += 2
    wait = m.run(8192)
    if wait != {"pc": 0x52001ea4, "reason": "wfi"}:
        raise ValueError(f"unexpected BED sigmoid downstream worker wait: {wait}")
    m.pc += 2
    launch = m.run(8192)
    if launch != {"pc": 0x520023cc, "reason": "unsupported opcode", "opcode": 10}:
        raise ValueError(f"unexpected BED sigmoid downstream compute: {launch}")
    return {"state_one_stop": stopped, "state_zero_request": request,
            "member_offset": bound.offset + 0x100, "slots": bound.slots,
            "next_compute": launch, "arguments": tuple(m.reg[4:8]),
            "following_projection": trace_fifth_after_sigmoid(m, dictionary, weight_base, second_binding)}


def trace_fifth_after_sigmoid(m, dictionary, weight_base, second_binding):
    """Advance scalar control independently of the Q16 consumer's output."""
    def expect(expected):
        stop = m.run(8192)
        if stop != expected:
            raise ValueError(f"unexpected BED post-sigmoid stop: {stop}, expected {expected}")
        return stop

    if m.link != 0x52001ec6:
        raise ValueError(f"unexpected BED post-sigmoid return: {m.link:#x}")
    m.pc = m.link
    expect({"pc": 0x52001ecc, "reason": "wfi"})
    m.pc += 2
    state = expect({"pc": 0x52001ede, "reason": "unmodeled read", "address": 0x481cc, "size": 4})
    m.fixture_reads[0x481cc] = deque((0,))
    begin = len(m.dma_submissions)
    expect({"pc": 0x520010cc, "reason": "wfi"})
    (reload,) = m.dma_submissions[begin:]
    (source, target, a, b), = reload["rows"]
    slots = plane_slots(a, b)
    if target != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED post-sigmoid program: {reload}")
    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    overlay = program(image, source - BLOCK_VA, slots)
    m.pc += 2
    ready = expect({"pc": 0x520021a4, "reason": "unmodeled read", "address": 0x48230, "size": 4})
    m.fixture_reads[0x48230] = deque((0,))
    begin = len(m.dma_submissions)
    expect({"pc": 0x520021cc, "reason": "wfi"})
    (request,) = m.dma_submissions[begin:]
    targets = ((0x10080, 24), (0xffc0, 192), (0x10800, 192), (0x11000, 192), (0x11800, 192))
    offsets = [bind_linear_parameter(m, row, dictionary, weight_base, target, size)
               for row, (target, size) in zip(request["rows"], targets, strict=True)]
    m.pc += 2
    expect({"pc": 0x52002244, "reason": "wfi"})
    m.pc += 2
    calls = []
    begin = len(m.scratch_reads)
    for _ in range(3):
        expect({"pc": 0x5200060c, "reason": "unsupported opcode", "opcode": 10})
        calls.append(tuple(m.reg[4:8]))
        m.pc = m.link
    expect({"pc": 0x520022b8, "reason": "wfi"})
    m.pc += 2
    expect({"pc": 0x520022ca, "reason": "unmodeled read", "address": 0x481cc, "size": 4})
    m.fixture_reads[0x481cc] = deque((0,))
    expect({"pc": 0x520022d8, "reason": "wfi"})
    m.pc += 2
    stop = expect({"pc": 0x520007d2, "reason": "unmodeled read", "address": 0x48100, "size": 4})
    m.fixture_reads.update({0x48100: deque((0x20000000,)), 0x48160: deque((BLOCK_VA,))})
    directory = unpack_from("<I", image, 0x1c)[0]
    entry = BLOCK_VA + unpack_from("<I", image, directory + 5 * 4)[0]
    branch = m.run(8192, stop_at=entry)
    if branch != {"pc": entry, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED directory[5] handoff: {branch}")
    begin_dma = len(m.dma_submissions)
    expect({"pc": entry + 0x14c, "reason": "wfi"})
    (startup,) = m.dma_submissions[begin_dma:]
    (row,) = startup["rows"]
    startup_offset = bind_linear_parameter(m, row, image, BLOCK_VA, 0x18, 160)
    m.pc += 2
    startup_stop = m.run(8192)
    return {"state": state, "ready": ready, "overlay_request": reload,
            "member_offset": overlay.offset + 0x100, "slots": overlay.slots,
            "parameter_request": request, "dictionary_offsets": offsets, "calls": calls,
            "coefficient_reads": [r for r in m.scratch_reads[begin:] if r["pc"] == 0x52000c82],
            "stop": stop, "next_branch": branch, "next_member_offset": entry - BLOCK_VA + 0x100,
            "startup_request": startup, "startup_member_offset": startup_offset + 0x100,
            "startup_stop": startup_stop,
            "sixth_fixture": trace_sixth_boot(m, entry, startup_stop, image, dictionary, weight_base, second_binding)}


def trace_sixth_boot(m, entry, stop, image, dictionary, weight_base, second_binding):
    if stop != {"pc": entry + 0x284, "reason": "wfi"}:
        raise ValueError(f"unexpected BED sixth-block external code wait: {stop}")
    block = entry - BLOCK_VA
    start, end = (block + unpack_from("<I", image, block + offset)[0] for offset in (0x30, 0x10))
    writes = {w["address"]: w["value"] for w in m.mmio_writes if entry <= w["pc"] < stop["pc"]}
    target, last = (writes[a] for a in (0x49d04, 0x49d08))
    if not block < start < end <= len(image) or end - start != last - target + 1:
        raise ValueError("BED sixth-block external code extent disagrees with bootstrap")
    m.pc += 2
    jump = m.run(512, stop_at=target)
    if jump != {"pc": target, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED sixth-block external entry: {jump}")
    code = image[start:end]
    m.regions += ((target, code),)
    m.stack_high = pc_literal(lambda a, n: code[a - target:a - target + n], target, 15)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    m.dma_pointer = None
    layout, dictionary2 = second_binding
    loads = []
    for pc, dst, size in ((0x528006ec, 0x17e00, 76), (0x52800738, 0x17e4c, 260)):
        begin = len(m.dma_submissions)
        wait = m.run(8192)
        if wait != {"pc": pc, "reason": "wfi"}:
            raise ValueError(f"unexpected BED sixth-block parameter wait: {wait}")
        (request,) = m.dma_submissions[begin:]
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary2, weight_base + layout["base_offset"], dst, size)
        loads.append({"request": request, "offset": offset, "bytes": size})
        m.pc += 2
    catalog = writes[0x48168]
    stop = m.run(8192)
    return {"entry": jump, "external_transfer": (writes[0x49d0c], target, len(code)),
            "source_member_offset": start + 0x100, "parameter_loads": loads,
            "catalog_pointer": catalog, "stop": stop,
            "projection_fixture": trace_sixth_projection(m, stop, catalog, image, dictionary, weight_base,
                                                         second_binding)}


def trace_sixth_projection(m, stop, catalog, image, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED sixth-block projection stop: {result}")

    if stop != {"pc": 0x528007ae, "reason": "unmodeled read", "address": 0x48168, "size": 4}:
        raise ValueError(f"unexpected BED sixth-block catalog read: {stop}")
    m.fixture_reads[0x48168] = deque((catalog,) * 6)
    begin = len(m.dma_submissions)
    expect(0x52800808, "wfi")
    (overlay_request,) = m.dma_submissions[begin:]
    (src, dst, a, b), = overlay_request["rows"]
    slots = plane_slots(a, b)
    if slots is None or dst != 0x0400f000:
        raise ValueError(f"unexpected BED sixth-block program: {overlay_request}")
    overlay = program(image, src - BLOCK_VA, slots)
    m.pc += 2
    expect(0x52800be0, "unmodeled read", address=0x481cc, size=4)
    m.fixture_reads[0x481cc] = deque((0,))
    expect(0x52800d5e, "unmodeled read", address=0x48230, size=4)
    m.fixture_reads[0x48230] = deque((0,) * 3)
    begin = len(m.dma_submissions)
    expect(0x52800ea0, "wfi")
    batches = m.dma_submissions[begin:]
    if len(batches) != 2:
        raise ValueError(f"unexpected BED sixth-block DMA batches: {batches}")
    offsets = []
    for batch, targets in zip(batches, (
            (0xebd0, 0x4120, 0x4920, 0x5120, 0x5920, 0xeaa0, 0xf120),
            (0xebc0, 0x4000, 0x4800, 0x5000, 0x5800, 0xe980, 0xf000)), strict=True):
        offsets.append([bind_linear_parameter(m, row, dictionary, weight_base, dst, 12 if i == 0 else 288)
                        for i, (row, dst) in enumerate(zip(batch["rows"], targets, strict=True))])
    m.pc += 2
    begin = len(m.scratch_reads)
    compute = m.run(8192)
    if compute["reason"] != "unsupported opcode" or compute.get("opcode") != 10:
        raise ValueError(f"unexpected BED sixth-block first compute: {compute}")
    return {"overlay_request": overlay_request, "member_offset": overlay.offset + 0x100, "slots": overlay.slots,
            "parameter_requests": batches, "dictionary_offsets": offsets,
            "reads": [r for r in m.scratch_reads[begin:] if 0xebd0 <= r["address"] < 0xebdc],
            "compute": compute, "arguments": tuple(m.reg[4:8]),
            "tiles": trace_sixth_tiles(m, compute, image, dictionary, weight_base, second_binding)}


def trace_sixth_tiles(m, compute, image, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED sixth-block tile stop: {result}")
        return result

    launch = {"pc": 0x528003b4, "reason": "unsupported opcode", "opcode": 10}
    if compute != launch:
        raise ValueError(f"unexpected BED sixth-block tile launch: {compute}")
    tiles = []
    feature_base, tile_bytes = 0x2000e440, 0x24c0
    for i in range(5):
        if m.link != 0x528009ba:
            raise ValueError("unexpected BED sixth-block first tile return")
        m.pc = m.link
        expect(0x52800a1c, "wfi")
        m.pc += 2
        before = len(m.mmio_writes)
        expect(**launch)
        args = tuple(m.reg[4:8])
        aperture = tuple((w["address"], w["value"]) for w in m.mmio_writes[before:]
                         if 0x40320 <= w["address"] <= 0x40330)
        if m.link != 0x52800a52:
            raise ValueError("unexpected BED sixth-block second tile return")
        m.pc = m.link
        expect(0x52800f18, "wfi")
        m.pc += 2
        expect(0x52800f78, "unmodeled read", address=0x481cc, size=4)
        m.fixture_reads[0x481cc] = deque((0,))
        expect(0x52800fd6, "unmodeled read", address=0x48230, size=4)
        m.fixture_reads[0x48230] = deque((0,))
        before = len(m.dma_submissions)
        if i < 4:
            expect(0x52800ea0, "wfi")
        else:
            expect(0x52801004, "unmodeled read", address=0x48230, size=4)
        (request,) = m.dma_submissions[before:]
        feature, *parameters = request["rows"]
        if feature != [feature_base + i * tile_bytes, args[3] | 0x80000000, (tile_bytes - 1) << 16,
                       (tile_bytes - 1) << 16]:
            raise ValueError(f"unexpected BED sixth-block tile feature row: {feature}")
        targets = ((0xebd0, 0x4120, 0x4920, 0x5120, 0x5920, 0xeaa0, 0xf120) if i % 2 == 0 else
                   (0xebc0, 0x4000, 0x4800, 0x5000, 0x5800, 0xe980, 0xf000))
        alias = 0x40000 if i >= 3 else 0
        offsets = [bind_linear_parameter(m, row, dictionary, weight_base, dst | alias, 12 if j == 0 else 288)
                   for j, (row, dst) in enumerate(zip(parameters, targets, strict=True))]
        tiles.append({"arguments": args, "aperture": aperture, "request": request,
                      "dictionary_offsets": offsets})
        if i < 4:
            m.pc += 2
            expect(**launch)
    m.fixture_reads[0x48230] = deque((0,))
    expect(0x5280101c, "unmodeled read", address=0x481cc, size=4)
    m.fixture_reads[0x481cc] = deque((0,))
    before = len(m.dma_submissions)
    expect(0x52800b58, "wfi")
    (reload,) = m.dma_submissions[before:]
    (src, dst, a, b), = reload["rows"]
    slots = plane_slots(a, b)
    if dst != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED sixth-block final program: {reload}")
    overlay = program(image, src - BLOCK_VA, slots)
    m.pc += 2
    expect(0x52801094, "unmodeled read", address=0x48230, size=4)
    m.fixture_reads[0x48230] = deque((0,))
    before = len(m.dma_submissions)
    expect(0x528010bc, "wfi")
    (gather,) = m.dma_submissions[before:]
    if gather["rows"] != [[feature_base + offset, 0xa000 + offset, (size - 1) << 16, (size - 1) << 16]
                          for offset, size in ((0, 0x4000), (0x4000, 0x4000), (0x8000, 5 * tile_bytes - 0x8000))]:
        raise ValueError(f"unexpected BED sixth-block merged feature span: {gather}")
    m.pc += 2
    expect(0x5280113c, "wfi")
    m.pc += 2
    final = expect(0x52801bb0, "unsupported opcode", opcode=10)
    args = tuple(m.reg[4:8])
    if m.link != 0x5280115c:
        raise ValueError("unexpected BED sixth-block merged compute return")
    m.pc = m.link
    expect(0x52801160, "wfi")
    m.pc += 2
    expect(0x52801170, "unmodeled read", address=0x481cc, size=4)
    m.fixture_reads[0x481cc] = deque((0,))
    stop = expect(0x5280117c, "unmodeled read", address=0x157c0, size=2)
    return {"iterations": tiles, "final_overlay": reload, "member_offset": overlay.offset + 0x100,
            "slots": overlay.slots, "merged_request": gather, "final_compute": final,
            "arguments": args, "stop": stop, "copy_registers": tuple(m.reg[r] for r in (7, 2, 3)),
            "scalar_fixtures": trace_sixth_scalar_head(m, dictionary, weight_base, second_binding)}


def trace_sixth_scalar_head(snapshot, dictionary, weight_base, second_binding):
    source, target, end = (snapshot.reg[r] for r in (7, 2, 3))
    if (source, target, end) != (0x157c0, 0xa018, 0xa054):
        raise ValueError("unexpected BED sixth-block scalar input range")
    # Opposite vectors distinguish rectification from zero output caused by a skipped worker.
    signs = [1 if v > 0 else -1 for v in unpack_from("<30h", dictionary, 0x44110)]
    (image,) = (data for base, data in snapshot.regions if base == BLOCK_VA)
    workers = {}
    for src, dst, a, b in startup_rows(image, 0x1ad00):
        if dst in (0x800, 0x1800, 0x2800, 0x8000, 0x9000):
            size = (a >> 16) + 1
            if a & 0xffff != 0x4000 or b != a & 0xffff0000:
                raise ValueError("unexpected BED sixth-block worker transfer")
            workers[dst] = image[src:src + size]
    paths = []
    for direction in (-1, 1):
        m = deepcopy(snapshot)

        def expect(pc, reason, **fields):
            stop = m.run(8192)
            if stop != {"pc": pc, "reason": reason, **fields}:
                raise ValueError(f"unexpected BED sixth-block scalar head stop: {stop}")
            return stop

        values = tuple(direction * 16 * sign for sign in signs)
        m.extra_scratch += ((source, source + 30 * 8), (0xa000, end))
        for i, value in enumerate(values):
            m.scratch.update(enumerate(pack("<h", value), source + 8 * i))
        read_start, write_start = len(m.scratch_reads), len(m.scratch_trace)
        expect(0x5280118c, "unmodeled read", address=0x481cc, size=4)
        copied = m.scratch_trace[write_start:]
        input_reads = m.scratch_reads[read_start:]
        m.fixture_reads[0x481cc] = deque((0,))
        expect(0x52801216, "unmodeled read", address=0x48230, size=4)
        m.fixture_reads[0x48230] = deque((0,))
        before = len(m.dma_submissions)
        expect(0x52801240, "wfi")
        (request,) = m.dma_submissions[before:]
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary, weight_base, 0xa058, 640)
        m.pc += 2
        expect(0x52801280, "wfi")
        m.pc += 2
        begin, read_begin = len(m.scratch_trace), len(m.scratch_reads)
        expect(0x528012ec, "wfi")
        writes = [r for r in m.scratch_trace[begin:] if r["pc"] == 0x528012d6]
        weights = [r["address"] for r in m.scratch_reads[read_begin:] if r["pc"] == 0x528012a6]
        paths.append({"input": values, "input_reads": input_reads, "copy_writes": copied,
                      "request": request, "dictionary_offset": offset, "coefficient_addresses": weights,
                      "projection_writes": writes,
                      "combined": trace_sixth_scalar_workers(m, workers, dictionary, weight_base, second_binding)})
    return paths


def trace_sixth_scalar_workers(m, workers, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED sixth-block merged scalar stop: {stop}")

    def run_workers(entries, addresses, size):
        results = []
        exits = (0xf9c, 0x1f9c, 0x2f90, 0x87a4, 0x97a8)
        for base, start, stop, expected in zip(
                (0x800, 0x1800, 0x2800, 0x8000, 0x9000), entries, exits, addresses, strict=True):
            path = deepcopy(m)
            path.regions = ((base, workers[base]),) + path.regions
            path.pc = start
            before = len(path.scratch_trace)
            boundary = path.run(8192, stop_at=stop)
            writes = path.scratch_trace[before:]
            if (boundary != {"pc": stop, "reason": "slice boundary"}
                    or tuple((w["address"], w["size"]) for w in writes) != tuple((a, size) for a in expected)):
                raise ValueError(f"unexpected BED sixth-block worker slice: {boundary}, {writes}")
            for w in writes:
                addr = w["address"]
                m.scratch.update((i, path.scratch[i]) for i in range(addr, addr + size))
            results.append({"entry": start, "exit": stop, "writes": writes})
        return results

    first = run_workers((0xfe0, 0x1fe4, 0x2fd8, 0x87ec, 0x97f0),
                        ((0xa004, 0xa006), (0xa008, 0xa00a), (0xa00c, 0xa00e), (0xa010,), (0xa012,)), 2)
    hidden = tuple(m.read(a, 2, sign=True) for a in range(0xa000, 0xa014, 2))
    m.pc += 2
    expect(0x528012fc, "unmodeled read", address=0x481cc, size=4)
    m.fixture_reads[0x481cc] = deque((0,))
    expect(0x528013ea, "unmodeled read", address=0x48230, size=4)
    m.fixture_reads[0x48230] = deque((0,))
    before = len(m.dma_submissions)
    expect(0x52801414, "wfi")
    (request,) = m.dma_submissions[before:]
    (row,) = request["rows"]
    offset = bind_linear_parameter(m, row, dictionary, weight_base, 0xa018, 720)
    m.pc += 2
    expect(0x52801448, "wfi")
    lo, hi = 0xa2e8, 0xa360
    m.extra_scratch += ((lo, hi),)
    second = run_workers((0xfa8, 0x1fa8, 0x2f9c, 0x87b0, 0x97b4),
                         (tuple(range(a, a + 20, 4)) for a in range(lo + 20, hi, 20)), 4)
    m.pc += 2
    expect(0x52801488, "wfi")
    logits = tuple(m.read(a, 4, sign=True) for a in range(lo, hi, 4))
    m.pc += 2
    expect(0x52801498, "unmodeled read", address=0x481cc, size=4)
    m.fixture_reads[0x481cc] = deque((0,))
    before = len(m.scratch_trace)
    boundary = m.run(8192, stop_at=0x528015b8)
    if boundary != {"pc": 0x528015b8, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED sixth-block sigmoid end: {boundary}")
    return {"first_workers": first, "hidden": hidden, "second_workers": second,
            "second_request": request, "second_dictionary_offset": offset, "logits": logits,
            "sigmoid_writes": m.scratch_trace[before:], "boundary": boundary,
            "q16_consumer_fixture": trace_sixth_q16_consumer(m, workers, dictionary, weight_base, second_binding)}


def trace_sixth_q16_consumer(m, workers, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED sixth-block Q16 consumer stop: {result}")
        return result

    def reply(pc, address):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((0,))

    reply(0x528015ba, 0x481cc)
    begin = len(m.dma_submissions)
    expect(0x52800b58, "wfi")
    (request,) = m.dma_submissions[begin:]
    (src, dst, a, b), = request["rows"]
    slots = plane_slots(a, b)
    if dst != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED Q16 consumer program: {request}")
    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    overlay = program(image, src - BLOCK_VA, slots)
    m.pc += 2
    begin = len(m.dma_submissions)
    for pc in (0x52801684, 0x52801742, 0x5280175c):
        reply(pc, 0x48230)
    expect(0x528017dc, "wfi")
    first = m.dma_submissions[begin:]

    launches = []
    batches = list(first)
    for i in range(7):
        mailbox = tuple(m.read(addr, 4) for addr in range(0x17ad8, 0x17af0, 4))
        jobs = trace_sixth_q16_workers(m, workers) if i == 0 else []
        m.pc += 2
        launch = expect(0x52801bb0, "unsupported opcode", opcode=10)
        launches.append({"stop": launch, "arguments": tuple(m.reg[4:8]), "workers": jobs,
                         "mailbox": mailbox})
        if m.link != 0x52801808:
            raise ValueError("unexpected BED Q16 consumer return")
        m.pc = m.link
        expect(0x52801814, "wfi")
        m.pc += 2
        reply(0x52801904, 0x48230)
        begin = len(m.dma_submissions)
        if i < 6:
            expect(0x528017dc, "wfi")
        else:
            reply(0x5280192c, 0x48230)
        (batch,) = m.dma_submissions[begin:]
        batches.append(batch)
    geometry = []
    for i, batch in enumerate(batches):
        source, target, a, b = batch["rows"][-1]
        pitch, width, rows = (a & 0xffff) * 8, (a >> 16) + 1, (b & 255) + 1
        if (pitch, width, rows, b) != (1568, 224, 30, 0xdf1d1d):
            raise ValueError(f"unexpected BED Q16 feature geometry: {batch}")
        base = 0x2000e440
        if i < 7 and source != base + i * width:
            raise ValueError(f"unexpected BED Q16 feature stripe: {batch}")
        geometry.append({"source": source, "target_raw": target,
                         "window": PackedRead(source - base, pitch, width, rows) if i < 7 else None})
    reply(0x52801944, 0x481cc)
    expect(0x52801958, "wfi")
    m.pc += 2
    stop = expect(0x52800564, "unmodeled read", address=0x48100, size=4)
    return {"member_offset": overlay.offset + 0x100, "slots": slots, "program_request": request,
            "launches": launches, "feature_batches": batches, "feature_geometry": geometry, "stop": stop,
            "seventh_fixture": trace_seventh_boot(m, image, dictionary, weight_base, second_binding)}


def trace_seventh_boot(m, image, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block stop: {result}")
        return result

    def receive(pc):
        begin = len(m.dma_submissions)
        expect(pc, "wfi")
        (request,) = m.dma_submissions[begin:]
        m.pc += 2
        return request

    def reply(pc, address, values=(0,)):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque(values)

    m.fixture_reads.update({0x48100: deque((0x20000000,)), 0x48160: deque((BLOCK_VA,))})
    directory = unpack_from("<I", image, 0x1c)[0]
    block = unpack_from("<I", image, directory + 6 * 4)[0]
    entry = BLOCK_VA + block
    jump = m.run(8192, stop_at=entry)
    if jump != {"pc": entry, "reason": "slice boundary"} or m.link != 0x528005ca:
        raise ValueError(f"unexpected BED directory[6] handoff: {jump}")
    descriptors = receive(entry + 0x14c)
    (row,) = descriptors["rows"]
    table_offset = bind_linear_parameter(m, row, image, BLOCK_VA, 0x17818, 160)
    locals_request = receive(entry + 0x284)
    writes = {w["address"]: w["value"] for w in m.mmio_writes if entry <= w["pc"] < m.pc}
    target, last = (writes[a] for a in (0x49d04, 0x49d08))
    start, end = (block + unpack_from("<I", image, block + offset)[0] for offset in (0x30, 0x10))
    if not block < start < end <= len(image) or end - start != last - target + 1:
        raise ValueError("BED seventh-block external code extent disagrees with bootstrap")
    external = m.run(512, stop_at=target)
    if external != {"pc": target, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED seventh-block external entry: {external}")
    code = image[start:end]
    m.regions += ((target, code),)
    m.stack_high = pc_literal(lambda a, n: code[a - target:a - target + n], target, 15)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    m.dma_pointer = None
    first_write = expect(0x53000f5c, "unmodeled write", address=0x5240, size=4, value=0)
    m.extra_scratch += ((0x5080, 0x5600),)

    layout, dictionary2 = second_binding
    loads = []
    for pc, dst, size in ((0x53001058, 0x17b00, 232), (0x530010a4, 0x17be8, 720)):
        request = receive(pc)
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary2, weight_base + layout["base_offset"], dst, size)
        loads.append({"request": request, "offset": offset, "bytes": size})
    reply(0x53001ba6, 0x481cc)
    reply(0x5300199c, 0x48168, (writes[0x48168],) * 2)
    request = receive(0x530019f8)
    (src, dst, a, b), = request["rows"]
    slots = plane_slots(a, b)
    if dst != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED seventh-block program: {request}")
    overlay = program(image, src - BLOCK_VA, slots)
    reply(0x53001c1c, 0x48230)
    features = receive(0x53001c50)
    expect(0x53001c8c, "wfi")
    m.pc += 2
    compute = expect(0x53004044, "unsupported opcode", opcode=10)
    return {"entry": jump, "member_offset": block + 0x100, "descriptor_request": descriptors,
            "descriptor_offset": table_offset + 0x100, "local_requests": locals_request,
            "external": external, "external_member_offset": start + 0x100,
            "external_transfer": (writes[0x49d0c], target, len(code)), "first_write": first_write,
            "parameter_loads": loads, "program_request": request, "program_offset": overlay.offset + 0x100,
            "slots": slots, "feature_request": features, "compute": compute, "arguments": tuple(m.reg[4:8]),
            "projection_fixture": trace_seventh_projection(m, image, dictionary, weight_base, writes[0x48168],
                                                           second_binding)}


def trace_seventh_projection(m, image, dictionary, weight_base, catalog, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block projection stop: {result}")
        return result

    def receive(pc):
        begin = len(m.dma_submissions)
        expect(pc, "wfi")
        (request,) = m.dma_submissions[begin:]
        m.pc += 2
        return request

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    if m.link != 0x53001d02:
        raise ValueError("unexpected BED seventh-block initial compute return")
    m.pc = m.link
    expect(0x53001d0c, "wfi")
    m.pc += 2
    reply(0x53001d5a, 0x481cc)
    reply(0x53001a4a, 0x48168, catalog)
    reply(0x53001a4e, 0x48168, catalog)
    reload = receive(0x53001aa4)
    (src, dst, a, b), = reload["rows"]
    slots = plane_slots(a, b)
    if dst != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED seventh-block projection program: {reload}")
    overlay = program(image, src - BLOCK_VA, slots)
    reply(0x53001e5c, 0x48230)
    request = receive(0x53001ef8)
    targets = (0xfb88, 0x5800, 0x6000, 0x6800, 0x10808, 0xfa68, 0x10000)
    offsets = [bind_linear_parameter(m, row, dictionary, weight_base, addr, 36 if i == 0 else 288)
               for i, (row, addr) in enumerate(zip(request["rows"], targets, strict=True))]
    expect(0x53001f7c, "wfi")
    m.pc += 2
    reads = len(m.scratch_reads)
    calls = []
    for i in range(4):
        expect(0x53000c38, "unsupported opcode", opcode=10)
        calls.append(tuple(m.reg[4:8]))
        if m.link != (0x53001854 if i else 0x530017a6):
            raise ValueError("unexpected BED seventh-block projection return")
        m.pc = m.link
        if i < 3:
            expect(0x53001830, "wfi")
            m.pc += 2
    scalar_parameters = [r for r in m.scratch_reads[reads:] if r["pc"] == 0x5300178c]
    expect(0x53002000, "wfi")
    m.pc += 2
    reply(0x5300204a, 0x481cc)
    copies = [(receive(0x53001314), receive(0x53001398)) for _ in range(18)]
    reply(0x53002110, 0x481cc)
    reply(0x5300225e, 0x48230)
    mixed = receive(0x53002290)
    targets = (0x160c8, 0x5800, 0x6000, 0x6800, 0x17008, 0x15e28, 0x16800)
    next_offsets = [bind_linear_parameter(m, row, dictionary, weight_base, addr, 84 if i == 0 else 672)
                    for i, (row, addr) in enumerate(zip(mixed["rows"][:7], targets, strict=True))]
    for i, (src, dst, a, b) in enumerate(mixed["rows"][7:]):
        if (src, dst, a, b) != (0x2000a0c0 + i * 0x4d0, 0x60c0a8 + i * 0x540, 0x5f000b, 0x5f0d0d):
            raise ValueError("unexpected BED seventh-block mixed feature DMA row")
    return {"program": reload, "member_offset": overlay.offset + 0x100, "slots": slots,
            "parameter_request": request, "dictionary_offsets": offsets, "calls": calls,
            "scalar_parameters": scalar_parameters, "copy_requests": copies,
            "mixed_request": mixed, "mixed_dictionary_offsets": next_offsets,
            "late_fixture": trace_seventh_late(m, image, dictionary, weight_base, catalog, second_binding)}


def trace_seventh_late(m, image, dictionary, weight_base, catalog, second_binding):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block late stop: {result}")
        return result

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def receive(pc):
        begin = len(m.dma_submissions)
        wait(pc)
        (request,) = m.dma_submissions[begin:]
        return request

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    def calls(count, setup_return, loop_return, wait_pc, wait_count):
        results = []
        for i in range(count):
            expect(0x53000c38, "unsupported opcode", opcode=10)
            results.append(tuple(m.reg[4:8]))
            if m.link != (loop_return if i else setup_return):
                raise ValueError("unexpected BED seventh-block late compute return")
            m.pc = m.link
            if i < wait_count:
                wait(wait_pc)
        return results

    wait(0x5300237c)
    reads = len(m.scratch_reads)
    mixed_calls = calls(8, 0x530017a6, 0x53001854, 0x53001830, 3)
    mixed_parameters = [r for r in m.scratch_reads[reads:] if r["pc"] == 0x5300178c]
    wait(0x530023f4)
    reply(0x5300243a, 0x481cc)
    copies = [(receive(0x53001314), receive(0x53001398)) for _ in range(42)]
    reply(0x5300246c, 0x481cc)
    reply(0x530025d8, 0x48230)
    request = receive(0x5300260c)
    targets = (0x15ed0, 0x5800, 0x6000, 0x6800, 0x16808, 0x15e28, 0x16000)
    offsets = [bind_linear_parameter(m, row, dictionary, weight_base, dst, 84 if i == 0 else 168)
               for i, (row, dst) in enumerate(zip(request["rows"], targets, strict=True))]
    wait(0x53002694)
    reads = len(m.scratch_reads)
    last_calls = calls(8, 0x5300151a, 0x530015b2, 0x5300157c, 7)
    last_parameters = [r for r in m.scratch_reads[reads:] if r["pc"] == 0x5300145c]
    wait(0x53002778)
    reply(0x5300278a, 0x481cc)
    overlays = []
    for target in range(0x04000000, 0x04006000, 0x1000):
        reply(0x530018f4, 0x48168, catalog)
        reply(0x530018f8, 0x48168, catalog)
        load = receive(0x53001958)
        (src, dst, a, b), = load["rows"]
        slots = plane_slots(a, b)
        if dst != target or slots is None:
            raise ValueError(f"unexpected BED seventh-block final program: {load}")
        overlay = program(image, src - BLOCK_VA, slots)
        overlays.append({"request": load, "member_offset": overlay.offset + 0x100, "slots": slots})
    wait(0x530027f0)
    compute = expect(0x53004044, "unsupported opcode", opcode=10)
    arguments = tuple(m.reg[4:8])
    if m.link != 0x53002812:
        raise ValueError("unexpected BED seventh-block final compute return")
    m.pc = m.link
    wait(0x5300281c)
    reply(0x5300282c, 0x481cc)
    stop = expect(0x53002842, "unmodeled read", address=0x10aa0, size=2)
    return {"mixed_calls": mixed_calls, "mixed_parameters": mixed_parameters, "copy_requests": copies,
            "late_request": request, "dictionary_offsets": offsets, "late_calls": last_calls,
            "late_parameters": last_parameters, "overlays": overlays, "compute": compute,
            "arguments": arguments, "stop": stop, "copy_registers": tuple(m.reg[i] for i in (7, 2, 3)),
            "head_fixture": trace_seventh_head(m, dictionary, weight_base, second_binding)}


def trace_seventh_head(snapshot, dictionary, weight_base, second_binding):
    m = deepcopy(snapshot)

    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block head stop: {result}")
        return result

    def reply(pc, address):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((0,))

    source, target, end = (m.reg[i] for i in (7, 2, 3))
    if (source, target, end) != (0x10aa0, 0x10a48, 0x10a9c):
        raise ValueError("unexpected BED seventh-block halfword ranges")
    values = tuple(range(-21, 21))
    m.extra_scratch += ((source, source + 8 * len(values)), (target, end))
    for i, value in enumerate(values):
        m.scratch.update(enumerate(pack("<h", value), source + 8 * i))
    before = len(m.scratch_trace)
    reply(0x53002852, 0x481cc)
    copy_writes = m.scratch_trace[before:]
    calls = []
    for ready, wait, dst, size, ret, status in (
            (0x53002902, 0x53002934, 0x10aa0, 1232, 0x53002964, 0x53002968),
            (0x53002a9a, 0x53002acc, 0x10a48, 1344, 0x53002aea, 0x53002aec)):
        reply(ready, 0x48230)
        before = len(m.dma_submissions)
        expect(wait, "wfi")
        (request,) = m.dma_submissions[before:]
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary, weight_base, dst, size)
        m.pc += 2
        expect(0x53004044, "unsupported opcode", opcode=10)
        calls.append({"request": request, "offset": offset, "size": size, "arguments": tuple(m.reg[4:8])})
        if m.link != ret:
            raise ValueError("unexpected BED seventh-block head compute return")
        m.pc = m.link
        reply(status, 0x481cc)

    stop = expect(0x53002b1c, "unmodeled read", address=0x10f88, size=4)
    exponent = m.read(m.reg[0], 1, sign=True)
    if exponent != 25:
        raise ValueError(f"unexpected BED seventh-block head input exponent: {exponent}")
    scores = deepcopy(m)
    logits = tuple(v << exponent for v in (-2, -1, 0, 1, 2, 5, -5)) * 6
    fp32_base = scores.read(scores.reg[15] + 0x14, 4)
    scores.extra_scratch += ((0x10f88, 0x11030), (0x10a28, 0x10b78),
                             (fp32_base, fp32_base + 4 * len(logits)))
    scores.scratch.update(enumerate(pack("<42i", *logits), 0x10f88))
    before = len(scores.scratch_trace)
    fp32_stop = scores.run(8192)
    if fp32_stop != {"pc": 0x53002bae, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED seventh-block float write boundary: {fp32_stop}")
    fp32_writes = scores.scratch_trace[before:]
    scores.fixture_reads[0x481cc] = deque((0,))
    before = len(scores.scratch_trace)
    boundary = scores.run(8192, stop_at=0x53002bfe)
    if boundary != {"pc": 0x53002bfe, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED seventh-block sigmoid boundary: {boundary}")
    return {"copy_input": values, "copy_writes": copy_writes, "calls": calls, "unseeded_stop": stop,
            "sigmoid_fixture": {"input": logits, "exponent": exponent, "boundary": boundary,
                                "fp32_writes": fp32_writes, "writes": scores.scratch_trace[before:],
                                "consumer_fixture": trace_seventh_q16_consumer(
                                    scores, dictionary, weight_base, second_binding)}}


def trace_seventh_q16_consumer(m, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block Q16 trial stop: {stop}")
        return stop

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    def receive(pc):
        start = len(m.dma_submissions)
        wait(pc)
        (request,) = m.dma_submissions[start:]
        return request

    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    catalog = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)

    def overlay(pc, target):
        request = receive(pc)
        (src, dst, a, b), = request["rows"]
        slots = plane_slots(a, b)
        if dst != target or slots is None:
            raise ValueError(f"unexpected BED seventh-block Q16 program: {request}")
        code = program(image, src - BLOCK_VA, slots)
        return {"request": request, "member_offset": code.offset + 0x100, "slots": code.slots}

    def bind_batch(pc, targets, coefficient_bytes):
        request = receive(pc)
        offsets = [bind_linear_parameter(m, row, dictionary, weight_base, dst, coefficient_bytes if i else 84)
                   for i, (row, dst) in enumerate(zip(request["rows"], targets, strict=True))]
        return {"request": request, "offsets": offsets}

    def projection(setup_ret, loop_ret, wait_pc, waits, param_pc):
        start = len(m.scratch_reads)
        calls = []
        for i in range(8):
            expect(0x53000c38, "unsupported opcode", opcode=10)
            calls.append(tuple(m.reg[4:8]))
            if m.link != (loop_ret if i else setup_ret):
                raise ValueError("unexpected BED seventh-block Q16 projection return")
            m.pc = m.link
            if i < waits:
                wait(wait_pc)
        return calls, [r for r in m.scratch_reads[start:] if r["pc"] == param_pc]

    reply(0x53002c00, 0x481cc)
    reply(0x5300199c, 0x48168, catalog)
    reply(0x530019a0, 0x48168, catalog)
    activation_program = overlay(0x530019f8, 0x0400f000)
    wait(0x53002c58)
    consumers = []
    for ret in (0x53002c7a, 0x53002d18, 0x53002d30):
        expect(0x53004044, "unsupported opcode", opcode=10)
        if m.link != ret:
            raise ValueError("unexpected BED seventh-block Q16 consumer return")
        consumers.append({"arguments": tuple(m.reg[4:8]),
                          "stack_arguments": tuple(m.read(m.reg[15] + i, 4) for i in (0, 4, 8))})
        m.pc = m.link
    wait(0x53002d3c)
    reply(0x53002d88, 0x481cc)
    reply(0x53001a4a, 0x48168, catalog)
    reply(0x53001a4e, 0x48168, catalog)
    projection_program = overlay(0x53001aa4, 0x0400f000)
    reply(0x53002e9e, 0x48230)
    first = bind_batch(0x53002f38, (0x15aa8, 0x5800, 0x6000, 0x6800, 0x16808, 0x15808, 0x16000), 672)
    wait(0x53002fc0)
    first["calls"], first["parameters"] = projection(0x530017a6, 0x53001854, 0x53001830, 3, 0x5300178c)
    wait(0x53003044)
    reply(0x5300308a, 0x481cc)
    copies = [(receive(0x53001314), receive(0x53001398)) for _ in range(42)]
    reply(0x530030aa, 0x481cc)
    reply(0x530031ee, 0x48230)
    second = bind_batch(0x53003218, (0x15ed0, 0x5800, 0x6000, 0x6800, 0x16808, 0x15e28, 0x16000), 168)
    wait(0x53003300)
    second["calls"], second["parameters"] = projection(0x5300151a, 0x530015b2, 0x5300157c, 7, 0x5300145c)
    wait(0x5300337c)
    reply(0x5300338e, 0x481cc)
    tail = []
    for target in range(0x04000000, 0x04006000, 0x1000):
        reply(0x530018f4, 0x48168, catalog)
        reply(0x530018f8, 0x48168, catalog)
        tail.append(overlay(0x53001958, target))
    wait(0x530033ec)
    compute = expect(0x53004044, "unsupported opcode", opcode=10)
    arguments = tuple(m.reg[4:8])
    if m.link != 0x5300340e:
        raise ValueError("unexpected BED seventh-block next halfword compute return")
    m.pc = m.link
    wait(0x53003418)
    reply(0x53003428, 0x481cc)
    stop = expect(0x53003434, "unmodeled read", address=0xc120, size=2)
    return {"activation_program": activation_program, "consumers": consumers, "projection_program": projection_program,
            "first_projection": first, "copy_requests": copies, "second_projection": second, "tail_programs": tail,
            "compute": compute, "arguments": arguments, "unseeded_stop": stop,
            "copy_registers": tuple(m.reg[i] for i in (7, 2, 3)),
            "final_fixture": trace_seventh_final(m, image, dictionary, weight_base, catalog, second_binding)}


def trace_seventh_final(snapshot, image, dictionary, weight_base, catalog, second_binding):
    m = deepcopy(snapshot)

    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED seventh-block final stop: {result}")
        return result

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def receive(pc):
        before = len(m.dma_submissions)
        wait(pc)
        (request,) = m.dma_submissions[before:]
        return request

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    def launch(ret):
        expect(0x53004044, "unsupported opcode", opcode=10)
        if m.link != ret:
            raise ValueError("unexpected BED seventh-block final compute return")
        args = tuple(m.reg[4:8])
        m.pc = m.link
        return args

    values = tuple(range(-21, 21))
    m.extra_scratch += ((0xc120, 0xc270), (0xc0a8, 0xc11c))
    for i, v in enumerate(values):
        m.scratch.update(enumerate(pack("<h", v), 0xc120 + 8 * i))
    before = len(m.scratch_trace)
    reply(0x53003444, 0x481cc)
    copies = m.scratch_trace[before:]
    coefficient_loads = []
    for ready, waited, dst, size, ret, status in (
            (0x53003528, 0x53003550, 0xc120, 1232, 0x53003580, 0x53003584),
            (0x53003662, 0x5300368c, 0xc170, 1344, 0x530036aa, 0x530036ac)):
        reply(ready, 0x48230)
        request = receive(waited)
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary, weight_base, dst, size)
        coefficient_loads.append({"request": request, "offset": offset, "arguments": launch(ret)})
        reply(status, 0x481cc)

    # The skipped compute supersedes the packed halfwords in this overlapping range.
    source, end, target = 0xc0c8, 0xc170, 0xd308
    for addr in range(source, end):
        m.scratch.pop(addr, None)
    unseeded = expect(0x530036d6, "unmodeled read", address=source, size=4)
    exponents = tuple(m.read(m.reg[r], 1, sign=True) for r in (4, 10, 8))
    if exponents != (25, 25, -16):
        raise ValueError(f"unexpected BED seventh-block final exponents: {exponents}")
    values32 = tuple(v << exponents[0] for v in (-2, -1, 0, 1, 2, 5, -5)) * 6
    m.extra_scratch += ((source, end), (target, target + 8 * len(values32)))
    m.scratch.update(enumerate(pack("<42i", *values32), source))
    before = len(m.scratch_trace)
    reply(0x53003774, 0x481cc)
    scores = m.scratch_trace[before:]

    def overlay():
        reply(0x5300199c, 0x48168, catalog)
        reply(0x530019a0, 0x48168, catalog)
        request = receive(0x530019f8)
        (src, dst, a, b), = request["rows"]
        slots = plane_slots(a, b)
        if dst != 0x0400f000 or slots is None:
            raise ValueError(f"unexpected BED seventh-block final overlay: {request}")
        code = program(image, src - BLOCK_VA, slots)
        return {"request": request, "member_offset": code.offset + 0x100, "slots": slots}

    overlays = [overlay()]
    wait(0x530037c4)
    mailbox = tuple(m.read(a, 4) for a in range(0xc064, 0xc070, 4))
    consumers = [launch(0x530037f6) for _ in range(7)]
    wait(0x53003808)
    reply(0x53003818, 0x481cc)
    overlays.append(overlay())
    wait(0x530038cc)
    further = [launch(0x530038e8) for _ in range(7)]
    wait(0x530038f8)
    reply(0x530039d6, 0x48230)
    request = receive(0x53003a00)
    offsets = [bind_linear_parameter(m, row, dictionary, weight_base, row[1], 72 if i else 36)
               for i, row in enumerate(request["rows"])]
    wait(0x53003af8)
    last = launch(0x53003b2c)
    wait(0x53003b34)
    reply(0x53003b42, 0x481cc)
    wait(0x53003b50)
    stop = expect(0x53000e32, "unmodeled read", address=0x48100, size=4)
    m.fixture_reads.update({0x48100: deque((0x20000000,)), 0x48160: deque((BLOCK_VA,))})
    directory = unpack_from("<I", image, 0x1c)[0]
    block = unpack_from("<I", image, directory + 7 * 4)[0]
    entry = BLOCK_VA + block
    branch = m.run(8192, stop_at=entry)
    if branch != {"pc": entry, "reason": "slice boundary"} or m.link != 0x53000e8e:
        raise ValueError(f"unexpected BED directory[7] handoff: {branch}")
    descriptors = receive(entry + 0x14c)
    (row,) = descriptors["rows"]
    descriptor_offset = bind_linear_parameter(m, row, image, BLOCK_VA, 0x18, 160)
    locals_request = receive(entry + 0x284)
    control = {w["address"]: w["value"] for w in m.mmio_writes if entry <= w["pc"] < m.pc}
    code_base, last_addr = (control[a] for a in (0x49d04, 0x49d08))
    lo, hi = (block + unpack_from("<I", image, block + at)[0] for at in (0x30, 0x10))
    if not block < lo < hi <= len(image) or hi - lo != last_addr - code_base + 1:
        raise ValueError("BED eighth-block external code extent disagrees with bootstrap")
    external = m.run(512, stop_at=code_base)
    if external != {"pc": code_base, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED eighth-block external entry: {external}")
    code = image[lo:hi]
    m.regions += ((code_base, code),)
    m.stack_high = pc_literal(lambda addr, size: code[addr - code_base:addr - code_base + size], code_base, 15)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    m.dma_pointer = None
    first_write = expect(0x53800ab4, "unmodeled write", address=0x3b74, size=4, value=0)
    return {"copy_writes": copies, "coefficient_loads": coefficient_loads, "unseeded_logits": unseeded,
            "input_q25": values32, "exponents": exponents, "q16_writes": scores, "overlays": overlays,
            "mailbox": mailbox, "consumers": consumers, "further_calls": further, "last_request": request,
            "last_offsets": offsets, "last_call": last, "stop": stop, "next_entry": branch,
            "next_member_offset": block + 0x100, "descriptor_request": descriptors,
            "descriptor_member_offset": descriptor_offset + 0x100, "local_request": locals_request,
            "external_transfer": (control[0x49d0c], code_base, len(code)), "external_member_offset": lo + 0x100,
            "eighth_stop": first_write,
            "eighth_fixture": trace_eighth_first_projections(m, image, dictionary, weight_base,
                                                            second_binding, control[0x48168])}


def trace_eighth_first_projections(m, image, dictionary, weight_base, second_binding, catalog):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED eighth-block stop: {stop}")
        return stop

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def receive(pc):
        begin = len(m.dma_submissions)
        wait(pc)
        (request,) = m.dma_submissions[begin:]
        return request

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    work = m.reg[4], m.reg[2]
    if work != (0x3880, 0x43b0):
        raise ValueError(f"unexpected BED eighth-block clear range: {work}")
    m.extra_scratch += (work,)
    layout, dictionary2 = second_binding
    loads = []
    for pc, dst, size in ((0x53800ce4, 0xa000, 272), (0x53800d30, 0xa110, 1116)):
        request = receive(pc)
        (row,) = request["rows"]
        offset = bind_linear_parameter(m, row, dictionary2, weight_base + layout["base_offset"], dst, size)
        loads.append({"offset": offset, "bytes": size, "request": request})
    clear = [w["address"] for w in m.scratch_trace if w["pc"] == 0x53800ab4 and not w["value"]]
    if clear != list(range(*work, 4)):
        raise ValueError("BED eighth-block clear did not cover its original work range")
    reply(0x53800d98, 0x48168, catalog)
    reply(0x53800d9c, 0x48168, catalog)
    overlay_request = receive(0x53800df4)
    (src, dst, a, b), = overlay_request["rows"]
    slots = plane_slots(a, b)
    if dst != 0x0400f000 or slots is None:
        raise ValueError(f"unexpected BED eighth-block first program: {overlay_request}")
    overlay = program(image, src - BLOCK_VA, slots)
    initial_copies = [receive(pc) for pc in (0x53801af8, 0x53801b18, 0x53801b34)]
    reply(0x53801b3c, 0x481cc)
    groups = []
    for i, (ready, dma_wait, compute_wait, final_wait, status) in enumerate((
            (0x53801d08, 0x53801d30, 0x53801d78, 0x53801df0, 0x53801dfe),
            (0x53801fb2, 0x53801fe4, 0x53802068, 0x53802140, 0x53802150),
            (0x53802232, 0x538022c4, 0x53802348, 0x538023c0, 0x538023ce),
            (0x5380252a, 0x5380255c, 0x538025e0, 0x538026b8, 0x538026c8),
            (0x538027c6, 0x538027f8, 0x538028e4, 0x53802958, 0x53802968))):
        reply(ready, 0x48230)
        request = receive(dma_wait)
        targets = (0xab00, 0x4000, 0x4800, 0x5000, 0x5800, 0xaa00, 0xb000) if i else (0xaa00, 0xaa18)
        sizes = (24,) + (256,) * 6 if i else (24, 1536)
        offsets = [bind_linear_parameter(m, row, dictionary, weight_base, dst, size)
                   for row, dst, size in zip(request["rows"], targets, sizes, strict=True)]
        wait(compute_wait)
        begin = len(m.scratch_reads)
        calls = []
        for _ in range(2):
            expect(0x53800630, "unsupported opcode", opcode=10)
            if m.link != 0x53801678:
                raise ValueError("unexpected BED eighth-block first projection return")
            calls.append(tuple(m.reg[4:8]))
            m.pc = m.link
        parameters = [r for r in m.scratch_reads[begin:] if r["pc"] == 0x538015de]
        groups.append({"request": request, "offsets": offsets, "calls": calls, "parameters": parameters})
        wait(final_wait)
        reply(status, 0x481cc)
    following_copies = [receive(0x5380172c) for _ in range(5)]
    stop = expect(0x53802aee, "unmodeled read", address=0x48230, size=4)
    return {"work_range": work, "clear_words": len(clear), "parameter_loads": loads,
            "overlay_request": overlay_request, "overlay_offset": overlay.offset + 0x100, "slots": slots,
            "initial_copies": initial_copies, "projections": groups, "following_copies": following_copies,
            "stop": stop, "wide_fixture": trace_eighth_wide(m, dictionary, weight_base, second_binding)}


def dictionary_span(source, size, regions):
    """Resolve a transfer against known file bytes, without filling gaps."""
    cursor, end = source, source + size
    data, fragments = [], []
    if size <= 0:
        raise ValueError("BED dictionary transfer must have positive length")
    while cursor < end:
        owners = [(i, base, blob) for i, (base, blob) in enumerate(regions, 1)
                  if base <= cursor < base + len(blob)]
        if len(owners) != 1:
            raise ValueError(f"BED dictionary transfer has no unique source at {cursor:#x}")
        key, base, blob = owners[0]
        offset = cursor - base
        take = min(len(blob) - offset, end - cursor)
        if any(cursor < other < cursor + take for other, _ in regions):
            raise ValueError(f"BED dictionary transfer has overlapping sources at {cursor:#x}")
        fragments.append((key, offset, take))
        data.append(blob[offset:offset + take])
        cursor += take
    return b"".join(data), fragments


def trace_eighth_wide(m, dictionary, weight_base, second_binding):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED eighth-block wide projection stop: {stop}")
        return stop

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def receive(pc):
        begin = len(m.dma_submissions)
        wait(pc)
        (request,) = m.dma_submissions[begin:]
        return request

    def reply(pc, address):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((0,))

    layout, dictionary2 = second_binding
    regions = ((weight_base, dictionary), (weight_base + layout["base_offset"], dictionary2))
    m.fixture_reads[0x48230] = deque((0,))
    groups = []
    for i, (param_wait, worker_wait, end_wait, feature, result_addr) in enumerate((
            (0x53802b20, 0x53802b9c, 0x53802bdc, 0xcd10, 0xaa00),
            (0x53802d9c, 0x53802e14, 0x53802e4c, 0xc7d0, 0xacd0),
            (0x53802ff0, 0x53803068, 0x53803114, 0xc290, 0xafa0),
            (0x53803248, 0x5380332c, 0x53803364, 0xbd50, 0xb270),
            (0x53803504, 0x5380357c, 0x538035c0, 0xb810, 0xb540))):
        request = receive(param_wait)
        if len(request["rows"]) != 7:
            raise ValueError(f"unexpected BED eighth-block wide parameters: {request}")
        transfers = []
        for j, row in enumerate(request["rows"]):
            src, dst, *_ = row
            size = 4800 if j else 60
            data, fragments = dictionary_span(src, size, regions)
            bind_linear_parameter(m, row, data, src, dst, size)
            transfers.append({"fragments": fragments, "sha256": sha256(data).hexdigest()})
        wait(worker_wait)
        reads = len(m.scratch_reads)
        calls = []
        for j in range(5):
            expect(0x53800630, "unsupported opcode", opcode=10)
            args = tuple(m.reg[4:8])
            if m.link != 0x5380145a or args != (0x53800038, feature, 0x4000 + j * 0x3c0, result_addr + j * 0x18):
                raise ValueError(f"unexpected BED eighth-block wide compute launch: {args}")
            calls.append(args)
            m.pc = m.link
        param_addr = request["rows"][0][1]
        parameters = [r for r in m.scratch_reads[reads:] if param_addr <= r["address"] < param_addr + 10]
        wait(end_wait)
        if i in (1, 3):
            reply((0x53802f32, 0x538033e6)[i // 2], 0x481cc)
        copies = [receive(0x5380172c) for _ in range(6)] if i < 4 else []
        if i < 4:
            reply((0x53802d6a, 0x53802fc8, 0x53803220, 0x538034dc)[i], 0x48230)
        groups.append({"request": request, "transfers": transfers, "calls": calls,
                       "parameters": parameters, "copy_requests": copies})
    reply(0x538035d0, 0x481cc)
    wait(0x53803608)
    boundary = m.run(8192, stop_at=0x53803632)
    if boundary != {"pc": 0x53803632, "reason": "slice boundary"} or tuple(m.reg[4:7]) != (0xaa00, 0xaa02, 0xaa04):
        raise ValueError(f"unexpected BED eighth-block scalar reduction: {boundary}")
    # These nine values are produced by the skipped compute, not by prior loads at the same address.
    for address in range(0xaa00, 0xaa18, 8):
        for lane in range(6):
            m.scratch.pop(address + lane, None)
    stop = expect(0x53803632, "unmodeled read", address=0xaa00, size=2)
    ninth_resources = (dictionary2, weight_base + layout["base_offset"])
    return {"groups": groups, "scalar_inputs": tuple(m.reg[4:7]), "stop": stop,
            "reduction_fixture": trace_eighth_scalar_reduction(m, dictionary, weight_base, ninth_resources)}


def trace_eighth_scalar_reduction(snapshot, dictionary, weight_base, ninth_resources):
    m = deepcopy(snapshot)
    source, lo, hi = m.reg[4], m.reg[9], m.reg[12]
    config, shift_addr, multiplier = m.reg[8], m.reg[10], m.reg[11]
    shift = m.read(shift_addr, 1, sign=True)
    negative_bias, positive_bias, ceiling, floor = (m.read(config + off, 4, sign=True) for off in (12, 16, 20, 24))
    if (source, lo, hi, config, shift_addr) != (0xaa00, 0xb960, 0xba28, 0xa0bc, 0xa34e):
        raise ValueError("unexpected BED eighth-block scalar reduction ranges")
    values = (tuple(range(9)), tuple(range(-30, -21)), tuple(range(30, 39)),
              (32767,) * 9, (-32768,) * 9) * 5
    m.extra_scratch += ((source, source + 24 * len(values)), (lo, hi))
    seed_eighth_reduction(m, source, values)
    start, reads = len(m.scratch_trace), len(m.scratch_reads)
    wait = m.run(8192)
    if wait != {"pc": 0x538036f4, "reason": "wfi"}:
        raise ValueError(f"unexpected BED eighth-block reduction wait: {wait}")
    writes = [w for w in m.scratch_trace[start:] if w["pc"] == 0x53803676]
    inputs = [r for r in m.scratch_reads[reads:] if r["pc"] in (0x53803632, 0x53803634, 0x5380363c)]
    m.pc += 2
    status = m.run(8192)
    if status != {"pc": 0x53803704, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block reduction merge status: {status}")
    m.fixture_reads[0x481cc] = deque((0,))
    boundary = m.run(128, stop_at=0x53803710)
    first, target, end = (m.reg[i] for i in (3, 2, 7))
    if boundary != {"pc": 0x53803710, "reason": "slice boundary"} or (first, target, end) != (lo, 0xb830, 0xb95c):
        raise ValueError(f"unexpected BED eighth-block packed reduction range: {boundary}")
    # Only the controller's computed words may be read; the five workers have not run.
    for address in range(hi, first + 4 * (end - target), 8):
        m.scratch.pop(address, None)
        m.scratch.pop(address + 1, None)
    m.extra_scratch += ((target, end),)
    complete = deepcopy(m)
    start = len(m.scratch_trace)
    stop = m.run(2048)
    if stop != {"pc": 0x53803710, "reason": "unmodeled read", "address": hi, "size": 2}:
        raise ValueError(f"unexpected BED eighth-block worker reduction boundary: {stop}")
    return {"inputs": values, "reads": inputs, "config_address": config, "shift_address": shift_addr,
            "multiplier": multiplier, "shift": shift, "bias": (negative_bias, positive_bias),
            "limits": (floor, ceiling), "writes": writes, "merged_range": (first, target, end),
            "merge_writes": m.scratch_trace[start:], "worker_stop": stop,
            "workers_fixture": trace_eighth_worker_reduction(complete, values, dictionary, weight_base, ninth_resources)}


def seed_eighth_reduction(machine, source, values):
    for i, group in enumerate(values):
        for j, value in enumerate(group):
            addr = source + 24 * i + 8 * (j // 3) + 2 * (j % 3)
            machine.scratch.update(enumerate(pack("<h", value), addr))


def trace_eighth_worker_reduction(m, main_inputs, dictionary, weight_base, ninth_resources):
    (image,) = (data for addr, data in m.regions if addr == BLOCK_VA)
    mailbox = tuple(m.read(addr, 4) for addr in range(0xa99c, 0xa9ac, 4))
    if mailbox != (0xb960, 0xaa00, 0xa000, 0xa110):
        raise ValueError(f"unexpected BED eighth-block worker mailbox: {mailbox}")
    clips = {
        0x800: (0x13e8, 0x1460), 0x1800: (0x23e8, 0x2460), 0x2800: (0x33d8, 0x3450),
        0x8000: (0x8bec, 0x8c64), 0x9000: (0x9bec, 0x9c64),
    }
    values = tuple(tuple(i - 90 + j for j in range(9)) for i in range(25, 150))
    source, target = mailbox[1], mailbox[0]
    m.extra_scratch += ((target, target + 150 * 8),)
    jobs = []
    for src, base, a, b in startup_rows(image, 0x31400):
        if base not in clips:
            continue
        if a & 0xffff != 0x4000 or b != a & 0xffff0000:
            raise ValueError("unexpected BED eighth-block worker transfer")
        index = len(jobs) + 1
        entry, end = clips[base]
        code = image[src:src + 1 + (a >> 16)]
        worker = DmaTraceMachine(code, scratch_base=0xa000, scratch_size=0x56c)
        worker.pc, worker.regions = entry, ((base, code),)
        input_start = source + index * 25 * 24
        output_start = target + index * 25 * 8
        worker.extra_scratch = ((0xa99c, 0xa9ac), (input_start, input_start + 25 * 24),
                                (output_start, output_start + 25 * 8))
        for lo, hi in ((0xa000, 0xa56c), (0xa99c, 0xa9ac)):
            worker.scratch.update((p, m.scratch[p]) for p in range(lo, hi))
        seed_eighth_reduction(worker, input_start, values[(index - 1) * 25:index * 25])
        boundary = worker.run(8192, stop_at=end)
        writes = worker.scratch_trace
        if (boundary != {"pc": end, "reason": "slice boundary"}
                or [(w["address"], w["size"]) for w in writes] != [(output_start + j * 8, 2) for j in range(25)]):
            raise ValueError(f"unexpected BED eighth-block worker reduction at {entry:#x}: {boundary}")
        for w in writes:
            addr = w["address"]
            m.scratch.update((p, worker.scratch[p]) for p in (addr, addr + 1))
        reads = [r for r in worker.scratch_reads if input_start <= r["address"] < input_start + 25 * 24]
        jobs.append({"entry": entry, "boundary": boundary, "input_start": input_start, "reads": reads, "writes": writes})
    if len(jobs) != 5:
        raise ValueError("did not resolve all five BED eighth-block worker slices")
    start = len(m.scratch_trace)
    stop = m.run(2048, stop_at=0x5380371e)
    if stop != {"pc": 0x5380371e, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED complete reduction pack: {stop}")
    return {"mailbox": mailbox, "inputs": (*main_inputs, *values), "workers": jobs,
            "packed_writes": m.scratch_trace[start:], "boundary": stop,
            "consumer": trace_eighth_packed_consumer(m, image, dictionary, weight_base, ninth_resources)}


def trace_eighth_packed_consumer(m, image, dictionary, weight_base, ninth_resources):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED eighth-block packed consumer stop: {stop}")
        return stop

    def receive(pc):
        start = len(m.dma_submissions)
        expect(pc, "wfi")
        (request,) = m.dma_submissions[start:]
        m.pc += 2
        return request

    catalog = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)
    for pc, addr, value in ((0x53803720, 0x481cc, 0), (0x53801764, 0x48168, catalog),
                            (0x53801768, 0x48168, catalog)):
        expect(pc, "unmodeled read", address=addr, size=4)
        m.fixture_reads[addr] = deque((value,))
    overlay_request = receive(0x538017c0)
    (source, target, a, b), = overlay_request["rows"]
    slots = plane_slots(a, b)
    if slots is None or target != 0x0400f000:
        raise ValueError(f"unexpected BED eighth-block packed consumer program: {overlay_request}")
    overlay = program(image, source - BLOCK_VA, slots)
    expect(0x538037c6, "unmodeled read", address=0x48230, size=4)
    m.fixture_reads[0x48230] = deque((0,))
    coefficients = receive(0x538037f8)
    (row,) = coefficients["rows"]
    offset = bind_linear_parameter(m, row, dictionary, weight_base, 0xb960, 4560)
    expect(0x53803840, "wfi")
    mailbox = tuple(m.read(addr, 4) for addr in range(0xa9ac, 0xa9c0, 4))
    m.pc += 2
    compute = expect(0x53804bb8, "unsupported opcode", opcode=10)
    stack_args = tuple(m.read(m.reg[15] + offset, 4) for offset in (0, 4, 8))
    return {"overlay_offset": overlay.offset + 0x100, "slots": slots, "request": coefficients,
            "dictionary_offset": offset, "mailbox": mailbox, "compute": compute,
            "arguments": tuple(m.reg[4:8]), "stack_arguments": stack_args,
            "following_projection": trace_eighth_projection_output(m, dictionary, weight_base, ninth_resources)}


def trace_eighth_projection_output(m, dictionary, weight_base, ninth_resources):
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED eighth-block output projection stop: {stop}")
        return stop

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def reply(pc, addr):
        expect(pc, "unmodeled read", address=addr, size=4)
        m.fixture_reads[addr] = deque((0,))

    if m.link != 0x53803862:
        raise ValueError("unexpected BED eighth-block packed projection return")
    m.pc = m.link
    wait(0x53803868)
    reply(0x53803876, 0x481cc)
    reply(0x5380398e, 0x48230)
    start = len(m.dma_submissions)
    wait(0x538039c0)
    (request,) = m.dma_submissions[start:]
    (row,) = request["rows"]
    offset = bind_linear_parameter(m, row, dictionary, weight_base, 0xba88, 4800)
    wait(0x538039f8)
    mailbox = tuple(m.read(a, 4) for a in range(0xa9c0, 0xa9cc, 4))
    compute = expect(0x53804bb8, "unsupported opcode", opcode=10)
    args = tuple(m.reg[4:8])
    if m.link != 0x53803a18:
        raise ValueError("unexpected BED eighth-block output projection return")
    m.pc = m.link
    wait(0x53803a20)
    reply(0x53803a2e, 0x481cc)
    boundary = m.run(8192, stop_at=0x53803b1a)
    source, target = m.reg[7], m.reg[9]
    if boundary != {"pc": 0x53803b1a, "reason": "slice boundary"} or (source, target) != (0xb830, 0xc2a0):
        raise ValueError(f"unexpected BED eighth-block sigmoid boundary: {boundary}")
    count = 150
    for a in range(source, source + 4 * count):
        m.scratch.pop(a, None)
    stop = expect(0x53803b1a, "unmodeled read", address=source, size=4)
    exponents = tuple(m.read(m.reg[r], 1, sign=True) for r in (10, 8, 4))
    if exponents != (22, 22, -16):
        raise ValueError(f"unexpected BED eighth-block sigmoid exponents: {exponents}")
    trial = deepcopy(m)
    values = ((-5, -2, -1, 0, 1, 2, 5) * 22)[:count]
    logits = tuple(v << exponents[0] for v in values)
    trial.extra_scratch += ((source, source + 4 * count), (target, target + 8 * count))
    trial.scratch.update(enumerate(pack("<150i", *logits), source))
    start, reads = len(trial.scratch_trace), len(trial.scratch_reads)
    end = trial.run(20000)
    if end != {"pc": 0x53803ed6, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block sigmoid end: {end}")
    return {"request": request, "dictionary_offset": offset, "mailbox": mailbox, "compute": compute,
            "arguments": args, "unseeded_stop": stop,
            "sigmoid_fixture": {"input": logits, "exponents": exponents, "end": end,
                                "input_reads": [r for r in trial.scratch_reads[reads:] if r["pc"] == 0x53803b1a],
                                "writes": trial.scratch_trace[start:],
                                "consumer_fixture": trace_eighth_q16_consumer(
                                    trial, dictionary, weight_base, ninth_resources)}}


def trace_eighth_q16_consumer(m, dictionary, weight_base, ninth_resources):
    def expect(pc, reason, **fields):
        result = m.run(8192)
        if result != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED eighth-block Q16 consumer stop: {result}")
        return result

    def wait(pc):
        expect(pc, "wfi")
        m.pc += 2

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    (image,) = (data for a, data in m.regions if a == BLOCK_VA)
    catalog = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)

    def overlay():
        reply(0x53801764, 0x48168, catalog)
        reply(0x53801768, 0x48168, catalog)
        begin = len(m.dma_submissions)
        wait(0x538017c0)
        (request,) = m.dma_submissions[begin:]
        (src, dst, a, b), = request["rows"]
        slots = plane_slots(a, b)
        if slots is None or dst != 0x0400f000:
            raise ValueError(f"unexpected BED eighth-block Q16 program: {request}")
        code = program(image, src - BLOCK_VA, slots)
        return {"request": request, "member_offset": code.offset + 0x100, "slots": slots}

    reply(0x53803ed6, 0x481cc)
    first = overlay()
    wait(0x53803f24)
    early = expect(0x53804bb8, "unsupported opcode", opcode=10)
    early_args = tuple(m.reg[4:8])
    stack_args = tuple(m.read(m.reg[15] + at, 4) for at in (0, 4, 8, 12))
    if m.link != 0x53803f48:
        raise ValueError("unexpected BED eighth-block Q16 compute return")
    m.pc = m.link
    wait(0x53803f50)
    reply(0x53803f5c, 0x481cc)
    begin = len(m.dma_submissions)
    for _ in range(4):
        wait(0x5380172c)
    copies = m.dma_submissions[begin:]
    reply(0x53803f8a, 0x481cc)
    second = overlay()
    reply(0x53803fb2, 0x48100, 0x20000000)
    reply(0x53803fe0, 0x48100, 0x20000000)
    reply(0x5380411c, 0x48230)
    begin = len(m.dma_submissions)
    reply(0x53804176, 0x48230)
    reply(0x53804190, 0x48230)
    wait(0x53804220)
    batches = list(m.dma_submissions[begin:])
    calls = []
    for i in range(12):
        expect(0x53804bb8, "unsupported opcode", opcode=10)
        args = tuple(m.reg[4:8])
        if args != (0x538004dc, i, 0, 2) or m.link != 0x5380424c:
            raise ValueError(f"unexpected BED eighth-block Q16 downstream call: {args}")
        calls.append(args)
        m.pc = m.link
        wait(0x53804254)
        reply(0x53804350, 0x48230)
        begin = len(m.dma_submissions)
        if i < 11:
            wait(0x53804220)
        else:
            reply(0x53804388, 0x48230)
        (request,) = m.dma_submissions[begin:]
        batches.append(request)
    reply(0x538043a0, 0x481cc)
    stop = expect(0x538043c6, "unmodeled read", address=0x172b0, size=2)
    return {"programs": (first, second), "compute": early, "arguments": early_args,
            "stack_arguments": stack_args, "copies": copies, "batches": batches,
            "following_calls": calls, "stop": stop,
            "projection_registers": tuple(m.reg[r] for r in (3, 4, 5, 8, 9, 10, 11)),
            "scalar_heads_fixture": trace_eighth_scalar_heads(m, dictionary, weight_base, ninth_resources)}


def trace_eighth_scalar_heads(snapshot, dictionary, weight_base, ninth_resources):
    m = deepcopy(snapshot)
    source, weights, out, end, stride = (m.reg[r] for r in (3, 4, 9, 10, 11))
    if (source, weights, end, stride) != (0x172b0, 0xa004b450, 0x17378, 200):
        raise ValueError("unexpected BED eighth-block scalar head ranges")
    values = tuple(range(-50, 50))
    m.extra_scratch += ((source, end),)
    m.scratch.update(enumerate(pack("<100h", *values), source))

    def bind_alias(address, size):
        dma_address = 0x20000000 | (address & 0x1fffffff)
        offset = dma_address - weight_base
        if not 0 <= offset <= len(dictionary) - size:
            raise ValueError(f"BED scalar head coefficient extent outside dictionary: {address:#x}")
        m.extra_scratch += ((address, address + size),)
        m.scratch.update(enumerate(dictionary[offset:offset + size], address))
        return offset

    first_offset = bind_alias(weights - 8, 8 + 4 * stride)
    start = len(m.scratch_reads)
    first = m.run(8192)
    if first != {"pc": 0x538043f2, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block first head stop: {first}")
    first_read = [r for r in m.scratch_reads[start:] if r["pc"] == 0x538043e0]
    first_values = tuple(m.read(out + 4 * j, 4, sign=True) for j in range(4))
    m.fixture_reads[0x481cc] = deque((0,))
    quantize = m.run(8192)
    if quantize != {"pc": 0x538044fe, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block first head quantization: {quantize}")
    m.fixture_reads[0x481cc] = deque((0,))
    stop = m.run(8192)
    if stop != {"pc": 0x53804526, "reason": "unmodeled read", "address": 0xa004b778, "size": 2}:
        raise ValueError(f"unexpected BED eighth-block second head start: {stop}")
    if (m.reg[4], m.reg[8], m.reg[10], m.reg[11]) != (0xa004b778, 5, end, stride):
        raise ValueError("unexpected BED eighth-block second head ranges")
    second_out = m.reg[9]
    second_offset = bind_alias(0xa004b770, 8 + 2 * stride)
    start = len(m.scratch_reads)
    second = m.run(8192)
    if second != {"pc": 0x53804550, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block second head stop: {second}")
    second_read = [r for r in m.scratch_reads[start:] if r["pc"] == 0x5380453e]
    second_values = tuple(m.read(second_out + 4 * j, 4, sign=True) for j in range(2))
    m.fixture_reads[0x481cc] = deque((0, 0))
    third_start = m.run(8192)
    if third_start != {"pc": 0x53804662, "reason": "unmodeled read", "address": 0xa004b910, "size": 2}:
        raise ValueError(f"unexpected BED eighth-block third head start: {third_start}")
    third_offset = bind_alias(0xa004b908, 8 + 2 * stride)
    start = len(m.scratch_reads)
    third = m.run(8192)
    if third != {"pc": 0x53804690, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block third head stop: {third}")
    third_read = [r for r in m.scratch_reads[start:] if r["pc"] == 0x5380467a]
    third_out = m.reg[15] + m.read(0x5380479c, 2, fetch=True)
    third_values = tuple(m.read(third_out + 4 * j, 4, sign=True) for j in range(2))
    m.fixture_reads[0x481cc] = deque((0, 0, 0))
    fourth_start = m.run(8192)
    if fourth_start != {"pc": 0x5380475a, "reason": "unmodeled read", "address": 0xa004baa8, "size": 2}:
        raise ValueError(f"unexpected BED eighth-block fourth head start: {fourth_start}")
    fourth_offset = bind_alias(0xa004baa0, 8 + stride)
    fourth = m.run(8192)
    if fourth != {"pc": 0x53804776, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block fourth head stop: {fourth}")
    fourth_sum, fourth_bias, fourth_shift = (m.reg[r] for r in (7, 2, 3))
    m.fixture_reads[0x481cc] = deque((0, 0, 0))
    fifth_start = m.run(8192)
    if fifth_start != {"pc": 0x5380488a, "reason": "unmodeled read", "address": 0xa004bb78, "size": 2}:
        raise ValueError(f"unexpected BED eighth-block fifth head start: {fifth_start}")
    fifth_offset = bind_alias(0xa004bb70, 8 + stride)
    fifth = m.run(8192)
    if fifth != {"pc": 0x538048a6, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block fifth head stop: {fifth}")
    fifth_sum, fifth_bias, fifth_shift = (m.reg[r] for r in (7, 2, 3))
    m.fixture_reads[0x481cc] = deque((0,))
    wait = m.run(8192)
    if wait != {"pc": 0x53804958, "reason": "wfi"}:
        raise ValueError(f"unexpected BED eighth-block result wait: {wait}")
    float_words = tuple(m.read(a, 4) for a in (0x7fc4, 0x7fc8))
    m.pc += 2
    start = len(m.scratch_trace)
    m.fixture_reads.update({0x48100: deque((0x20000000,)), 0x481cc: deque((0,))})
    result = m.run(8192)
    if result != {"pc": 0x53800a34, "reason": "unmodeled read", "address": 0x48160, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block result boundary: {result}")
    work_writes = [w for w in m.scratch_trace[start:] if 0xe0004040 <= w["address"] < 0xe00040d8]
    m.fixture_reads[0x48160] = deque((BLOCK_VA,))
    handoff = m.run(8192)
    if handoff != {"pc": 0x5004d7ba, "reason": "unmodeled read", "address": 0x481c4, "size": 4}:
        raise ValueError(f"unexpected BED eighth-block downstream handoff: {handoff}")
    first_dma = len(m.dma_submissions)
    m.fixture_reads[0x481c4] = deque((0,))
    downstream = m.run(8192)
    if downstream != {"pc": 0x5004d834, "reason": "unmodeled read", "address": 0x48020, "size": 4}:
        raise ValueError(f"unexpected BED downstream startup boundary: {downstream}")
    return {"input": values, "first_offset": first_offset, "first_bias_reads": first_read,
            "first_output": first_values, "second_unseeded_stop": stop, "second_offset": second_offset,
            "second_bias_reads": second_read, "second_output": second_values, "second_stop": second,
            "third_offset": third_offset, "third_bias_reads": third_read, "third_output": third_values,
            "third_stop": third, "fourth_start": fourth_start, "fourth_offset": fourth_offset,
            "fourth_sum": fourth_sum, "fourth_bias": fourth_bias, "fourth_shift": fourth_shift,
            "fourth_stop": fourth, "fifth_offset": fifth_offset, "fifth_sum": fifth_sum,
            "fifth_bias": fifth_bias, "fifth_shift": fifth_shift, "float_words": float_words,
            "result_wait": wait, "work_writes": work_writes, "handoff": handoff,
            "downstream": downstream, "downstream_dma": m.dma_submissions[first_dma:first_dma + 1],
            "ninth_fixture": trace_ninth_bed_windows(m, ninth_resources)}


def trace_ninth_bed_windows(m, ninth_resources):
    """Replay the original ninth-block consumer without synthesizing window pixels."""
    (image,) = (data for base, data in m.regions if base == BLOCK_VA)
    block = 0x5004d700 - BLOCK_VA
    target = 0x54000000

    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED ninth-block stop: {stop}")
        return stop

    def wait(pc):
        stop = expect(pc, "wfi")
        m.pc += 2
        return stop

    (row,) = m.dma_submissions[-1]["rows"]
    bind_linear_parameter(m, row, image, BLOCK_VA, 0x17818, 160)
    catalog = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)
    m.fixture_reads.update({0x48020: deque((0x33440000,) * 2), 0x481c4: deque((0,))})
    start = len(m.mmio_writes)
    waits = (wait(0x5004d84c), wait(0x5004d984))
    jump = expect(target, "unmapped fetch", address=target, size=2)
    writes = {w["address"]: w["value"] for w in m.mmio_writes[start:]}
    source = block + unpack_from("<I", image, block + 0x30)[0]
    end = block + unpack_from("<I", image, block + 0x10)[0]
    if (writes[0x49d04], writes[0x49d08] + 1 - target) != (target, end - source):
        raise ValueError("BED ninth-block program extent differs from its header")
    code = image[source:end]
    m.regions += ((target, code),)
    m.stack_high = pc_literal(lambda a, n: code[a - target:a - target + n], target, 15)
    m.stack_low = m.stack_high - 0x1000
    m.stack = bytearray(0x1000)
    expect(0x5400494e, "unmodeled read", address=0x49c80, size=4)
    m.fixture_reads[0x49c80] = deque((0,))
    compute_writes = len(m.mmio_writes)
    reads, stores = len(m.scratch_reads), len(m.scratch_trace)
    output_stop = expect(0x54000b4e, "unmodeled write", address=0xe00040d8, size=2, value=0x22)
    m.extra_scratch += ((0xe00040d8, 0xe00040dc),)
    start_dma = len(m.dma_submissions)
    input_wait = wait(0x540018bc)
    window_waits = tuple(wait(0x54001b58) for _ in range(9))
    stop = expect(0x54001d88, "unmodeled read", address=0x48168, size=4)
    inputs = m.dma_submissions[start_dma:]
    overlays = []
    for first, second, ready in (
            (0x54001d88, 0x54001d8a, 0x54001de0), (0x54001df0, 0x54001df2, 0x54001e94),
            (0x54001ea4, 0x54001ea6, 0x54001efc), (0x54001f0c, 0x54001f0e, 0x54001f64),
            (0x54001f74, 0x54001f76, 0x54001fcc), (0x54001fe2, 0x54001fe4, 0x5400206c)):
        if m.pc != first:
            expect(first, "unmodeled read", address=0x48168, size=4)
        m.fixture_reads[0x48168] = deque((catalog,))
        expect(second, "unmodeled read", address=0x48168, size=4)
        m.fixture_reads[0x48168] = deque((catalog,))
        begin = len(m.dma_submissions)
        wait(ready)
        (request,) = m.dma_submissions[begin:]
        (src, dst, a, b), = request["rows"]
        overlay = program(image, src - BLOCK_VA, plane_slots(a, b))
        overlays.append((overlay.offset + 0x100, overlay.slots, dst))
    compute = expect(0x54000718, "unsupported opcode", opcode=0xa)
    arguments, continuation = tuple(m.reg[4:8]), m.link
    record_offset = source + arguments[0] - target
    record = image[record_offset:source + compute["pc"] - target]
    setup = tuple((w["address"], w["value"]) for w in m.mmio_writes[compute_writes:]
                  if 0x40000 <= w["address"] <= 0x40038)
    return {"boot_waits": waits, "jump": jump, "code_extent": (source + 0x100, len(code)),
            "work_reads": [r for r in m.scratch_reads[reads:] if 0xe0004040 <= r["address"] < 0xe00040dc],
            "work_writes": [w for w in m.scratch_trace[stores:] if 0xe0004040 <= w["address"] < 0xe00040dc],
            "output_stop": output_stop, "input_wait": input_wait, "window_waits": window_waits,
            "input_requests": inputs, "stop": stop, "overlays": overlays,
            "compute": compute, "arguments": arguments, "return": continuation, "setup": setup,
            "record_offset": record_offset + 0x100, "record_hex": record.hex(),
            "consumer_fixture": trace_ninth_output_consumer(m, image, catalog, arguments[3], ninth_resources)}


def trace_ninth_output_consumer(m, image, catalog, output, ninth_resources):
    m.pc = m.link  # Isolated control replay, not an executed compute return.
    # Do not let older block fixtures stand in for this compute output.
    for address in range(output, output + 0x480):
        m.scratch.pop(address, None)
    requests = []
    for first, second, ready in (
            (0x5400208a, 0x5400208c, 0x540020dc), (0x54002102, 0x54002104, 0x54002158),
            (0x54002168, 0x5400216a, 0x540021c0), (0x540021d0, 0x540021d2, 0x54002228),
            (0x54002238, 0x5400223a, 0x54002290), (0x540022a0, 0x540022a2, 0x540022f8)):
        for pc in (first, second):
            if m.run(8192) != {"pc": pc, "reason": "unmodeled read", "address": 0x48168, "size": 4}:
                raise ValueError("unexpected BED ninth-block consumer catalog access")
            m.fixture_reads[0x48168] = deque((catalog,))
        start = len(m.dma_submissions)
        if m.run(8192) != {"pc": ready, "reason": "wfi"}:
            raise ValueError("unexpected BED ninth-block consumer overlay wait")
        (request,) = m.dma_submissions[start:]
        (src, dst, a, b), = request["rows"]
        overlay = program(image, src - BLOCK_VA, plane_slots(a, b))
        requests.append((overlay.offset + 0x100, overlay.slots, dst))
        m.pc += 2
        if ready == 0x540020dc:
            if m.run(8192) != {"pc": 0x540020f0, "reason": "wfi"}:
                raise ValueError("unexpected BED ninth-block consumer worker wait")
            m.pc += 2

    stop = m.run(8192)
    if stop != {"pc": 0x540025d4, "reason": "unmodeled read", "address": output, "size": 1}:
        raise ValueError(f"unexpected BED ninth-block first consumer read: {stop}")
    source, destination, size = m.reg[5] + m.reg[0], m.reg[2], m.reg[6]
    data = bytes((i * 17 + 3) & 0xff for i in range(size))
    m.extra_scratch += ((source, source + size), (destination, destination + size))
    m.scratch.update(enumerate(data, source))
    copy = m.run(8192, stop_at=0x540025de)
    if copy != {"pc": 0x540025de, "reason": "slice boundary"}:
        raise ValueError(f"unexpected BED ninth-block byte copy boundary: {copy}")
    copied = bytes(m.scratch[destination + i] for i in range(size))
    if copied != data:
        raise ValueError("BED ninth-block output copy changed bytes")
    return {"overlays": requests, "stop": stop, "copy_source": source,
            "copy_destination": destination, "copy_size": size, "copy_sha256": sha256(copied).hexdigest(),
            "next_fixture": trace_ninth_center_crop(m, image, catalog, destination, copied, ninth_resources)}


def trace_ninth_center_crop(m, image, catalog, source, data, ninth_resources):
    """Follow the copied shared-kernel output through the original center-crop setup."""
    def expect(pc, reason, **fields):
        stop = m.run(8192)
        if stop != {"pc": pc, "reason": reason, **fields}:
            raise ValueError(f"unexpected BED ninth-block center-crop boundary: {stop}")
        return stop

    def wait(pc):
        start = len(m.dma_submissions)
        expect(pc, "wfi")
        m.pc += 2
        return m.dma_submissions[start:]

    def reply(pc, address, value=0):
        expect(pc, "unmodeled read", address=address, size=4)
        m.fixture_reads[address] = deque((value,))

    work = [*wait(0x54000e58), *wait(0x54000ea8)]
    dictionary, base = ninth_resources
    tail_offsets = tuple(bind_linear_parameter(m, row, dictionary, base, dst, size)
                         for ((row,), (dst, size)) in zip(
                             (r["rows"] for r in work), ((0x17e04, 180), (0x17c00, 516)), strict=True))
    reply(0x54002bcc, 0x481cc)
    (broadcast,) = wait(0x54002c2c)
    destinations = (0xe800, 0xf000, 0xf800, 0x10000, 0x10800)
    copies = copy_local_rows(broadcast["rows"], m.scratch, source_span=(source, source + len(data)),
                             destination_span=(destinations[0], destinations[-1] + len(data)))
    if len(copies) != len(destinations) * len(data) or any(
            bytes(copies[dst + i] for i in range(len(data))) != data for dst in destinations):
        raise ValueError("BED ninth-block broadcast does not carry the copied bytes")
    reply(0x54002c36, 0x481cc)
    (crop,) = wait(0x54002970)
    (dst, tagged, total_word, flags), = crop["rows"]
    byte_width, rows, pitch = 40, 20, 48
    start = 2 * pitch + 4
    if (len(data) != 24 * pitch or tagged != (0xc0000000 | (pitch << 16) | (source + start))
            or total_word != (byte_width * rows - 1) << 16
            or flags != (((byte_width - 1) << 16) | ((rows - 1) << 8))):
        raise ValueError("BED ninth-block center window differs from the original descriptor")
    patch = b"".join(data[i:i + byte_width] for i in range(start, start + rows * pitch, pitch))
    m.extra_scratch += ((dst, dst + len(patch)),)
    m.scratch.update(enumerate(patch, dst))

    reply(0x54002c5c, 0x481cc)
    reply(0x540029ac, 0x48168, catalog)
    reply(0x540029b0, 0x48168, catalog)
    (request,) = wait(0x54002a08)
    (src, target, a, b), = request["rows"]
    overlay = program(image, src - BLOCK_VA, plane_slots(a, b))
    if wait(0x54002d10):
        raise ValueError("unexpected BED ninth-block DMA before compute")
    compute = expect(0x5400477c, "unsupported opcode", opcode=0xa)
    arguments, continuation = tuple(m.reg[4:8]), m.link
    m.pc = continuation  # Control replay only; no compute result is supplied.
    reads = len(m.scratch_reads)
    if wait(0x54002d44):
        raise ValueError("unexpected BED ninth-block post-crop DMA")
    reply(0x54002d52, 0x481cc)
    reply(0x54002a5a, 0x48168, catalog)
    reply(0x54002a5e, 0x48168, catalog)
    (request,) = wait(0x54002ab4)
    (src, target_next, a, b), = request["rows"]
    overlay_next = program(image, src - BLOCK_VA, plane_slots(a, b))
    reply(0x54002e2c, 0x48230)
    (resources,) = wait(0x54002e58)
    spans = ((0xcf60, 12), *((dst, 24) for dst in (0x5800, 0x6000, 0x6800, 0xd008, 0xcac8, 0xd800)))
    offsets = tuple(bind_linear_parameter(m, row, dictionary, base, dst, size)
                    for row, (dst, size) in zip(resources["rows"], spans, strict=True))
    blocks = tuple(padded_3x3(dictionary[offset:offset + 24])[0] for offset in offsets[1:])
    if wait(0x54002ee0):
        raise ValueError("unexpected BED ninth-block DMA before next compute")
    compute_next = expect(0x54000718, "unsupported opcode", opcode=0xa)
    return {"resource_requests": work, "tail_offsets": tail_offsets, "broadcast": broadcast, "crop": crop,
            "raster_extent": (pitch, 24), "crop_extent": (byte_width, rows),
            "crop_sha256": sha256(patch).hexdigest(), "overlay": (overlay.offset + 0x100, overlay.slots, target),
            "compute": compute, "arguments": arguments, "return": continuation,
            "following_fixture": {
                "overlay": (overlay_next.offset + 0x100, overlay_next.slots, target_next),
                "resources": resources, "offsets": offsets, "blocks": blocks,
                "parameter_reads": [r for r in m.scratch_reads[reads:]
                                    if any(dst <= r["address"] < dst + size for dst, size in
                                           (*spans, (0x17e04, 180), (0x17c00, 516)))],
                "compute": compute_next, "arguments": tuple(m.reg[4:8]), "return": m.link}}


def trace_sixth_q16_workers(snapshot, workers):
    jobs = []
    for base, start, stop in ((0x800, 0xf68, 0x1134), (0x1800, 0x1f68, 0x213c), (0x2800, 0x2f5c, 0x3130),
                              (0x8000, 0x8778, 0x8920), (0x9000, 0x977c, 0x9924)):
        code = workers[base]
        worker = DmaTraceMachine(code, scratch_base=0x17ad8, scratch_size=24)
        worker.regions = ((base, code),)
        worker.stack_low = base + len(code)
        worker.stack_high = pc_literal(lambda a, n: code[a - base:a - base + n], base + 0x10, 15)
        worker.stack = bytearray(worker.stack_high - worker.stack_low)
        worker.reg[15], worker.pc = worker.stack_high, start
        worker.scratch.update((a, snapshot.scratch[a]) for a in range(0x17ad8, 0x17af0))
        launch = worker.run(128)
        if launch != {"pc": stop, "reason": "unsupported opcode", "opcode": 10}:
            raise ValueError(f"unexpected BED Q16 worker boundary: {launch}")
        stack_args = tuple(worker.read(worker.reg[15] + offset, 4) for offset in range(0, 20, 4))
        jobs.append({"entry": start, "launch": launch, "arguments": tuple(worker.reg[4:8]),
                     "stack_arguments": stack_args, "mailbox_reads": worker.scratch_reads})
    return jobs


def trace_coefficient_workers(controller, bed):
    loads = {target: bed[src:src + (a >> 16) + 1]
             for src, target, a, b in startup_rows(bed, 0x2200)
             if a & 0xffff == 0x4000 and b == a & 0xffff0000}
    # Rebuild only the scalar mailbox and the 24 real parameter bytes.
    shared = {a: controller.scratch[a] for lo, hi in
              ((0x17700, 0x17704), (0x17750, 0x17770), (0x16240, 0x16258)) for a in range(lo, hi)}
    (external,) = (image for base, image in controller.regions if base == 0x50800000)
    record = external[0x184:0x2b0]
    jobs = []
    for base, call_wait in ((0x800, 0x10fc), (0x1800, 0x20f8), (0x2800, 0x3110)):
        image = loads[base]
        worker = DmaTraceMachine(image, scratch_base=0x17700, scratch_size=0x100)
        worker.pc, worker.regions = base, tuple(loads.items())
        worker.stack_low = base + len(image)
        worker.stack_high = pc_literal(lambda a, n: image[a - base:a - base + n], base + 0x10, 15)
        worker.stack = bytearray(worker.stack_high - worker.stack_low)
        worker.extra_scratch = ((0x16240, 0x16258), (0x40c4, 0x40cc))
        worker.scratch.update(shared)
        if worker.run(8192)["reason"] != "wfi":
            raise ValueError("BED coefficient worker did not reach its initial wait")
        worker.pc += 2
        snapshot = deepcopy(worker)
        unseeded = worker.run(8192)
        if (unseeded["reason"], unseeded.get("address"), unseeded.get("size")) != ("unmodeled read", 0x40cb, 1):
            raise ValueError(f"unexpected BED coefficient control dependency: {unseeded}")
        paths = []
        controls = [(0, 0), (0, 13), (-9, 0)]
        config = tuple(controller.scratch.get(a) for a in (0x40c4, 0x40cb))
        if None not in config:
            controls.append(tuple(unpack_from("<b", bytes((v,)))[0] for v in config))
        for i, (left_source, control) in enumerate(controls):
            worker = deepcopy(snapshot)
            worker.scratch.update({0x40c4: left_source & 255, 0x40cb: control & 255})
            calls = []
            for _ in range(3 if i in (0, 3) else 1):
                begin = len(worker.mmio_writes)
                if worker.run(8192) != {"pc": call_wait, "reason": "wfi"}:
                    raise ValueError("BED coefficient worker did not reach its compute wait")
                worker.pc += 2
                stop = worker.run(8192)
                if stop != {"pc": base + 0x380, "reason": "unsupported opcode", "opcode": 0xa}:
                    raise ValueError(f"unexpected BED local coefficient compute: {stop}")
                offset = worker.reg[4] - base
                if offset < 0 or image[offset:offset + len(record)] != record:
                    raise ValueError("BED coefficient worker record differs from the controller")
                writes = {w["address"]: w["value"] for w in worker.mmio_writes[begin:]}
                calls.append({"arguments": tuple(worker.reg[4:8]),
                              "aperture": tuple((a, writes[a]) for a in range(0x40320, 0x40334, 4))})
                worker.pc = worker.link
            writes = {w["address"]: w["value"] for w in worker.mmio_writes if w["address"] == 0x40218}
            paths.append({"controls": (left_source, control), "aperture_control": writes[0x40218], "calls": calls})
        job = {"base": base, "unseeded_stop": unseeded, "paths": paths[:3]}
        if len(paths) == 4:
            job["bound_controls"] = paths[3]
        jobs.append(job)
    return jobs


def trace_next_program(m, stop, bed):
    """Bind a firmware program-plane request without completing the DMA."""
    if stop != {"pc": 0x50800e40, "reason": "unmodeled read", "address": 0x481cc, "size": 4}:
        raise ValueError(f"unexpected BED block-one status read: {stop}")
    pointer = next(w["value"] for w in reversed(m.mmio_writes) if w["address"] == 0x48168)
    before = len(m.fixture_trace)
    start = len(m.dma_submissions)
    m.fixture_reads.update({0x481cc: deque((0,)), 0x48168: deque((pointer, pointer))})
    wait = m.run(8192)
    if wait != {"pc": 0x50800ea8, "reason": "wfi"}:
        raise ValueError(f"unexpected BED program DMA boundary: {wait}")
    (request,) = m.dma_submissions[start:]
    (src, dst, a, b), = request["rows"]
    slots = plane_slots(a, b)
    if slots is None or dst != 0x0400f000:
        raise ValueError(f"unexpected BED program DMA request: {request}")
    offset = 0x100 + src - BLOCK_VA
    planes = program(bed, offset, slots)
    size = 3 * slots * 8
    return {"wait": wait, "request": request, "fixture_reads": m.fixture_trace[before:],
            "member_offset": offset, "slots": planes.slots,
            "sha256": sha256(bed[offset:offset + size]).hexdigest(),
            "compute_fixture": trace_next_compute(m, bed, planes)}


def trace_next_compute(m, bed, planes):
    """Observe the compute handoff without providing any copied data or outputs."""
    before = len(m.scratch_trace)
    start = len(m.dma_submissions)
    m.pc += 2  # Isolated control continuation after the program request.
    wait = m.run(8192)
    if wait != {"pc": 0x50800ee0, "reason": "wfi"}:
        raise ValueError(f"unexpected BED block-one worker wait: {wait}")
    m.pc += 2
    stop = m.run(8192)
    if stop != {"pc": 0x508018d4, "reason": "unsupported opcode", "opcode": 0xa}:
        raise ValueError(f"unexpected BED block-one compute boundary: {stop}")
    if len(m.dma_submissions) != start:
        raise ValueError("unexpected DMA between the BED program and compute boundaries")
    pointer, *args = m.reg[4:8]
    base, image = next((base, image) for base, image in m.regions if base <= pointer < m.pc < base + len(image))
    record = image[pointer - base:m.pc - base]
    marker = COMMAND_MARKER
    end = next(i for i in range(0, len(record), 4) if record[i:i + len(marker)] == marker)
    record = record[:end]
    applied = join_record(planes, record)
    offset = bed.find(image)
    if offset < 0 or bed.find(image, offset + 1) >= 0:
        raise ValueError("expected a unique BED block-one program in the pinned member")
    mailbox = tuple(w for w in m.scratch_trace[before:] if 0x17700 <= w["address"] < 0x17800)
    return {"wait": wait, "stop": stop, "return": m.link, "record": pointer,
            "arguments": args, "member_offset": offset + pointer - base,
            "record_sha256": sha256(record).hexdigest(),
            "slot_indices": tuple(row["index"] for row in applied),
            "mailbox_writes": mailbox, "workers": trace_next_workers(bed, mailbox, record + marker)}


def trace_next_workers(bed, mailbox, record):
    """Replay local copies of block one's worker code with the original mailbox writes."""
    loads = {target: (src, bed[src:src + (a >> 16) + 1])
             for src, target, a, b in startup_rows(bed, 0x2200)
             if a & 0xffff == 0x4000 and b == a & 0xffff0000}
    regions = tuple((target, image) for target, (_, image) in loads.items())
    jobs = []
    for base in (0x800, 0x1800, 0x2800):
        source, image = loads[base]
        worker = DmaTraceMachine(image, scratch_base=0x17700, scratch_size=0x100)
        worker.pc, worker.regions = base, regions
        worker.stack_low = base + len(image)
        worker.stack_high = pc_literal(lambda a, n: image[a - base:a - base + n], base + 0x10, 15)
        worker.stack = bytearray(worker.stack_high - worker.stack_low)
        for w in mailbox:
            worker.scratch.update(enumerate(w["value"].to_bytes(w["size"], "little"), w["address"]))
        wait = worker.run(8192)
        if wait["reason"] != "wfi":
            raise ValueError(f"unexpected BED block-one worker wait: {wait}")
        worker.pc += 2
        stop = worker.run(8192)
        if (stop["reason"], stop.get("opcode")) != ("unsupported opcode", 0xa):
            raise ValueError(f"unexpected BED block-one worker compute: {stop}")
        pointer, *args = worker.reg[4:8]
        offset = pointer - base
        if not 0 <= offset < len(image) or image[offset:offset + len(record)] != record:
            raise ValueError("BED block-one worker command differs from controller")
        jobs.append({"base": base, "wait": wait, "stop": stop, "return": worker.link,
                     "record": pointer, "member_offset": source + offset,
                     "arguments": args, "mailbox_reads": worker.scratch_reads})
    return jobs


def trace_bed(cp, bundle, config, *, center=(280, 192), sides=(64, 64), work=bytes(WORK_BYTES),
              stages=False, exec_id=0, roll=0, dictionary=None, dictionary2=None):
    if len(work) != WORK_BYTES:
        raise ValueError("expected the explicit initial work-buffer bytes")
    if dictionary is not None and (not stages or sha256(dictionary).hexdigest() != BED_DICT1_SHA):
        raise ValueError("expected --stages and the pinned release bed_dic1.bin")
    if dictionary2 is not None and dictionary is None:
        raise ValueError("expected both published BED dictionaries")
    member = members(bundle)[JOB]
    if member.sha256 != BED_SHA or member.header_job_id != JOB:
        raise ValueError("expected pinned A7 IV BED member")
    bed = bundle[member.offset:member.offset + member.size]
    regions = tuple((target, bed[src:src + (flags2 >> 16) + 1])
                    for src, target, flags2, flags3 in startup_rows(bed)
                    if flags2 & 0xffff == 0x4000 and flags3 == flags2 & 0xffff0000)
    origin, image = regions[0]
    parameter = probe_keypoint(cp, config, make_candidate(center, sides, exec_id, roll))["parameter"]
    (row,) = probe_request(cp, "bed", ((0x80000000, 0x500020200, 0, 0),))["rows"]
    if row[3] != JOB or row[-3:-1] != (0x40000000, unpack_from("<I", parameter, 8)[0]):
        raise ValueError("CP request and parameter addresses disagree")

    m = DmaTraceMachine(image, scratch_base=0x9000, scratch_size=0xf000)
    m.pc = origin
    m.regions = (*regions, (0x49e00, bytes(0x160)))
    m.stack_high = pc_literal(lambda address, size: image[address - origin:address - origin + size],
                              origin, 15)
    if m.stack_high != STACK:
        raise ValueError("unexpected original BED stack")
    m.stack_low = m.stack_high - 0x800
    m.stack = bytearray(0x800)
    m.fixture_reads = {a: deque(v) for a, v in
                       ((0x48004, (0,)), (0x48024, (0, 0)), (0x49c80, (0,) * 8))}
    m.extra_scratch = ((0, 0x100), (0x49002, 0x49003),
                       (PARAMETER_ALIAS, PARAMETER_ALIAS + len(parameter)),
                       (0xe0004040, 0xe00040d8))
    m.scratch[0x49002] = 0
    m.scratch.update(enumerate(parameter, PARAMETER_ALIAS))
    m.capture_dma = True
    check = m.run(8192, stop_at=VALIDATOR_RETURN)
    if check != {"pc": VALIDATOR_RETURN, "reason": "slice boundary"}:
        raise ValueError(f"BED did not reach parameter validation: {check}")
    result = {"parameter_hex": parameter.hex(), "queue_row": row, "validator_mask": m.reg[0],
              "submissions": m.dma_submissions}
    result["parameter_reads"] = parameter_reads = []

    def record_parameter_reads():
        parameter_reads[:] = [(r["pc"], r["address"] - PARAMETER_ALIAS, r["size"])
                              for r in m.scratch_reads
                              if PARAMETER_ALIAS <= r["address"] < PARAMETER_ALIAS + len(parameter)]

    record_parameter_reads()
    if m.reg[0] != 0:
        return result
    first = m.run(8192)
    result["first_stop"] = first
    if first != {"pc": WORK_WAIT, "reason": "wfi"}:
        return result

    (work_request,) = m.dma_submissions
    (src, dst, a, b), = work_request["rows"]
    start = (dst & 0x7fffffff) - m.stack_low
    if (src != 0x20004040 or dst >> 31 != 1 or a != b or a != (len(work) - 1) << 16 or
            not 0 <= start <= len(m.stack) - len(work)):
        raise ValueError("unexpected work-buffer DMA request")
    m.stack[start:start + len(work)] = work
    m.pc += 2
    image_stop = m.run(8192)
    result["image_stop"] = image_stop
    if image_stop != {"pc": IMAGE_WAIT, "reason": "wfi"}:
        return result

    result["image_read"] = bind_packed_read(parameter, m.dma_submissions[-1])
    m.pc += 2  # Observe setup without writing or reading invented image pixels.
    for _ in range(256):
        submitted = len(m.dma_submissions)
        stop = m.run(8192)
        if stop["reason"] != "wfi" or len(m.dma_submissions) != submitted + 1:
            break
        # Border helpers only enqueue local copy requests; do not supply bytes.
        if m.dma_submissions[-1]["submit_pc"] not in EDGE_SUBMITS:
            break
        m.pc += 2
    result["after_image"] = stop
    record_parameter_reads()
    if stop == COMPUTE:
        result["compute_arguments"] = tuple(m.reg[4:8])
        result["local_image"] = bind_bed_local_image(parameter, result["image_read"], m.dma_submissions[1:],
                                                    m.reg[5], m.reg[6])
        result["compute_return"] = m.link
        result["compute_setup"] = tuple((w["address"], w["value"]) for w in m.mmio_writes
                                        if 0x40000 <= w["address"] < 0x40100)
        record = image[m.reg[4] - origin:m.pc - origin]
        result["compute_record"] = {"bytes": len(record), "sha256": sha256(record).hexdigest()}
        if stages:
            main = dict(result["compute_setup"])
            result["controller_partition"] = (main[0x40038], main[0x40000])
            result["workers"] = first_stage_workers(m, regions, record)
            result["shared_image_entry"] = shared_image_entry(bundle, record)
            result["handoff"] = trace_handoff(m, bed, dictionary, dictionary2)
        # A separate slice of the known instructions inside the wrapper, not a
        # continuation of an executed 0x000a.
        m.pc += 2
        result["wrapper_inner_stop"] = m.run(32)
        result["wrapper_record_pointer"] = m.reg[0]
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("bundle", type=Path)
    cli.add_argument("firmware_tar", type=Path)
    cli.add_argument("--stages", action="store_true", help="Trace workers and isolated handoff slices")
    cli.add_argument("--dictionary", type=Path, help="Bind the first post-compute parameter read")
    cli.add_argument("--dictionary2", type=Path, help="Check the second resource's packed file layout")
    args = cli.parse_args()
    if args.dictionary2 and (not args.dictionary or not args.stages):
        cli.error("--dictionary2 requires --stages and --dictionary")
    cp = args.cp.read_bytes()
    with tarfile.open(args.firmware_tar, "r") as archive:
        config = published_config(cp, archive.extractfile(BACKUP_MEMBER).read())
    dictionary = args.dictionary.read_bytes() if args.dictionary else None
    dictionary2 = args.dictionary2.read_bytes() if args.dictionary2 else None
    witness = trace_bed(cp, args.bundle.read_bytes(), config, stages=args.stages,
                        dictionary=dictionary, dictionary2=dictionary2)
    print(json.dumps(witness, default=vars, indent=2))
