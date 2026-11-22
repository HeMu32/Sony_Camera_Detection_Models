"""Execute original PDN scalar call setup and program-loader slices."""

import argparse
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from struct import iter_unpack, pack

import numpy as np

from compare_sa_payload_records import MARKER
from parsers.fdn_geometry import APERTURE, table_for_pdn
from parsers.pdn_sections import SCALAR_HEAD_OFFSET
from parsers.sony_sa_program_planes import PROGRAM_BASE, PROGRAM_BANK_BYTES, overlays, plane_slots, startup_rows
from probe_sa_fdn_boot import BootMachine, DmaTraceMachine
from probe_sa_fdn_dispatch import word
from run_sa_fdn_dispatch import Stop, signed


PDN_SHA = "6a9178e4a7ff06295aac70cc36f8a0118a8d9832c16483ac2bc2196bec8c7e8f"
PDN_DICTIONARY_SHA = "b2ac10a58b02620ff838d52787abd7347050719b48d62a6a60f4a89df10968b7"
BASE = 0x50000000
CATALOG_VA = 0x50010000
COMPUTE_STOP = {"pc": 0x50000b5c, "reason": "unsupported opcode", "opcode": 0xa}
FP32_OPS = (np.add, np.subtract, np.multiply, np.divide)
DICTIONARY_CPU_VA = 0xa0010000
DICTIONARY_DMA_BASE = 0x20010000
DOT_ENTRY = BASE + 0x39e8
DOT_END = BASE + 0x3a22
SCORE_FP_ENTRY = BASE + 0x3a5c
SCORE_TO_Q30 = BASE + 0xe7e
SCORE_WRITTEN = BASE + 0xe94
HEAD_SOURCE = 0xe4e0
MAP_RETURN = BASE + 0x24d6
MAP_PARAMETER_ADDRS = (0x17970, 0x17975)
FINAL_WRAPPER = BASE + 0x5908
FINAL_RETURN = BASE + 0x393e
FP32_SLICES = ((0x15cc, 0x161e), (0x3a5c, 0x3d34), (0x5098, 0x50cc), (0xe7e, 0xe8e))


class PdnGeometryMachine(DmaTraceMachine):
    def step(self):
        if not any(lo <= self.pc - BASE <= hi for lo, hi in FP32_SLICES):
            return super().step()
        word16 = self.read(self.pc, 2, fetch=True)
        op, dst, src = word16 >> 8, word16 & 15, word16 >> 4 & 15
        if not 0x34 <= op <= 0x39:
            return super().step()
        operand = np.uint32(self.reg[src]).view(np.float32)
        if op == 0x39:
            result = np.float32(signed(self.reg[src], 32)).view(np.uint32)
        elif op == 0x38:
            if not np.isfinite(operand) or not -0x80000000 <= operand < 0x80000000:
                raise Stop("unmodeled FP32 to signed conversion", value=self.reg[src])
            result = int(operand) & 0xffffffff
        else:
            previous = np.uint32(self.reg[dst]).view(np.float32)
            result = FP32_OPS[op - 0x34](previous, operand).view(np.uint32)
        self.reg[dst] = int(result)
        self.pc += 2


def build(pdn, *, machine_cls=DmaTraceMachine):
    if sha256(pdn).hexdigest() != PDN_SHA:
        raise ValueError("expected pinned A7 IV PDN member")
    start = 0x100 + word(pdn, 0x130)
    end = 0x100 + word(pdn, 0x110)
    image = pdn[start:end]
    m = machine_cls(image)
    m.regions = ((BASE, image),)
    m.stack_low, m.stack_high = 0x7800, 0x8000
    m.stack = bytearray(0x800)
    m.reg[15] = 0x7f00
    return m, start


def call_setup(pdn, entry, fixture_bytes):
    m, file_base = build(pdn)
    m.pc = entry
    m.reg[4:8] = [0x6000, 0x4000, 0xe000, 0x14000]
    m.write(0x7f04, 4, 0x6200)  # Original stack argument, filled only for this control slice.
    m.regions += ((0x6000, fixture_bytes), (0x6200, fixture_bytes))
    stop = m.run(4096)
    if stop != COMPUTE_STOP or m.link != 0x500022bc:
        raise ValueError(f"PDN setup did not reach the original wrapper: {stop}")
    return {"entry": entry, "return_pc": m.link, "stop": stop,
            "record_member_offset": file_base + m.reg[4] - BASE,
            "registers_r4_to_r7": m.reg[4:8],
            "aperture": {hex(w["address"]): w["value"] for w in m.mmio_writes if w["address"] >= 0x40190}}


def program_request(pdn):
    m, _ = build(pdn)
    catalog = 0x100 + word(pdn, 0x10c)
    m.regions += ((CATALOG_VA, pdn[catalog:]),)
    m.fixture_reads[0x48168] = deque([CATALOG_VA] * 4)
    m.capture_dma = True
    m.pc = 0x500014e2
    stop = m.run(1024)
    if stop != {"pc": 0x50001540, "reason": "wfi"}:
        raise ValueError(f"unexpected PDN program loader stop: {stop}")
    request, = m.dma_submissions
    source, _, _, _ = request["rows"][0]
    return {"stop": stop, "member_offset": source - CATALOG_VA + catalog, **request}


def geometry_machine(pdn):
    m, file_base = build(pdn, machine_cls=PdnGeometryMachine)
    catalog = 0x100 + word(pdn, 0x10c)
    m.regions += ((CATALOG_VA, pdn[catalog:]), (0x100, table_for_pdn(pdn)))
    m.scratch_base, m.scratch_end = 0x17800, 0x18000
    m.fixture_reads[0x48168] = deque([CATALOG_VA] * 2)
    m.capture_dma = True
    return m, file_base, catalog


def trace_geometry_call(m, file_base, catalog):
    routed = m.run(512, stop_at=BASE + 0x124c)
    if routed != {"pc": BASE + 0x124c, "reason": "slice boundary"} or m.link != BASE + 0x1742:
        raise ValueError(f"unexpected PDN geometry caller: {routed}")
    call_args = {
        "registers_r4_to_r7": m.reg[4:8],
        "stack_words": [m.read(m.reg[15] + offset, 4) for offset in range(0, 20, 4)],
    }
    wait = m.run(4096)
    if wait != {"pc": BASE + 0x14b4, "reason": "wfi"}:
        raise ValueError(f"unexpected PDN first-stage loader stop: {wait}")
    request = m.dma_submissions[-1]
    (source, dest, flags2, flags3), = request["rows"]
    slots = plane_slots(flags2, flags3)
    if dest != 0x0400f000 or slots is None:
        raise ValueError("unexpected PDN first-stage program descriptor")

    m.pc += 2  # Synthetic wake only. Do not execute the loaded program.
    stop = m.run(512)
    if stop != COMPUTE_STOP or m.link != BASE + 0x14d2:
        raise ValueError(f"unexpected PDN first-stage compute stop: {stop}")
    return {"geometry_call": call_args, "wait": wait, "request": request,
            "aperture_writes": [w for w in m.mmio_writes if APERTURE <= w["address"] < APERTURE + 0x4c],
            "table_offset": source - CATALOG_VA + catalog,
            "slots": slots, "stop": stop,
            "return_pc": m.link, "record_member_offset": file_base + m.reg[4] - BASE,
            "registers_r4_to_r7": m.reg[4:8]}


def first_stage_request(pdn, *, option=0, node=0, extents=(48, 48)):
    if option not in range(3) or node not in range(15) or any(not 1 <= n <= 128 for n in extents):
        raise ValueError("expected option 0..2, node 0..14, and positive fixture extents up to 128")
    m, file_base, catalog = geometry_machine(pdn)
    # Enter after the upstream arithmetic; none of the supplied values is a pixel.
    m.pc = BASE + 0x1632
    m.reg[0], m.reg[9], m.reg[10] = 0x600, node, option
    m.reg[13], m.reg[12] = extents
    m.write(m.reg[15] + 0x54, 4, 0xb000)
    return trace_geometry_call(m, file_base, catalog)


def image_geometry_request(pdn, *, center_q18=(281 << 18, 193 << 18), scales=(64, 80),
                           option=1, node=8):
    if (option not in range(3) or node not in range(15) or
            any(not 2 <= n <= 128 for n in scales) or
            not all(0 <= n < limit << 18 for n, limit in zip(center_q18, (640, 480), strict=True))):
        raise ValueError("expected valid PDN fixture center, scales, option and node")
    m, file_base, catalog = geometry_machine(pdn)
    m.pc = BASE + 0x15a8
    m.extra_scratch = ((0, 0x100),)
    m.reg[4:8] = [0xa0008000, 640, 480, 704]
    for i, val in enumerate((*center_q18, *scales, option, node, 0xb000)):
        m.write(m.reg[15] + 4 * i, 4, val)
    return trace_image_geometry(m, file_base, catalog)


def trace_image_geometry(m, file_base, catalog):
    entry = m.run(512, stop_at=BASE + 0xefc)
    if entry != {"pc": BASE + 0xefc, "reason": "slice boundary"} or m.link != BASE + 0x1632:
        raise ValueError(f"unexpected PDN image arguments: {entry}")
    frame, window = ([m.read(m.reg[reg] + off, 4) for off in range(0, 16, 4)] for reg in (5, 6))
    wait = m.run(4096)
    if wait != {"pc": BASE + 0x17ec, "reason": "wfi"}:
        raise ValueError(f"unexpected PDN image request stop: {wait}")
    image_request, = m.dma_submissions

    edge_waits = []
    for _ in range(256):
        m.pc += 2  # Only acknowledge requests; no buffer contents are fabricated.
        stop = m.run(4096, stop_at=BASE + 0x1632)
        if stop == {"pc": BASE + 0x1632, "reason": "slice boundary"}:
            break
        if stop["reason"] != "wfi":
            raise ValueError(f"unexpected PDN image continuation: {stop}")
        edge_waits.append(stop["pc"])
    else:
        raise ValueError("PDN edge continuation did not return")

    return {"frame": frame, "window_xywh": [signed(n, 32) for n in window], "wait": wait,
            "image_request": image_request, "edge_waits": edge_waits,
            "edge_requests": m.dma_submissions[1:],
            "compute": trace_geometry_call(m, file_base, catalog)}


def worker_images(pdn):
    bases = (0x800, 0x2800, 0x4800, 0x8000, 0xa000)
    return {dst: (offset, pdn[offset:offset + (a >> 16) + 1])
            for offset, dst, a, b in startup_rows(pdn)
            if dst in bases and a & 0xffff == 0x4000 and b == a & 0xffff0000}


def worker_after_crop(base, images, crop_data):
    _, image = images[base]
    worker = BootMachine(b"", scratch_base=0xc000, scratch_size=0xc000)
    worker.regions = tuple((dst, data) for dst, (_, data) in images.items()) + ((0x48170, b"\0"),)
    worker.pc, worker.stack_high = base, word(image, 0x30)
    worker.stack_low = base + len(image)
    worker.stack = bytearray(worker.stack_high - worker.stack_low)
    worker.fixture_reads = {addr: deque([0] * 16) for addr in (0x48004, 0x48024, 0x48100, 0x48230)}
    worker.scratch = crop_data.copy()
    boot = worker.run(8192)
    if boot["reason"] != "wfi":
        raise ValueError(f"PDN worker {base:#x} did not wait at startup: {boot}")
    worker.pc += 2
    compute = {"pc": base + 0xb74, "reason": "unsupported opcode", "opcode": 0xa}
    crop = worker.run(8192)
    if crop != compute or worker.reg[4] != base + 0xac8:
        raise ValueError(f"unexpected PDN worker crop boundary: {crop}")

    # Resume an isolated control slice at the real return, without a crop result.
    worker.pc = worker.link
    return worker, boot["pc"], compute


def trace_weight_workers(pdn, crop_data, weight_data):
    """Replay separately mapped workers using the appropriate producer snapshot."""
    images = worker_images(pdn)
    reports = []
    for base, (source, _) in images.items():
        worker, boot, compute = worker_after_crop(base, images, crop_data)
        crop_registers = worker.reg[4:8]
        worker.scratch = weight_data.copy()
        start_write, start_read = len(worker.mmio_writes), len(worker.scratch_reads)
        waits = []
        for _ in range(2):
            wait = worker.run(8192)
            if wait["reason"] != "wfi":
                raise ValueError(f"unexpected PDN worker after crop: {wait}")
            waits.append(wait["pc"])
            worker.pc += 2
        stop = worker.run(8192)
        if stop != compute or worker.reg[4] != base + 0x44:
            raise ValueError(f"unexpected PDN worker weight boundary: {stop}")
        reports.append({
            "local_base": base, "boot_wait": boot, "crop_registers_r4_to_r7": crop_registers,
            "waits": waits, "stop": stop, "record_member_offset": source + worker.reg[4] - base,
            "registers_r4_to_r7": worker.reg[4:8],
            "parameter_reads": [r for r in worker.scratch_reads[start_read:]
                                if r["address"] in (0x1782f, 0x17834) or 0xc1f8 <= r["address"] < 0xc204],
            "aperture": {hex(w["address"]): w["value"] for w in worker.mmio_writes[start_write:]
                         if 0x40190 <= w["address"] <= 0x40330},
        })
    return reports


def pdn_output_border(scratch):
    """Confirm byte coverage of the scalar zero writes, without filling interiors."""
    zero = {(x, y) for y in range(22) for x in range(22)
            if x in (0, 21) or y in (0, 21)}
    planes = [0xdaa0 + 0x420 * i for i in range(6)]
    for base in planes:
        expected = {base + y * 48 + 2 * x + b for x, y in zero for b in (0, 1)}
        actual = {addr for addr in range(base, base + 0x420) if addr in scratch}
        if actual != expected or any(scratch[addr] for addr in actual):
            raise ValueError(f"unexpected PDN scalar border writes at {base:#x}")
    return {"plane_bases": planes, "row_bytes": 48, "logical_shape": [22, 22],
            "unwritten_interior_shape": [20, 20], "known_zero_bytes": len(zero) * 2 * len(planes)}


def worker_scratch(m):
    return {addr: value for addr, value in m.scratch.items() if 0xc000 <= addr < 0x18000}


def copy_dictionary_rows(m, dictionary, requests, copies, skipped):
    for request in requests:
        for source, target, a, b in request["rows"]:
            offset = source - DICTIONARY_DMA_BASE
            if not 0 <= offset < len(dictionary):
                skipped.append([source, target, a, b])
                continue
            size = (a >> 16) + 1
            if (a != b or a & 0xffff or offset + size > len(dictionary) or
                    not m.in_scratch(target, size)):
                raise ValueError("unexpected PDN dictionary copy format")
            m.scratch.update(enumerate(dictionary[offset:offset + size], target))
            copies.append({"submit_pc": request["submit_pc"], "offset": offset,
                           "target": target, "size": size})


def trace_weight_setup(m, pdn, dictionary):
    """Start an isolated post-crop slice; materialize only known dictionary copies."""
    if sha256(dictionary).hexdigest() != PDN_DICTIONARY_SHA:
        raise ValueError("expected the pinned A7 IV PDN dictionary")
    if (m.pc, m.link) != (COMPUTE_STOP["pc"], BASE + 0x14d2):
        raise ValueError("expected to resume at the recorded original post-crop return")
    crop_data = worker_scratch(m)
    m.pc = m.link
    m.fixture_reads[0x48168] = deque([CATALOG_VA] * 8)
    zero_registers = (0x48100, 0x48024, 0x48004, 0x48230)
    m.fixture_reads.update({addr: deque([0] * 16) for addr in zero_registers})
    m.regions += ((DICTIONARY_CPU_VA, dictionary), (0x48170, b"\0"))
    m.extra_scratch += ((0x6800, 0x7800), (0x8000, 0x17000))
    last_submission = len(m.dma_submissions)
    start_write, start_read = len(m.mmio_writes), len(m.scratch_reads)
    copies, skipped, waits = [], [], []
    for _ in range(24):
        stop = m.run(12000)
        if stop == COMPUTE_STOP:
            break
        if stop["reason"] != "wfi":
            raise ValueError(f"unexpected stop while tracing PDN parameter setup: {stop}")
        waits.append(stop["pc"])
        copy_dictionary_rows(m, dictionary, m.dma_submissions[last_submission:], copies, skipped)
        last_submission = len(m.dma_submissions)
        m.pc += 2  # Acknowledge the request, without computing or moving activations.
    else:
        raise ValueError("PDN did not reach its second compute call")
    if (m.link != BASE + 0x22bc or m.reg[4] != BASE + 0x38 or
            any(0x17000 <= address < 0x173c8 for address in m.scratch)):
        raise ValueError("unexpected PDN second-stage call or synthetic activation")
    weight_data = worker_scratch(m)
    return {
        "entry": BASE + 0x14d2, "waits": waits, "dictionary_copies": copies,
        "unexecuted_rows": skipped, "stop": stop, "registers_r4_to_r7": m.reg[4:8],
        "zero_mmio_fixture": [*zero_registers, 0x48170],
        "parameter_reads": [r for r in m.scratch_reads[start_read:]
                            if r["pc"] in (BASE + 0x1f62, BASE + 0x1f66, BASE + 0x227c)],
        "aperture": {hex(w["address"]): w["value"] for w in m.mmio_writes[start_write:]
                     if 0x40190 <= w["address"] <= 0x40330},
        "worker_calls": trace_weight_workers(pdn, crop_data, weight_data),
        "output_borders": pdn_output_border(weight_data),
    }


def scalar_head_fixture(m, values):
    """Independent oracle for the original scalar tail, not computed input."""
    if len(values) != 96:
        raise ValueError("expected 96 input fixture halfwords")
    replay = deepcopy(m)
    for index, value in enumerate(values):
        replay.scratch.update(enumerate(pack("<h", value), HEAD_SOURCE + 8 * index))
    stop = replay.run(6000, stop_at=DOT_END)
    if stop != {"pc": DOT_END, "reason": "slice boundary"}:
        raise ValueError(f"unexpected PDN scalar dot-product stop: {stop}")
    dot = signed(replay.reg[7], 32)
    stop = replay.run(256, stop_at=SCORE_FP_ENTRY)
    if stop != {"pc": SCORE_FP_ENTRY, "reason": "slice boundary"}:
        raise ValueError(f"unexpected PDN scalar quantization stop: {stop}")
    quantized = signed(replay.reg[1], 32)
    for _ in range(3):
        stop = replay.run(6000, stop_at=SCORE_TO_Q30)
        if stop == {"pc": SCORE_TO_Q30, "reason": "slice boundary"}:
            break
        if stop["reason"] != "wfi":
            raise ValueError(f"unexpected PDN FP32 head stop: {stop}")
        replay.pc += 2
    else:
        raise ValueError("PDN scalar FP32 head did not reach integer conversion")

    fp32_bits = replay.reg[3]
    stop = replay.run(128, stop_at=SCORE_WRITTEN)
    if stop != {"pc": SCORE_WRITTEN, "reason": "slice boundary"}:
        raise ValueError(f"unexpected PDN score write stop: {stop}")
    address = replay.reg[7]
    score = replay.read(address, 4)
    return {"dot": dot, "i16": quantized, "fp32_bits": fp32_bits,
            "written_address": address, "q30": signed(score, 32)}


def record_info(image, offset):
    if not 0 <= offset <= len(image) - len(MARKER) or offset % 4:
        raise ValueError("PDN record pointer outside code image")
    end = next((i for i in range(offset, len(image) - len(MARKER) + 1, 4)
                if image[i:i + len(MARKER)] == MARKER), None)
    if end is None:
        raise ValueError("expected a marker after the original PDN final record")
    return {
        "record_image_offset": offset,
        "record_body_words": (end - offset) // 4,
        "record_sha256_including_marker": sha256(image[offset:end + len(MARKER)]).hexdigest(),
    }


def stack_words(m):
    return [m.read(m.reg[15] + 4 * i, 4) for i in range(4)]


def aperture_writes(m, start):
    return [w for w in m.mmio_writes[start:] if 0x40000 <= w["address"] < 0x40400]


def final_wrapper_args(machine):
    """Replay the scalar frame setup independently of either unknown opcode."""
    wrapper = deepcopy(machine)
    entry = wrapper.pc
    wrapper.pc += 2
    stop = wrapper.run(16)
    if stop != {"pc": entry + 0x16, "reason": "unsupported opcode", "opcode": 9}:
        raise ValueError(f"unexpected PDN final wrapper stop: {stop}")
    if wrapper.reg[:15] != [machine.reg[4], *machine.reg[1:15]]:
        raise ValueError("PDN final wrapper did not preserve the argument registers")
    frame = wrapper.reg[15]
    if frame != machine.reg[15] - 0x24 or stack_words(machine) != [
            wrapper.read(frame + 0x24 + 4 * i, 4) for i in range(4)]:
        raise ValueError("PDN final wrapper lost its caller stack arguments")
    return {"stop": stop, "r0_to_r7": wrapper.reg[:8], "caller_stack_offset": 0x24}


def worker_job(base, images, crop_data, shared_data):
    worker, _, _ = worker_after_crop(base, images, crop_data)
    worker.scratch = shared_data.copy()
    waits = []
    for _ in range(2):
        stop = worker.run(8192)
        if stop["reason"] != "wfi":
            raise ValueError(f"unexpected PDN worker wait: {stop}")
        waits.append(stop["pc"])
        worker.pc += 2
    return worker, waits


def map_call(machine, stop, first_write):
    return {
        "stop": stop, "return_pc": machine.link, "registers_r4_to_r7": machine.reg[4:8],
        "aperture": {hex(w["address"]): w["value"] for w in aperture_writes(machine, first_write)
                     if 0x40190 <= w["address"] <= 0x40330},
    }


def map_worker_calls(images, crop_data, shared_data):
    reports = []
    for base, (source, image) in images.items():
        worker, waits = worker_job(base, images, crop_data, shared_data)
        first_read = len(worker.scratch_reads)
        first_write = len(worker.mmio_writes)
        calls = []
        for _ in range(8):
            stop = worker.run(8192)
            if stop.get("opcode") != 0xa or worker.reg[4] != base + 0x184:
                raise ValueError(f"unexpected PDN map worker call: {stop}")
            calls.append(map_call(worker, stop, first_write))
            first_write = len(worker.mmio_writes)
            worker.pc = worker.link  # No output is supplied to the next scalar call.
        stop = worker.run(8192)
        if stop["reason"] != "wfi":
            raise ValueError(f"expected PDN map worker completion wait: {stop}")
        reports.append({"local_base": base, "waits": waits, "completion_wait": stop["pc"],
                        "record_member_offset": source + 0x184, **record_info(image, 0x184),
                        "parameter_reads": [r for r in worker.scratch_reads[first_read:]
                                            if r["address"] in MAP_PARAMETER_ADDRS],
                        "calls": calls})
    return reports


def final_worker_calls(images, crop_data, final_data):
    reports = []
    for base, (source, image) in images.items():
        worker, waits = worker_job(base, images, crop_data, final_data)
        start_read, start_write = len(worker.scratch_reads), len(worker.mmio_writes)
        stop = worker.run(8192)
        if stop.get("opcode") != 0xa or worker.reg[4] != base + 0x994:
            raise ValueError(f"unexpected PDN final worker call: {stop}")
        reports.append({
            "local_base": base, "waits": waits, "stop": stop, "return_pc": worker.link,
            "registers_r4_to_r7": worker.reg[4:8], "stack_words": stack_words(worker),
            "isolated_wrapper": final_wrapper_args(worker),
            "record_member_offset": source + worker.reg[4] - base,
            **record_info(image, worker.reg[4] - base),
            "mailbox_reads": [r for r in worker.scratch_reads[start_read:] if 0xc1ac <= r["address"] < 0xc1dc],
            "aperture_writes": aperture_writes(worker, start_write),
        })
    return reports


def bind_final_programs(pdn, calls):
    """Bind the final stage's records to the program tables from actual DMA submits."""
    if [c["registers_r4_to_r7"][1] for c in calls] != [0, 1]:
        raise ValueError("expected both original PDN final batches in order")
    table, banks = overlays(pdn)
    bound = []
    loads = {}
    for call in calls:
        setup = call["setup_since_previous_compute"]
        for request in setup["dma_submissions"]:
            for source, target, a, b in request["rows"]:
                if not PROGRAM_BASE <= target < PROGRAM_BASE + len(banks) * PROGRAM_BANK_BYTES:
                    continue
                bank, rem = divmod(target - PROGRAM_BASE, PROGRAM_BANK_BYTES)
                image = banks[bank][4]
                if (rem or bank in loads or plane_slots(a, b) != image.slots or
                        source - CATALOG_VA + table != image.offset):
                    raise ValueError("PDN final program load does not match the original overlay")
                loads[bank] = image
        records = [setup, *setup["workers"]]
        if len(loads) != len(banks) or len(records) != len(banks):
            raise ValueError("PDN final stage does not have a program for each worker")
        bindings = []
        for bank, record in enumerate(records):
            start = record["record_member_offset"]
            size = 4 * record["record_body_words"]
            end = start + size
            if pdn[end:end + len(MARKER)] != MARKER:
                raise ValueError("PDN final stage record marker is missing")
            image = loads[bank]
            indices = sorted({(w >> 24) & 0x7f for w, in iter_unpack("<I", pdn[start:end])})
            if not indices or max(indices) >= image.slots:
                raise ValueError("PDN final record references an unavailable program slot")
            bindings.append({"bank": bank, "program_member_offset": image.offset,
                             "record_member_offset": start, "referenced_slots": indices})
        bound.append(bindings)
    return bound


def trace_later_weights(m, pdn, dictionary, crop_data, *, head_fixture=None):
    """Isolate later controller calls up to the first scalar head read."""
    if (m.pc, m.link) != (COMPUTE_STOP["pc"], BASE + 0x22bc):
        raise ValueError("expected the original second-stage boundary")
    m.pc = m.link
    m.extra_scratch += ((0x17000, 0x17800),)
    m.fixture_reads[0x48168] = deque([CATALOG_VA] * 128)
    m.fixture_reads.update({addr: deque([0] * 384) for addr in (0x48100, 0x48024, 0x48004, 0x48230)})
    last_submission = len(m.dma_submissions)
    first_write, first_submission = len(m.mmio_writes), last_submission
    first_scratch, first_read = len(m.scratch_trace), len(m.scratch_reads)
    images = worker_images(pdn)
    waits, copies, skipped, calls = [], [], [], []
    maps = None
    for _ in range(128):
        stop = m.run(24000, stop_at=DOT_ENTRY)
        if stop == {"pc": DOT_ENTRY, "reason": "slice boundary"}:
            break
        if stop.get("opcode") == 0xa and stop["pc"] in (COMPUTE_STOP["pc"], FINAL_WRAPPER):
            call = {"stop": stop, "return_pc": m.link, "registers_r4_to_r7": m.reg[4:8],
                    "dictionary_copy_count": len(copies)}
            if stop == COMPUTE_STOP and m.link == MAP_RETURN:
                if maps is None:
                    base, image = m.regions[0]
                    maps = {
                        **record_info(image, m.reg[4] - base), "controller_calls": [],
                        "setup_dma_submissions": m.dma_submissions[first_submission:],
                        "parameter_reads": [r for r in m.scratch_reads[first_read:]
                                            if r["address"] in MAP_PARAMETER_ADDRS],
                        "workers": map_worker_calls(images, crop_data, worker_scratch(m)),
                    }
                maps["controller_calls"].append(map_call(m, stop, first_write))
            elif stop["pc"] == FINAL_WRAPPER and m.link == FINAL_RETURN:
                base, image = m.regions[0]
                call["setup_since_previous_compute"] = {
                    "record_member_offset": 0x100 + word(pdn, 0x130) + m.reg[4] - base,
                    **record_info(image, m.reg[4] - base), "stack_words": stack_words(m),
                    "isolated_wrapper": final_wrapper_args(m),
                    "aperture_writes": aperture_writes(m, first_write),
                    "mailbox_writes": [w for w in m.scratch_trace[first_scratch:]
                                       if w["address"] == 0xc000 or 0xc1ac <= w["address"] < 0xc1dc],
                    "dma_submissions": m.dma_submissions[first_submission:],
                    "workers": final_worker_calls(images, crop_data, worker_scratch(m)),
                }
            calls.append(call)
            first_write, first_submission = len(m.mmio_writes), len(m.dma_submissions)
            first_scratch, first_read = len(m.scratch_trace), len(m.scratch_reads)
            m.pc = m.link  # Another isolated return; no computed buffer or return value supplied.
        elif stop["reason"] == "wfi":
            waits.append(stop["pc"])
            copy_dictionary_rows(m, dictionary, m.dma_submissions[last_submission:], copies, skipped)
            last_submission = len(m.dma_submissions)
            m.pc += 2
        else:
            raise ValueError(f"unexpected PDN later stop: {stop}")
    else:
        raise ValueError("PDN did not reach its scalar head")
    if maps is None or len(maps["controller_calls"]) != 8:
        raise ValueError("expected eight original PDN map calls on the controller")
    weight_base = m.read(m.reg[15] + 0x98, 4)
    if weight_base != DICTIONARY_CPU_VA + SCALAR_HEAD_OFFSET:
        raise ValueError("unexpected scalar head coefficient pointer")
    result = {"calls": calls, "waits": waits, "dictionary_copies": copies,
              "unexecuted_rows": skipped, "map_stage": maps, "scalar_head_entry": stop,
              "scalar_head_dictionary_offset": weight_base - DICTIONARY_CPU_VA,
              "final_program_bindings": bind_final_programs(
                  pdn, [c for c in calls if c["return_pc"] == FINAL_RETURN])}
    if head_fixture is not None:
        result["scalar_head_on_explicit_fixture"] = scalar_head_fixture(m, head_fixture)
    result["earlier_head_base_writes"] = [w for w in m.scratch_trace if
                                          w["address"] <= HEAD_SOURCE < w["address"] + w["size"]]
    first_read = len(m.scratch_reads)
    result["unseeded_stop"] = m.run(6000)
    result["unseeded_head_reads"] = m.scratch_reads[first_read:]
    return result


def probe(pdn):
    fixture = bytes(range(0x100))
    return {"mode_one": call_setup(pdn, 0x50001fd4, fixture),
            "mode_two": call_setup(pdn, 0x50002060, fixture),
            "isolated_program_request": program_request(pdn),
            "first_stage_request": first_stage_request(pdn),
            "image_geometry_request": image_geometry_request(pdn)}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pdn", type=Path)
    args = cli.parse_args()
    print(json.dumps(probe(args.pdn.read_bytes()), indent=2))
