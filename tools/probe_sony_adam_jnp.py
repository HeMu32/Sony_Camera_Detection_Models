"""Replay the original Adam JNP register interface and CPU pose packing offline."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import iter_unpack, pack, unpack_from
import sys

from unicorn import (
    UC_ARCH_ARM, UC_HOOK_CODE, UC_HOOK_MEM_WRITE, UC_MODE_ARM, UC_PROT_EXEC,
    UC_PROT_READ, UC_PROT_WRITE, Uc,
)
from unicorn.arm_const import (
    UC_ARM_REG_FP, UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2,
    UC_ARM_REG_SB, UC_ARM_REG_SP,
)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_sony_cp_region_scale import BASE as CP_BASE, RETURN as CP_RAM, CpSlices
from probe_sony_adam_xhd import (
    ARGUMENT_REGISTERS, HEADER, LAYOUTS, PARAMETERS, PHYSICAL, RETURN, STACK,
)


JNP = 0x03003000
INPUT = PARAMETERS + 0x400
OUTPUT = PARAMETERS + 0x900
CONTROL_BYTES = 0xe4
COMMAND6_BYTES = 0xec
POSE_COMMAND_OFFSET = 0x530
NPU_COMMAND_BYTES = 0x414
POSE_BYTES = 0x68
POLL_REGISTERS = (0x03002204, 0x03002284, 0x03002304, 0x03002384, 0x03002404, 0x03002484)


@dataclass(frozen=True)
class JnpLayout:
    registers: int
    npu_args: int
    prepare_pose: int
    configure: int
    peak_counts: int
    pose_count: int
    pack_poses: int
    cache: int


JNP_LAYOUTS = {
    "A7R V": JnpLayout(0x0fe82cb8, 0x1bfe4, 0x16098, 0x34a18, 0x34e80, 0x34ff0, 0x161b4, 0x35038),
    "A1 II": JnpLayout(0x0fe92af0, 0x212b4, 0x16440, 0x41bc4, 0x4202c, 0x4219c, 0x1655c, 0x421e4),
}
CP_PROJECTION = {"A7R V": (0x1ad06c, 0x1ad0ce), "A1 II": (0x1e093c, 0x1e099e)}
CP_POSTURE = {"A7R V": (0x1ac5f0, 0x1ac76e), "A1 II": (0x1dfec0, 0x1e003e)}


@dataclass(frozen=True)
class PostureJob:
    npu: bytes
    jnp: bytes
    postprocess: tuple[int, ...]
    procedure_list: tuple[tuple[int, int, int], ...]


class JnpProbe:
    def __init__(self, adam):
        pinned = LAYOUTS.get(sha256(adam).hexdigest())
        if pinned is None or unpack_from("<2I", adam, 8) != (PHYSICAL, len(adam) - HEADER):
            raise ValueError("expected a pinned Adam image")
        self.model = pinned.model
        self.layout = JNP_LAYOUTS[self.model]
        expected = (*POLL_REGISTERS, JNP)
        if unpack_from("<7I", adam, self.layout.registers - PHYSICAL + HEADER) != expected:
            raise ValueError("unexpected JNP register table in the original image")
        m = self.machine = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        payload = adam[HEADER:]
        m.mem_map(PHYSICAL, (len(payload) + 0xfff) & ~0xfff, UC_PROT_READ | UC_PROT_EXEC)
        m.mem_write(PHYSICAL, payload)
        for address, size in ((JNP - 0x1000, 0x2000), (PARAMETERS, 0x1000), (STACK, 0x4000)):
            m.mem_map(address, size, UC_PROT_READ | UC_PROT_WRITE)

    def call(self, entry, *args):
        m = self.machine
        for reg, value in zip(ARGUMENT_REGISTERS, args):
            m.reg_write(reg, value)
        m.reg_write(UC_ARM_REG_LR, RETURN)
        m.reg_write(UC_ARM_REG_SP, STACK + 0x3f00)
        m.emu_start(PHYSICAL + entry - HEADER, RETURN, count=100_000)
        if m.reg_read(UC_ARM_REG_PC) != RETURN:
            raise ValueError("JNP slice did not return")
        return m.reg_read(ARGUMENT_REGISTERS[0])

    def configure(self, block):
        if len(block) != CONTROL_BYTES:
            raise ValueError("expected a 0xe4-byte JNP register input block")
        m = self.machine
        m.mem_write(PARAMETERS, bytes(block))
        writes = []

        def record(_uc, _access, address, size, value, _data):
            if size != 4:
                raise ValueError("unexpected JNP register write width")
            writes.append((address - JNP, value & 0xffffffff))

        hook = m.hook_add(UC_HOOK_MEM_WRITE, record, begin=JNP, end=JNP + 0xfff)
        try:
            return self.call(self.layout.configure, PARAMETERS), writes
        finally:
            m.hook_del(hook)

    def prepare_pose(self, payload):
        """Execute the original command-6 preparation, before the outer start call."""
        if len(payload) != COMMAND6_BYTES:
            raise ValueError("expected the original-size JNP command-6 block")
        m = self.machine
        m.mem_write(PARAMETERS, bytes(0x13c))
        address = STACK + POSE_COMMAND_OFFSET
        m.mem_write(address, bytes(payload))
        if self.call(self.layout.prepare_pose, PARAMETERS, address) != 0:
            raise ValueError("original Adam JNP preparation failed")
        return bytes(m.mem_read(PARAMETERS + 0x34, CONTROL_BYTES))

    def npu_boundary(self, payload):
        """Execute the original NPU descriptor copy, stopping before the backend call."""
        if len(payload) != NPU_COMMAND_BYTES:
            raise ValueError("expected the original-size Adam NPU parameter block")
        m = self.machine
        context, backend, vtable = PARAMETERS, OUTPUT, OUTPUT + 0x40
        m.mem_write(context + 8, pack("<I", backend))
        m.mem_write(backend, pack("<I", vtable))
        m.mem_write(vtable + 0x14, pack("<I", RETURN))
        m.mem_write(INPUT, bytes(payload))
        start = PHYSICAL + self.layout.npu_args - HEADER
        stop = start + 0xc4
        m.reg_write(UC_ARM_REG_SP, STACK + 0x3f00)
        m.reg_write(UC_ARM_REG_LR, RETURN)
        m.reg_write(UC_ARM_REG_R0, context)
        m.reg_write(UC_ARM_REG_R1, INPUT)
        m.emu_start(start, stop, count=1000)
        if m.reg_read(UC_ARM_REG_PC) != stop:
            raise ValueError("original NPU parameters did not reach the backend call")
        this, graph, ins, num_inputs = (m.reg_read(r) for r in ARGUMENT_REGISTERS)
        outs, num_outputs = unpack_from("<2I", m.mem_read(m.reg_read(UC_ARM_REG_SP), 8))
        if this != backend or not 1 <= num_inputs <= 64 or not 1 <= num_outputs <= 64:
            raise ValueError("unexpected arguments at the original NPU call")
        return {
            "graph": graph,
            "inputs": tuple(iter_unpack("<2I", m.mem_read(ins, num_inputs * 8))),
            "outputs": tuple(iter_unpack("<2I", m.mem_read(outs, num_outputs * 8))),
        }

    def counts(self, peaks, poses, *, busy=False):
        """Exercise register decoders with explicitly synthetic hardware results."""
        if len(peaks) != 24 or any(not 0 <= n <= 31 for n in peaks) or not 0 <= poses <= 255:
            raise ValueError("expected 24 five-bit peak counts and one byte pose count")
        m = self.machine
        m.mem_write(JNP + 0xa8, bytes(n | 0xe0 for n in peaks))
        m.mem_write(JNP + 0xc, pack("<I", 0xff00ff00 | poses << 16))
        m.mem_write(POLL_REGISTERS[0], pack("<I", int(busy) << 31))
        peak_status = self.call(self.layout.peak_counts, 1, OUTPUT)
        pose_status = self.call(self.layout.pose_count, 1, OUTPUT + 24)
        counts = list(m.mem_read(OUTPUT, 25))
        return (peak_status, pose_status), counts[:24], counts[24]

    def pack_poses(self, records, *, parts, tag):
        """Run the CPU copy from synthetic JNP records; variables are not model output."""
        if not 1 <= parts <= 24 or not 0 <= tag <= 255 or len(records) > 10:
            raise ValueError("expected up to ten records for 1..24 parts")
        stride = (parts + 3) * 4
        if any(len(row) != stride for row in records):
            raise ValueError("incorrect JNP source record size")
        m = self.machine
        m.mem_write(PARAMETERS, bytes(0x13c))
        m.mem_write(PARAMETERS + 0x6f, bytes((tag,)))
        m.mem_write(PARAMETERS + 0x73, bytes((parts - 1,)))
        m.mem_write(PARAMETERS + 0x114, pack("<3I", stride, INPUT, OUTPUT))
        m.mem_write(PARAMETERS + 0x138, bytes((len(records),)))
        if records:
            m.mem_write(INPUT, b"".join(records))
        m.mem_write(OUTPUT, b"\xa5" * (4 + 10 * POSE_BYTES))
        expected_size = 4 + len(records) * POSE_BYTES if records and tag else 4
        calls = []

        def flush(uc, _pc, _size, _data):
            args = tuple(uc.reg_read(r) for r in ARGUMENT_REGISTERS)
            if args != (OUTPUT, expected_size, 3, 2):
                raise ValueError(f"unexpected JNP cache request: {args}")
            calls.append(args)
            uc.reg_write(ARGUMENT_REGISTERS[0], 0)
            uc.reg_write(UC_ARM_REG_PC, uc.reg_read(UC_ARM_REG_LR))

        cache = PHYSICAL + self.layout.cache - HEADER
        hook = m.hook_add(UC_HOOK_CODE, flush, begin=cache, end=cache)
        try:
            status = self.call(self.layout.pack_poses, PARAMETERS)
        finally:
            m.hook_del(hook)
        if status != 0 or len(calls) != 1:
            raise ValueError("unexpected JNP pack result")
        return bytes(m.mem_read(OUTPUT, expected_size))


class PoseProjection(CpSlices):
    """Replay the CP geometry conversion for one packed result, before publication."""

    def project(self, result, *, canvas):
        if len(result) < 4 or result[1] > 10:
            raise ValueError("expected complete Adam JNP pose records")
        if len(canvas) != 2 or not all(0 < n <= 0xffff for n in canvas):
            raise ValueError("expected two positive CP canvas dimensions")
        count = min(result[0], result[1])
        if count == 0:
            return []
        if len(result) != 4 + POSE_BYTES * result[1]:
            raise ValueError("expected complete Adam JNP pose records")
        m = self.machine
        sp, source, geometry = (CP_RAM + off for off in (0x80, 0x800, 0x500))
        m.mem_write(source, result)
        m.mem_write(geometry + 0xa, pack("<2H", *canvas))
        m.mem_write(sp + 0x4c, pack("<I", geometry))
        start, stop = (CP_BASE + pc for pc in CP_PROJECTION[self.firmware.model])
        boxes = []
        # This upper bound comes from the record loop; later publication is not replayed.
        for i in range(count):
            offset = 8 + i * POSE_BYTES
            x1, y1, x2, y2 = result[offset:offset + 4]
            if x2 < x1 or y2 < y1:
                raise ValueError("inverted JNP rectangle")
            for register, value in ((UC_ARM_REG_SP, sp), (UC_ARM_REG_R0, POSE_BYTES),
                                    (UC_ARM_REG_R1, source), (UC_ARM_REG_R2, 0),
                                    (UC_ARM_REG_SB, i), (UC_ARM_REG_LR, 0x30c30c31)):
                m.reg_write(register, value)
            self.execute(start, stop)
            x, y = unpack_from("<2I", m.mem_read(sp + 8, 8))
            w, h = unpack_from("<2I", m.mem_read(sp + 0x10, 8))
            boxes.append((x, y, x + w, y + h))
        return boxes


def original_posture_job(image, *, expected_model=None, source_size=(512, 336)):
    """Run the CP PostureEstimationParam builders without queues or transfer."""
    cp = CpSlices(image)
    if expected_model is not None and expected_model != cp.firmware.model:
        raise ValueError("CP and Adam images must describe the same camera")
    if len(source_size) != 2 or not all(0 < n <= 0xffff for n in source_size):
        raise ValueError("expected two positive CP source dimensions")
    m = cp.machine
    base, obj, sp = CP_RAM + 0x1000, CP_RAM + 0x800, CP_RAM + 0x400
    cp.map_ram(base)
    m.mem_write(obj + 0x40, pack("<I", base))
    m.mem_write(obj + 0x1e, pack("<2H", *source_size))
    m.mem_write(sp + 0x8c, pack("<2I", base, STACK))
    m.reg_write(UC_ARM_REG_FP, obj)
    m.reg_write(UC_ARM_REG_SP, sp)
    cp.execute(*(CP_BASE + pc for pc in CP_POSTURE[cp.firmware.model]), limit=2048)
    schedule = unpack_from("<6I", m.mem_read(base + 0x61c, 24))
    if schedule != (STACK + 0x74, 2, STACK + POSE_COMMAND_OFFSET, 6, STACK + 0x634, 2):
        raise ValueError("unexpected original JNP schedule")
    count, = unpack_from("<I", m.mem_read(base, 4))
    if count != 14:
        raise ValueError("unexpected CP PostureEstimationParam procedure count")
    return PostureJob(
        bytes(m.mem_read(base + 0x11c, NPU_COMMAND_BYTES)),
        bytes(m.mem_read(base + POSE_COMMAND_OFFSET, COMMAND6_BYTES)),
        schedule,
        tuple(iter_unpack("<HHI", m.mem_read(base + 4, count * 8))),
    )


def original_pose_parameters(image, *, expected_model=None):
    job = original_posture_job(image, expected_model=expected_model)
    return job.jnp, job.postprocess


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("adam", type=Path)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("--parameter-block", type=Path, help="Explicit 0xe4-byte JNP register input")
    source.add_argument("--cp-defaults", type=Path, help="Original CP PostureEstimationParam initializer")
    args = cli.parse_args()
    probe = JnpProbe(args.adam.read_bytes())
    metadata = {}
    if args.cp_defaults:
        job = original_posture_job(args.cp_defaults.read_bytes(), expected_model=probe.model)
        schedule = job.postprocess
        block = probe.prepare_pose(job.jnp)
        metadata = {"command6_sha256": sha256(job.jnp).hexdigest(),
                    "npu_boundary": probe.npu_boundary(job.npu),
                    "procedure_list": job.procedure_list,
                    "procedure_records": [[hex(address), command]
                                          for address, command in zip(schedule[::2], schedule[1::2])]}
    else:
        block = args.parameter_block.read_bytes()
    status, writes = probe.configure(block)
    print(json.dumps({"model": probe.model, "status": status, **metadata,
                      "writes": {hex(JNP + offset): hex(value) for offset, value in writes}}, indent=2))
