"""Run the original Adam ARM XHD backend on explicit host tensor buffers."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct, pack, unpack_from
import sys
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from unicorn import UC_ARCH_ARM, UC_MODE_ARM, UC_PROT_EXEC, UC_PROT_READ, UC_PROT_WRITE, Uc
from unicorn import UC_HOOK_CODE
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_PC, UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7, UC_ARM_REG_SP,
)
from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers.xhd_cp_result import COUNT_OFFSET, Geometry, decode
from probe_sony_cp_xhd_nms import BLOCK, NMS_OFFSET, PARAMETER_SIZE, RESULT_SIZE, NmsProbe, TuningFixture


HEADER = 0x20
PHYSICAL, LINKED = 0x0fe10000, 0x00e10000
INPUTS = (0x10000000, 0x10100000, 0x10200000)
PARAMETERS, STACK, RETURN = 0x10300000, 0x10400000, 0x10500000
INPUT_LIMIT = 0x80000
AUX_OFFSET, AUX_BYTES = 0x32030, 0xc80a
GRID = (80, 40, 20, 10, 5, 3, 1)
CLASS_BYTES, REGRESSION_BYTES = 9 * AUX_BYTES, 4 * AUX_BYTES
PROFILES = {8: (5, 8, 3), 9: (7, 9, 5), 10: (3, 10, 4), 11: (7, 11, 4)}
RECORD = Struct("<4h2BH")
SHORT_PROCEDURE = (0x10, 3, NMS_OFFSET)
ARGUMENT_REGISTERS = (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3)


@dataclass(frozen=True)
class Layout:
    model: str
    entry: int
    cache: int
    copy: int
    scratch: int
    bss_end: int
    constructor: int
    accessor: int
    init_tail: tuple[int, int]
    dispatch: tuple[int, int]
    method: int
    selector: int
    selector_branch: int


LAYOUTS = {
    "8183d014a20d5317c15d37437baec73253f156ca999ec285b5c3b37d7d719c8c":
        Layout("A7R V", 0x18fb0, 0x35038, 0xdd50, 0x0fe88100, 0x0feb7000,
               0x67048, 0x16b94, (0x10cc4, 0x10d08), (0x10f4c, 0x10ff4), 0x16ac8, 0x16b18, 0x16b5c),
    "43c2d1c2817dc74db0634f2cef0a2c98312c4c06ecef5ff1e19821c5036a5d57":
        Layout("A1 II", 0x1a6ec, 0x421e4, 0xdd54, 0x0fe99aa0, 0x0fec4000,
               0x741d0, 0x17010, (0x10f64, 0x10fa8), (0x111ec, 0x11294), 0x16f14, 0x16f64, 0x16fc4),
}


def packed_parameters(cp, driver_id, *, published=None):
    if published is None:
        cp.describe(driver_id, TuningFixture(((128, 76),) * 10, 229, 129, 0), inputs=INPUTS, prefill=0)
        minimum_side = 0
    else:
        target, member = published
        addr, metadata = cp.load_published_tuning(target, member)
        cp.describe_at(driver_id, addr, inputs=INPUTS, prefill=0)
        minimum_side = metadata["minimum_side_byte"]
    return bytes(cp.machine.mem_read(BLOCK + NMS_OFFSET, PARAMETER_SIZE - NMS_OFFSET)), minimum_side


def verify_dispatch(m, layout, procedure):
    """Run the original table initialization and stop at the controller call."""
    if tuple(procedure) != SHORT_PROCEDURE:
        raise ValueError("expected the original short XHD procedure descriptor")
    proc, operation, _ = procedure
    context, descriptor = PARAMETERS + 0x100, PARAMETERS + 0x200

    def execute(start, stop=RETURN):
        end = stop if stop == RETURN else PHYSICAL + stop - HEADER
        m.reg_write(UC_ARM_REG_LR, RETURN)
        m.reg_write(UC_ARM_REG_SP, STACK + 0x3f00)
        m.emu_start(PHYSICAL + start - HEADER, end, count=1000)
        if m.reg_read(UC_ARM_REG_PC) != end:
            raise ValueError("original Adam dispatch did not reach its slice boundary")

    execute(layout.constructor)
    execute(layout.accessor)
    controller = m.reg_read(UC_ARM_REG_R0)
    m.reg_write(UC_ARM_REG_R4, context)
    execute(*layout.init_tail)
    if bytes(m.mem_read(context + 0x5c + 8 * proc, 1)) != b"\x03":
        raise ValueError("original procedure table did not select controller 3")
    # Attach the original global object to the context without calling the OS initializer.
    m.mem_write(context + 0x40, pack("<I", controller))
    m.mem_write(descriptor, pack("<HHI", proc, operation, PARAMETERS))
    m.reg_write(UC_ARM_REG_R0, context)
    m.reg_write(UC_ARM_REG_R1, descriptor)
    execute(*layout.dispatch)
    regs = tuple(m.reg_read(r) for r in ARGUMENT_REGISTERS)
    vtable, = unpack_from("<I", m.mem_read(controller, 4))
    method, = unpack_from("<I", m.mem_read(vtable + 0x18, 4))
    if (regs != (controller, context + 0x3c, operation, PARAMETERS) or
            method != PHYSICAL + layout.method - HEADER or m.reg_read(UC_ARM_REG_R5) != method):
        raise ValueError("original Adam dispatch did not forward the descriptor")

    # Execute the method's switch separately from OS acquire/release.
    m.reg_write(UC_ARM_REG_R7, operation)
    m.reg_write(UC_ARM_REG_R6, PARAMETERS)
    execute(layout.selector, layout.selector_branch)
    execute(layout.selector_branch, layout.entry)
    if m.reg_read(UC_ARM_REG_R0) != PARAMETERS:
        raise ValueError("original postprocessing command did not select the pinned CPU function")


def run(adam, params, classification, regression, auxiliary, *, procedure=SHORT_PROCEDURE):
    """Run the CPU backend, not the unavailable VIP9000 or hardware DMA."""
    layout = LAYOUTS.get(sha256(adam).hexdigest())
    if layout is None or unpack_from("<2I", adam, 8) != (PHYSICAL, len(adam) - HEADER):
        raise ValueError("expected one of the pinned original Adam images")
    if (len(params) != PARAMETER_SIZE - NMS_OFFSET or unpack_from("<3I", params) != INPUTS
            or unpack_from("<I", params, 0x54)[0] != INPUTS[1] + AUX_OFFSET):
        raise ValueError("expected an original CP XHD block for the local buffers")
    if (len(classification), len(regression), len(auxiliary)) != (CLASS_BYTES, REGRESSION_BYTES, AUX_BYTES):
        raise ValueError("invalid XHD input buffer sizes")

    m = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    payload = adam[HEADER:]
    size = (len(payload) + 0xfff) & ~0xfff
    for address in (LINKED, PHYSICAL):
        m.mem_map(address, size, UC_PROT_READ | UC_PROT_EXEC)
        m.mem_write(address, payload)
    m.mem_map(PHYSICAL + size, layout.bss_end - PHYSICAL - size, UC_PROT_READ | UC_PROT_WRITE)
    for address, extent in ((INPUTS[0], INPUT_LIMIT), (INPUTS[1], INPUT_LIMIT), (INPUTS[2], 0x1000),
                            (PARAMETERS, 0x1000), (STACK, 0x4000)):
        m.mem_map(address, extent, UC_PROT_READ | UC_PROT_WRITE)
    m.mem_write(PARAMETERS, params)
    m.mem_write(INPUTS[0], classification)
    m.mem_write(INPUTS[1], regression)
    m.mem_write(INPUTS[1] + AUX_OFFSET, auxiliary)
    m.mem_write(INPUTS[2], b"\xa5" * RESULT_SIZE)
    verify_dispatch(m, layout, procedure)

    services = []

    def service(uc, address, _size, kind):
        args = tuple(uc.reg_read(register) for register in ARGUMENT_REGISTERS)
        if kind == "copy":
            if args[:3] != (layout.scratch, INPUTS[1] + AUX_OFFSET, AUX_BYTES):
                raise ValueError("unexpected original XHD DMA call")
            result = 1  # Enter the firmware's own software-copy fallback.
        else:
            pointer, length, operation, scope = args
            prepare = args == (layout.scratch, AUX_BYTES, 2, 2)
            flush = (pointer == INPUTS[2] and 4 <= length <= RESULT_SIZE and
                     (length - 4) % RECORD.size == 0 and (operation, scope) == (3, 2))
            if not (prepare or flush):
                raise ValueError(f"unmodeled XHD cache operation: {args}")
            result = 0  # No cache maintenance is needed in Unicorn's RAM.
        services.append((kind, args))
        uc.reg_write(UC_ARM_REG_R0, result)
        uc.reg_write(UC_ARM_REG_PC, uc.reg_read(UC_ARM_REG_LR))

    for base in (LINKED, PHYSICAL):
        for kind, offset in (("cache", layout.cache), ("copy", layout.copy)):
            address = base + offset - HEADER
            m.hook_add(UC_HOOK_CODE, service, kind, begin=address, end=address)
    m.reg_write(UC_ARM_REG_R0, PARAMETERS)
    m.reg_write(UC_ARM_REG_SP, STACK + 0x3f00)
    m.reg_write(UC_ARM_REG_LR, RETURN)
    m.emu_start(PHYSICAL + layout.entry - HEADER, RETURN, count=24_000_000)
    if m.reg_read(UC_ARM_REG_PC) != RETURN or m.reg_read(UC_ARM_REG_R0) != 1:
        raise ValueError("original XHD backend did not return success")
    raw = bytes(m.mem_read(INPUTS[2], RESULT_SIZE))
    count, = unpack_from("<I", raw)
    if count > 50 or [kind for kind, _ in services] != ["cache", "copy", "cache"]:
        raise ValueError("unexpected original XHD output or service sequence")
    used = 4 + count * RECORD.size
    if services[-1][1][1] != used or raw[used:] != b"\xa5" * (RESULT_SIZE - used):
        raise ValueError("original XHD output does not match the published CP result layout")
    if bytes(m.mem_read(layout.scratch, AUX_BYTES)) != auxiliary:
        raise ValueError("firmware fallback did not copy the original auxiliary bytes")
    return layout.model, raw[:used]


def tensor_buffers(prefix, slot, *, transpose_aux=False):
    """Bind the supported slots to the native per-scale buffers, without dequantizing."""
    classes, _, _ = PROFILES[slot]
    parts = [[], [], []]
    for i, grid in enumerate(GRID):
        pixels = grid * grid
        for group, channels in enumerate((6 * classes, 24, 6)):
            path = prefix.with_name(f"{prefix.name}.{7 * group + i}.raw")
            data = path.read_bytes()
            if slot == 8 and i == 0:
                if len(data) != 6:
                    raise ValueError(f"expected the unused 80x80 interface slot: {path}")
                data = bytes(pixels * channels)
            if len(data) != pixels * channels:
                raise ValueError(f"unexpected slot-{slot:02d} tensor size: {path}")
            if group == 0:
                # The Adam code advances each scale by nine classes, not the graph's active classes.
                data += bytes(pixels * 6 * (9 - classes))
            if group == 2 and transpose_aux:
                data = np.frombuffer(data, dtype=np.uint8).reshape(channels, pixels).T.tobytes()
            parts[group].append(data)
    return tuple(b"".join(group) for group in parts)


def describe(raw, min_side, *, manifest=None):
    geometry = Geometry((320, 320), (320, 320), (160, 160), min_side_raw=min_side)
    records = []
    if manifest is not None and (manifest["input_size"] != [320, 320] or
                                 any(v <= 0 for v in manifest["scale_xy"])):
        raise ValueError("overlay requires the matching 320x320 input manifest")
    for record, result in zip(RECORD.iter_unpack(raw[4:]), decode(bytes(COUNT_OFFSET) + raw, geometry),
                              strict=True):
        row = {"xyxy_q14": record[:4], "type": record[4], "part": record[5], "confidence_q15": record[6],
               "xyxy_raster": result.box, "confidence": result.confidence}
        if manifest is not None:
            row["xyxy_source"] = [
                min(max((x - manifest["offset"][i % 2]) / manifest["scale_xy"][i % 2], 0),
                    manifest["source_size"][i % 2])
                for i, x in enumerate(result.box)
            ]
        records.append(row)
    return records


def overlay(records, source, manifest, output):
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    w, h = image.size
    if [w, h] != manifest["source_size"]:
        raise ValueError("source picture does not match its preprocessing manifest")
    image.thumbnail((1100, 750), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for row in records:
        x0, y0, x1, y1 = row["xyxy_source"]
        if row["confidence"] <= 0 or x1 <= x0 or y1 <= y0:
            continue
        box = (x0 * image.width / w, y0 * image.height / h, x1 * image.width / w, y1 * image.height / h)
        draw.rectangle(box, outline="#68f5da", width=3)
        draw.text((box[0] + 4, max(0, box[1] - 14)),
                  f"type {row['type']} / part {row['part']}  {row['confidence']:.3f}",
                  fill="#68f5da", stroke_width=2, stroke_fill="#000000")
    image.save(output)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("adam", type=Path)
    cli.add_argument("prefix", type=Path, help="Prefix of 21 audited raw tensor files")
    cli.add_argument("--slot", type=int, choices=PROFILES, required=True)
    cli.add_argument("--tuning-target", type=int, choices=range(1, 7),
                     help="Override the experimental profile's CP tuning target")
    cli.add_argument("--published-backup", type=Path, required=True)
    cli.add_argument("--backup-member", required=True)
    cli.add_argument("--source", type=Path)
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--overlay", type=Path)
    args = cli.parse_args()
    cp = NmsProbe(args.cp.read_bytes())
    if args.slot in (9, 11) and cp.firmware.model != "A1 II":
        cli.error("slot 9 and 11 profiles are only verified on A1 II")
    if not args.backup_member.startswith("0110_backup/"):
        cli.error("--backup-member must be an exact 0110_backup archive member")
    if len({args.source is None, args.manifest is None, args.overlay is None}) > 1:
        cli.error("--source, --manifest and --overlay must be used together")
    _, driver, target = PROFILES[args.slot]
    if args.tuning_target is not None:
        target = args.tuning_target
    with tarfile.open(args.published_backup, "r") as tar:
        with tar.extractfile(args.backup_member) as file:
            params, min_side = packed_parameters(cp, driver, published=(target, file.read()))
    if args.slot == 8 and params[0x10] != 1:
        raise ValueError("slot 8 requires the original skip flag for the unused 80x80 bank")
    (procedure,) = cp.trace_schedule_reader(0)
    model, output = run(args.adam.read_bytes(), params, *tensor_buffers(args.prefix, args.slot),
                        procedure=procedure)
    if model != cp.firmware.model:
        raise ValueError("Adam and CP models differ")
    manifest = (json.loads(args.manifest.read_text(encoding="utf-8"))[args.source.stem + ".raw"]
                if args.manifest is not None else None)
    records = describe(output, min_side, manifest=manifest)
    if manifest is not None:
        overlay(records, args.source, manifest, args.overlay)
    print(json.dumps({"model": model, "slot": args.slot, "driver_id": driver,
                      "tuning_target": target, "result_hex": output.hex(" "), "records": records}, indent=2))


if __name__ == "__main__":
    main()
