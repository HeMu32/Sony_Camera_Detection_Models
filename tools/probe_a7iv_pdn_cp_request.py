"""Replay A7 IV CP's PDN parameter packer and the original SA image request."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import pack
import sys

from unicorn import UC_ARCH_ARM, UC_MODE_ARM, UC_PROT_ALL, UC_PROT_EXEC, UC_PROT_READ, Uc
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R4, UC_ARM_REG_R6, UC_ARM_REG_R8,
    UC_ARM_REG_R10, UC_ARM_REG_SB, UC_ARM_REG_SP,
)

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_a7iv_fdn_cp_request import (
    CP_BASE, CP_SHA256, DRIVER, INPUT, MANAGER, PARAM_VA, PYRAMID_RECORD, STACK, execute,
)
from probe_sa_pdn_shared import (
    BASE, geometry_machine, trace_image_geometry, trace_later_weights, trace_weight_setup, worker_scratch,
)


PDN_MANAGER_VTABLE = 0xc17fb4
PDN_PACK = 0x266be4
RETURN = 0x51000000
PARAM_SIZE = 0x30
CROP_ENTRY = BASE + 0x15a8
VALIDATOR = BASE + 0xc34
EYE_DISPATCH = BASE + 0xe38


def pyramid_frame_slice(m, levels, result_level):
    """Replay the scale-index transfer and the original PDN frame-record lookup."""
    if len(levels) != 5 or result_level not in range(5):
        raise ValueError("expected five frame records and an accepted result level 0..4")
    table, process, context, result, metadata = (DRIVER + off for off in
                                                 (0x6000, 0x6100, 0x6200, 0x6800, 0x6900))
    m.mem_write(table, b"".join(PYRAMID_RECORD.pack(*entry) for entry in levels))
    m.mem_write(process + 0xc, pack("<I", context))
    m.mem_write(context + 0x464, b"\1")
    m.mem_write(context + 0x4f4, b"\1")
    m.mem_write(context + 0x51c, pack("<I", 5))
    m.mem_write(result + 0x34, pack("<I", result_level))
    m.mem_write(metadata, pack("<I", 42))
    execute(m, 0x726eac, 0x726eb2, ((UC_ARM_REG_R0, result), (UC_ARM_REG_R4, context)))
    # In this slice the original driver request begins at SP+8.
    execute(m, 0x28ae14, 0x28ae7e, (
        (UC_ARM_REG_R10, process), (UC_ARM_REG_R6, metadata), (UC_ARM_REG_R4, table),
        (UC_ARM_REG_SP, INPUT - 8),
    ))
    m.reg_write(UC_ARM_REG_SP, STACK)


def pack_parameters(cp, *, left=(281 << 18, 193 << 18), right=(300 << 18, 190 << 18),
                    frame=(640, 480, 704), scales=(64, 64), angle=1, node=8,
                    pyramid=None, result_level=None):
    if sha256(cp).hexdigest() != CP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 CP image")
    if (pyramid is None) != (result_level is None):
        raise ValueError("pyramid records and result level must be supplied together")
    slot = PDN_MANAGER_VTABLE + 0x20 - CP_BASE
    if cp[slot:slot + 4] != pack("<I", PDN_PACK | 1):
        raise ValueError("PDN manager vtable does not select the expected parameter packer")

    m = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    m.mem_map(CP_BASE, (len(cp) + 0xfff) & ~0xfff, UC_PROT_READ | UC_PROT_EXEC)
    m.mem_write(CP_BASE, cp)
    m.mem_map(DRIVER, 0x10000, UC_PROT_ALL & ~UC_PROT_EXEC)
    m.mem_map(PARAM_VA, 0x1000, UC_PROT_ALL & ~UC_PROT_EXEC)
    m.reg_write(UC_ARM_REG_SP, STACK)
    m.mem_write(MANAGER, pack("<I", PDN_MANAGER_VTABLE))
    m.mem_write(MANAGER + 8, pack("<I", PARAM_VA))
    m.mem_write(MANAGER + 0x20, pack("<2Q", 0x500000000, 0x500000000))
    m.mem_write(MANAGER + 0x48, pack("<2I", 0x80000, 0x40000))
    if pyramid is None:
        m.mem_write(INPUT + 4, pack("<IQ", 42, 0x500020000))
        m.mem_write(INPUT + 0x18, pack("<3H", *frame))
    else:
        pyramid_frame_slice(m, pyramid, result_level)
    m.mem_write(INPUT + 0x24, pack("<2i4x2i", *left, *right))
    m.mem_write(INPUT + 0x40, pack("<4H", *scales, angle, node))
    execute(m, 0x2668f4, 0x266930, ((UC_ARM_REG_R8, MANAGER), (UC_ARM_REG_SB, MANAGER + 0x20)))
    execute(m, PDN_PACK, 0x266c3a, ((UC_ARM_REG_R0, MANAGER), (UC_ARM_REG_R1, INPUT)))
    raw = bytes(m.mem_read(PARAM_VA, PARAM_SIZE))

    m.mem_write(INPUT, pack("<Q", 0x500001000))
    stop = 0xac95dc
    execute(m, 0x2759a8, stop, ((UC_ARM_REG_R0, INPUT), (UC_ARM_REG_LR, stop | 1)))
    return m.reg_read(UC_ARM_REG_R0), raw


def probe(cp, pdn, *, eye="left", dictionary=None, trace_later=False, head_fixture=None, **fixtures):
    if eye not in ("left", "right"):
        raise ValueError("expected left or right eye selection")
    if (trace_later and dictionary is None) or (head_fixture is not None and not trace_later):
        raise ValueError("the later replay requires dictionary data, and head fixtures require that replay")
    address, raw = pack_parameters(cp, **fixtures)
    m, file_base, catalog = geometry_machine(pdn)
    m.pc, m.link, m.reg[4] = VALIDATOR, RETURN, address
    m.regions += ((0x49e00, bytes(0x160)),)
    converter_stop = m.run(4096, stop_at=BASE + 0xc4a)
    if converter_stop != {"pc": BASE + 0xc4a, "reason": "slice boundary"}:
        raise ValueError(f"unexpected PDN parameter conversion: {converter_stop}")
    converted = m.reg[0]
    if (address, converted) != (0x40000400, 0xa0000400):
        raise ValueError("unexpected address translation for the pinned RAM fixture")

    # The original validator can write metadata to the supplied output buffer.
    # No detector score, dictionary result, or image pixels populate these regions.
    m.regions += ((converted, raw),)
    m.extra_scratch = ((0, 0x100), (0xe0020000, 0xe0020100))
    stop = m.run(8192, stop_at=EYE_DISPATCH)
    if stop == {"pc": EYE_DISPATCH, "reason": "slice boundary"}:
        # Enter the second loop iteration independently of any left-eye compute.
        if eye == "right":
            m.reg[8] = 1
        stop = m.run(512, stop_at=CROP_ENTRY)
    mask_writes = [w["value"] for w in m.mmio_writes if w["address"] == 0x48188]
    result = {"cp_parameter_address": address, "converted_address": converted,
              "cp_parameter_hex": raw.hex(" "), "dispatch_stop": stop,
              "validator_mask": mask_writes[-1] if mask_writes else 0}
    if stop == {"pc": CROP_ENTRY, "reason": "slice boundary"}:
        result["crop_call"] = {"eye": eye, "r4_r7": m.reg[4:8],
                               "stack_words": [m.read(m.reg[15] + i, 4) for i in range(0, 0x1c, 4)]}
        result["image_geometry"] = trace_image_geometry(m, file_base, catalog)
        crop_data = worker_scratch(m) if trace_later else None
        if dictionary is not None:
            result["isolated_weight_setup"] = trace_weight_setup(m, pdn, dictionary)
        if trace_later:
            result["isolated_later_control"] = trace_later_weights(
                m, pdn, dictionary, crop_data, head_fixture=head_fixture,
            )
    elif stop != {"pc": RETURN, "reason": "unmapped fetch", "address": RETURN, "size": 2}:
        raise ValueError(f"PDN dispatch did not reject or reach the crop call: {stop}")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("pdn", type=Path)
    cli.add_argument("--scales", type=int, nargs=2, default=(64, 64))
    cli.add_argument("--eye", choices=("left", "right"), default="left")
    cli.add_argument("--dictionary", type=Path, help="Replay the post-crop setup with pinned pdn_dic.bin")
    cli.add_argument("--trace-later", action="store_true", help="Continue scalar replay up to the unread head")
    args = cli.parse_args()
    print(json.dumps(probe(args.cp.read_bytes(), args.pdn.read_bytes(),
                           scales=args.scales, eye=args.eye,
                           trace_later=args.trace_later,
                           dictionary=args.dictionary.read_bytes() if args.dictionary else None), indent=2))
