"""Replay the A7 IV Sequence_Face pyramid descriptors, without running hardware."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct

from unicorn import UC_ARCH_ARM, UC_MODE_ARM, UC_PROT_EXEC, UC_PROT_READ, UC_PROT_WRITE, Uc
from unicorn.arm_const import (
    UC_ARM_REG_LR, UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3, UC_ARM_REG_R4,
    UC_ARM_REG_R5, UC_ARM_REG_SB, UC_ARM_REG_SP,
)

from probe_a7iv_fdn_cp_request import CP_BASE, CP_SHA256, PYRAMID_RECORD, U32, execute


OUTPUT_RECORD = Struct("<Q4I")
BUS = 0xF35A0000
STATE, FRAME, MESSAGE = 0x30000000, 0x30002000, 0x30004000
GLOBAL = 0x01450000
FRAME_BASE = 0x500020000
RC_HANDLE, RC_SOURCE, RC_WORD = 0x00123456, 0x500200000, 0x00112233
MAX_FRAME_BYTES = 0xca800 + 2 * 64 * 30
RETURN = 0xac95dc


def replay_resize(m):
    """Read the original RCCT submission and compare its nonzero selector to zero."""
    execute(m, 0x4436aa, 0x4436ba, ((UC_ARM_REG_SP, FRAME), (UC_ARM_REG_SB, STATE)))
    handle, desc, callback, argument = (
        m.reg_read(r) for r in (UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3)
    )
    if desc != FRAME + 0x148:
        raise ValueError("unexpected RCCT descriptor pointer at the original submission")
    data = bytes(m.mem_read(desc, 0x12c))
    result = {
        "handle": handle, "callback": callback, "callback_arg": argument,
        "input_words": Struct("<QI").unpack_from(data, 0x10),
        "input_canvas": Struct("<2H").unpack_from(data, 0x50),
        "input_regions": tuple(Struct("<4H").iter_unpack(data[0x54:0x64])),
        "output_canvas": Struct("<2H").unpack_from(data, 0xd8),
        "output_regions": tuple(Struct("<4H").iter_unpack(data[0xdc:0xec])),
        "output_kind": U32.unpack_from(data, 0x100)[0],
        "output_selector": U32.unpack_from(data, 0x108)[0],
        "output_option": U32.unpack_from(data, 0x120)[0],
    }
    config, writeback, sp = STATE + 0x1700, STATE + 0x1800, FRAME - 0x100
    m.mem_write(sp, Struct("<4I").pack(0, 0, config, writeback))
    for label, selector in (("native", result["output_selector"]), ("zero_control", 0)):
        m.mem_write(desc + 0x108, U32.pack(selector))
        # The zero selector additionally calls the original output-buffer builder.
        stop = RETURN if selector else 0xa85488
        execute(m, 0xa853fc, stop, (
            (UC_ARM_REG_SP, sp), (UC_ARM_REG_R1, desc), (UC_ARM_REG_LR, RETURN | 1),
        ))
        result[label + "_config_word"] = U32.unpack(m.mem_read(config, 4))[0]
        result[label + "_writeback_call_reached"] = stop == 0xa85488
    m.mem_write(desc + 0x108, U32.pack(result["output_selector"]))
    result["config_word_controls"] = replay_resize_controls(m, desc, config)
    return result


def replay_resize_controls(m, desc, config):
    """Perturb one descriptor field at a time in the original word builder."""
    results = {}
    for name, offset, value in (
        ("selector_1", 0x108, 1),
        ("selector_ae", 0x108, 0xae),
        ("selector_100", 0x108, 0x100),
        ("input_kind_3", 4, 3),
        ("option_1b", 0x10c, 0x1b),
    ):
        original = bytes(m.mem_read(desc + offset, 4))
        m.mem_write(desc + offset, U32.pack(value))
        execute(m, 0xa854c6, RETURN, (
            (UC_ARM_REG_R1, desc), (UC_ARM_REG_R3, config),
            (UC_ARM_REG_SP, FRAME - 0x100), (UC_ARM_REG_LR, RETURN | 1),
        ))
        results[name] = U32.unpack(m.mem_read(config, 4))[0]
        m.mem_write(desc + offset, original)
    return results


def replay_input_route(m, input_desc, *, reconfigure):
    """Replay the mode-4 request and its low-level writer, without running AIF state management."""
    execute(m, 0x25da7c, 0x25daa8, (
        (UC_ARM_REG_R0, STATE + 0x1000), (UC_ARM_REG_R1, input_desc),
    ))
    # Supply the gate result explicitly. Neither branch calls the hardware.
    stop, writer = (0x25dc62, 0xa4b112) if reconfigure else (0x25dc96, 0xa4afdc)
    execute(m, 0x25daac, stop, ((UC_ARM_REG_R0, int(reconfigure)),))
    device, request = (m.reg_read(r) for r in (UC_ARM_REG_R0, UC_ARM_REG_R1))
    if device != 0xad or request != m.reg_read(UC_ARM_REG_SP):
        raise ValueError("unexpected source device or request pointer")
    # The higher-level AIF wrapper validates the request before calling this writer.
    execute(m, writer, RETURN, ((UC_ARM_REG_LR, RETURN | 1),))
    cfg = Struct("<10I").unpack(m.mem_read(BUS + 0x1200, 40))
    return {"device_id": device, "call_site": hex(stop), "writer": hex(writer),
            "control": cfg[0], "start_low": cfg[8], "start_high": cfg[6], "size": cfg[9]}


def probe(cp, *, mode=1, frame_base=FRAME_BASE, reconfigure=False):
    if sha256(cp).hexdigest() != CP_SHA256:
        raise ValueError("expected pinned A7 IV 6.02 CP image")
    if mode not in range(7):
        raise ValueError("expected a Sequence_Face mode 0..6")
    if not 0x500000000 <= frame_base <= 0x600000000 - MAX_FRAME_BYTES:
        raise ValueError("expected all five frame allocations in the first physical range")

    m = Uc(UC_ARCH_ARM, UC_MODE_ARM)
    m.mem_map(CP_BASE, (len(cp) + 0xfff) & ~0xfff, UC_PROT_READ | UC_PROT_EXEC)
    m.mem_write(CP_BASE, cp)
    for base, length in ((STATE, 0x5000), (GLOBAL, 0xb000), (BUS, 0x2000)):
        m.mem_map(base, length, UC_PROT_READ | UC_PROT_WRITE)
    m.mem_write(STATE + 0x458, U32.pack(mode))
    m.mem_write(STATE + 0x990, Struct("<Q").pack(frame_base))
    m.mem_write(STATE + 0x9a0, U32.pack(1))
    m.mem_write(STATE + 0x9dc, b"\1")
    m.mem_write(STATE + 0x9e4, U32.pack(RC_HANDLE))
    m.mem_write(STATE + 0x9f0, Struct("<QI").pack(RC_SOURCE, 0))
    m.mem_write(STATE + 0xd24, U32.pack(RC_WORD))

    # Select the original mode dimensions, then bypass unrelated frame scheduling.
    execute(m, 0x442f66, 0x4430ca, (
        (UC_ARM_REG_SP, FRAME), (UC_ARM_REG_SB, STATE), (UC_ARM_REG_R2, 0),
    ))
    height, width = Struct("<2I").unpack(m.mem_read(FRAME + 0x28, 8))
    # Input canvas and crop sizes are fixture values normally supplied by 0x442f38..0x442f4e.
    m.mem_write(FRAME + 0x30, Struct("<4I").pack(1920, 1080, 1920, 1080))
    execute(m, 0x44323a, 0x443388, ((UC_ARM_REG_R4, 0), (UC_ARM_REG_R5, 1)))
    input_desc, output_desc = (m.reg_read(r) for r in (UC_ARM_REG_R1, UC_ARM_REG_R3))

    # Both helpers write only emulator RAM. Never execute the accelerator start.
    source = replay_input_route(m, input_desc, reconfigure=reconfigure)
    execute(m, 0x25dd34, RETURN, ((UC_ARM_REG_SP, FRAME), (UC_ARM_REG_R1, output_desc),
                                  (UC_ARM_REG_LR, RETURN | 1)))
    if m.reg_read(UC_ARM_REG_R0) != 0:
        raise ValueError("firmware rejected the output records")
    outputs = tuple(OUTPUT_RECORD.iter_unpack(m.mem_read(output_desc, 5 * OUTPUT_RECORD.size)))
    bus_words = Struct("<7I").unpack(m.mem_read(BUS + 0x120, 7 * 4))
    resize = replay_resize(m)

    # Re-enter the completion path after the asynchronous wait, with no pixels.
    execute(m, 0x443a92, 0x443c48, ((UC_ARM_REG_SP, FRAME), (UC_ARM_REG_SB, STATE)))
    table = FRAME + 0x40
    m.mem_write(MESSAGE, b"\xa5" * 0x80)
    execute(m, 0x287834, 0x2878c8, ((UC_ARM_REG_R4, table), (UC_ARM_REG_R5, MESSAGE)))
    raw = bytes(m.mem_read(table, 5 * PYRAMID_RECORD.size))
    copied = bytes(m.mem_read(MESSAGE + 8, len(raw)))
    if copied != b"".join(raw[i:i + 14] + b"\xa5\xa5" for i in range(0, len(raw), 16)):
        raise ValueError("unexpected FC message frame copy")
    return {
        "mode": mode,
        "dimensions": [width, height],
        "source": source,
        "resize": resize,
        "output_records": outputs,
        "frame_records": tuple(PYRAMID_RECORD.iter_unpack(copied)),
        "message_hex": bytes(m.mem_read(MESSAGE, 0x80)).hex(" "),
        "mmio": {"geometry": bus_words[0], "destination_low_words": bus_words[2:7],
                 "destination_high_control": U32.unpack(m.mem_read(BUS + 0x1298, 4))[0],
                 "output_mask": U32.unpack(m.mem_read(BUS + 0x13c, 4))[0]},
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cp", type=Path)
    cli.add_argument("--mode", type=int, default=1)
    cli.add_argument("--reconfigure", action="store_true", help="Supply the nonzero PRC input gate fixture")
    args = cli.parse_args()
    print(json.dumps(probe(args.cp.read_bytes(), mode=args.mode, reconfigure=args.reconfigure), indent=2))
