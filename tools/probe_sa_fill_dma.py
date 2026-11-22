"""Replay the original SA_FILL controller's first local-copy submissions."""

import argparse
from collections import Counter
import json
from pathlib import Path
from struct import pack

from probe_sa_compute_controls import regions
from probe_sa_fdn_boot import DmaTraceMachine
from sa_local_dma import copy_local_rows


FILL_SHA = "b8880b2054f8f799b1c16f83de47e6dfeec5a02b6e074f8506f51ba0524ac9c0"
PATTERN = (0xdeadbeef, 0x78)


def probe_fill_local(blob, *, pattern=PATTERN, width=32):
    """Exercise the 5-byte pattern branch; stop before any external copy."""
    if not 20 <= width <= 48 or width % 2:
        raise ValueError("expected a bounded even width")
    _, image = regions(blob, FILL_SHA)[0]
    stride = 64
    params = pack("<10I", 0x40008000, stride, 0, 0, width, 3, 1, 0, *pattern)
    m = DmaTraceMachine(image, scratch_base=0x2800, scratch_size=0x1800)
    m.capture_dma = True
    m.pc, m.reg[4] = 0, 0x40000400
    m.regions = ((0, image), (0x49e00, bytes(0x160)), (0xe0000400, params))
    m.stack_low, m.stack_high, m.stack = 0x2000, 0x2800, bytearray(0x800)
    local = []
    for _ in range(8):
        before = len(m.dma_submissions)
        stop = m.run(8192)
        batch, = m.dma_submissions[before:]
        rows = batch["rows"]
        if stop == {"pc": 0xa5c, "reason": "wfi"}:
            if len(rows) != 1:
                raise ValueError(f"unexpected SA_FILL external rows: {rows}")
            address, local_source, a, b = rows[0]
            if address != 0x20008000 or local_source != 0x80003000 or a != b or a & 0xffff:
                raise ValueError(f"unexpected SA_FILL external row: {rows}")
            length = (a >> 16) + 1
            output = bytes(m.scratch[i] for i in range(0x3000, 0x3000 + length))
            return {"local_requests": local, "external_request": batch,
                    "external_stop": stop, "buffer_hex": output.hex(" ")}
        if stop != {"pc": 0xb80, "reason": "wfi"} or batch["submit_pc"] != 0xb7a:
            raise ValueError(f"unexpected SA_FILL local-copy stop: {stop}")
        updates = copy_local_rows(rows, m.scratch, source_span=(0x2850, 0x2890),
                                  destination_span=(0x3000, 0x3100))
        if any(i in m.scratch for i in updates):
            raise ValueError("expected the SA_FILL destination to be unknown before its request")
        m.scratch.update(updates)
        local.append(batch)
        m.pc += 2  # Standalone host replay of the identified request only.
    raise ValueError("SA_FILL did not reach the external transfer boundary")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("fill", type=Path)
    args = cli.parse_args()
    print(json.dumps(probe_fill_local(args.fill.read_bytes()), indent=2))
