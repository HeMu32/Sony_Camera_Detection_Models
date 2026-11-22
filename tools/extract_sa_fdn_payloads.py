"""Extract three FDN payload candidates selected by the original DSP dispatch."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from probe_sa_fdn_dispatch import MEMBER_SHA
from run_sa_fdn_dispatch import bank_entry


def payloads(blob, block):
    selectors = {"A": (0, 1), "B": (2,)}[block]
    copies = {selector: [] for selector in selectors}
    for bank in range(5):
        runs = [bank_entry(blob, block, bank, selector) for selector in selectors]
        info = runs[0]["bank"]
        dst, src = info["destination"], info["file_offset"]
        stops = [r["after_fixture"] for r in runs]
        if any(s["reason"] != "unsupported opcode" or s["opcode"] != 0x000a or
               s["pc"] != stops[0]["pc"] for s in stops):
            raise ValueError("expected dispatch to the shared unmapped instruction")
        starts = [r["registers_at_stop"]["r4"] - dst for r in runs]
        end = stops[0]["pc"] - dst
        bounds = starts + [end]
        if not (0 <= bounds[0] and bounds[-1] + 2 <= info["length"] and
                all(a < b for a, b in zip(bounds, bounds[1:])) and
                all(bound % 4 == 0 for bound in bounds)):
            raise ValueError("r4 and the shared entry do not delimit valid ranges")
        for i, selector in enumerate(selectors):
            start, end = (src + offset for offset in bounds[i:i + 2])
            raw = blob[start:end]
            copies[selector].append((raw, {
                "bank": bank, "local_r4": dst + bounds[i], "file_offset": start,
                "selector_read": runs[i]["first_read_after_wake"]["address"],
                "dispatch_call_pc": stops[i]["pc"],
            }))
    for selector, instances in copies.items():
        first = instances[0][0]
        if any(raw != first for raw, _ in instances):
            raise ValueError(f"{block}/{selector}: candidate ranges differ between banks")
        yield first, {"block": block, "selector": selector, "bytes": len(first),
                      "sha256": sha256(first).hexdigest(),
                      "copies": [meta for _, meta in instances]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("member", type=Path)
    cli.add_argument("--extract-dir", type=Path)
    args = cli.parse_args()
    blob = args.member.read_bytes()
    if sha256(blob).hexdigest() != MEMBER_SHA:
        cli.error("expected pinned A7 IV FDN member")
    results = [(raw, meta) for block in "AB" for raw, meta in payloads(blob, block)]
    if args.extract_dir:
        args.extract_dir.mkdir(parents=True, exist_ok=True)
        for raw, meta in results:
            name = f"fdn-{meta['block']}-selector{meta['selector']}.bin"
            with (args.extract_dir / name).open("xb") as file:
                file.write(raw)
    print(json.dumps({"member_sha256": MEMBER_SHA,
                      "payload_candidates": [meta for _, meta in results]}, indent=2))
