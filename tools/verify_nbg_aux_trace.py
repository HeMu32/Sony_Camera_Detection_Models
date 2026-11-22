"""Verify the recorded A1 II slot 9/10/11 1x1 auxiliary path on the Flight sample."""

import argparse
import json
from pathlib import Path

from inspect_nbg_nn_writes import coordinates, read_writes
from inspect_nbg_tp_trace import read_trace
from nbg_integer_quantization import integer_formula
from sony_legacy_pose_backbone import read_instructions


def verify(root):
    results = {}
    for slot, tp_count, lut_count, address, ordinal, expected in (
            ("09", 85, 0, 0x005f2900, 75, 128),
            ("11", 84, 0, 0x0854a7c0, None, 0),
            ("10", 109, 25, 0x005f2900, 95, 128)):
        n, tp = read_trace(root / f"{slot}.nbt")
        nn = read_instructions(root / f"{slot}.nnc", n)
        lookup = [i for i, t in enumerate(tp, 1) if t.words[6] == 0xe8000018]
        assert (n, len(tp), len(lookup), int(nn[66, 7])) == (74, tp_count, lut_count, address)
        observed, = read_writes(root / f"{slot}-nn67.nnw", first=67).values()
        ch, y, x, code = (v.tolist() for v in coordinates(observed))
        assert (ch, y, x, code) == (list(range(6)), [0] * 6, [0] * 6, [0] * 6)
        following = [(i, t) for i, t in enumerate(tp, 1) if t.after_nn == 67]
        assert [i for i, _ in following] == ([] if ordinal is None else [ordinal])
        if following:
            words = following[0][1].words
            assert (words[0], words[1], words[10]) == (1, 0x60001, address)
            if slot == "09":
                mode, inp, out = (words[i] for i in (24, 29, 30))
                assert (mode, inp, out) == (0x80a, 0, 0x80)
                assert integer_formula(0, out & 255, 0x8000 | (out >> 8), 15 + (mode >> 10)) == expected
            else:
                assert ordinal in lookup and words[24] == 0xc00600a
        assert all((root / f"{slot}-nn67.{i}.raw").read_bytes() ==
                   (root / f"{slot}.{i}.raw").read_bytes() for i in range(21))
        assert (root / f"{slot}.20.raw").read_bytes() == bytes([expected]) * 6
        results[slot] = {"nn_count": n, "tp_count": tp_count, "lookup_commands": lookup,
                         "nn67_address": hex(address), "nn67_codes": code,
                         "post_nn67_tp": ordinal, "output20_codes": [expected] * 6}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_directory", type=Path)
    print(json.dumps(verify(parser.parse_args().trace_directory), indent=2))
