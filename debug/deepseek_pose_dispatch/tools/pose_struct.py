"""Structural analysis of a pose NBG output set: channel statistics and PAF pairs."""
import argparse
import json
import pathlib

import numpy as np


def load(prefix, index, shape, scale, zero):
    raw = np.fromfile(prefix.with_name(f"{prefix.name}.{index}.raw"), dtype=np.uint8)
    return ((raw.astype(np.float32) - zero) * scale).reshape(shape[::-1])


def specs(log):
    out = []
    for line in pathlib.Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        f = line.split()
        if len(f) >= 7 and f[0] == "param" and f[2] == "output":
            spec = dict(x.split("=", 1) for x in f[3:])
            out.append((tuple(map(int, spec["shape"].split("x"))),
                         float(spec["scale"]), int(spec["zp"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("log")
    args = ap.parse_args()
    args.prefix = pathlib.Path(args.prefix)
    spec = specs(args.log)
    arrays = [load(args.prefix, i, *s) for i, s in enumerate(spec)]
    report = []
    for i, (a, (shape, scale, zero)) in enumerate(zip(arrays, spec)):
        entry = {"index": i + 1, "shape": list(shape), "scale": scale, "zp": zero,
                 "min": float(a.min()), "max": float(a.max()),
                 "mean": float(a.mean())}
        if a.shape[0] > 4:
            per = a.reshape(a.shape[0], -1)
            entry["channel_max"] = [round(float(v), 5) for v in per.max(axis=1)]
        report.append(entry)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
