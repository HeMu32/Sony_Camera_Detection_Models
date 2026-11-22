"""Compare SDK outputs before and after pre-filling output tensors."""

import argparse
import json
from pathlib import Path

import numpy as np


def inspect(original, first, second, seed_first, seed_second, count=21):
    if seed_first == seed_second or any(not 0 <= seed <= 255 for seed in (seed_first, seed_second)):
        raise ValueError("expected two distinct byte seeds")
    results = []
    for i in range(count):
        a, b, c = (np.fromfile(p.with_name(f"{p.name}.{i}.raw"), dtype=np.uint8)
                   for p in (original, first, second))
        if not a.size or a.shape != b.shape or a.shape != c.shape:
            raise ValueError(f"missing or inconsistent tensor {i} size")
        kept = (b == seed_first) & (c == seed_second)
        results.append({"output": i, "bytes": a.size, "both_seeds_retained": int(kept.sum()),
                        "seed_runs_differ": int((b != c).sum()),
                        "first_differs_from_unseeded": int((b != a).sum()),
                        "second_differs_from_unseeded": int((c != a).sum())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--seed-first", type=lambda s: int(s, 0), required=True)
    parser.add_argument("--seed-second", type=lambda s: int(s, 0), required=True)
    parser.add_argument("--outputs", type=int, default=21)
    args = parser.parse_args()
    if args.outputs <= 0:
        parser.error("--outputs must be positive")
    print(json.dumps(inspect(args.original, args.first, args.second,
                             args.seed_first, args.seed_second, args.outputs), indent=2))


if __name__ == "__main__":
    main()
