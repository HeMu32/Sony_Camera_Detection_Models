"""Read captured NN output writes from the pinned V8 simulator."""

import argparse
import gzip
from pathlib import Path

import numpy as np


def read_writes(path, *, first=1):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as file:
        if file.read(8) != b"NBNW001\0":
            raise ValueError("invalid NN-write trace header")
        if path.suffix == ".gz":
            data = np.frombuffer(file.read(), dtype="<u4")
        else:
            data = np.memmap(path, dtype="<u4", offset=8, mode="r")
    rows = data.reshape(-1, 2)
    start = np.flatnonzero(rows[:, 0] == 0xffffffff)
    end = np.flatnonzero(rows[:, 0] == 0xfffffffe)
    if (len(start) != len(end) or not len(start) or start[0] != 0 or end[-1] != len(rows) - 1 or
            np.any(start[1:] != end[:-1] + 1) or np.any(end <= start) or
            np.any(end - start - 1 != rows[end, 1]) or
            not np.array_equal(rows[start, 1], np.arange(first, first + len(start)))):
        raise ValueError("unexpected NN-write trace framing")
    return {int(rows[a, 1]): rows[a + 1:b] for a, b in zip(start, end)}


def coordinates(rows):
    key, value = rows.T
    return key >> 20, (key >> 10) & 1023, key & 1023, value.view("<i4")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path)
    cli.add_argument("--from-op", type=int, default=1)
    args = cli.parse_args()
    print("op  count        min(C,Y,X)    max(C,Y,X)    min..max")
    for op, rows in read_writes(args.file, first=args.from_op).items():
        if len(rows):
            c, y, x, value = coordinates(rows)
            print(f"{op:2d}  {len(rows):7d}      "
                  f"{str(tuple(int(i.min()) for i in (c, y, x))):14s}"
                  f"{str(tuple(int(i.max()) for i in (c, y, x))):15s}"
                  f"{int(value.min())}..{int(value.max())}")
        else:
            print(f"{op:2d}        0")


if __name__ == "__main__":
    main()
