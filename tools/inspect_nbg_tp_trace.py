"""Decode the pinned V8 simulator's TP command and ALU-conversion trace."""

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from struct import Struct


EVENT = Struct("<4I")
COMMAND = Struct("<32I")


@dataclass
class Invocation:
    index: int
    after_nn: int
    base: int | None = None
    words: tuple = ()
    pairs: Counter = field(default_factory=Counter)
    samples: int = 0
    sequence: list | None = None


def read_trace(path, *, capture=()):
    nn_count = 0
    calls = []
    active = None
    with path.open("rb") as file:
        if file.read(8) != b"NBTP001\0":
            raise ValueError("invalid TP trace header")
        while record := file.read(EVENT.size):
            if len(record) != EVENT.size:
                raise ValueError("truncated trace event")
            kind, ordinal, a, b = EVENT.unpack(record)
            if kind == 1:
                if active or ordinal != nn_count + 1 or b != len(calls):
                    raise ValueError("unexpected NN ordering")
                nn_count += 1
            elif kind == 2:
                if active or ordinal != len(calls) + 1 or b != nn_count:
                    raise ValueError("unexpected TP ordering")
                active = Invocation(a, b, sequence=[] if ordinal in capture else None)
            elif active is None or ordinal != len(calls) + 1:
                raise ValueError("no current TP invocation")
            elif kind == 3:
                if active.words or b != active.index:
                    raise ValueError("unexpected TP command")
                payload = file.read(COMMAND.size)
                if len(payload) != COMMAND.size:
                    raise ValueError("truncated TP command")
                active.base, active.words = a, COMMAND.unpack(payload)
            elif kind == 4:
                active.pairs[a, b] += 1
                active.samples += 1
                if active.sequence is not None:
                    active.sequence.append((a, b))
            elif kind == 5:
                if a != active.samples or b not in (0, 1) or not active.words:
                    raise ValueError("incomplete TP invocation")
                calls.append(active)
                active = None
            else:
                raise ValueError(f"unknown event code {kind}")
    if active is not None:
        raise ValueError("trace ends before TP completion")
    return nn_count, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--tp", type=int, action="append")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()
    nn_count, calls = read_trace(args.trace)
    report = []
    for tp, call in enumerate(calls, 1):
        if args.tp is not None and tp not in args.tp:
            continue
        row = {
            "tp": tp, "after_nn": call.after_nn, "index": call.index,
            "command_address": hex(call.base + 128 * call.index),
            "fetched_words": [f"{word:08x}" for word in call.words],
            "sample_count": call.samples, "distinct_pairs": len(call.pairs),
        }
        if args.pairs:
            row["observed_pairs"] = [[a, b, count] for (a, b), count in sorted(call.pairs.items())]
        report.append(row)
    print(json.dumps({"nn_commands": nn_count, "tp_commands": len(calls), "calls": report}, indent=2))


if __name__ == "__main__":
    main()
