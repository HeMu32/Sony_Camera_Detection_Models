"""Bind FDN B's original compact-head arguments and materialize coefficient bytes."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from struct import Struct

from .fdn_sections import parse


HEADS = ((4, "score", 0), (0, "left_x", 8), (1, "left_y", 12), (2, "right_x", 16),
         (3, "right_y", 20), (5, "nose_x", 24), (6, "nose_y", 28))
I32 = Struct("<i")


@dataclass(frozen=True)
class HeadInput:
    role: str
    head: int
    activation_pointer: int
    weight_pointer: int
    temporary_pointer: int
    dictionary_offset: int
    bias: int
    weight_bytes: bytes


def materialize(data, trace):
    fixture = trace["head_fixture"]
    if fixture["outer_call"] is None or not trace["dictionary_reads"]:
        raise ValueError("trace did not reach the original compact heads")
    commands = [fixture["outer_call"], *fixture["worker_calls"]]
    heads = parse(data)["dense_candidates_144x1"]
    dictionary = trace["dictionary_reads"][0]["dictionary"]
    if len(commands) != len(HEADS) or len(heads) != len(HEADS):
        raise ValueError("unexpected number of compact heads")
    inputs = []
    for (index, role, offset), command in zip(HEADS, commands):
        activation, weights, result = command["registers_r5_to_r7"]
        if command["count_word"] != 12 or result != trace["writer_temporary_pointer"] + offset:
            raise ValueError(f"unexpected control arguments for {role}")
        head = heads[index]
        coeff = head["coefficient_offset"]
        weight_bytes = data[coeff:head["bias_offset"]]
        bias, = I32.unpack_from(data, head["bias_offset"])
        native = [r for r in fixture["dictionary_transfers"]
                  if (r["dictionary"], r["offset"], r["destination"], r["size"]) ==
                  (dictionary, coeff, weights, len(weight_bytes))]
        if (len(weight_bytes) != 288 or len(native) != 1 or
                native[0]["sha256"] != sha256(weight_bytes).hexdigest() or
                command["bias_word"] != bias & 0xffffffff):
            raise ValueError(f"dictionary transfer or bias does not match {role}")
        inputs.append(HeadInput(role, index, activation, weights, result, coeff, bias, weight_bytes))
    by_address = sorted(inputs, key=lambda item: item.weight_pointer)
    if any(left.weight_pointer + len(left.weight_bytes) > right.weight_pointer
           for left, right in zip(by_address, by_address[1:])):
        raise ValueError("overlapping compact weight buffers")
    return inputs


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path, help="JSON from probe_sa_fdn_b_results.py --trace-heads")
    cli.add_argument("dict_dir", type=Path)
    args = cli.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    name = trace["dictionary_reads"][0]["dictionary"]
    if name not in ("front", "side"):
        cli.error("unexpected dictionary")
    inputs = materialize((args.dict_dir / f"fdn_{name}_dict.bin").read_bytes(), trace)
    print(json.dumps({"dictionary": name, "inputs": [
        {"role": h.role, "head": h.head, "activation_pointer": h.activation_pointer,
         "weight_pointer": h.weight_pointer, "temporary_pointer": h.temporary_pointer,
         "dictionary_offset": h.dictionary_offset, "bias": h.bias, "weight_bytes": len(h.weight_bytes),
         "weight_sha256": sha256(h.weight_bytes).hexdigest()} for h in inputs
    ]}, indent=2))
