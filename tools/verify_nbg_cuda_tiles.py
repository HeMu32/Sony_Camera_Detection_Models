"""Compare GPU convolution tile deltas with native NBG simulator accumulators."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read_rows(path):
    with path.open("rb") as f:
        if f.read(8) != b"NBCOEF1\0":
            raise ValueError("invalid trace header")
        return np.fromfile(f, dtype="<i4").reshape(-1, 9)


def rows_for_op(rows, op):
    data = rows[rows[:, 1] == op]
    if data[0, 0] != 0 or data[-1, 0] != 3 or not data[-1, 2]:
        raise ValueError(f"incomplete coefficient trace for operation {op}")
    return data


def index_key(tile_lane):
    return tile_lane[:, 0].astype(np.int64) * 64 + tile_lane[:, 1]


def check_tiles(data):
    taps, cpu = data[data[:, 0] == 4], data[data[:, 0] == 5]
    if not len(cpu):
        return None
    cpu_key = index_key(cpu[:, 2:4])
    tap_key = index_key(taps[:, 2:4])
    if len(cpu_key) != len(np.unique(cpu_key)) or not np.isin(tap_key, cpu_key).all():
        raise ValueError("nonmatching CPU and operand records")
    if len(np.unique(tap_key)) != len(cpu_key):
        raise ValueError("missing operands for CPU accumulator sites")
    reference = np.ascontiguousarray(cpu[:, 7:9], dtype="<i4").view("<i8").ravel()
    device = torch.device("cuda")
    # Signed inputs and weights enter the GPU separately. The CPU sum is only a reference.
    x = torch.from_numpy(taps[:, 5].astype(np.int64)).to(device)
    w = torch.from_numpy(taps[:, 6].astype(np.int64)).to(device)
    keys = torch.from_numpy(tap_key).to(device)
    acc = torch.zeros(int(cpu_key.max()) + 1, dtype=torch.int64, device=device)
    acc.index_add_(0, keys, x * w)
    predicted = acc[torch.from_numpy(cpu_key).to(device)].cpu().numpy()
    return {
        "tile_calls": int(len(np.unique(cpu[:, 2]))),
        "multiply_terms": len(taps), "cpu_accumulator_deltas": len(cpu),
        "nonzero_cpu_accumulator_deltas": int(np.count_nonzero(reference)),
        "mismatches": int((predicted != reference).sum()),
        "max_absolute_difference": int(np.abs(predicted - reference).max()),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("--report", type=Path)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    rows = read_rows(args.trace)
    operations = []
    for op in np.unique(rows[:, 1]):
        data = rows_for_op(rows, op)
        coeff = data[data[:, 0] == 1, 2:7]
        row = {
            "number": int(op), "simulator_argument": int(data[0, 2]),
            "ordinary_coefficient_records": len(coeff),
            "special_coefficient_records": int((data[:, 0] == 2).sum()),
            "coefficient_shape_kzyx": (coeff[:, :4].max(axis=0)[::-1] + 1).tolist()
            if len(coeff) else None,
            "coefficient_records_sha256": hashlib.sha256(coeff.tobytes()).hexdigest(),
            "cuda": check_tiles(data),
        }
        operations.append(row)
    result = {
        "cuda_device": torch.cuda.get_device_name(),
        "trace": str(args.trace), "operations": operations,
        "scope": "sampled pre-bias convolution tile MAC deltas, not full layer or model",
    }
    serialized = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(serialized, encoding="ascii")
    print(serialized, end="")
    checks = [op["cuda"] for op in operations if op["cuda"]]
    if not checks or not any(check["nonzero_cpu_accumulator_deltas"] for check in checks):
        raise SystemExit("no nonzero tile accumulator captured")
    if any(check["mismatches"] for check in checks):
        raise SystemExit("GPU/CPU accumulator deltas did not match")


if __name__ == "__main__":
    main()
