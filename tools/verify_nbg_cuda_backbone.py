"""Verify successive slot-08 GPU blocks from the first CUDA activation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from verify_nbg_cuda_first_conv import layer_parameters
from verify_nbg_cuda_second_conv import coefficients, compare_writes, fused_block
from verify_nbg_cuda_tiles import read_rows, rows_for_op


# Commands 4 and 7 are two vertical tiles of the same logical block.
BLOCKS = (
    ((32, 32, 3, 3), True, ((4, 0), (7, 58))),
    ((32, 32, 3, 3), False, ((11, 0),)),
    ((32, 32, 3, 3), True, ((12, 0),)),
    ((32, 32, 1, 1), False, ((13, 0),)),
    ((64, 32, 3, 3), False, ((14, 0),)),
    ((32, 64, 1, 1), False, ((15, 0),)),
    ((64, 32, 3, 3), True, ((16, 0),)),
    ((64, 64, 1, 1), False, ((17, 0),)),
    ((128, 64, 3, 3), False, ((18, 0),)),
    ((64, 128, 1, 1), False, ((19, 0),)),
    ((128, 64, 3, 3), False, ((20, 0),)),
    ((128, 128, 1, 1), False, ((21, 0),)),
)


def verify(activation, rows, blocks=BLOCKS):
    result = []
    torch.backends.cudnn.allow_tf32 = False
    tensor = torch.from_numpy(activation).to("cuda")
    for shape, pool, sites in blocks:
        op = rows_for_op(rows, sites[0][0])
        w = coefficients(op, shape)
        b, s = layer_parameters(op, shape[0])
        # Reject configurations where an FP32 dot product is not provably exact.
        max_dot = 128 * np.abs(w.astype(np.int64)).sum((1, 2, 3)).max()
        if max_dot + np.abs(b.astype(np.int64)).max() >= 2**24:
            raise ValueError("FP32 exact-integer bound exceeded")
        accum, tensor = fused_block(tensor, w, b, s, pool=pool)
        accum, raw = accum.cpu().numpy(), tensor.cpu().numpy()
        checks = []
        for number, y0 in sites:
            op = rows_for_op(rows, number)
            actual = coefficients(op, shape), *layer_parameters(op, shape[0])
            if not all(np.array_equal(lhs, rhs) for lhs, rhs in zip(actual, (w, b, s))):
                raise ValueError(f"inconsistent coefficients or requantization for command {number}")
            check = compare_writes(op, accum, raw, y0)
            if check["biased_i32_mismatches"] or check["uint8_mismatches"]:
                raise ValueError(f"command {number} did not match: {check}")
            checks.append({"operation": number, **check})
        result.append({
            "max_pool_2x2": pool, "weight_shape_oihw": shape,
            "output_shape_chw": raw.shape,
            "output_uint8_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "comparisons": checks,
        })
    return result, tensor.cpu().numpy()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("first_cuda_activation", type=Path)
    cli.add_argument("report", type=Path)
    cli.add_argument("--output-raw", type=Path)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    activation = np.fromfile(args.first_cuda_activation, dtype=np.uint8).reshape(32, 320, 320)
    blocks, raw = verify(activation, read_rows(args.trace))
    report = {
        "scope": "first slot-08 backbone blocks after the first CUDA convolution",
        "cuda_device": torch.cuda.get_device_name(),
        "first_cuda_activation_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
        "blocks": blocks,
    }
    text = json.dumps(report, indent=2) + "\n"
    args.report.write_text(text, encoding="ascii")
    if args.output_raw:
        with args.output_raw.open("xb") as file:
            file.write(raw.tobytes())
    print(text, end="")


if __name__ == "__main__":
    main()
