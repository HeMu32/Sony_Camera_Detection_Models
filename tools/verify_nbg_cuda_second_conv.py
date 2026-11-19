"""Run the slot-08 second fused block using the first CUDA activation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nbg_cuda_quantization import round_scaled_u8
from verify_nbg_cuda_first_conv import layer_parameters
from verify_nbg_cuda_tiles import read_rows, rows_for_op


# The simulator's local output Y coordinates restart at each vertical tile.
Y_ORIGIN = {4: 0, 7: 58}


def coefficients(op, shape=(32, 32, 3, 3)):
    x, y, z, k, value = op[op[:, 0] == 1, 2:7].T
    indices = np.ravel_multi_index((k, z, y, x), shape)
    if (len(indices) != np.prod(shape) or len(np.unique(indices)) != len(indices) or
            np.any((value < 0) | (value > 255)) or np.any(op[:, 0] == 2)):
        raise ValueError("incomplete slot-08 convolution coefficient grid")
    weight = np.empty(shape, dtype=np.int8)
    weight.flat[indices] = value.astype(np.uint8).view(np.int8)
    return weight


def fused_block(activation, weight, bias, scale, *, pool):
    tensor = activation.to(device="cuda", dtype=torch.float32) - 128
    kernels = torch.from_numpy(weight).to(device="cuda", dtype=torch.float32)
    pad = weight.shape[-1] // 2
    # Zero-byte padding becomes -128 after centering.
    convolved = F.conv2d(F.pad(tensor[None], (pad,) * 4, value=-128), kernels)
    spatial = F.max_pool2d(convolved, kernel_size=2, stride=2) if pool else convolved
    b = torch.from_numpy(bias).to(device="cuda", dtype=torch.float32)[:, None, None]
    s = torch.from_numpy(scale).to(device="cuda")[:, None, None]
    biased = spatial[0] + b
    raw = round_scaled_u8(biased, s, relu=True)
    return biased.to(torch.int32), raw


def compare_writes(op, accum, raw, y0=0):
    post = op[op[:, 0] == 8]
    c, x, y, expected = op[op[:, 0] == 9, 2:6].T
    if (not len(post) or not np.array_equal(post[:, 2], c) or
            not np.array_equal(post[:, 7], expected) or np.any(post[:, 5:7])):
        raise ValueError("postprocess samples and writes disagree, or include a skip operand")
    sites = np.ravel_multi_index((c, y + y0, x), raw.shape)
    if len(np.unique(sites)) != len(sites):
        raise ValueError("duplicate comparison sites")
    # SDK integer postprocessing sign-extends only the low 32 bits before ReLU.
    return {
        "output_y_origin": y0, "sites": len(c),
        "biased_i32_mismatches": int(np.count_nonzero(accum[c, y + y0, x] != post[:, 3])),
        "uint8_mismatches": int(np.count_nonzero(raw[c, y + y0, x] != expected)),
    }


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
    rows = read_rows(args.trace)
    first = rows_for_op(rows, 4)
    weight = coefficients(first)
    bias, scale = layer_parameters(first)
    torch.backends.cudnn.allow_tf32 = False
    accum, raw = fused_block(torch.from_numpy(activation), weight, bias, scale, pool=True)
    accum, raw = accum.cpu().numpy(), raw.cpu().numpy()
    checks = []
    for operation, y0 in Y_ORIGIN.items():
        op = rows_for_op(rows, operation)
        b, s = layer_parameters(op)
        if not all(np.array_equal(actual, expected) for actual, expected in
                   ((weight, coefficients(op)), (bias, b), (scale, s))):
            raise ValueError("inconsistent block parameters across simulator tiles")
        checks.append({"operation": operation, **compare_writes(op, accum, raw, y0)})
    report = {
        "scope": "second slot-08 fused convolution, maxpool, bias, ReLU, and uint8 requantization",
        "cuda_device": torch.cuda.get_device_name(),
        "first_cuda_activation_sha256": hashlib.sha256(activation.tobytes()).hexdigest(),
        "weight_int8_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "bias_i32_sha256": hashlib.sha256(bias.tobytes()).hexdigest(),
        "post_scale_f32_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "output_shape_chw": list(raw.shape),
        "output_uint8_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "comparisons": checks,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    args.report.write_text(text, encoding="ascii")
    if any(c["biased_i32_mismatches"] or c["uint8_mismatches"] for c in checks):
        raise SystemExit("second block CUDA/CPU comparison failed")
    if args.output_raw:
        with args.output_raw.open("xb") as file:
            file.write(raw.tobytes())


if __name__ == "__main__":
    main()
