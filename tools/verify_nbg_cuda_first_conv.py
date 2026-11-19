"""Run the original slot-08 first spatial convolution directly on the host GPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nbg_cuda_quantization import round_scaled_u8
from verify_nbg_cuda_tiles import read_rows, rows_for_op


# Local V8 accumulator coordinates for observed vertically tiled invocations.
Y_ORIGIN = {3: 59, 5: 117, 6: 175, 8: 233}


def first_conv_coefficients(data):
    rows = data[data[:, 0] == 1, 2:7]
    x, y, z, k, value = rows.T
    if (len(rows) != 32 * 3 * 3 * 3 or
            not np.array_equal((x.max(), y.max(), z.max(), k.max()), (2, 2, 2, 31)) or
            np.any(value < 0) or np.any(value > 255) or len(np.unique(rows[:, :4], axis=0)) != len(rows)):
        raise ValueError("unexpected slot-08 first convolution coefficients")
    weight = np.empty((32, 3, 3, 3), dtype=np.int8)
    weight[k, z, y, x] = value.astype(np.uint8).view(np.int8)
    return weight


def cpu_sites(op, weight):
    taps = op[(op[:, 0] == 4) & (op[:, 3] == 0)]
    cpu = op[op[:, 0] == 5]
    accum = {}
    for tile in np.unique(cpu[:, 2]):
        ref = cpu[cpu[:, 2] == tile]
        kernel = taps[taps[:, 2] == tile]
        if not len(kernel):
            continue
        channel = int(ref[0, 6])
        w = kernel[np.argsort(kernel[:, 4]), 6]
        if w.shape != (9,):
            raise ValueError("expected nine ordered coefficient lanes")
        matches = np.flatnonzero((weight[channel].reshape(3, 9) == w).all(axis=1))
        if len(matches) != 1:
            raise ValueError("tile does not identify an unambiguous input channel")
        z = int(matches[0])
        reference = np.ascontiguousarray(ref[:, 7:9], dtype="<i4").view("<i8").ravel()
        for (x, y, c), delta in zip(ref[:, 4:7], reference):
            group = accum.setdefault((int(c), int(y), int(x)), {})
            if z in group:
                raise ValueError("duplicate channel contribution")
            group[z] = int(delta)
    return {site: sum(terms.values()) for site, terms in accum.items() if len(terms) == 3}


def layer_parameters(op, channels=32, *, zero_point=0):
    bias = op[op[:, 0] == 6, 2:5]
    cfg = op[op[:, 0] == 7]
    if not np.array_equal(bias[:, 0], np.arange(channels)):
        raise ValueError(f"expected {channels} decoded biases")
    # The accumulator's integer postprocessor consumes the low signed 32 bits.
    bias = np.ascontiguousarray(bias[:, 1], dtype="<i4")
    scales = []
    for channel in range(channels):
        rows = cfg[cfg[:, 2] == channel]
        if (not len(rows) or not np.all(rows[:, 3] == rows[0, 3]) or
                np.any(rows[:, 7] != zero_point) or np.any(rows[:, 8])):
            raise ValueError("unsupported output zero point or scaling mode")
        scales.append(rows[0, 3])
    scales = np.array(scales, dtype="<i4").view("<f4")
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError("invalid layer scaling factor")
    return bias, scales


def compare_layer(rows, pre_bias, args):
    bias, scale = layer_parameters(rows_for_op(rows, 3))
    biases = torch.from_numpy(bias).to(device="cuda", dtype=torch.float32)[:, None, None]
    scales = torch.from_numpy(scale).to(device="cuda")[:, None, None]
    biased = pre_bias + biases
    quantized = round_scaled_u8(biased, scales, relu=True).cpu().numpy()
    biased = biased.to(torch.int32).cpu().numpy()
    checks = []
    for index, y0 in Y_ORIGIN.items():
        op = rows_for_op(rows, index)
        b, s = layer_parameters(op)
        if not (np.array_equal(bias, b) and np.array_equal(scale, s)):
            raise ValueError("layer parameters differ across simulator tiles")
        check = op[op[:, 0] == 8]
        target = op[op[:, 0] == 9]
        channel, x, y, value = target[:, 2:6].T
        if not len(check) or not (np.array_equal(check[:, 2], channel) and
                                  np.array_equal(check[:, 7], value)):
            raise ValueError("requantization samples do not align with writes")
        checks.append({
            "op": index, "input_y_origin": y0, "sites": len(value),
            "bias_mismatches": int(np.count_nonzero(biased[channel, y + y0, x] != check[:, 3])),
            "uint8_mismatches": int(np.count_nonzero(quantized[channel, y + y0, x] != value)),
        })
    if any(c["bias_mismatches"] or c["uint8_mismatches"] for c in checks):
        raise SystemExit(f"first fused layer CUDA/CPU comparison failed: {checks}")
    if args.layer_raw:
        with args.layer_raw.open("xb") as f:
            f.write(quantized.tobytes())
    return {
        "bias_i32_sha256": hashlib.sha256(bias.tobytes()).hexdigest(),
        "post_scale_f32_sha256": hashlib.sha256(scale.tobytes()).hexdigest(),
        "output_uint8_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
        "checked_tiles": checks,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--layer-trace", type=Path)
    cli.add_argument("--layer-raw", type=Path)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    if args.layer_raw and not args.layer_trace:
        cli.error("--layer-raw requires --layer-trace")
    rows = read_rows(args.trace)
    weight = first_conv_coefficients(rows_for_op(rows, 3))
    image = np.fromfile(args.input, dtype=np.uint8).reshape(1, 3, 320, 320)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.from_numpy(image).to(device="cuda", dtype=torch.float32) - 128
    w = torch.from_numpy(weight).to(device="cuda", dtype=torch.float32)
    # FP32 sums are exact here: 27 signed products cannot exceed 2**24 in magnitude.
    first_conv = F.conv2d(x, w, stride=1, padding=1)[0]
    raw = first_conv.to(torch.int32).cpu().numpy()
    comparisons = []
    for index, y0 in Y_ORIGIN.items():
        op = rows_for_op(rows, index)
        if not np.array_equal(first_conv_coefficients(op), weight):
            raise ValueError(f"coefficient mismatch for operation {index}")
        sites = cpu_sites(op, weight)
        if not sites:
            raise ValueError(f"no complete CPU sums for operation {index}")
        diff = np.array([int(raw[c, y + y0, xx]) - value
                         for (c, y, xx), value in sites.items()])
        comparisons.append({
            "op": index, "input_y_origin": y0, "sites": len(sites),
            "nonzero_cpu_sums": sum(value != 0 for value in sites.values()),
            "mismatches": int(np.count_nonzero(diff)),
            "max_absolute_difference": int(np.max(np.abs(diff))),
        })
    result = {
        "scope": "first slot-08 spatial convolution only, pre-bias and pre-requantization",
        "cuda_device": torch.cuda.get_device_name(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "weight_shape_oihw": list(weight.shape),
        "weight_int8_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
        "output_shape_chw": list(raw.shape),
        "output_int32_sha256": hashlib.sha256(raw.astype("<i4").tobytes()).hexdigest(),
        "comparisons": comparisons,
    }
    if args.layer_trace:
        layer_trace = read_rows(args.layer_trace)
        if not np.array_equal(weight, first_conv_coefficients(rows_for_op(layer_trace, 3))):
            raise ValueError("weight mismatch between convolution and fused-layer traces")
        result["scope"] = "first slot-08 fused convolution, bias, ReLU, and uint8 requantization"
        result["fused_layer"] = compare_layer(layer_trace, first_conv, args)
    report = json.dumps(result, indent=2) + "\n"
    print(report, end="")
    if any(row["mismatches"] for row in comparisons):
        raise SystemExit("first-convolution CUDA/CPU comparison failed")
    args.output.write_text(report, encoding="ascii")


if __name__ == "__main__":
    main()
