"""Verify slot-08 40x40 GPU head commands and the final auxiliary output."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nbg_cuda_nn import nn_forward
from verify_nbg_cuda_backbone import BLOCKS, verify
from verify_nbg_cuda_first_conv import layer_parameters
from verify_nbg_cuda_second_conv import coefficients, compare_writes
from verify_nbg_cuda_tiles import read_rows, rows_for_op


def head_block(rows, number, activation, outputs, kernel, *, relu=True, zero_point=0, padding=None):
    op = rows_for_op(rows, number)
    weight = coefficients(op, (outputs, activation.shape[0], kernel, kernel))
    bias, scale = layer_parameters(op, outputs, zero_point=zero_point)
    max_dot = 128 * np.abs(weight.astype(np.int64)).sum((1, 2, 3)).max()
    if max_dot + np.abs(bias.astype(np.int64)).max() >= 2**24:
        raise ValueError(f"command {number}: FP32 exact-integer bound exceeded")

    matrix = torch.from_numpy(weight).cuda().to(dtype=torch.float32).flatten(1)
    bias = torch.from_numpy(bias).cuda()[:, None, None]
    scale = torch.from_numpy(scale).cuda()[:, None, None]
    accum, quantized = nn_forward(activation.cuda(), matrix, bias, scale, kernel,
                                  relu=relu, zero_point=zero_point, padding=padding)
    check = compare_writes(op, accum.to(torch.int32).cpu().numpy(), quantized.cpu().numpy())
    if check["biased_i32_mismatches"] or check["uint8_mismatches"]:
        raise ValueError(f"command {number} did not match: {check}")
    return quantized, {"operation": number, "zero_point": zero_point, **check}


def run_heads(activation, rows):
    _, from_20 = verify(activation, rows, BLOCKS[:-1])
    from_20 = torch.from_numpy(from_20)
    classification, c21 = head_block(rows, 21, from_20, 128, 1)
    logits, c22 = head_block(rows, 22, classification, 30, 3, relu=False, zero_point=149)
    auxiliary, c23 = head_block(rows, 23, classification, 6, 3)
    regression, c24 = head_block(rows, 24, from_20, 128, 1)
    boxes, c25 = head_block(rows, 25, regression, 24, 3, relu=False, zero_point=129)
    return logits, auxiliary, boxes, [c21, c22, c23, c24, c25]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("first_cuda_activation", type=Path)
    cli.add_argument("cpu_aux_output", type=Path)
    cli.add_argument("report", type=Path)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    activation = np.fromfile(args.first_cuda_activation, dtype=np.uint8).reshape(32, 320, 320)
    rows = read_rows(args.trace)
    _, auxiliary, _, checks = run_heads(activation, rows)

    # Output 15 has OpenVX order (6, 40, 40), so channels are contiguous.
    result = (torch.sigmoid(auxiliary.float()) * 256).round().clamp(0, 255)
    raw = result.to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
    reference = args.cpu_aux_output.read_bytes()
    if len(reference) != len(raw):
        raise ValueError("wrong byte count for CPU auxiliary output")
    mismatches = sum(a != b for a, b in zip(raw, reference))
    report = {
        "scope": "slot-08 head commands 21..25; final output parity for auxiliary output 15 only",
        "cuda_device": torch.cuda.get_device_name(),
        "head_comparisons": checks,
        "final_aux_output": {
            "bytes": len(raw), "mismatches": mismatches,
            "gpu_sha256": hashlib.sha256(raw).hexdigest(),
            "cpu_sha256": hashlib.sha256(reference).hexdigest(),
            "observed_pre_sigmoid_values": torch.unique(auxiliary).cpu().tolist(),
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    args.report.write_text(text, encoding="ascii")
    print(text, end="")
    if mismatches:
        raise SystemExit("final 40x40 auxiliary output did not match")


if __name__ == "__main__":
    main()
