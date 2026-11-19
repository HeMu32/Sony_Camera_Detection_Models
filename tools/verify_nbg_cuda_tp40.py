"""Verify integer TP conversion for slot-08's 40x40 class and box heads."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inspect_nbg_tp_trace import read_trace
from nbg_cuda_quantization import tp_requantize_u8
from verify_nbg_cuda_head40 import run_heads
from verify_nbg_cuda_tiles import read_rows


def decode_quantization(words):
    mode, input_quant, output_quant = words[24], words[29], words[30]
    mantissa = (output_quant >> 8) & 0x7fff
    if not mantissa or mode & ~0x0c007c00 != 0x0a or input_quant & 0x00ffffff:
        raise ValueError("not the supported TP uint8-to-uint8 conversion")
    if output_quant >> 23:
        raise ValueError("unsupported extended scale bits")
    shift = ((mode >> 10) & 31) | (((mode >> 26) & 3) << 5)
    shift = (shift ^ 64) - 64
    if not -14 <= shift <= 15:
        raise ValueError("TP scale shift out of supported range")
    return input_quant >> 24, output_quant & 0xff, mantissa | 0x8000, 15 + shift


def verify_head(tensor, calls, nn, expected_path):
    channels, height, width = tensor.shape
    reference = np.fromfile(expected_path, dtype=np.uint8).reshape(channels, height, width)
    tiles = []
    gpu_output = np.empty_like(reference)
    coverage = np.zeros(tensor.shape, dtype=bool)
    source_differences = np.zeros(tensor.shape, dtype=bool)
    output_differences = np.zeros(tensor.shape, dtype=bool)
    for call in calls:
        words = call.words
        h, c = words[1] & 0xffff, words[1] >> 16
        c0, offset = divmod(words[13] - calls[0].words[13], height * width)
        y0, x0 = divmod(offset, width)
        w = words[0]
        if (call.after_nn != nn or not w or x0 + w > width or
                words[2] != width or words[3] != height * width or
                not h or not c or y0 + h > height or not 0 <= c0 < c0 + c <= channels or
                len(call.sequence) != c * h * w):
            raise ValueError("unexpected TP command geometry")
        izp, ozp, multiplier, shift = decode_quantization(words)
        shape = (c, h, w)
        inputs, outputs = np.array(call.sequence, dtype="<u4").T
        inputs = inputs.view("<i4").reshape(shape)
        outputs = outputs.reshape(shape)
        if np.any((inputs < -izp) | (inputs > 255 - izp)):
            raise ValueError("TP source is outside the expected centered uint8 range")
        rendered = tp_requantize_u8(torch.from_numpy(inputs.copy()).cuda(), ozp, multiplier, shift)
        section = np.s_[c0:c0 + c, y0:y0 + h, x0:x0 + w]
        if np.any(coverage[section]):
            raise ValueError("overlapping TP stripes")
        coverage[section] = True
        if not (np.array_equal(outputs, reference[section]) and
                np.array_equal(rendered.cpu().numpy(), outputs)):
            raise ValueError("TP arithmetic or sample order is not bit-exact")
        gpu_inputs = tensor[section].to(torch.int32) - izp
        gpu_rendered = tp_requantize_u8(gpu_inputs, ozp, multiplier, shift)
        source_differences[section] = gpu_inputs.cpu().numpy() != inputs
        gpu_output[section] = gpu_rendered.cpu().numpy()
        output_differences[section] = gpu_output[section] != outputs
        tiles.append({
            "channel": c0, "channels": c, "y": y0, "height": h, "x": x0, "width": w,
            "bytes": outputs.size,
            "input_zero_point": izp,
            "output_zero_point": ozp, "multiplier": multiplier, "right_shift": shift,
            "tp_arithmetic_mismatches": 0, "cpu_trace_to_final_mismatches": 0,
        })
    if not np.all(coverage) or np.any(output_differences & ~source_differences):
        raise ValueError("incomplete TP coverage or inconsistent mismatch attribution")
    report = {
        "nn": nn, "tiles": tiles,
        "gpu_nn_source_mismatches": int(source_differences.sum()),
        "gpu_final_mismatches": int(output_differences.sum()),
        "gpu_nn_mismatches_by_y": source_differences.sum(axis=(0, 2)).tolist(),
    }
    return report, gpu_output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("nn_trace", type=Path)
    cli.add_argument("tp_trace", type=Path)
    cli.add_argument("first_cuda_activation", type=Path)
    cli.add_argument("cpu_output_prefix", type=Path)
    cli.add_argument("report", type=Path)
    cli.add_argument("--output-prefix", type=Path, help="Export exact GPU outputs 1 and 8")
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    activation = np.fromfile(args.first_cuda_activation, dtype=np.uint8).reshape(32, 320, 320)
    logits, _, boxes, _ = run_heads(activation, read_rows(args.nn_trace))
    _, calls = read_trace(args.tp_trace, capture=(*range(1, 5), *range(10, 14)))
    result = {}
    arrays = {}
    for output, tensor, group, nn in ((1, logits, calls[:4], 22), (8, boxes, calls[9:13], 25)):
        expected = Path(f"{args.cpu_output_prefix}.{output}.raw")
        result[output], arrays[output] = verify_head(tensor, group, nn, expected)
    text = json.dumps(result, indent=2) + "\n"
    args.report.write_text(text, encoding="ascii")
    print(text, end="")
    if any(r["gpu_nn_source_mismatches"] or r["gpu_final_mismatches"] for r in result.values()):
        raise SystemExit("GPU head parity failed; see report")
    if args.output_prefix:
        for output, raw in arrays.items():
            with Path(f"{args.output_prefix}.{output}.raw").open("xb") as file:
                file.write(raw.tobytes())


if __name__ == "__main__":
    main()
