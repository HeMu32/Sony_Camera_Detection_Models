"""Compute and validate the original slot-08 multiscale predictions on CUDA."""

import argparse
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inspect_nbg_tp_trace import read_trace
from verify_nbg_cuda_backbone import BLOCKS, verify
from verify_nbg_cuda_head40 import head_block
from verify_nbg_cuda_tiles import read_rows
from verify_nbg_cuda_tp40 import verify_head


@dataclass(frozen=True)
class Scale:
    grid: int
    before: tuple
    branch: int
    class_zp: int
    bbox_zp: int
    output: tuple
    class_tp: tuple
    bbox_tp: tuple


SCALES = (
    Scale(40, (), 21, 149, 129, (1, 8, 15), (1, 4), (10, 13)),
    Scale(20, (26, 27, 28, 29), 30, 127, 130, (2, 9, 16), (18, 21), (27, 30)),
    Scale(10, (35, 36, 37, 38), 39, 142, 130, (3, 10, 17), (35, 38), (44, 47)),
    Scale(5, (44, 45, 46, 47), 48, 160, 134, (4, 11, 18), (52, 55), (61, 64)),
    Scale(3, (), 65, 190, 146, (5, 12, 19), (80, 83), (89, 92)),
    Scale(1, (), 59, 164, 154, (6, 13, 20), (77, 77), (79, 79)),
)


def interval(bounds):
    return range(bounds[0], bounds[1] + 1)


def sigmoid_hwc(tensor):
    value = torch.sigmoid(tensor.float()).mul(256).round().clamp(0, 255)
    return value.to(torch.uint8).permute(1, 2, 0).contiguous()


def aux_output(tensor, reference):
    raw = sigmoid_hwc(tensor).cpu().numpy()
    expected = np.fromfile(reference, dtype=np.uint8).reshape(raw.shape)
    return {"bytes": raw.size, "gpu_final_mismatches": int(np.count_nonzero(raw != expected))}, raw


def checked_block(rows, checks, cmd, x, channels, kernel, *, relu=True, zp=0, padding=None):
    result, comparison = head_block(rows, cmd, x, channels, kernel,
                                    relu=relu, zero_point=zp, padding=padding)
    checks.append(comparison)
    return result


def shared_tail(block, x, first=53):
    x = block(first, x, 128, 1)
    levels = {}
    for cmd, start in ((first + 1, 1), (first + 3, 1), (first + 5, 0)):
        # Raw TP pads the leading edge for odd spatial dimensions, then packs
        # each 2x2 block as four phases for every original channel.
        padded = F.pad(x[None], (start, 2, start, 2), value=0)
        packed = F.pixel_unshuffle(padded, 2)[0]
        h = block(cmd, packed, 256, 2, padding=(0, 0, 0, 0))
        levels[h.shape[-1]] = h
        if cmd != first + 5:
            x = block(cmd + 1, h, 128, 1)
    return levels


def compute_branches(block, shared, scale, class_channels=30, transition_channels=128):
    if scale.before:
        shared = F.max_pool2d(shared[None].float(), 2)[0].to(torch.uint8)
        channels = (transition_channels, 2 * transition_channels) * 2
        for cmd, channels, kernel in zip(scale.before, channels, (1, 3, 1, 3)):
            shared = block(cmd, shared, channels, kernel)
    start = scale.branch
    # An unrelated command is interleaved between the 1x1 class and auxiliary producers.
    aux_cmd = start + (3 if scale.grid == 1 else 2)
    cls_feature = block(start, shared, 128, 1)
    classification = block(start + 1, cls_feature, class_channels, 3,
                           zp=scale.class_zp, relu=False)
    auxiliary = block(aux_cmd, cls_feature, 6, 3)
    reg_feature = block(aux_cmd + 1, shared, 128, 1)
    regression = block(aux_cmd + 2, reg_feature, 24, 3, zp=scale.bbox_zp, relu=False)
    return shared, classification, regression, auxiliary


def run_scales(block, shared):
    small = None
    for scale in SCALES:
        if scale.grid == 3:
            small = shared_tail(block, shared)
        feed = small[scale.grid] if small is not None else shared
        shared, classification, regression, auxiliary = compute_branches(block, feed, scale)
        yield scale, classification, regression, auxiliary


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("nn_trace", type=Path)
    cli.add_argument("tp_trace", type=Path)
    cli.add_argument("first_cuda_activation", type=Path)
    cli.add_argument("cpu_output_prefix", type=Path)
    cli.add_argument("report", type=Path)
    cli.add_argument("--output-prefix", type=Path)
    args = cli.parse_args()
    if not torch.cuda.is_available():
        cli.error("CUDA required")
    activation = np.fromfile(args.first_cuda_activation, dtype=np.uint8).reshape(32, 320, 320)
    rows = read_rows(args.nn_trace)
    _, shared = verify(activation, rows, BLOCKS[:-1])
    shared = torch.from_numpy(shared).cuda()
    capture = {i for scale in SCALES for bounds in (scale.class_tp, scale.bbox_tp)
               for i in interval(bounds)}
    _, calls = read_trace(args.tp_trace, capture=capture)
    checks, results, arrays = [], {}, {}
    for scale, classification, regression, auxiliary in run_scales(
        partial(checked_block, rows, checks), shared
    ):
        cls, reg, aux = scale.output
        for idx, tensor, bounds, nn in ((cls, classification, scale.class_tp,
                                         scale.branch + (2 if scale.grid == 1 else 1)),
                                        (reg, regression, scale.bbox_tp,
                                         scale.branch + (5 if scale.grid == 1 else 4))):
            expected = Path(f"{args.cpu_output_prefix}.{idx}.raw")
            results[idx], arrays[idx] = verify_head(tensor, [calls[i - 1] for i in interval(bounds)],
                                                    nn, expected)
        results[aux], arrays[aux] = aux_output(auxiliary, Path(f"{args.cpu_output_prefix}.{aux}.raw"))
    text = json.dumps({"outputs": results, "nn_checks": checks}, indent=2) + "\n"
    args.report.write_text(text, encoding="ascii")
    for idx in sorted(results):
        report = results[idx]
        print(f"output {idx}: {arrays[idx].size} bytes, {report['gpu_final_mismatches']} mismatches")
    if any(report.get("gpu_nn_source_mismatches", 0) or report["gpu_final_mismatches"]
           for report in results.values()):
        raise SystemExit("CUDA multiscale parity failed; see report")
    if args.output_prefix:
        for output, raw in arrays.items():
            with Path(f"{args.output_prefix}.{output}.raw").open("xb") as file:
                file.write(raw.tobytes())


if __name__ == "__main__":
    main()
