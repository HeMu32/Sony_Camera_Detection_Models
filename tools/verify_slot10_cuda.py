"""Compare all standalone slot-10 outputs, optionally checking internal TP tiles."""

import argparse
from pathlib import Path

import numpy as np
import torch

from inspect_nbg_tp_trace import read_trace
from slot10_cuda import SCALES, Slot10
from verify_nbg_cuda_multiscale import interval
from verify_nbg_cuda_tp40 import verify_head


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("package", "input", "cpu_output_prefix"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--tp-trace", type=Path)
    args = cli.parse_args()
    model = Slot10(args.package)
    image = torch.from_numpy(np.fromfile(args.input, dtype=np.uint8).reshape(3, 320, 320)).cuda()
    outputs = model.infer(image)
    total = 0
    for idx, output in outputs.items():
        expected = Path(f"{args.cpu_output_prefix}.{idx}.raw").read_bytes()
        actual = output.cpu().numpy().tobytes()
        if actual != expected:
            raise SystemExit(f"output {idx} differs from the CPU reference")
        total += len(actual)
    print(f"{args.input.stem}: {len(outputs)} outputs, {total} bytes, 0 mismatches")
    if args.tp_trace:
        capture = {i for scale in SCALES for bounds in (scale.class_tp, scale.bbox_tp)
                   for i in interval(bounds)}
        count, calls = read_trace(args.tp_trace, capture=capture)
        if (count, len(calls)) != (74, 109):
            raise ValueError("unexpected slot-10 trace schedule")
        for scale, cls, reg, _ in model.heads(image):
            for idx, tensor, bounds, nn in zip(
                scale.output[:2], (cls, reg), (scale.class_tp, scale.bbox_tp),
                (scale.branch + (2 if scale.grid == 1 else 1),
                 scale.branch + (5 if scale.grid == 1 else 4)),
            ):
                report, _ = verify_head(
                    tensor, [calls[i - 1] for i in interval(bounds)], nn,
                    Path(f"{args.cpu_output_prefix}.{idx}.raw"),
                )
                if report["gpu_nn_source_mismatches"] or report["gpu_final_mismatches"]:
                    raise SystemExit(f"output {idx} differs at its TP stage")
        print(f"{len(capture)} TP tiles: input, arithmetic, and output all exact")


if __name__ == "__main__":
    main()
