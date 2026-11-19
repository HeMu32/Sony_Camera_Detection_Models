"""A/B test two no-FP64 rounding modes on identical raw NBG inputs."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from nbg_cuda_quantization import ROUNDING_MODES
from render_anchor_probes import HEADS, iou, predict
from slot08_cuda import Slot08, output_specs as specs08
from slot10_cuda import Slot10, output_specs as specs10


def decoded_boxes(slot, outputs, specs, threshold):
    head = HEADS[slot]
    arrays = {}
    for idx in (head.cls, head.reg):
        shape, scale, zero = specs[idx]
        arrays[idx] = (outputs[idx].astype(np.float32).reshape(shape[::-1]) - zero) * float(scale)
    rows, peak = predict(slot, arrays, threshold)
    for row in rows:
        row["xyxy_raster"] = row["xyxy_raster"].tolist()
    return rows, peak


def match_boxes(first, second):
    if not first or not second:
        return [], len(first), len(second)
    a, b = (np.array([row["xyxy_raster"] for row in rows]) for rows in (first, second))
    overlaps = np.array([[iou(x, y) for y in b] for x in a])
    left, right = linear_sum_assignment(overlaps, maximize=True)
    matches = [{
        "simulator_index": int(i), "simple_index": int(j),
        "iou": float(overlaps[i, j]),
        "max_edge_shift_px": float(np.abs(a[i] - b[j]).max()),
        "score_abs_delta": abs(first[i]["classification_softmax"] -
                               second[j]["classification_softmax"]),
    } for i, j in zip(left, right) if overlaps[i, j] >= .5]
    return matches, len(a) - len(matches), len(b) - len(matches)


def preview(path, image, slot, boxes):
    source = Image.fromarray(image.transpose(1, 2, 0)).resize((480, 480), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (960, 505), "#141b23")
    draw = ImageDraw.Draw(canvas)
    for side, (mode, rows) in enumerate(zip(ROUNDING_MODES, boxes)):
        left = side * 480
        panel = source.copy()
        panel_draw = ImageDraw.Draw(panel)
        draw.text((left + 10, 6), f"{mode}  |  {HEADS[slot].name}  |  {len(rows)} boxes", fill="white")
        for row in rows:
            xyxy = tuple(np.array(row["xyxy_raster"]) * 1.5)
            panel_draw.rectangle(xyxy, outline=HEADS[slot].color, width=2)
        canvas.paste(panel, (left, 25))
    with path.open("xb") as file:
        canvas.save(file, format="PNG")


def benchmark(models, image, iterations):
    timings = {name: [] for name in models}
    for model in models.values():
        for _ in range(8):
            model.infer(image)
    torch.cuda.synchronize()
    for repeat in range(6):
        order = ROUNDING_MODES if repeat % 2 == 0 else ROUNDING_MODES[::-1]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                models[name].infer(image)
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) / iterations)
    return {k: {"median": median(v), "samples": v} for k, v in timings.items()}


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("slot", choices=("08", "10"))
    cli.add_argument("package", type=Path)
    cli.add_argument("inputs", nargs="+", type=Path)
    cli.add_argument("--report", type=Path, required=True)
    cli.add_argument("--preview-dir", type=Path)
    cli.add_argument("--threshold", type=float, default=.85)
    cli.add_argument("--iterations", type=int, default=20)
    args = cli.parse_args()
    if not 0 < args.threshold <= 1 or args.iterations < 0:
        cli.error("threshold must be in (0, 1] and iterations must be nonnegative")
    model_type, get_specs = {"08": (Slot08, specs08), "10": (Slot10, specs10)}[args.slot]
    models = {name: model_type(args.package, rounding=name) for name in ROUNDING_MODES}
    specs = get_specs(models["simulator"])
    report = {
        "slot": args.slot, "model_sha256": models["simulator"].model_hash,
        "cuda_device": torch.cuda.get_device_name(), "threshold": args.threshold,
        "decoder": "experimental host anchor probe; not camera-native",
        "match_iou_minimum": .5, "inputs": [],
    }
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(args.inputs):
        data = path.read_bytes()
        image = np.frombuffer(data, dtype=np.uint8).copy().reshape(3, 320, 320)
        gpu = torch.from_numpy(image).cuda()
        baseline, simple = ({i: x.cpu().numpy() for i, x in m.infer(gpu).items()}
                            for m in models.values())
        tensors = {}
        for i, before in sorted(baseline.items()):
            abs_delta = np.abs(before.astype(np.int16) - simple[i])
            tensors[i] = {"bytes": before.size, "different_bytes": int(np.count_nonzero(abs_delta)),
                          "max_abs_delta": int(abs_delta.max())}
        rows_a, peak_a = decoded_boxes(args.slot, baseline, specs, args.threshold)
        rows_b, peak_b = decoded_boxes(args.slot, simple, specs, args.threshold)
        matches, only_a, only_b = match_boxes(rows_a, rows_b)
        report["inputs"].append({
            "path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "outputs": tensors,
            "total_bytes": sum(t["bytes"] for t in tensors.values()),
            "different_bytes": sum(t["different_bytes"] for t in tensors.values()),
            "max_abs_delta": max(t["max_abs_delta"] for t in tensors.values()),
            "simulator": {"boxes": rows_a, "peak": peak_a, "unmatched": only_a},
            "simple": {"boxes": rows_b, "peak": peak_b, "unmatched": only_b},
            "matches": matches,
        })
        if args.preview_dir:
            png = args.preview_dir / f"{args.slot}-{index:02d}-{path.stem}.png"
            preview(png, image, args.slot, (rows_a, rows_b))
        if index == 0 and args.iterations:
            report["forward_ms_first_input"] = benchmark(models, gpu, args.iterations)
    with args.report.open("x", encoding="ascii") as file:
        json.dump(report, file, indent=2)
    for item in report["inputs"]:
        a = len(item["simulator"]["boxes"])
        b = len(item["simple"]["boxes"])
        print(f"{item['path']}: {item['different_bytes']}/{item['total_bytes']} changed bytes; "
              f"max delta {item['max_abs_delta']}; boxes {a}/{b}; {len(item['matches'])} matches")
    if args.iterations:
        print({k: round(v["median"], 3)
               for k, v in report["forward_ms_first_input"].items()})


if __name__ == "__main__":
    main()
