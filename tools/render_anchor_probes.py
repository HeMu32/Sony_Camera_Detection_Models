"""Render provisional slot-08/10 boxes; the camera priors are not recovered."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from inspect_nbg_spatial_outputs import outputs


@dataclass(frozen=True)
class Head:
    cls: int
    reg: int
    class_count: int
    grid: int
    center_scales: dict
    name: str
    color: str


# Center scales estimated from neighboring cells and translated inputs.
# Absolute priors use an experimental common center variance of 0.1.
HEADS = {
    "08": Head(1, 8, 5, 40, {0: (2.28, 5.88), 1: (2.63, 4.28),
                             2: (3.20, 3.37), 5: (4.12, 3.89)}, "Bird", "#4ef2e2"),
    "10": Head(2, 9, 3, 20, {3: (8.94, 4.91), 4: (10.71, 4.07)}, "Plane", "#ffd65d"),
}


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return overlap / ((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - overlap)


def class_scores(logits, class_count):
    cls = logits.reshape(6, class_count, *logits.shape[-2:])
    exp = np.exp(cls - cls.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def suppress(rows, coordinates):
    kept = []
    for row in sorted(rows, key=lambda r: -r["classification_softmax"]):
        if all(iou(row[coordinates], other[coordinates]) < .35 for other in kept):
            kept.append(row)
    return kept


def predict(slot, arrays, threshold):
    head = HEADS[slot]
    g = head.grid
    ps = class_scores(arrays[head.cls][0], head.class_count)
    reg = arrays[head.reg][0].reshape(6, 4, g, g)
    rows = []
    for a, (sx, sy) in head.center_scales.items():
        for y, x in np.argwhere(ps[a] >= threshold):
            dx, dy, dw, dh = reg[a, :, y, x]
            center = (np.array([x, y]) + .5) * (320 / g) + [sx * dx, sy * dy]
            half_size = 5 * np.array([sx, sy]) * np.exp(.2 * np.array([dw, dh]))
            row = {
                "slot": int(slot), "label": head.name, "class_index": 1,
                "grid_xy": [int(x), int(y)], "anchor": a,
                "classification_softmax": float(ps[a, y, x]),
                "delta": [float(v) for v in (dx, dy, dw, dh)],
                "xyxy_raster": np.concatenate((center - half_size, center + half_size)),
            }
            rows.append(row)
    return suppress(rows, "xyxy_raster"), float(ps.max())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("png", type=Path)
    cli.add_argument("--prediction", nargs=3, action="append", required=True,
                     metavar=("SLOT", "PREFIX", "LOG"))
    cli.add_argument("--manifest", type=Path)
    cli.add_argument("--probes", type=Path)
    cli.add_argument("--threshold", type=float, default=.85)
    args = cli.parse_args()
    if not 0 < args.threshold <= 1 or (args.probes and not args.manifest):
        cli.error("invalid threshold or probe without base manifest")
    geom = (json.loads(args.manifest.read_text(encoding="utf-8"))[args.source.stem + ".raw"]
            if args.manifest else None)
    probes = json.loads(args.probes.read_text(encoding="ascii")) if args.probes else {}
    with Image.open(args.source) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    size = image.size
    if not geom and size != (320, 320):
        cli.error("a source image without an input manifest must be the 320x320 raster")
    if not geom:
        image = image.resize((800, 800), Image.Resampling.NEAREST)
    image.thumbnail((1050, 800), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    detections, max_scores = [], {}
    for slot, prefix, log in args.prediction:
        prefix = Path(prefix)
        head = HEADS[slot]
        rows, maximum = predict(slot, outputs(prefix, Path(log), indices={head.cls, head.reg}),
                                args.threshold)
        max_scores[slot] = maximum
        for row in rows:
            xy = row["xyxy_raster"].reshape(2, 2)
            if probes:
                probe = probes[prefix.name + ".raw"]
                if probe["base_sha256"] != geom["sha256"]:
                    raise ValueError("probe does not match this source input")
                xy = (xy - probe["raster_offset"]) / probe["raster_scale"]
            if geom:
                xy = (xy - geom["offset"]) / geom["scale_xy"]
            xy = np.clip(np.sort(xy, axis=0), 0, size)
            if np.any(xy[1] <= xy[0]):
                continue
            row["xyxy_source"] = xy.ravel().tolist()
            row["xyxy_raster"] = row["xyxy_raster"].tolist()
            display = tuple(np.rint(xy * (np.array(image.size) / size)).ravel())
            color = HEADS[slot].color
            draw.rectangle(display, outline=color, width=3)
            draw.text((display[0] + 3, display[1] + 3),
                      f"{row['label']} (probe)", fill=color, stroke_width=2, stroke_fill="black")
            detections.append(row)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.png)
    args.png.with_suffix(".json").write_text(json.dumps({
        "method": "limited host probe, not camera-native priors or full detector",
        "threshold": args.threshold, "class1_softmax_max_by_head": max_scores,
        "boxes": detections,
    }, indent=2) + "\n", encoding="ascii")
    print(f"{args.png.stem}: {len(detections)} boxes; max by slot {max_scores}")


if __name__ == "__main__":
    main()
