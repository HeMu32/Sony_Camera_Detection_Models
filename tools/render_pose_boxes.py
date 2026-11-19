"""Render host-derived person boxes from the Sony slot-01 keypoint heatmaps."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from inspect_nbg_spatial_outputs import outputs


def peaks(heatmap, threshold):
    points = np.argwhere((heatmap >= threshold) &
                         (heatmap == maximum_filter(heatmap, size=3)))
    return [(float(x) + .5, float(y) + .5, float(heatmap[y, x])) for y, x in points]


def pose_boxes(heatmaps, threshold):
    if heatmaps.ndim != 3 or heatmaps.shape[0] != 19:
        raise ValueError("expected the slot-01 19-channel pose heatmap")
    points = [peaks(layer, threshold) for layer in heatmaps]
    upper, lower = points[17:19]
    if not upper or not lower:
        return []
    cost = np.full((len(upper), len(lower)), np.inf)
    for i, (x, y, _) in enumerate(upper):
        for j, (lx, ly, _) in enumerate(lower):
            dy = ly - y
            if 2 <= dy <= heatmaps.shape[1] / 2 and abs(lx - x) <= max(3, dy * .75):
                cost[i, j] = 3 * abs(lx - x) + dy
    # Dummy assignments allow legitimate incomplete or occluded poses.
    unmatched = np.full((len(upper), len(upper)), 100.)
    ii, jj = linear_sum_assignment(np.column_stack((cost, unmatched)))
    bodies = [(upper[i], lower[j]) for i, j in zip(ii, jj) if j < len(lower)]
    joints = [[] for _ in bodies]
    relative_y = [-.6] * 5 + [0.] * 2 + [.5] * 2 + [.55] * 2 + [1.] * 2 + [1.65] * 2 + [2.35] * 2
    for c, (y_offset, candidates) in enumerate(zip(relative_y, points[:17])):
        if not candidates or not bodies:
            continue
        cost = np.full((len(bodies), len(candidates)), np.inf)
        for i, ((x, y, _), (lx, ly, _)) in enumerate(bodies):
            torso = ly - y
            for j, (px, py, score) in enumerate(candidates):
                horizontal = abs(px - (x + lx) / 2)
                vertical = abs(py - (y + y_offset * torso))
                slack = .85 if 7 <= c <= 10 else .55
                if horizontal <= max(2, torso * .85) and vertical <= max(2, torso * slack):
                    cost[i, j] = (horizontal + vertical * .7) / torso - score * .18
        ii, jj = linear_sum_assignment(
            np.column_stack((cost, np.full((len(bodies), len(bodies)), 100.)))
        )
        for i, j in zip(ii, jj):
            if j < len(candidates):
                joints[i].append((c, *candidates[j]))
    boxes = []
    for ((x, y, us), (lx, ly, ls)), selected in zip(bodies, joints):
        if (not any(c < 7 for c, *_ in selected) or
                not any(c >= 13 for c, *_ in selected)):
            continue
        torso = ly - y
        xs = [x, lx] + [p[1] for p in selected]
        ys = [y, ly] + [p[2] for p in selected]
        boxes.append({
            "xyxy_grid": [min(xs) - .28 * torso, min(ys) - .44 * torso,
                          max(xs) + .28 * torso, max(ys) + .25 * torso],
            "center_pair_score": min(us, ls),
            "joints_grid": selected,
        })
    return boxes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("prefix", type=Path)
    cli.add_argument("log", type=Path)
    cli.add_argument("input_manifest", type=Path)
    cli.add_argument("photo", type=Path)
    cli.add_argument("png", type=Path)
    cli.add_argument("--threshold", type=float, default=.25)
    args = cli.parse_args()
    geom = json.loads(args.input_manifest.read_text(encoding="utf-8"))[
        args.photo.stem + ".raw"
    ]
    heatmaps = outputs(args.prefix, args.log)[1][0]
    boxes = pose_boxes(heatmaps, args.threshold)
    with Image.open(args.photo) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    source_w, source_h = image.size
    image.thumbnail((1050, 700), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    sx, sy = geom["scale_xy"]
    ox, oy = geom["offset"]
    iw, ih = geom["input_size"]
    _, gh, gw = heatmaps.shape
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = box["xyxy_grid"]
        rect = np.clip([(x1 * iw / gw - ox) / sx, (y1 * ih / gh - oy) / sy,
                        (x2 * iw / gw - ox) / sx, (y2 * ih / gh - oy) / sy],
                       0., [source_w - 1., source_h - 1.] * 2).tolist()
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            continue
        box["xyxy_source"] = rect
        display = tuple(round(p * size / original) for p, size, original in
                        zip(rect, image.size * 2, (source_w, source_h) * 2))
        draw.rectangle(display, outline="#37fce3", width=2)
        draw.text((display[0] + 3, display[1] + 2), f"Pose {i}", fill="white",
                  stroke_width=2, stroke_fill="black")
    args.png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.png)
    rendered = [box for box in boxes if "xyxy_source" in box]
    args.png.with_suffix(".json").write_text(json.dumps({
        "method": "host grouping of Sony slot-01 pose outputs, not firmware bbox decoder",
        "threshold": args.threshold, "heatmap_max": float(heatmaps.max()),
        "boxes": rendered,
    }, indent=2) + "\n", encoding="ascii")
    print(f"{args.photo.stem}: {len(rendered)} host-derived pose boxes")


if __name__ == "__main__":
    main()
