"""Build experimental R5 slot-12 animal-pose boxes from heatmaps and limb fields."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment

from inspect_nbg_spatial_outputs import outputs
from render_pose_boxes import peaks


# Zero-based vector-pair index and its supported directed heatmap channels.
EDGES = (
    (0, 17, 3), (6, 6, 4), (7, 17, 5), (8, 17, 6),
    (9, 5, 7), (10, 6, 8), (11, 7, 9), (12, 8, 10),
    (13, 5, 11), (14, 6, 12), (15, 11, 13), (16, 12, 14),
)


def affinity(field, first, second):
    x, y, a = first
    u, v, b = second
    delta = np.array([u - x, v - y])
    length = np.linalg.norm(delta)
    if length < 1 or length > field.shape[1] * .65:
        return 0.
    coords = [np.linspace(y - .5, v - .5, 16),
              np.linspace(x - .5, u - .5, 16)]
    vectors = np.stack([map_coordinates(layer, coords, order=1) for layer in field], axis=1)
    scores = vectors @ (delta / length)
    if (scores > .025).mean() < .55 or scores.mean() < .035:
        return 0.
    return float(scores.mean() * np.sqrt(a * b))


def components(vector, heatmap):
    if vector.shape != (40, 36, 52) or heatmap.shape != (19, 36, 52):
        raise ValueError("expected original R5 slot-12 output tensors")
    joints = [peaks(layer, .024) for layer in heatmap]
    groups = {}
    for k, a, b in EDGES:
        if not joints[a] or not joints[b]:
            continue
        field = vector[2*k:2*k + 2]
        scores = np.array([[affinity(field, u, v) for v in joints[b]] for u in joints[a]])
        ar, br = linear_sum_assignment(scores, maximize=True)
        for i, j in zip(ar, br):
            if not scores[i, j]:
                continue
            left, right = (a, int(i)), (b, int(j))
            first = groups.get(left, {left})
            second = groups.get(right, {right})
            if first is not second and {c for c, _ in first} & {c for c, _ in second}:
                continue
            union = first | second
            for key in union:
                groups[key] = union
    results = []
    for group in {frozenset(value) for value in groups.values()}:
        ids = {c for c, _ in group}
        if len(ids) < 4 or not ids & {9, 10} or not ids & {3, 4, 17}:
            continue
        points = [(c, *joints[c][i]) for c, i in sorted(group)]
        if max(p[3] for p in points) < .18:
            continue
        results.append(points)
    return sorted(results, key=lambda group: min(p[1] for p in group))


def to_boxes(joints):
    boxes = []
    for points in joints:
        xy = np.array([p[1:3] for p in points])
        low, high = xy.min(axis=0), xy.max(axis=0)
        height = high[1] - low[1]
        if height < 3:
            continue
        # These margins cover anatomy outside the detected joint centers, not the tail.
        low -= [.12 * height, .23 * height]
        high += [.19 * height, .09 * height]
        boxes.append({"xyxy_grid": [*low, *high], "joints_grid": points})
    return boxes


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("prefix", type=Path)
    cli.add_argument("log", type=Path)
    cli.add_argument("manifest", type=Path)
    cli.add_argument("photo", type=Path)
    cli.add_argument("png", type=Path)
    cli.add_argument("--probe-manifest", type=Path)
    args = cli.parse_args()
    geom = json.loads(args.manifest.read_text(encoding="utf-8"))[args.photo.stem + ".raw"]
    with Image.open(args.photo) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    size = image.size
    image.thumbnail((1050, 700), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    vector, heatmap = (t[0] for t in outputs(args.prefix, args.log))
    boxes = to_boxes(components(vector, heatmap))
    scale = np.array(geom["scale_xy"])
    offset = np.array(geom["offset"])
    probe_scale, probe_offset = np.ones(2), np.zeros(2)
    if args.probe_manifest:
        probe = json.loads(args.probe_manifest.read_text(encoding="ascii"))[
            args.prefix.name + ".raw"]
        if probe["base_sha256"] != geom["sha256"]:
            raise ValueError("probe does not match this source input")
        probe_scale, probe_offset = probe["raster_scale"], probe["raster_offset"]
    for n, box in enumerate(boxes, 1):
        raster = np.array(box["xyxy_grid"]).reshape(2, 2) * 8
        xy = ((raster - probe_offset) / probe_scale - offset) / scale
        rect = np.clip(np.sort(xy, axis=0), 0, size).ravel()
        box["xyxy_source"] = rect.tolist()
        display = tuple(np.rint(rect * (np.array(image.size * 2) / (size * 2))))
        draw.rectangle(display, outline="#35f5da", width=3)
        draw.text((display[0] + 4, display[1] + 4), f"Animal pose {n}", fill="white",
                  stroke_width=2, stroke_fill="black")
    args.png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.png)
    args.png.with_suffix(".json").write_text(json.dumps({
        "method": "experimental host pose grouping; not the Sony firmware box decoder",
        "boxes": boxes,
    }, indent=2) + "\n", encoding="ascii")
    print(f"{args.prefix.name}: {len(boxes)} animal-pose boxes")


if __name__ == "__main__":
    main()
