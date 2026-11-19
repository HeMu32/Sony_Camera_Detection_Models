"""Compare full-frame and fixed tiled inputs to the original slot-08 graph."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from prepare_nbg_inputs import prepare_rgb
from render_anchor_probes import HEADS, class_scores, predict, suppress
from slot08_cuda import Slot08, output_specs


def crops(width, height):
    yield 0, 0, width, height
    cw, ch = (width * 2 + 2) // 3, (height * 2 + 2) // 3
    if cw < width and ch < height:
        for top in (0, height - ch):
            for left in (0, width - cw):
                yield left, top, left + cw, top + ch


def project(row, geom, crop):
    xy = (row["xyxy_raster"].reshape(2, 2) - geom["offset"]) / geom["scale_xy"]
    if np.any(xy.mean(axis=0) < 0) or np.any(xy.mean(axis=0) >= geom["source_size"]):
        return None
    xy = (np.clip(xy, 0, geom["source_size"]) + crop[:2]).ravel()
    if np.any(xy[2:] <= xy[:2]):
        return None
    return xy


def run(model, image, threshold):
    specs = output_specs(model)
    head = HEADS["08"]
    all_boxes, tiles = [], []
    for crop in crops(*image.size):
        tile = len(tiles)
        raw, geom = prepare_rgb(image.crop(crop), 320, 320, "letterbox", 128, "rgb", "chw")
        raster = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy().reshape(3, 320, 320))
        inferred = model.infer(raster)
        decoded = {}
        for index in (*range(1, 7), *range(8, 14)):
            _, scale, zp = specs[index]
            decoded[index] = (inferred[index].cpu().numpy().astype(np.float32) - zp) * float(scale)

        rows, _ = predict("08", {i: a[None] for i, a in decoded.items()}, threshold)
        per_level = []
        for index in range(1, 7):
            ps = class_scores(decoded[index], head.class_count)
            grid = ps.shape[-1]
            reg = decoded[index + 7].reshape(6, 4, grid, grid)
            cells = []
            for anchor, y, x in np.argwhere(ps >= threshold):
                center = ((np.array([x, y]) + .5) * (320 / grid) - geom["offset"])
                source_xy = center / geom["scale_xy"] + crop[:2]
                cells.append({"anchor": int(anchor), "cell_xy": [int(x), int(y)],
                              "score": float(ps[anchor, y, x]), "unregressed_source_xy": source_xy,
                              "delta": reg[anchor, :, y, x]})
            per_level.append({"grid": grid, "max": float(ps.max()), "over_threshold": cells})
        tiles.append({"source_xyxy": crop, "input_sha256": sha256(raw).hexdigest(),
                      "classification": per_level})
        for row in rows:
            xy = project(row, geom, crop)
            if xy is not None:
                all_boxes.append(dict(row, xyxy_source=xy, tile=tile))
    return all_boxes, tiles


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("png", type=Path)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--threshold", type=float, default=.85)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("threshold must be in (0, 1]")
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    all_boxes, tiles = run(Slot08(args.package), image, args.threshold)
    kept = suppress(all_boxes, "xyxy_source")
    full = [row for row in all_boxes if row["tile"] == 0]
    original_size = image.size
    image.thumbnail((1050, 800), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for i, row in enumerate(kept, 1):
        xy = row["xyxy_source"].reshape(2, 2) * (np.array(image.size) / original_size)
        color = "#4ef2e2" if row["tile"] == 0 else "#ffd65d"
        draw.rectangle(xy.ravel().tolist(), outline=color, width=3)
        draw.text((xy[0, 0] + 3, xy[0, 1] + 3), f"{i} T{row['tile']}",
                  fill=color, stroke_width=2, stroke_fill="black")
    args.png.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.png)
    report = {
        "method": "host tiling with experimental priors, not camera-native",
        "model_sha256": Slot08.model_hash,
        "decoded_grid": HEADS["08"].grid,
        "threshold": args.threshold, "tiles": tiles, "full_frame_boxes": full,
        "merged_boxes": kept,
    }
    args.png.with_suffix(".json").write_text(
        json.dumps(report, indent=2, default=np.ndarray.tolist) + "\n", encoding="ascii")
    print(f"{args.photo.name}: {len(full)} full-frame, {len(kept)} after tiled merge")


if __name__ == "__main__":
    cli()
