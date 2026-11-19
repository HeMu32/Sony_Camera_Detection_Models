"""Build an A1 II slot-18 animal-pose box from its 18-channel heatmap and PAF pairs."""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from pose_struct import load, specs  # noqa: E402


def peaks(plane, threshold):
    mask = (plane >= threshold) & (plane == maximum_filter(plane, size=3))
    ys, xs = np.nonzero(mask)
    return [(int(x), int(y), float(plane[y, x])) for y, x in zip(ys, xs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("log")
    ap.add_argument("manifest")
    ap.add_argument("photo")
    ap.add_argument("png")
    ap.add_argument("--heat", type=int, default=2)
    ap.add_argument("--thresholds", default="0.02,0.05,0.1,0.15")
    args = ap.parse_args()
    prefix = pathlib.Path(args.prefix)
    spec = specs(args.log)
    heat = load(prefix, args.heat, *spec[args.heat])
    heat = heat.reshape(heat.shape[1], heat.shape[2], heat.shape[3])
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    key = pathlib.Path(args.photo).stem + ".raw"
    geom = manifest[key] if key in manifest else {
        "input_size": manifest["input_size"], "scale_xy": [1.0, 1.0],
        "offset": [0.0, 0.0]}
    with Image.open(args.photo) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
    size = image.size
    image.thumbnail((1050, 700), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    sx, sy = geom["scale_xy"]
    ox, oy = geom["offset"]
    iw, ih = geom["input_size"]
    _, gh, gw = heat.shape
    boxes = []
    for threshold in [float(t) for t in args.thresholds.split(",")]:
        points = [(c, *p) for c in range(heat.shape[0]) for p in peaks(heat[c], threshold)]
        if not points:
            boxes.append({"threshold": threshold, "xyxy_source": None, "joints": []})
            continue
        xy = np.array([[(x + .5) * iw / gw, (y + .5) * ih / gh] for _, x, y, _ in points])
        low, high = xy.min(axis=0), xy.max(axis=0)
        height = max(high[1] - low[1], 1.)
        low -= [.06 * height, .12 * height]
        high += [.10 * height, .05 * height]
        raster = np.array([*low, *high])
        raw = [(raster[0] - ox) / sx, (raster[1] - oy) / sy,
               (raster[2] - ox) / sx, (raster[3] - oy) / sy]
        rect = np.clip(raw, 0., [size[0] - 1, size[1] - 1] * 2).tolist()
        boxes.append({"threshold": threshold, "xyxy_source": rect,
                       "joints": [[c, x, y, round(v, 4)] for c, x, y, v in points]})
    main_box = boxes[-1]
    if main_box["xyxy_source"]:
        rect = main_box["xyxy_source"]
        display = tuple(round(p * size / orig) for p, size, orig in
                        zip(rect, image.size * 2, (size[0], size[1]) * 2))
        draw.rectangle(display, outline="#4dff88", width=3)
        draw.text((display[0] + 4, display[1] + 3), "A1 II slot18 animal", fill="white",
                  stroke_width=2, stroke_fill="black")
        for c, x, y, v in main_box["joints"]:
            u = ((x + .5) * iw / gw - ox) / sx * image.width / size[0]
            w = ((y + .5) * ih / gh - oy) / sy * image.height / size[1]
            draw.ellipse((u - 3, w - 3, u + 3, w + 3), outline="#4dff88", width=1)
    pathlib.Path(args.png).parent.mkdir(parents=True, exist_ok=True)
    image.save(pathlib.Path(args.png))
    pathlib.Path(args.png).with_suffix(".json").write_text(json.dumps({
        "method": "host grouping of A1 II slot-18 18-channel heatmap, not the firmware decoder",
        "heatmap_max": float(heat.max()), "boxes": boxes}, indent=1) + "\n",
        encoding="ascii")
    print(json.dumps([{"threshold": b["threshold"], "xyxy_source": b["xyxy_source"],
                        "joints": len(b["joints"])} for b in boxes], indent=0))


if __name__ == "__main__":
    main()
