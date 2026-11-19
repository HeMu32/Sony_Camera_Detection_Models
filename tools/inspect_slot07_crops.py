"""Visualize the original slot-07 crop outputs without claiming detection boxes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from inspect_nbg_spatial_outputs import outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Raw crops, manifests, logs and outputs")
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="ascii"))
    canvas = Image.new("RGB", (384 * len(manifest), 450), "#15191d")
    draw = ImageDraw.Draw(canvas)
    records = {}
    for index, (name, spec) in enumerate(manifest.items()):
        prefix = args.results / name
        data = prefix.with_suffix(".raw").read_bytes()
        if sha256(data).hexdigest() != spec["sha256"] or spec["storage"] != "chw":
            raise ValueError(f"unexpected input layout or hash for {name}")
        arrays = outputs(prefix, prefix.with_suffix(".log"))
        if [a.shape for a in arrays] != [(1, 3), (1, 3, 32, 32), (1, 3), (1, 3)]:
            raise ValueError(f"unexpected slot-07 interface for {name}")
        raster = np.frombuffer(data, dtype=np.uint8).reshape(3, 128, 128).transpose(1, 2, 0)
        x0 = index * 384
        canvas.paste(Image.fromarray(raster).resize((384, 384)), (x0, 30))
        values, points = [], {}
        for c, heat in enumerate(arrays[1][0]):
            y, x = np.unravel_index(heat.argmax(), heat.shape)
            peak = float(heat[y, x])
            values.append({"channel": c, "max": peak, "grid_xy": [int(x), int(y)] if peak else None})
            if peak:
                points.setdefault((x, y), []).append(str(c))
        for (x, y), channels in points.items():
            cx, cy = x0 + x * 12 + 6, 30 + y * 12 + 6
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline="#58ffdc", width=2)
            label = "/".join(channels)
            draw.text((cx + 8, cy - 5 if len(channels) > 1 else cy + 6), label,
                      fill="#58ffdc", stroke_width=2, stroke_fill="black")
        draw.text((x0 + 7, 9), f"{name}: manual crop, activation peaks only", fill="white")
        draw.text((x0 + 7, 424), "Output 1 peaks: " + ", ".join(f"{r['max']:.4f}" for r in values),
                  fill="white")
        records[name] = {
            "crop_source_xyxy": spec["crop"], "input_sha256": spec["sha256"],
            "output0": arrays[0].ravel().tolist(), "output1_peaks": values,
            "output2": arrays[2].ravel().tolist(), "output3": arrays[3].ravel().tolist(),
        }
    canvas.save(args.png)
    args.png.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
