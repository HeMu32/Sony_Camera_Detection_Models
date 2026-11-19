"""Render raw activation maxima for visual diagnostics, not detection boxes."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def outputs(prefix, specs_file, *, indices=None):
    specs = []
    for line in specs_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[0] == "param" and fields[2] == "output":
            spec = dict(field.split("=", 1) for field in fields[3:])
            shape = tuple(map(int, spec["shape"].split("x")))
            if spec["dtype"] != "0x3":
                raise ValueError("this diagnostic expects unsigned 8-bit outputs")
            specs.append((shape, float(spec.get("scale", "1")), int(spec.get("zp", "0"))))
    if not specs:
        raise ValueError("no output specifications")
    arrays = []
    for index, (shape, scale, zero) in enumerate(specs):
        if indices is not None and index not in indices:
            arrays.append(None)
            continue
        path = prefix.with_name(f"{prefix.name}.{index}.raw")
        raw = np.fromfile(path, dtype=np.uint8).reshape(shape[::-1])
        arrays.append((raw.astype(np.float32) - zero) * scale)
    return arrays


def candidates(family, arrays):
    if family == "slot09":
        for i, tensor in enumerate(arrays[:7]):
            logits = tensor[0].reshape(6, 7, *tensor.shape[-2:])
            values = np.exp(logits - logits.max(axis=1, keepdims=True))
            data = values / values.sum(axis=1, keepdims=True)
            a, c, y, x = np.unravel_index(data[:, 1:].argmax(), data[:, 1:].shape)
            yield f"out{i}:a{a}/c{c + 1}", data[a, c + 1], int(x), int(y)
    elif family == "slot01":
        for c, data in enumerate(arrays[1][0]):
            y, x = np.unravel_index(data.argmax(), data.shape)
            yield f"out1:c{c}", data, int(x), int(y)
    elif family == "slot18":
        for i in (0, 1, 2, 6, 7, 10, 11):
            data = arrays[i][0].max(axis=0)
            y, x = np.unravel_index(data.argmax(), data.shape)
            yield f"out{i}:max", data, int(x), int(y)
    elif family == "slot08":
        for i in range(1, 7):
            tensor = arrays[i][0]
            logits = tensor.reshape(6, 5, *tensor.shape[-2:])
            values = np.exp(logits - logits.max(axis=1, keepdims=True))
            data = values / values.sum(axis=1, keepdims=True)
            a, c, y, x = np.unravel_index(data[:, 1:].argmax(), data[:, 1:].shape)
            yield f"out{i}:a{a}/c{c + 1}", data[a, c + 1], int(x), int(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("slot01", "slot08", "slot09", "slot18"))
    parser.add_argument("prefix", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("photo", type=Path)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    geometry = json.loads(args.manifest.read_text(encoding="utf-8"))[
        args.photo.stem + ".raw"
    ]
    with Image.open(args.photo) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((840, 560), Image.Resampling.LANCZOS)
    sx, sy = geometry["scale_xy"]
    ox, oy = geometry["offset"]
    iw, ih = geometry["input_size"]
    sw, sh = geometry["source_size"]
    dx, dy = image.width / sw, image.height / sh
    arrays = outputs(args.prefix, args.log)
    rows = []
    for key, data, x, y in candidates(args.family, arrays):
        h, w = data.shape
        px = ((x + 0.5) * iw / w - ox) / sx
        py = ((y + 0.5) * ih / h - oy) / sy
        rows.append({
            "key": key, "cell": [x, y], "value": float(data[y, x]),
            "source_xy": [round(px, 2), round(py, 2)],
            "inside_source": 0 <= px < sw and 0 <= py < sh,
            "span": [float(data.min()), float(data.max())],
        })
    canvas = Image.new("RGB", (image.width + 225, max(image.height + 34, 35 + 20 * len(rows))),
                       (25, 27, 30))
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), f"{args.photo.stem} / {args.family} - activation peaks, not boxes",
              fill="white")
    colors = ("#ffed51", "#54f0eb", "#fc7f73", "#a6ff93", "#ef9dff")
    for n, row in enumerate(rows, 1):
        px, py = row["source_xy"]
        color = colors[(n - 1) % len(colors)]
        if row["inside_source"]:
            u, v = px * dx, 34 + py * dy
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=color, width=2)
            draw.text((u + 6, v - 5), str(n), fill=color, stroke_width=1, stroke_fill="black")
        draw.text((image.width + 10, 15 + 20 * n),
                  f"{n:2d}  {row['key']:15s}  {row['value']:.5f}", fill=color)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.png)
    args.png.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
