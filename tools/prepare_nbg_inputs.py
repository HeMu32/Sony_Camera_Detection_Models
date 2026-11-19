"""Produce explicit image tensor candidates and record the inverse geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def prepare(path, width, height, mode, pad, channels, layout):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return prepare_rgb(image, width, height, mode, pad, channels, layout)


def prepare_rgb(image, width, height, mode, pad, channels, layout):
    iw, ih = image.size
    if mode == "letterbox":
        ratio = min(width / iw, height / ih)
        rw, rh = max(1, round(iw * ratio)), max(1, round(ih * ratio))
    else:
        rw, rh = width, height
    left, top = (width - rw) // 2, (height - rh) // 2
    raster = Image.new("RGB", (width, height), (pad,) * 3)
    raster.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
    data = np.asarray(raster)
    if channels == "bgr":
        data = data[..., ::-1]
    if layout == "chw":
        data = data.transpose(2, 0, 1)
    return data.tobytes(), {
        "source_size": [iw, ih],
        "input_size": [width, height],
        "resized": [rw, rh],
        "offset": [left, top],
        "scale_xy": [rw / iw, rh / ih],
        "channel_order": channels,
        "storage": layout,
        "pad": pad,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("images", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("width", type=int)
    cli.add_argument("height", type=int)
    cli.add_argument("--mode", choices=("letterbox", "stretch"), default="letterbox")
    cli.add_argument("--pad", type=int, choices=range(256), default=128, metavar="0..255")
    cli.add_argument("--channels", choices=("rgb", "bgr"), default="rgb")
    cli.add_argument("--layout", choices=("chw", "hwc"), default="chw")
    args = cli.parse_args()
    if min(args.width, args.height) < 1:
        cli.error("dimensions must be positive")
    photos = sorted(args.images.glob("*.jpg"))
    if not photos:
        cli.error("no JPG input images")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in photos:
        data, geometry = prepare(path, args.width, args.height, args.mode,
                                 args.pad, args.channels, args.layout)
        filename = f"{path.stem}.raw"
        (args.output / filename).write_bytes(data)
        manifest[filename] = dict(
            geometry, source=str(path), sha256=hashlib.sha256(data).hexdigest()
        )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
