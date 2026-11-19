"""Prepare controlled geometric probes from a previously recorded RGB raster."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


# Maps original raster edge coordinates to the probe raster, about its center.
PROBES = {
    "shift_a": (1.0, 1.0, 3, -5),
    "shift_b": (1.0, 1.0, -11, 7),
    "shift_c": (1.0, 1.0, 13, 10),
    "mirror": (-1.0, 1.0, 0, 0),
    "zoom_in": (1.25, 1.25, 0, 0),
    "zoom_out": (0.8, 0.8, 0, 0),
}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    meta = json.loads(args.input.with_name("manifest.json").read_text(encoding="utf-8"))
    base = meta[args.input.name]
    assert base["channel_order"] == "rgb" and base["storage"] == "chw"
    w, h = base["input_size"]
    data = args.input.read_bytes()
    if hashlib.sha256(data).hexdigest() != base["sha256"]:
        raise ValueError("input hash differs from manifest")
    image = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(3, h, w)
                            .transpose(1, 2, 0))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, (sx, sy, dx, dy) in PROBES.items():
        tx, ty = (1 - sx) * w / 2 + dx, (1 - sy) * h / 2 + dy
        probe = image.transform(
            (w, h), Image.Transform.AFFINE,
            (1 / sx, 0, -tx / sx, 0, 1 / sy, -ty / sy),
            resample=Image.Resampling.BILINEAR, fillcolor=(base["pad"],) * 3,
        )
        raw = np.asarray(probe).transpose(2, 0, 1).tobytes()
        filename = f"{name}.raw"
        (args.output / filename).write_bytes(raw)
        manifest[filename] = dict(
            base_input=str(args.input), base_sha256=base["sha256"],
            raster_scale=[sx, sy], raster_offset=[tx, ty],
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
