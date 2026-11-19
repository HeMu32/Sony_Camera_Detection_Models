"""Create a single slot-08/10 bird-and-plane test raster from supplied inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("inputs", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--bird-shift", type=int, default=0)
    args = cli.parse_args()
    if not 0 <= args.bird_shift <= 50:
        cli.error("bird shift must be 0..50 pixels")
    tags = ("Flight_DSC02983.raw", "Bird_DSC02510.raw")
    inputs = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    rasters = []
    for tag in tags:
        meta = inputs[tag]
        data = (args.inputs / tag).read_bytes()
        if (meta["input_size"], meta["channel_order"], meta["storage"]) != (
                [320, 320], "rgb", "chw") or hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError(f"invalid raster: {tag}")
        rasters.append(np.frombuffer(data, dtype=np.uint8).reshape(3, 320, 320))
    seam = 175 + args.bird_shift
    rows = np.arange(320)
    blend = np.clip((rows + .5 - seam) / 5., 0., 1.)[None, :, None]
    bird = rasters[1][:, np.maximum(0, rows - args.bird_shift), :]
    raster = np.rint(rasters[0] * (1. - blend) + bird * blend).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=False)
    data = raster.tobytes()
    (args.output / "BirdPlane.raw").write_bytes(data)
    Image.fromarray(raster.transpose(1, 2, 0)).save(args.output / "BirdPlane.png")
    (args.output / "manifest.json").write_text(json.dumps({
        "input_size": [320, 320], "layout": "rgb_chw",
        "sources": {tag: inputs[tag]["sha256"] for tag in tags},
        "bird_shift_y": args.bird_shift, "blend_rows": [seam, seam + 5],
        "sha256": hashlib.sha256(data).hexdigest(),
    }, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
