"""Verify saved ONNX and TorchScript models against the original host pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw, ImageOps

from export_sony_portable import INPUT_SHAPE, ROOT, SLOTS, sha256
from prepare_nbg_inputs import prepare
from render_anchor_probes import HEADS, predict


STEMS = (
    "Human_DSC03106", "Bird_DSC02510", "Flight_DSC02983", "Cat_DSC04051", "Empty_DSC03284",
)
CPU_REFERENCES = {
    "08": {STEMS[1]: "NoTrace", STEMS[4]: "EmptyLayer"},
    "10": {STEMS[2]: "Flight", STEMS[4]: "Empty"},
}


def compare(actual, reference, specs):
    if len(actual) != len(reference) or len(actual) != len(specs):
        raise AssertionError("missing or extra output tensors")
    rows = []
    for a, b, spec in zip(actual, reference, specs):
        if (a.shape != tuple(spec["shape"]) or a.dtype != np.uint8 or
                b.dtype != np.uint8 or a.shape != b.shape):
            raise AssertionError(f"invalid interface at {spec['name']}")
        error = np.abs(a.astype(np.int16) - b)
        rows.append({"index": spec["index"], "bytes": a.size,
                     "different_bytes": int(np.count_nonzero(error)),
                     "max_abs_code_error": int(error.max())})
    return rows


def decode(slot, arrays, specs, geometry):
    head = HEADS[slot]
    slots = [None] * len(arrays)
    for spec in specs:
        i = spec["index"]
        if i in (head.cls, head.reg):
            slots[i] = ((arrays[i].astype(np.float32) - spec["zero_point"]) *
                        spec["scale"])[None], spec["scale"], spec["zero_point"]
    rows, maximum = predict(slot, slots, .85)
    boxes = []
    for row in rows:
        xy = row["xyxy_raster"].reshape(2, 2)
        xy = (xy - geometry["offset"]) / geometry["scale_xy"]
        xy = np.clip(np.sort(xy, axis=0), 0, geometry["source_size"])
        if np.all(xy[1] > xy[0]):
            boxes.append(dict(row, xyxy_raster=row["xyxy_raster"].tolist(),
                              xyxy_source=xy.ravel().tolist()))
    return {"boxes": boxes, "class1_softmax_max": maximum}


def render_boxes(source, output, boxes):
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    size = image.size
    image.thumbnail((1050, 800), Image.Resampling.LANCZOS)
    scale = np.tile(np.array(image.size) / size, 2)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        coords = tuple(np.rint(box["xyxy_source"] * scale))
        color = HEADS[f"{box['slot']:02}"].color
        draw.rectangle(coords, outline=color, width=3)
        draw.text((coords[0] + 4, coords[1] + 4), box["label"], fill=color,
                  stroke_width=2, stroke_fill="black")
    image.save(output)


def load_cpu_exports(prefix):
    meta = json.loads(prefix.with_suffix(".json").read_text(encoding="ascii"))
    specs = meta["outputs"]
    for ext in ("onnx", "pt"):
        if sha256(prefix.with_suffix(f".{ext}")) != meta["sha256"][ext]:
            raise ValueError(f"model file hash mismatch: {prefix} {ext}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    runtime = ort.InferenceSession(str(prefix.with_suffix(".onnx")), options,
                                   providers=["CPUExecutionProvider"])
    if [item.name for item in runtime.get_outputs()] != [spec["name"] for spec in specs]:
        raise AssertionError("unexpected ONNX output names or order")
    script = torch.jit.load(str(prefix.with_suffix(".pt")), map_location="cpu").eval()
    return meta, runtime, script


def verify_slot(directory, slot, extra_sdk=None):
    model_type, _, placeholders, weights_dir = SLOTS[slot]
    prefix = directory / f"slot{slot}" / f"sony-slot{slot}"
    meta, runtime, script = load_cpu_exports(prefix)
    specs = meta["outputs"]
    weights = ROOT / "debug/cpu_raw" / weights_dir
    cuda = model_type(weights / f"sony-slot{slot}-cuda.npz")
    reports = []
    for stem in STEMS:
        sample_path = ROOT / "assets_aux" / f"{stem}.jpg"
        raw, geometry = prepare(sample_path, 320, 320, "letterbox", 128, "rgb", "chw")
        bundled_input = ROOT / "debug/cpu_inputs/320x320_rgb" / f"{stem}.raw"
        if bundled_input.exists() and bundled_input.read_bytes() != raw:
            raise AssertionError(f"input preprocessing changed: {stem}")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(INPUT_SHAPE).copy()
        x = torch.from_numpy(image)
        ref = cuda.infer(x)
        original = []
        for spec in specs:
            i = spec["index"]
            original.append(np.zeros(spec["shape"], dtype=np.uint8)
                            if i in placeholders else ref[i].cpu().numpy())
        observed = {
            "onnx_cpu": runtime.run(None, {"rgb": image}),
            "torchscript_cpu": [item.numpy() for item in script(x)],
        }
        expected_boxes = decode(slot, original, specs, geometry)
        report = {"sample": stem, "input_sha256": hashlib.sha256(raw).hexdigest(),
                  "reference": expected_boxes, "checks": {}}
        cpu_prefix = None
        if stem in CPU_REFERENCES[slot]:
            cpu_prefix = weights / CPU_REFERENCES[slot][stem]
        elif extra_sdk is not None:
            cpu_prefix = extra_sdk / f"slot{slot}-{stem}"
        if cpu_prefix is not None:
            cpu = [np.fromfile(f"{cpu_prefix}.{spec['index']}.raw", dtype=np.uint8)
                   .reshape(spec["shape"]) for spec in specs]
            report["checks"]["cuda_reference_vs_sdk"] = compare(original, cpu, specs)
        for key, arrays in observed.items():
            rows = compare(arrays, original, specs)
            report["checks"][key] = rows
            detections = decode(slot, arrays, specs, geometry)
            report[f"{key}_boxes_equal"] = detections == expected_boxes
            if key == "onnx_cpu" and stem in CPU_REFERENCES[slot]:
                target = prefix.parent / f"{stem}-onnx.png"
                render_boxes(sample_path, target, detections["boxes"])
                report["onnx_overlay"] = target.name
        print(f"slot {slot} {stem}: {len(expected_boxes['boxes'])} boxes; "
              + ", ".join(f"{key}={sum(row['different_bytes'] for row in rows)} differing bytes"
                          for key, rows in report["checks"].items()), flush=True)
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--slots", nargs="+", choices=SLOTS, default=list(SLOTS))
    parser.add_argument("--extra-sdk", type=Path,
                        help="references for the other three images per model")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    reports = {slot: verify_slot(args.directory, slot, args.extra_sdk) for slot in args.slots}
    result = {"onnxruntime_version": ort.__version__, "pytorch_version": torch.__version__,
              "reference": "original CUDA reconstruction and fixed x86 SDK",
              "slots": reports}
    path = args.directory / "verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if any(row["different_bytes"] or not report.get(f"{backend}_boxes_equal", True)
           for slot in reports.values() for report in slot
           for backend, rows in report["checks"].items() for row in rows):
        raise SystemExit(f"conversion mismatch: {path}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
