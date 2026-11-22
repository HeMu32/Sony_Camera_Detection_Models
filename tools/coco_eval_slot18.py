"""Run the pinned 1M2 slot-18 portable graph under two requantization modes."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from coco_eval_decode_sdk import slot18_boxes
from coco_eval_infer import ANNOTATIONS, PINNED, ROOT, RUN, append_predictions, digest, load_predictions, read_json
from coco_eval_simulator import MODELS


SOURCE_SHA = "66809988312873dddc265522731ea1079460d4e376d77760d18c4c7a81d29576"
WEIGHTS_SHA = "c73a628ef286ad8c72ddd940de6a64769b0e2492c0027542b1fb30058fa6a43b"
WEIGHTS = ROOT / "runs/portable-release/pose-parameters/sony-cat-1m2-weights.pt"
SHAPE = (3, 288, 416)


class SimpleBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, activation):
        layer = self.layer
        x = activation.float() - 128
        if layer.padding:
            x = F.pad(x, (1, 1, 1, 1), value=-128)
        x = F.conv2d(x, layer.weight.float())
        if layer.pool:
            x = F.max_pool2d(x, 2)
        return ((x + layer.bias) * layer.scale).round().add(
            layer.zero_point).clamp(0, 255).byte()


def get_model(run, mode, device):
    source = run / "sony_slot18_candidate.py"
    if digest(source) != SOURCE_SHA or digest(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError("unexpected slot-18 portable source or parameter SHA-256")
    spec = importlib.util.spec_from_file_location("_pinned_sony_slot18", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.INPUT_SHAPE != SHAPE:
        raise ValueError("unexpected slot-18 input shape")
    model = module.Slot18(WEIGHTS)
    if mode == "simple":
        model.layers = nn.ModuleList(SimpleBlock(layer) for layer in model.layers)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return model.to(device).eval()


@torch.inference_mode()
def infer(model, pixels, device):
    inp = np.frombuffer(pixels, dtype=np.uint8).copy().reshape(SHAPE)
    return model(torch.from_numpy(inp).to(device))


def validate(run, device):
    sdk = run / "sdk-pilot"
    job = read_json(sdk / "job.json")
    samples = []
    for name in ("Cat_DSC04051", "Empty_DSC03284"):
        samples.append((
            name, ROOT / f"debug/cpu_inputs/416x288_rgb/{name}.raw",
            ROOT / f"runs/portable-slot18-stage1/{name}",
        ))
    for iid in job["subsets"]["animal"]:
        samples.append((
            str(iid), sdk / f"inputs/416x288-{iid:012d}.raw", sdk / f"out/animal-1m2-{iid:012d}",
        ))
    modes = ["simulator", "simple"]
    models = {mode: get_model(run, mode, device) for mode in modes}
    checks = []
    for sample, image, prefix in samples:
        result = {"input": sample, "sha256": digest(image), "outputs": {}}
        for mode in modes:
            outputs = infer(models[mode], image.read_bytes(), device)
            if len(outputs) != 12:
                raise ValueError("expected all 12 slot-18 outputs")
            stats = []
            for index, tensor in enumerate(outputs):
                actual = tensor.cpu().numpy().ravel()
                ref = np.fromfile(prefix.with_name(f"{prefix.name}.{index}.raw"), np.uint8)
                if actual.shape != ref.shape:
                    raise ValueError(f"tensor length mismatch for {sample} output {index}")
                diff = np.abs(actual.astype(np.int16) - ref)
                stats.append([len(ref), int(np.count_nonzero(diff)), int(diff.max())])
            if mode == "simulator" and any(row[1] for row in stats):
                raise ValueError(f"simulator-equivalent mode disagrees on {sample}: {stats}")
            result["outputs"][mode] = stats
        checks.append(result)
        print(f"{sample}: simulator exact; simple differs at "
              f"{sum(x[1] for x in result['outputs']['simple'])} bytes", flush=True)
    path = run / "slot18-reference-check.json"
    path.write_text(json.dumps({
        "device": device, "source_sha256": SOURCE_SHA, "weights_sha256": WEIGHTS_SHA,
        "torch": torch.__version__, "columns": ["bytes", "different_bytes", "max_abs_code_error"],
        "samples": checks,
    }, indent=2) + "\n", encoding="utf-8")


def run_full(run, mode, device):
    check = read_json(run / "slot18-reference-check.json")
    if (check["device"], check["source_sha256"], check["weights_sha256"], check["torch"]) != (
        device, SOURCE_SHA, WEIGHTS_SHA, torch.__version__,
    ):
        raise ValueError("validate this build and device against SDK before full evaluation")
    sys.path.insert(0, str(run / PINNED / "tools"))
    from prepare_nbg_inputs import prepare_rgb
    from render_pose_boxes import peaks

    ann = run / "data/annotations" / ANNOTATIONS[0]
    images = sorted(read_json(ann)["images"], key=lambda im: im["id"])
    model = get_model(run, mode, device)
    graph_sha = digest(ROOT / "nab_graphs/1m2_slots/slot18.nb")
    if graph_sha != MODELS["animal-1m2"][4]:
        raise ValueError("unexpected original model SHA-256")
    manifest = {
        "backend": "sony-slot18-portable", "task": "detect", "source_graph_sha256": graph_sha,
        "source_sha256": SOURCE_SHA, "weights_sha256": WEIGHTS_SHA, "mode": mode,
        "image_ids": [im["id"] for im in images], "device": device, "torch": torch.__version__,
        "annotations_sha256": digest(ann), "reference_sha256": digest(run / "slot18-reference-check.json"),
    }

    def produce(done):
        for im in images:
            if im["id"] in done:
                continue
            with Image.open(run / "data/val2017" / im["file_name"]) as src:
                pic = ImageOps.exif_transpose(src).convert("RGB")
            if pic.size != (im["width"], im["height"]):
                raise ValueError(f"image size mismatch: {im['file_name']}")
            raw, geometry = prepare_rgb(pic, SHAPE[2], SHAPE[1], "letterbox", 128, "rgb", "chw")
            heat = (infer(model, raw, device)[2].cpu().numpy().astype(np.float32) - 15) * .00450269459
            yield im["id"], slot18_boxes(heat, geometry, peaks)

    append_predictions(run / f"sony-18-{mode}.jsonl", manifest, produce)


def compare(run):
    paths = [run / f"sony-18-{m}.jsonl" for m in ("simulator", "simple")]
    exact, simple = ({q["image_id"]: q["pred"] for q in load_predictions(p)} for p in paths)
    ann = run / "data/annotations" / ANNOTATIONS[0]
    ids = {im["id"] for im in read_json(ann)["images"]}
    if len(ids) != 5000 or exact.keys() != ids or simple.keys() != ids:
        raise ValueError("comparison requires exactly the full COCO val2017 image set")
    pilot = list(load_predictions(run / "sdk-pilot/sony-animal-1m2.jsonl"))
    if not pilot or any(exact[q["image_id"]] != q["pred"] for q in pilot):
        raise ValueError("decoded pilot predictions disagree with SDK")
    check = read_json(run / "slot18-reference-check.json")
    stats = [o for sample in check["samples"] for o in sample["outputs"]["simple"]]
    diff = {
        "images": len(ids), "pilot_predictions_exact": len(pilot),
        "prediction_digests": {p.name: digest(p) for p in paths},
        "prediction_records_changed": sum(exact[i] != simple[i] for i in ids),
        "box_count_changed": sum(len(exact[i]) != len(simple[i]) for i in ids),
        "box_coordinates_or_presence_changed": sum(
            [p["bbox"] for p in exact[i]] != [p["bbox"] for p in simple[i]] for i in ids
        ),
        "reference_tensor_bytes": sum(o[0] for o in stats),
        "simple_different_reference_bytes": sum(o[1] for o in stats),
        "simple_max_reference_code_error": max(o[2] for o in stats),
    }
    report = run / "slot18-ab-check.json"
    report.write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
    print(report, json.dumps(diff), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, default=RUN)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("action", choices=("validate", "simulator", "simple", "compare"))
    args = cli.parse_args()
    if args.action == "validate":
        validate(args.run, args.device)
    elif args.action == "compare":
        compare(args.run)
    else:
        run_full(args.run, args.action, args.device)


if __name__ == "__main__":
    main()
