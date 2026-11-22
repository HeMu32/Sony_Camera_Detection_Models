"""Run Sony pose graphs on CUDA without FP64 and compare against saved SDK outputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from export_sony_legacy_pose import BASE, CASES, build, output_specs, sdk_paths
from export_sony_slot18 import INPUTS, NATIVE, specs
from export_sony_portable import ROOT, sha256
from nbg_cuda_quantization import ROUNDING_MODES
from prepare_nbg_inputs import prepare
from sony_legacy_pose_backbone import Depthwise
from sony_slot18_portable import MODEL_HASH, Slot18, Slot18Block
from verify_sony_portable import STEMS, compare


def setup(name, evidence):
    if name == "cat-1m2":
        if sha256(ROOT / "nab_graphs/1m2_slots/slot18.nb") != MODEL_HASH:
            raise ValueError("unexpected source NBG")
        model = Slot18(evidence / "pose-parameters/sony-cat-1m2-weights.pt").eval()
        return model, 416, 288, INPUTS, STEMS[3]
    width, height, positive, _ = CASES[name]
    model, *_ = build(name, evidence, ROOT / "runs/portable-legacy-stem/stem-verification.json")
    return model, width, height, ROOT / f"debug/cpu_inputs/{width}x{height}_rgb", positive


def sdk_case(name, stem, evidence):
    if name != "cat-1m2":
        prefix, log = sdk_paths(name, stem, evidence, ROOT / "runs/portable-legacy-fullnn/extra")
        return prefix, log, output_specs(name, log)
    if stem == STEMS[3]:
        prefix, log = NATIVE / "Cat", NATIVE / "reloc_run.log"
    elif stem == STEMS[-1]:
        prefix, log = NATIVE / "Empty", NATIVE / "Empty.log"
    else:
        prefix = ROOT / "runs/portable-slot18-stage1/sdk" / stem
        log = prefix.with_suffix(".log")
    return prefix, log, specs(log)


def render(name, prefix, arrays, log, inputs, photo):
    for i, array in enumerate(arrays):
        array.tofile(f"{prefix}.{i}.raw")
    tool, extra = {
        "human": ("tools/render_pose_boxes.py", ()),
        "cat-r5": ("tools/render_animal_pose_boxes.py", ()),
        "cat-1m2": ("debug/deepseek_pose_dispatch/tools/a1ii_animal_box.py", ("--thresholds", "0.02")),
    }[name]
    subprocess.run([sys.executable, "-B", str(ROOT / tool), str(prefix), str(log),
                    str(inputs / "manifest.json"), str(photo), str(prefix.with_suffix(".png")), *extra],
                   check=True, stdout=subprocess.DEVNULL)
    result = json.loads(prefix.with_suffix(".json").read_text(encoding="ascii"))
    return [box["xyxy_source"] for box in result["boxes"] if box["xyxy_source"] is not None]


@torch.inference_mode()
def verify(name, output, evidence, repeats):
    model, width, height, inputs, positive = setup(name, evidence)
    model.cuda()
    layers = [layer for layer in model.modules() if isinstance(layer, (Slot18Block, Depthwise))]
    if not layers:
        raise AssertionError("no pose quantization layers")
    output.mkdir()
    samples = {}
    for stem in STEMS:
        photo = ROOT / "assets_aux" / f"{stem}.jpg"
        raw, _ = prepare(photo, width, height, "letterbox", 128, "rgb", "chw")
        if raw != (inputs / f"{stem}.raw").read_bytes():
            raise ValueError(f"preprocessing differs from the SDK reference for {stem}")
        rgb = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).reshape(3, height, width).copy()).cuda()
        ref, log, outputs = sdk_case(name, stem, evidence)
        gold = [np.fromfile(f"{ref}.{i}.raw", dtype=np.uint8).reshape(d["shape"])
                for i, d in enumerate(outputs)]
        samples[stem] = {}
        for rounding in ROUNDING_MODES:
            for layer in layers:
                layer.cuda_rounding = rounding
            observed = model(rgb)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(rgb)
            end.record()
            end.synchronize()
            tensors = [a.cpu().numpy() for a in observed]
            rows = compare(tensors, gold, outputs)
            stats = {"outputs": rows, "gpu_ms": start.elapsed_time(end) / repeats}
            if stem in (positive, STEMS[-1]):
                stats["experimental_boxes"] = render(name, output / f"{stem}-{rounding}",
                                                     tensors, log, inputs, photo)
            samples[stem][rounding] = stats
            print(f"{name} {stem} {rounding}: "
                  f"{sum(r['different_bytes'] for r in rows)} differing bytes, {stats['gpu_ms']:.2f} ms",
                  flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence", type=Path, default=BASE)
    parser.add_argument("--models", nargs="+", choices=("human", "cat-r5", "cat-1m2"),
                        default=("human", "cat-r5", "cat-1m2"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    args.output.mkdir(parents=True)
    report = {"reference": "original Sony NBG in pinned x86 SDK", "gpu": torch.cuda.get_device_name(),
              "pytorch": torch.__version__, "models": {}}
    for name in args.models:
        report["models"][name] = verify(name, args.output / name, args.evidence, args.repeats)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    if any(r["different_bytes"] for model in report["models"].values() for sample in model.values()
           for r in sample["simulator"]["outputs"]):
        raise SystemExit("simulator rounding differs from reference; see report.json")


if __name__ == "__main__":
    main()
