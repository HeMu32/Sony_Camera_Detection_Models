"""Export and independently verify the ILCE-1M2 slot-18 cat model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from export_sony_portable import ROOT, sha256, write_models
from inspect_nbg_nn_writes import coordinates, read_writes
from prepare_nbg_inputs import prepare
from sony_slot18_portable import INPUT_SHAPE, MODEL_HASH, OUTPUT_COMMANDS, Slot18, stem_raster
from verify_sony_portable import STEMS, compare, load_cpu_exports


INPUTS = ROOT / "debug/cpu_inputs/416x288_rgb"
NATIVE = ROOT / "debug/cpu_raw/m2_s18"
BASE = ROOT / "runs/portable-release"
SAMPLE_STEM = (STEMS[3], STEMS[4])


def specs(log, *, grid=(36, 52), count=len(OUTPUT_COMMANDS)):
    outputs = []
    for line in log.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields[:1] != ["param"] or fields[2:3] != ["output"]:
            continue
        i = int(fields[1]) - 1
        data = dict(field.split("=", 1) for field in fields[3:])
        width, height, channels, batch = map(int, data["shape"].split("x"))
        if (i != len(outputs) or (height, width) != grid or batch != 1 or
                data["dtype"] != "0x3" or data["quant"] != "0x2"):
            raise ValueError(f"unexpected slot-18 output in {log}: {i}")
        outputs.append({"index": i, "name": f"output_{i:02}", "layout": "CHW",
                        "dtype": "uint8", "shape": [channels, height, width],
                        "openvx_shape": [width, height, channels, batch],
                        "scale": float(data["scale"]), "zero_point": int(data["zp"])})
    if len(outputs) != count:
        raise ValueError(f"missing output specifications in {log}")
    return outputs


def export(prefix, package):
    if sha256(ROOT / "nab_graphs/1m2_slots/slot18.nb") != MODEL_HASH:
        raise ValueError("unexpected slot-18 NBG SHA-256")
    model = Slot18(package).eval()
    example = torch.full(INPUT_SHAPE, 128, dtype=torch.uint8)
    outputs = specs(NATIVE / "reloc_run.log")
    for tensor, spec, command in zip(model(example), outputs, OUTPUT_COMMANDS, strict=True):
        if (list(tensor.shape) != spec["shape"] or tensor.dtype != torch.uint8 or
                model.layers[command - 1].zero_point != spec["zero_point"]):
            raise ValueError(f"reconstructed output does not match SDK interface: {spec['name']}")
    hashes = write_models(model, prefix, example, outputs)
    metadata = {
        "slot": 18, "cameras": ["ILCE-1M2"], "original_nbg_sha256": MODEL_HASH,
        "weight_package_sha256": sha256(package), "output_nn_commands": OUTPUT_COMMANDS,
        "interface": {"name": "rgb", "shape": INPUT_SHAPE, "dtype": "uint8",
                      "preprocessing": "EXIF transpose, RGB, bilinear letterbox, pad 128, CHW"},
        "outputs": outputs, "sha256": hashes,
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print("slot 18: wrote checked ONNX and TorchScript (12 outputs)")


def verify_trace(model, x, base):
    sram = np.fromfile(base.with_suffix(".sram"), dtype=np.uint8)
    count = 12 * 145 * 209
    if sram.size != 0x59300 or not np.array_equal(stem_raster(x).numpy().ravel(), sram[:count]):
        raise AssertionError("source raster does not reproduce captured TP output")
    writes = read_writes(base.with_suffix(".nnw.gz"))
    if len(writes) != 69:
        raise AssertionError("expected all 69 NN commands")
    checks = []
    for i, actual in enumerate(model.activations(x), 1):
        c, y, x1, values = coordinates(writes[i])
        actual = actual[0].numpy()
        if (actual.size != len(values) or len(np.unique(writes[i][:, 0])) != len(values) or
                np.any((values < 0) | (values > 255))):
            raise AssertionError(f"incomplete or invalid observations for NN {i}")
        diff = np.abs(actual[c, y, x1].astype(np.int16) - values)
        checks.append({"command": i, "bytes": len(values),
                       "different_bytes": int(np.count_nonzero(diff)),
                       "max_abs_code_error": int(diff.max())})
    return {"tp_bytes": count, "nn": checks}


def overlay(prefix, stem, arrays, log):
    out = prefix.parent / f"{stem}-onnx"
    for i, a in enumerate(arrays):
        a.tofile(f"{out}.{i}.raw")
    subprocess.run([
        sys.executable, "-B", str(ROOT / "debug/deepseek_pose_dispatch/tools/a1ii_animal_box.py"),
        str(out), str(log), str(INPUTS / "manifest.json"), str(ROOT / "assets_aux" / f"{stem}.jpg"),
        str(out.with_suffix(".png")), "--thresholds", "0.02",
    ], check=True, stdout=subprocess.DEVNULL)
    result = json.loads(out.with_suffix(".json").read_text(encoding="ascii"))
    name = "cat-1m2" if stem == SAMPLE_STEM[0] else "cat-1m2-empty"
    expected = json.loads((BASE / "pose-sdk-overlays" / f"{name}.json").read_text(encoding="ascii"))
    if result != expected:
        raise AssertionError(f"portable box differs from saved SDK baseline: {stem}")
    return result["boxes"][0]["xyxy_source"]


def verify(prefix, package, sdk, trace):
    metadata, onnx, script = load_cpu_exports(prefix)
    model = Slot18(package).eval()
    if sha256(package) != metadata["weight_package_sha256"]:
        raise AssertionError("package changed since export")
    samples = {}
    for stem in STEMS:
        raw, _ = prepare(ROOT / "assets_aux" / f"{stem}.jpg",
                         416, 288, "letterbox", 128, "rgb", "chw")
        if raw != (INPUTS / f"{stem}.raw").read_bytes():
            raise AssertionError("sample preprocessing changed")
        if stem in SAMPLE_STEM:
            native_prefix = NATIVE / ("Cat" if stem == SAMPLE_STEM[0] else "Empty")
            log = NATIVE / ("reloc_run.log" if stem == SAMPLE_STEM[0] else "Empty.log")
        else:
            native_prefix = sdk / stem
            log = native_prefix.with_suffix(".log")
        outputs = metadata["outputs"]
        if specs(log) != outputs:
            raise AssertionError("original SDK output specifications changed")
        reference = [np.fromfile(f"{native_prefix}.{i}.raw", dtype=np.uint8).reshape(d["shape"])
                     for i, d in enumerate(outputs)]
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(INPUT_SHAPE).copy()
        x = torch.from_numpy(rgb)
        observed = {"onnx_cpu": onnx.run(None, {"rgb": rgb}),
                    "torchscript_cpu": [t.numpy() for t in script(x)]}
        checks = {name: compare(tensors, reference, outputs) for name, tensors in observed.items()}
        sample = {"input_sha256": hashlib.sha256(raw).hexdigest(), "checks": checks}
        if stem in SAMPLE_STEM:
            trace_base = trace / stem
            sample["trace"] = verify_trace(model, x, trace_base)
            for i, gold in enumerate(reference):
                if Path(f"{trace_base}.{i}.raw").read_bytes() != gold.tobytes():
                    raise AssertionError("NN tracing changed SDK output")
            sample["onnx_experimental_box_xyxy"] = overlay(prefix, stem, observed["onnx_cpu"], log)
        samples[stem] = sample
        print(f"{stem}: " + ", ".join(f"{name}: {sum(x['different_bytes'] for x in rows)} bytes differ"
                                     for name, rows in checks.items()), flush=True)
    result = {"onnxruntime_version": ort.__version__, "pytorch_version": torch.__version__,
              "reference": "original slot-18 NBG in pinned x86 SDK", "samples": samples}
    report = prefix.parent / "slot18-verification.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    if any(item["different_bytes"] for sample in samples.values()
           for rows in (*sample["checks"].values(), sample.get("trace", {}).get("nn", ()))
           for item in rows):
        raise SystemExit(f"conversion mismatch: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--package", type=Path, default=BASE / "pose-parameters/sony-cat-1m2-weights.pt")
    parser.add_argument("--sdk", type=Path, default=ROOT / "runs/portable-slot18-stage1/sdk")
    parser.add_argument("--trace", type=Path, default=ROOT / "runs/portable-slot18-stage1/full")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    prefix = args.output_dir / "slot18" / "sony-slot18"
    with torch.no_grad():
        if args.action == "export":
            export(prefix, args.package)
        else:
            verify(prefix, args.package, args.sdk, args.trace)


if __name__ == "__main__":
    main()
