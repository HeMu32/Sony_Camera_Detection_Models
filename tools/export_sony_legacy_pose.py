"""Export and verify the shared Sony human graph and the ILCE-7RM5 cat graph."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
import torch

from export_sony_portable import ROOT, write_models
from export_sony_pose_weights import MODELS, digest
from export_sony_slot18 import specs
from inspect_nbg_nn_writes import coordinates, read_writes
from inspect_nbg_tp_trace import read_trace
from prepare_nbg_inputs import prepare
from sony_legacy_pose_backbone import read_instructions
from sony_legacy_pose_portable import GRAPHS, LegacyPose
from verify_legacy_pose_stem import CASES
from verify_sony_portable import STEMS, compare, load_cpu_exports


BASE = ROOT / "runs/portable-release"
CASES_BY_SLOT = {"cat-r5": "12", "human": "01"}
EMPTY = STEMS[-1]


def build(name, evidence, maps_path):
    model_hash, originals, _, _, _ = MODELS[name]
    if any(digest(ROOT / "nab_graphs" / file) != model_hash for file in originals):
        raise ValueError(f"{name}: original NBG changed")
    package_path = evidence / "pose-parameters" / f"sony-{name}-weights.pt"
    package = torch.load(package_path, weights_only=True, map_location="cpu")
    if package["metadata"]["source_nbg_sha256"] != model_hash:
        raise ValueError(f"{name}: wrong coefficient package")
    nn_path = evidence / "pose-evidence" / f"{name}-empty.nnc"
    tp_path = nn_path.with_suffix(".nbt")
    maps = json.loads(maps_path.read_text(encoding="ascii"))[name]
    if maps["source_nbg_sha256"] != model_hash:
        raise ValueError(f"{name}: wrong kernel mapping report")
    maps = {int(i): order for i, order in maps["logical_to_kernel_channel"].items()}
    groups = package["commands"]
    instructions = read_instructions(nn_path, len(groups))
    count, tp = read_trace(tp_path)
    if count != len(groups) or any(call.samples for call in tp):
        raise ValueError(f"{name}: wrong TP schedule")
    source_hashes = {label: digest(file) for label, file in (
        ("weights", package_path), ("nn", nn_path), ("tp", tp_path), ("channel_maps", maps_path),
    )}
    return LegacyPose(name, groups, maps, instructions, tp).eval(), groups, instructions, source_hashes


def output_specs(name, log):
    return specs(log, grid=GRAPHS[name][1], count=len(GRAPHS[name][2]))


def export(name, prefix, model, hashes, evidence):
    slot = CASES_BY_SLOT[name]
    width, height, _, _ = CASES[name]
    source_log = evidence / "pose-evidence" / f"{name}.log"
    outputs = output_specs(name, source_log)
    example = torch.full((3, height, width), 128, dtype=torch.uint8)
    for a, b in zip(model(example), outputs, strict=True):
        if list(a.shape) != b["shape"] or a.dtype != torch.uint8:
            raise ValueError(f"{name}: output contract mismatch")
    metadata = {
        "slot": int(slot), "cameras": ["ILCE-7RM5", "ILCE-1M2"] if name == "human" else ["ILCE-7RM5"],
        "original_nbg_sha256": MODELS[name][0], "source_files_sha256": hashes,
        "interface": {"name": "rgb", "shape": list(example.shape), "dtype": "uint8",
                      "preprocessing": "EXIF transpose, RGB, bilinear letterbox, pad 128, CHW"},
        "outputs": outputs,
        "sha256": write_models(model, prefix, example, outputs),
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"{name}: wrote checked ONNX and TorchScript")


def coeff_key(group):
    h = hashlib.sha256()
    for key in ("weight_kzyx", "bias_channel_lo_hi"):
        h.update(group[key].numpy().tobytes())
    return h.digest()


def verify_trace(name, model, groups, instructions, x, path):
    positive = read_instructions(path.with_suffix(".nnc"), len(groups))
    if not np.array_equal(positive[:, (0, 1, 2, 3, 6, 7, 15, 22)],
                          instructions[:, (0, 1, 2, 3, 6, 7, 15, 22)]):
        raise AssertionError("NN command layout changed across inputs")
    first, _, _ = GRAPHS[name]
    stages = {coeff_key(groups[i]): i for i in model.backbone.commands}
    values = {i: a[0].numpy() for i, a in model.activations(x) if i <= len(groups)}
    writes = read_writes(path.with_suffix(".nnw.gz"))
    if len(writes) != len(groups):
        raise AssertionError("incomplete NN write capture")
    rows_seen = defaultdict(int)
    report = []
    for command, pairs in writes.items():
        source = stages[coeff_key(groups[command])] if command < first else command
        a = values[source]
        c, y, x1, v = coordinates(pairs)
        y0 = rows_seen[source] if command < first else 0
        height = int(y.max()) + 1
        rows_seen[source] += height
        valid = c < a.shape[0]
        if (np.any(~valid) and
                (source != model.seed or set(c[~valid]) != set(range(128, 135)))):
            raise AssertionError(f"unexpected nonlogical channels at NN {command}")
        if (len(np.unique(pairs[:, 0])) != len(pairs) or
                valid.sum() != a.shape[0] * height * a.shape[2] or
                np.any((v[valid] < 0) | (v[valid] > 255))):
            raise AssertionError(f"incomplete NN {command} tensor observations")
        diff = np.abs(a[c[valid], y[valid] + y0, x1[valid]].astype(np.int16) - v[valid])
        result = {"command": command, "logical_stage": source, "start_row": y0,
                  "bytes": len(diff), "different_bytes": int(np.count_nonzero(diff)),
                  "max_abs_code_error": int(diff.max()),
                  "out_of_range_hook_calls": int(np.count_nonzero(~valid))}
        report.append(result)
    if any(rows_seen[i] != a.shape[1] for i, a in values.items()):
        raise AssertionError("legacy NN trace did not cover the full logical tensors")
    return report


def sdk_paths(name, stem, evidence, extra):
    _, _, native, tag, _ = MODELS[name]
    gold = ROOT / "debug/cpu_raw" / native
    if stem == CASES[name][2]:
        return gold / tag, evidence / "pose-evidence" / f"{name}.log"
    if stem == EMPTY:
        return gold / "Empty", gold / "Empty.log"
    prefix = extra / f"{name}-{stem}"
    return prefix, prefix.with_suffix(".log")


def check_boxes(name, stem, prefix, arrays, log, inputs, evidence):
    path = prefix.parent / f"{stem}-onnx"
    for i, array in enumerate(arrays):
        array.tofile(f"{path}.{i}.raw")
    renderer = "render_pose_boxes.py" if name == "human" else "render_animal_pose_boxes.py"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools" / renderer),
                    str(path), str(log), str(inputs / "manifest.json"),
                    str(ROOT / "assets_aux" / f"{stem}.jpg"), str(path.with_suffix(".png"))],
                   check=True, stdout=subprocess.DEVNULL)
    copied = json.loads(path.with_suffix(".json").read_text(encoding="ascii"))
    baseline = name + ("-empty" if stem == EMPTY else "")
    known = json.loads((evidence / "pose-sdk-overlays" / f"{baseline}.json").read_text(encoding="ascii"))
    if copied != known:
        raise AssertionError(f"{name}: portable output changed experiment boxes for {stem}")
    return len(copied["boxes"])


def verify(name, prefix, model, groups, instructions, hashes, args):
    meta, onnx, script = load_cpu_exports(prefix)
    if meta["source_files_sha256"] != hashes:
        raise AssertionError("source package changed after export")
    width, height, observed_stem, _ = CASES[name]
    inputs = ROOT / f"debug/cpu_inputs/{width}x{height}_rgb"
    reports, trace = {}, []
    for stem in STEMS:
        source = ROOT / "assets_aux" / f"{stem}.jpg"
        raw, _ = prepare(source, width, height, "letterbox", 128, "rgb", "chw")
        if (inputs / f"{stem}.raw").read_bytes() != raw:
            raise AssertionError("reference input no longer matches sample preprocessing")
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(3, height, width).copy()
        x = torch.from_numpy(rgb)
        native, log = sdk_paths(name, stem, args.evidence, args.extra)
        info = output_specs(name, log)
        if info != meta["outputs"]:
            raise AssertionError("SDK output metadata mismatch")
        gold = [np.fromfile(f"{native}.{d['index']}.raw", dtype=np.uint8).reshape(d["shape"]) for d in info]
        onnx_outputs = onnx.run(None, {"rgb": rgb})
        checks = {"onnx_cpu": compare(onnx_outputs, gold, info),
                  "torchscript_cpu": compare([a.numpy() for a in script(x)], gold, info)}
        result = {"input_sha256": hashlib.sha256(raw).hexdigest(), "checks": checks}
        if stem in (observed_stem, EMPTY):
            result["experimental_box_count"] = check_boxes(name, stem, prefix, onnx_outputs, log,
                                                            inputs, args.evidence)
        if stem == observed_stem:
            base = args.trace / name
            trace = verify_trace(name, model, groups, instructions, x, base)
            if any(Path(f"{base}.{i}.raw").read_bytes() != a.tobytes() for i, a in enumerate(gold)):
                raise AssertionError("traced simulator final output differs from native")
        reports[stem] = result
        print(f"{name} {stem}: " + ", ".join(
            f"{key} {sum(a['different_bytes'] for a in rows)} bytes differ"
            for key, rows in checks.items()), flush=True)
    out = prefix.parent / "verification.json"
    out.write_text(json.dumps({
        "reference": "original Sony NBG in pinned x86 SDK", "samples": reports,
        "onnxruntime_version": ort.__version__, "pytorch_version": torch.__version__,
        "nn_positive_sample_trace": trace,
    }, indent=2) + "\n", encoding="ascii")
    if any(row["different_bytes"] for r in reports.values() for v in r["checks"].values() for row in v
           ) or any(row["different_bytes"] for row in trace):
        raise SystemExit(f"verification failed: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--models", choices=CASES_BY_SLOT, nargs="+", default=list(CASES_BY_SLOT))
    parser.add_argument("--evidence", type=Path, default=BASE)
    parser.add_argument("--maps", type=Path, default=ROOT / "runs/portable-legacy-stem/stem-verification.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "runs/portable-legacy-fullnn")
    parser.add_argument("--extra", type=Path, default=ROOT / "runs/portable-legacy-fullnn/extra")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with torch.no_grad():
        for name in args.models:
            slot = CASES_BY_SLOT[name]
            prefix = args.directory / f"slot{slot}" / f"sony-slot{slot}"
            model, groups, instructions, hashes = build(name, args.evidence, args.maps)
            if args.action == "export":
                export(name, prefix, model, hashes, args.evidence)
            else:
                verify(name, prefix, model, groups, instructions, hashes, args)


if __name__ == "__main__":
    main()
