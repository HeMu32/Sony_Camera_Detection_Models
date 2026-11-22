"""Validate the independently reconstructed R5 cat and shared human pose stem."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_sony_pose_weights import MODELS, ROOT, digest, encode, read_operations
from inspect_nbg_nn_writes import coordinates, read_writes
from prepare_nbg_inputs import prepare
from sony_legacy_pose_backbone import LegacyBackbone, read_instructions
from sony_slot18_portable import stem_raster


CASES = {
    "cat-r5": (416, 288, "Cat_DSC04051", 0xfc700),
    "human": (512, 336, "Human_DSC03106", 0x92500),
}


def check_maps(path, commands):
    maps = {}
    for command, block in read_operations(path, extra_kinds=(10,)):
        observed, info = encode(block)
        expected = commands[command]
        if info["trace_hit_capture_cap"] or observed.keys() != expected.keys() or any(
            not torch.equal(observed[key], expected[key])
            for key in ("weight_kzyx", "bias_channel_lo_hi")
        ):
            raise AssertionError(f"weight or bias changed in mapping capture at NN {command}")
        rows = block[block[:, 0] == 10, 2:4]
        if len(rows):
            count = expected["weight_kzyx"].shape[0]
            if (sorted(rows[:, 0]) != list(range(count)) or
                    sorted(rows[:, 1]) != list(range(count))):
                raise AssertionError(f"expected a bijective channel map at NN {command}")
            maps[command] = rows[rows[:, 0].argsort(), 1].tolist()
    if command != len(commands) or not maps:
        raise AssertionError("incomplete depthwise mapping capture")
    return maps


def verify(name, evidence, packages, command_evidence):
    width, height, stem, sram_end = CASES[name]
    expected_hash, models, folder, sample, outputs = MODELS[name]
    if any(digest(ROOT / "nab_graphs" / path) != expected_hash for path in models):
        raise AssertionError(f"{name}: unexpected NBG source")
    for i in range(outputs):
        reference = ROOT / "debug/cpu_raw" / folder / f"{sample}.{i}.raw"
        if any(digest(reference) != digest(evidence / f"{prefix}.{i}.raw")
               for prefix in (name, f"{name}-map", f"{name}-dw")):
            raise AssertionError(f"{name}: instrumentation changed output {i}")
    raw, _ = prepare(ROOT / "assets_aux" / f"{stem}.jpg", width, height,
                     "letterbox", 128, "rgb", "chw")
    reference_input = ROOT / f"debug/cpu_inputs/{width}x{height}_rgb/{stem}.raw"
    if raw != reference_input.read_bytes():
        raise AssertionError(f"{name}: sample preprocessing changed")
    rgb = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy().reshape(3, height, width))
    staged = stem_raster(rgb, pad=0)
    recorded = np.fromfile(evidence / f"{name}.sram", dtype=np.uint8)
    if (len(recorded) != sram_end - 0x800 or
            not np.array_equal(staged.numpy().ravel(), recorded[:staged.numel()])):
        raise AssertionError(f"{name}: TP input staging differs from SRAM trace")
    package = torch.load(packages / f"sony-{name}-weights.pt", weights_only=True, map_location="cpu")
    if package["metadata"]["source_nbg_sha256"] != expected_hash:
        raise AssertionError(f"{name}: unexpected coefficient source")
    commands = package["commands"]
    writes = read_writes(evidence / f"{name}.nnw.gz")
    map_path = evidence / f"{name}-map.nbc.gz"
    maps = check_maps(map_path, commands)
    table = read_instructions(command_evidence / f"{name}-empty.nnc", len(commands))
    model = LegacyBackbone(name, commands, maps, table).eval()
    actual = {i: x[0].numpy() for i, x in model.activations(rgb)}
    checked = {}
    for command, rows in writes.items():
        block, = (i for i in actual if all(torch.equal(commands[i][key], commands[command][key])
                                           for key in ("weight_kzyx", "bias_channel_lo_hi")))
        if maps.get(block) != maps.get(command):
            raise AssertionError(f"{name}: NN {command} changed channel order across stripes")
        c, y, x, expected = coordinates(rows)
        offset = int(table[command - 1, 7]) - int(table[block - 1, 7])
        row, remainder = divmod(offset, actual[block].shape[-1])
        if (remainder or row < 0 or len(np.unique(rows[:, 0])) != len(rows) or
                not np.array_equal(actual[block][c, y + row, x], expected)):
            raise AssertionError(f"{name}: NN {command} stripe differs from SDK")
        checked[str(command)] = len(rows)
    print(f"{name}: TP {staged.numel():,} bytes and {len(checked)} NN commands "
          f"({sum(checked.values()):,} bytes) exact")
    return {
        "source_nbg_sha256": expected_hash, "input_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "TP preprocessing and first 25 NN commands, not an end-to-end model",
        "map_trace_sha256": digest(map_path), "logical_to_kernel_channel": maps,
        "tp_stem_exact_bytes": staged.numel(), "nn_exact_bytes_by_command": checked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("packages", type=Path)
    parser.add_argument("--command-evidence", type=Path, default=ROOT / "runs/portable-release/pose-evidence")
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    with torch.inference_mode():
        report = {name: verify(name, args.evidence, args.packages, args.command_evidence)
                  for name in CASES}
    (args.evidence / "stem-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
