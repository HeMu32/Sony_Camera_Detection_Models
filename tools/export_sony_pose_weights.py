"""Archive verified Sony pose NN coefficients as PyTorch tensors, not an executable graph."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from inspect_nbg_tp_trace import read_trace


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "human": ("a5d5ead9478e3b069598483bda95702283d7b0c3d825de02c67250eeea4394b6",
              ("r5_slots/slot01.nb", "1m2_slots/slot01.nb"), "r5_s01", "Human", 2),
    "cat-r5": ("6edaf8ed0e182ede460276e87f9841f6f29329d685f6237279b14dd3efc4e7bc",
               ("r5_slots/slot12.nb",), "r5_s12", "Cat", 2),
    "cat-1m2": ("84a35252118fa92ed7a5e36e7f16d5f8b00f0ab6b901d5ba3d83cbc5a4a10e75",
                ("1m2_slots/slot18.nb",), "m2_s18", "Cat", 12),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_operations(path, *, extra_kinds=()):
    """Stream the trace so peak memory is bounded by one command."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        if stream.read(8) != b"NBCOEF1\0":
            raise ValueError("invalid trace header")
        pending = []
        operation = 0
        while chunk := stream.read(36 * 131072):
            rows = np.frombuffer(chunk, dtype="<i4").reshape(-1, 9)
            start = 0
            for stop in np.flatnonzero(rows[:, 0] == 3):
                pending.append(rows[start:stop + 1])
                block = np.concatenate(pending)
                operation += 1
                if (block[0, 0] != 0 or not np.all(block[:, 1] == operation) or
                        not np.isin(block[1:-1, 0], (1, 2, 6, 7, *extra_kinds)).all()):
                    raise ValueError(f"unexpected trace framing at command {operation}")
                yield operation, block
                pending = []
                start = stop + 1
            if start < len(rows):
                pending.append(rows[start:])
        if pending or not operation:
            raise ValueError("incomplete trace")


def encode(block):
    weight = block[block[:, 0] == 1, 2:7]
    special = block[block[:, 0] == 2, 2:]
    bias = block[block[:, 0] == 6, 2:5]
    params = block[block[:, 0] == 7, 2:]
    tensors = {
        "bias_channel_lo_hi": torch.from_numpy(bias),
        "postprocess_records": torch.from_numpy(params),
    }
    if len(special):
        tensors["lane_coeff_x_y_z_k_value_flag_lane"] = torch.from_numpy(special)
    layout = "no_coefficients_observed"
    if len(weight):
        shape = tuple((weight[:, :4].max(0)[::-1] + 1).tolist())
        if not len(special) and np.prod(shape) == len(weight) and np.all(weight[:, :4] >= 0):
            indices = np.ravel_multi_index(tuple(weight[:, :4].T[::-1]), shape)
            if len(np.unique(indices)) == len(weight):
                values = weight[:, 4]
                dtype = np.uint8 if np.all((0 <= values) & (values <= 255)) else np.int16
                if np.any((values < np.iinfo(dtype).min) | (values > np.iinfo(dtype).max)):
                    raise ValueError("SetCoefData value exceeds the pinned ABI")
                dense = np.empty(shape, dtype=dtype)
                dense.flat[indices] = values
                tensors["weight_kzyx"] = torch.from_numpy(dense)
                layout = f"dense_{dense.dtype}_coefficient_grid"
        if "weight_kzyx" not in tensors:
            tensors["coeff_x_y_z_k_value"] = torch.from_numpy(weight)
            layout = "coordinate_records"
    elif len(special):
        layout = "lane_records"
    return tensors, {
        "layout": layout, "simulator_argument": int(block[0, 2]),
        "ordinary_coefficients": len(weight), "lane_coefficients": len(special),
        "bias_records": len(bias), "postprocess_records": len(params),
        "trace_hit_capture_cap": not bool(block[-1, 2]),
    }


def export(name, evidence, out):
    expected_hash, paths, directory, stem, outputs = MODELS[name]
    if any(digest(ROOT / "nab_graphs" / path) != expected_hash for path in paths):
        raise ValueError(f"{name}: source model SHA-256 mismatch")
    trace = evidence / f"{name}.nbc.gz"
    if not trace.exists():
        trace = evidence / f"{name}.nbc"
    hashes = {}
    for i in range(outputs):
        original = ROOT / "debug/cpu_raw" / directory / f"{stem}.{i}.raw"
        actual = evidence / f"{name}.{i}.raw"
        hashes[str(i)] = digest(actual)
        if hashes[str(i)] != digest(original):
            raise ValueError(f"{name}: traced simulator run changed raw output {i}")

    commands, description = {}, {}
    for i, block in read_operations(trace):
        commands[i], description[i] = encode(block)
    metadata = {
        "source_nbg_sha256": expected_hash, "source_graphs": list(paths),
        "trace_sha256": digest(trace), "reference_output_sha256": hashes,
        "contains_executable_graph": False, "commands": description,
        "scope": "Coefficient and postprocess observations, not recovered connectivity or full parameters",
    }
    package = out / f"sony-{name}-weights.pt"
    with package.open("xb") as stream:
        torch.save({"metadata": metadata, "commands": commands}, stream)
    saved = torch.load(package, weights_only=True, map_location="cpu")
    if saved["metadata"] != metadata or saved["commands"].keys() != commands.keys():
        raise AssertionError("checkpoint metadata failed round trip")
    for i, group in commands.items():
        loaded = saved["commands"][i]
        if group.keys() != loaded.keys() or any(not torch.equal(t, loaded[k]) for k, t in group.items()):
            raise AssertionError(f"checkpoint changed command {i}")
    for i, block in read_operations(trace):
        group = saved["commands"][i]
        if "weight_kzyx" in group:
            x, y, z, k, value = block[block[:, 0] == 1, 2:7].T
            if not np.array_equal(group["weight_kzyx"].numpy()[k, z, y, x], value):
                raise AssertionError(f"coefficient indexing changed command {i}")
    metadata["checkpoint_sha256"] = digest(package)
    package.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"{name}: {len(commands)} commands, "
          f"{sum('weight_kzyx' in c for c in commands.values())} dense grids, "
          f"{sum(c['trace_hit_capture_cap'] for c in description.values())} hit trace cap, "
          f"{outputs} SDK output hashes match")


def verify_empty(name, evidence, out):
    _, _, directory, _, outputs = MODELS[name]
    prefix = evidence / f"{name}-empty"
    for i in range(outputs):
        expected = ROOT / "debug/cpu_raw" / directory / f"Empty.{i}.raw"
        if digest(Path(f"{prefix}.{i}.raw")) != digest(expected):
            raise ValueError(f"{name}: traced empty-scene SDK output {i} changed")
    package = torch.load(out / f"sony-{name}-weights.pt", weights_only=True, map_location="cpu")
    commands = package["commands"]
    for i, block in read_operations(Path(f"{prefix}.nbc.gz")):
        tensors, meta = encode(block)
        expected = commands[i]
        if meta["trace_hit_capture_cap"] or tensors.keys() != expected.keys():
            raise ValueError(f"{name}: incomplete second capture at command {i}")
        for key, value in tensors.items():
            # Some per-command scratch words are not coefficients and may vary.
            if key != "postprocess_records" and not torch.equal(value, expected[key]):
                raise ValueError(f"{name}: coefficient or bias mismatch at command {i}: {key}")
    if i != len(commands):
        raise ValueError(f"{name}: NN schedule length changed between images")
    with Path(f"{prefix}.nnc").open("rb") as file:
        if file.read(8) != b"NBNNC01\0":
            raise ValueError("invalid NN command trace")
        nn = np.fromfile(file, dtype="<u4").reshape(-1, 34)
    nn_n, tp = read_trace(Path(f"{prefix}.nbt"))
    if nn_n != len(commands) or not np.array_equal(nn[:, 0], np.arange(1, nn_n + 1)):
        raise ValueError(f"{name}: NN and TP traces disagree with coefficient schedule")
    if any(call.samples for call in tp):
        raise ValueError(f"{name}: expected command-only TP trace")
    print(f"{name}: coefficients and biases match on skyline; {outputs} SDK outputs exact; "
          f"{nn_n} NN and {len(tp)} TP commands")
    return {
        "source_nbg_sha256": package["metadata"]["source_nbg_sha256"],
        "coefficient_and_bias_mismatch_count": 0,
        "sdk_empty_exact_output_count": outputs, "nn_commands": nn_n, "tp_commands": len(tp),
        "command_trace_sha256": {ext: digest(Path(f"{prefix}.{ext}")) for ext in ("nnc", "nbt")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--verify-empty", action="store_true",
                        help="verify existing checkpoints against independent skyline captures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_empty:
        result = {name: verify_empty(name, args.evidence, args.output) for name in args.models}
        (args.output / "pose-verification.json").write_text(json.dumps(result, indent=2) + "\n",
                                                            encoding="ascii")
        return
    for name in args.models:
        export(name, args.evidence, args.output)


if __name__ == "__main__":
    main()
