"""Export recovered Sony NBG detection graphs to standalone ONNX and TorchScript."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nbg_integer_quantization import integer_formula
from slot08_cuda import Slot08, PLACEHOLDERS, output_specs as specs08
from slot10_cuda import Slot10, output_specs as specs10, run_heads
from verify_nbg_cuda_backbone import BLOCKS
from verify_nbg_cuda_multiscale import run_scales


ROOT = Path(__file__).resolve().parents[1]
SLOTS = {
    "08": (Slot08, specs08, PLACEHOLDERS, "coeff_probe_01"),
    "10": (Slot10, specs10, (), "r5_s10"),
}
INPUT_SHAPE = (3, 320, 320)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ConvBlock(torch.nn.Module):
    def __init__(self, data, cmd):
        super().__init__()
        w, b, s = (data[f"{key}{cmd}"] for key in ("w", "b", "s"))
        bound = 128 * np.abs(w.astype(np.int64)).sum((1, 2, 3)) + np.abs(b.astype(np.int64))
        if (w.dtype != np.int8 or not np.issubdtype(b.dtype, np.signedinteger) or
                s.dtype != np.float32 or not np.all(np.isfinite(s) & (s > 0)) or
                b.shape != (w.shape[0],) or s.shape != b.shape or np.any(bound >= 2**24)):
            raise ValueError(f"unsupported coefficients for command {cmd}")
        self.register_buffer("weight", torch.from_numpy(w))
        self.register_buffer("bias", torch.from_numpy(b.astype(np.int32))[:, None, None])
        self.register_buffer("scale", torch.from_numpy(s)[:, None, None])
        self.zero_point = int(data[f"zp{cmd}"])
        self.pad_code = 128 if cmd == 3 else 0

    def forward(self, x, padding, pooled, relu):
        centered = F.pad(x[None].float() - 128, padding, value=self.pad_code - 128)
        acc = F.conv2d(centered, self.weight.float())
        if pooled:
            acc = F.max_pool2d(acc, 2)
        acc = acc[0] + self.bias
        if relu:
            acc = acc.clamp_min(0)
        # Exact product of the integer accumulator and binary32 scale, including
        # cases in which binary32 alone would round onto the wrong half-integer.
        return (acc.double() * self.scale.double()).round().add(self.zero_point).clamp(0, 255).byte()


class PortableDetector(torch.nn.Module):
    def __init__(self, slot, package):
        super().__init__()
        model_type, get_specs, self.placeholders, _ = SLOTS[slot]
        self.slot = slot
        self.layers = torch.nn.ModuleDict()
        with np.load(package, allow_pickle=False) as data:
            if data["model_sha256"].item() != model_type.model_hash:
                raise ValueError("weight package does not match the supported original model")
            self.tp = {i: tuple(map(int, data[f"tp{i}"])) for i in model_type.tp_outputs}
            for cmd in model_type.commands:
                self.layers[str(cmd)] = ConvBlock(data, cmd)
        for i, (iz, oz, multiplier, shift) in self.tp.items():
            self.register_buffer(f"tp{i}", torch.tensor([
                integer_formula(v - iz, oz, multiplier, shift) for v in range(256)
            ], dtype=torch.uint8))
        # For uint8 inputs, sigmoid rounds and saturates to 255 from code 6.
        self.register_buffer("sigmoid", torch.tensor(
            [128, 187, 225, 244, 251, 254] + [255] * 250, dtype=torch.uint8))
        self.specs = get_specs(self)
        self.indices = tuple(sorted(self.specs))

    def block(self, cmd, x, channels, kernel, *, relu=True, zp=0, padding=None, pool=False):
        layer = self.layers[str(cmd)]
        if not torch.jit.is_tracing() and (
            layer.weight.shape != (channels, x.shape[0], kernel, kernel) or layer.zero_point != zp
        ):
            raise ValueError(f"unexpected operands at command {cmd}")
        if padding is None:
            padding = (kernel // 2,) * 4
        return layer(x, padding, pool, relu)

    def forward(self, rgb):
        if self.slot == "08":
            shared = self.block(3, rgb, 32, 3)
            for shape, pool, sites in BLOCKS[:-1]:
                shared = self.block(sites[0][0], shared, shape[0], shape[-1], pool=pool)
            heads = run_scales(self.block, shared)
        else:
            heads = run_heads(self.block, rgb)
        output = {i: torch.zeros((6,), dtype=torch.uint8, device=rgb.device) for i in self.placeholders}
        for scale, cls, reg, aux in heads:
            for i, tensor in zip(scale.output[:2], (cls, reg)):
                output[i] = getattr(self, f"tp{i}")[tensor.long()]
            output[scale.output[2]] = self.sigmoid[aux.long()].permute(1, 2, 0).contiguous()
        return tuple(output[i] for i in self.indices)


def output_metadata(model, outputs):
    return [
        {"index": i, "name": f"output_{i:02}", "shape": list(tensor.shape),
         "dtype": "uint8", "openvx_shape": list(model.specs[i][0]),
         "scale": float(model.specs[i][1]), "zero_point": model.specs[i][2],
         "layout": ("placeholder" if i in model.placeholders else "HWC" if i >= 14 else "CHW")}
        for i, tensor in zip(model.indices, outputs, strict=True)
    ]


def write_models(model, prefix, example, outputs):
    import onnx

    prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(prefix.with_suffix(ext).exists() for ext in (".pt", ".onnx", ".json")):
        raise FileExistsError(prefix)
    with torch.no_grad():
        script = torch.jit.trace(model, example, check_trace=False)
        script.save(str(prefix.with_suffix(".pt")))
        torch.onnx.export(
            model, (example,), str(prefix.with_suffix(".onnx")), dynamo=False,
            opset_version=17, do_constant_folding=False, input_names=["rgb"],
            output_names=[item["name"] for item in outputs],
        )
    graph_path = prefix.with_suffix(".onnx")
    graph = onnx.load(graph_path, load_external_data=False)
    if ({node.domain for node in graph.graph.node} != {""} or graph.functions or
            any(onnx.external_data_helper.uses_external_data(t) for t in graph.graph.initializer)):
        raise ValueError("ONNX requires custom operators or external data")
    onnx.checker.check_model(graph, full_check=True)
    return {ext: sha256(prefix.with_suffix(f".{ext}")) for ext in ("pt", "onnx")}


def export(slot, path):
    model_type, _, _, directory = SLOTS[slot]
    orig = {camera: ROOT / "nab_graphs" / folder / f"slot{slot}.nb"
            for camera, folder in (("ILCE-7RM5", "r5_slots"), ("ILCE-1M2", "1m2_slots"))}
    if any(sha256(p) != model_type.model_hash for p in orig.values()):
        raise ValueError("one or more camera NBGs do not match the recovered model")
    package = ROOT / "debug/cpu_raw" / directory / f"sony-slot{slot}-cuda.npz"
    model = PortableDetector(slot, package).eval()
    example = torch.full(INPUT_SHAPE, 128, dtype=torch.uint8)
    prefix = path / f"sony-slot{slot}"
    with torch.no_grad():
        outputs = output_metadata(model, model(example))
    hashes = write_models(model, prefix, example, outputs)
    meta = {
        "slot": int(slot), "cameras": list(orig),
        "original_nbg_sha256": model_type.model_hash, "weight_package_sha256": sha256(package),
        "interface": {"name": "rgb", "shape": list(INPUT_SHAPE), "dtype": "uint8",
                      "preprocessing": "EXIF transpose, RGB, bilinear letterbox, pad 128, CHW"},
        "outputs": outputs,
        "sha256": hashes,
    }
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
    print(f"slot {slot}: wrote checked ONNX and TorchScript ({len(outputs)} interface outputs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--slots", nargs="+", choices=SLOTS, default=list(SLOTS))
    args = parser.parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    for slot in args.slots:
        export(slot, args.output_dir / f"slot{slot}")


if __name__ == "__main__":
    main()
