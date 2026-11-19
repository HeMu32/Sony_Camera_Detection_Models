"""Pack the known Sony slot-08 graph once, then run it without CPU traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nbg_cuda_nn import nn_forward
from nbg_cuda_quantization import ROUNDING_MODES, tp_requantize_u8
from verify_nbg_cuda_backbone import BLOCKS
from verify_nbg_cuda_multiscale import SCALES, interval, run_scales, sigmoid_hwc


MODEL_HASH = "28782d52ac73d7a537a2576bd8f9e4e6b5f7ccd23a420093e1a1a14b556d06f9"
COMMANDS = (3, *(sites[0][0] for _, _, sites in BLOCKS[:-1]),
            *(i for i in range(21, 70) if i != 61))
TP_OUTPUTS = tuple(i for scale in SCALES for i in scale.output[:2])
PLACEHOLDERS = (0, 7, 14)
BOX_SCALES = ("0.0827599987", "0.127638996", "0.129427999",
              "0.188388005", "0.201147005", "0.0560249984")


def output_specs(model):
    specs = {i: ((1, 1, 6, 1), "0.00390625", 0) for i in PLACEHOLDERS}
    for scale, regression_scale in zip(SCALES, BOX_SCALES):
        g = scale.grid
        cls, reg, aux = scale.output
        specs[cls] = ((g, g, 30, 1), "0.046875", model.tp[cls][1])
        specs[reg] = ((g, g, 24, 1), regression_scale, model.tp[reg][1])
        aux_shape = (1, 1, 6, 1) if g == 1 else (6, g, g, 1)
        specs[aux] = (aux_shape, "0.00390625", 0)
    return specs


def pack(args):
    pack_graph(args, MODEL_HASH, (69, 92), COMMANDS, SCALES)


def pack_graph(args, expected_hash, schedule, commands, scales):
    from inspect_nbg_tp_trace import read_trace
    from verify_nbg_cuda_first_conv import layer_parameters
    from verify_nbg_cuda_second_conv import coefficients
    from verify_nbg_cuda_tiles import read_rows, rows_for_op
    from verify_nbg_cuda_tp40 import decode_quantization

    nbg_hash = hashlib.sha256(args.nbg.read_bytes()).hexdigest()
    if nbg_hash != expected_hash:
        raise ValueError("NBG hash does not match the supported original model")
    rows = read_rows(args.nn_trace)
    count, tp = read_trace(args.tp_trace)
    if (count, len(tp)) != schedule:
        raise ValueError("unexpected model schedule")
    data = {"model_sha256": np.array(nbg_hash)}
    for cmd in commands:
        op = rows_for_op(rows, cmd)
        shape = tuple(op[op[:, 0] == 1, 2:6].max(0)[::-1] + 1)
        w = coefficients(op, shape)
        zp, = np.unique(op[op[:, 0] == 7, 7])
        b, s = layer_parameters(op, shape[0], zero_point=zp)
        maximum = 128 * np.abs(w.astype(np.int64)).sum((1, 2, 3)).max()
        if maximum + np.abs(b.astype(np.int64)).max() >= 2**24:
            raise ValueError(f"command {cmd} exceeds exact FP32 accumulation range")
        data[f"w{cmd}"], data[f"b{cmd}"], data[f"s{cmd}"], data[f"zp{cmd}"] = w, b, s, zp
    for scale in scales:
        for idx, bounds in zip(scale.output[:2], (scale.class_tp, scale.bbox_tp)):
            q, = {decode_quantization(tp[i - 1].words) for i in interval(bounds)}
            data[f"tp{idx}"] = np.array(q, dtype=np.int32)
    with args.package.open("xb") as file:
        np.savez_compressed(file, **data)
    print(f"Packed {len(commands)} original weight blocks and {2 * len(scales)} TP conversions")


class PackedModel:
    model_hash = None
    commands = ()
    tp_outputs = ()

    def __init__(self, package, *, rounding="simulator"):
        if rounding not in ROUNDING_MODES:
            raise ValueError(f"unsupported rounding mode: {rounding}")
        self.rounding = rounding
        self.ops = {}
        with np.load(package, allow_pickle=False) as data:
            if data["model_sha256"].item() != self.model_hash:
                raise ValueError("package hash does not match the supported model")
            for cmd in self.commands:
                w, b, s = (data[f"{k}{cmd}"] for k in ("w", "b", "s"))
                self.ops[cmd] = (
                    torch.from_numpy(w).to(device="cuda", dtype=torch.float32).flatten(1),
                    torch.from_numpy(b).to(device="cuda", dtype=torch.float32)[:, None, None],
                    torch.from_numpy(s).cuda()[:, None, None], w.shape, int(data[f"zp{cmd}"]),
                )
            self.tp = {i: tuple(map(int, data[f"tp{i}"])) for i in self.tp_outputs}

    def block(self, cmd, x, channels, kernel, *, relu=True, zp=0, padding=None, pool=False):
        w, b, s, shape, packed_zp = self.ops[cmd]
        if shape != (channels, x.shape[0], kernel, kernel) or packed_zp != zp:
            raise ValueError(f"unexpected parameter shape or zero point at command {cmd}")
        pad_code = 128 if cmd == 3 else 0
        _, output = nn_forward(x, w, b, s, kernel, relu=relu, zero_point=zp,
                               padding=padding, pad_code=pad_code, pool=pool,
                               match_trace=self.rounding == "simulator")
        return output

    def requantize_output(self, index, tensor):
        iz, oz, multiplier, shift = self.tp[index]
        return tp_requantize_u8(tensor.to(torch.int32) - iz, oz, multiplier, shift,
                                match_trace=self.rounding == "simulator")


class Slot08(PackedModel):
    model_hash = MODEL_HASH
    commands = COMMANDS
    tp_outputs = TP_OUTPUTS

    @torch.inference_mode()
    def infer(self, rgb_chw):
        if rgb_chw.shape != (3, 320, 320) or rgb_chw.dtype != torch.uint8:
            raise ValueError("expected a 3x320x320 planar uint8 RGB tensor")
        x = self.block(3, rgb_chw.cuda(), 32, 3)
        for shape, pool, sites in BLOCKS[:-1]:
            x = self.block(sites[0][0], x, shape[0], shape[-1], pool=pool)
        outputs = {}
        for scale, classification, regression, auxiliary in run_scales(self.block, x):
            cls, reg, aux = scale.output
            for i, tensor in ((cls, classification), (reg, regression)):
                outputs[i] = self.requantize_output(i, tensor)
            outputs[aux] = sigmoid_hwc(auxiliary)
        return outputs


def export_inference(args, model_type, get_specs, placeholders=()):
    from prepare_nbg_inputs import prepare

    if args.input.suffix.lower() == ".raw":
        data, geometry = args.input.read_bytes(), None
    else:
        data, geometry = prepare(args.input, 320, 320, "letterbox", 128, "rgb", "chw")
    image = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy().reshape(3, 320, 320))
    model = model_type(args.package, rounding=args.rounding)
    outputs = model.infer(image)
    specs = get_specs(model)
    lines = [f"# rounding={model.rounding}\n"]
    for idx, (shape, scale, zp) in sorted(specs.items()):
        size = int(np.prod(shape))
        # The original graph leaves these output slots unwritten.
        raw = bytes(size) if idx in placeholders else outputs[idx].cpu().numpy().tobytes()
        if len(raw) != size:
            raise ValueError(f"unexpected output {idx} size")
        with Path(f"{args.output_prefix}.{idx}.raw").open("xb") as file:
            file.write(raw)
        dims = "x".join(map(str, shape))
        lines.append(f"param {idx + 1} output dtype=0x3 quant=0x2 shape={dims} "
                     f"bytes={size} scale={scale} zp={zp}\n")
    with Path(f"{args.output_prefix}.log").open("x", encoding="ascii") as file:
        file.writelines(lines)
    if geometry is not None:
        key = args.input.stem + ".raw"
        geometry["sha256"] = hashlib.sha256(data).hexdigest()
        with Path(f"{args.output_prefix}.manifest.json").open("x", encoding="ascii") as file:
            json.dump({key: geometry}, file, indent=2)
    print(f"Computed {len(outputs)} outputs on {torch.cuda.get_device_name()}; "
          f"initialized {len(placeholders)} unwritten interface slots")


def infer(args):
    export_inference(args, Slot08, output_specs, PLACEHOLDERS)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    verbs = cli.add_subparsers(dest="action", required=True)
    p = verbs.add_parser("pack")
    for argument in ("nbg", "nn_trace", "tp_trace", "package"):
        p.add_argument(argument, type=Path)
    p = verbs.add_parser("infer")
    for argument in ("package", "input", "output_prefix"):
        p.add_argument(argument, type=Path)
    p.add_argument("--rounding", choices=ROUNDING_MODES, default="simulator")
    args = cli.parse_args()
    (pack if args.action == "pack" else infer)(args)


if __name__ == "__main__":
    main()
