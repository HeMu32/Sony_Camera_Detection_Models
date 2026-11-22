"""Pack and run the original A7R V/A1 II airplane NBG on CUDA without FP64."""

import argparse
from pathlib import Path

import torch

from nbg_cuda_quantization import ROUNDING_MODES
from slot08_cuda import PackedModel, export_inference, pack_graph
from verify_nbg_cuda_backbone import BLOCKS
from verify_nbg_cuda_multiscale import Scale, compute_branches, shared_tail, sigmoid_hwc


MODEL_HASH = "22a46a633b90f780026f57b21b2d34fa1dbd7b49237276a415eb109537da39be"
COMMANDS = (3, *(sites[0][0] for _, _, sites in BLOCKS[:7]),
            *(i for i in range(17, 75) if i != 66))
SCALES = (
    Scale(80, (), 17, 186, 132, (0, 7, 14), (1, 4), (10, 13)),
    Scale(40, (22, 23, 24, 25), 26, 155, 128, (1, 8, 15), (18, 21), (27, 30)),
    Scale(20, (31, 32, 33, 34), 35, 120, 128, (2, 9, 16), (35, 38), (44, 47)),
    Scale(10, (40, 41, 42, 43), 44, 127, 125, (3, 10, 17), (52, 55), (61, 64)),
    Scale(5, (49, 50, 51, 52), 53, 129, 128, (4, 11, 18), (69, 72), (78, 81)),
    Scale(3, (), 70, 128, 141, (5, 12, 19), (97, 100), (106, 109)),
    Scale(1, (), 64, 129, 152, (6, 13, 20), (94, 94), (96, 96)),
)
CLASS_SCALES = ("0.127384201", "0.129197836", "0.15894872", "0.158323452",
                "0.196404055", "0.131796598", "0.13619791")
BOX_SCALES = ("0.0617020018", "0.0841739997", "0.161569998", "0.193286002",
              "0.133892998", "0.171483994", "0.0713410005")


def output_specs(model):
    specs = {}
    for scale, cls_scale, reg_scale in zip(SCALES, CLASS_SCALES, BOX_SCALES):
        cls, reg, aux = scale.output
        g = scale.grid
        specs[cls] = ((g, g, 18, 1), cls_scale, model.tp[cls][1])
        specs[reg] = ((g, g, 24, 1), reg_scale, model.tp[reg][1])
        specs[aux] = ((g, g, 6, 1), "0.00390625", 0)
    return specs


def run_heads(block, rgb_chw):
    shared = block(3, rgb_chw, 32, 3)
    for shape, pool, sites in BLOCKS[:7]:
        cmd = sites[0][0]
        shared = block(cmd, shared, shape[0], shape[-1], pool=pool and cmd != 16)
    small = None
    for scale in SCALES:
        if scale.grid == 3:
            small = shared_tail(block, shared, first=58)
        feed = shared if small is None else small[scale.grid]
        shared, cls, reg, aux = compute_branches(
            block, feed, scale, class_channels=18,
            transition_channels=64 if scale.grid == 40 else 128,
        )
        yield scale, cls, reg, aux


class Slot10(PackedModel):
    model_hash = MODEL_HASH
    commands = COMMANDS
    tp_outputs = tuple(i for scale in SCALES for i in scale.output[:2])

    def heads(self, rgb_chw):
        if rgb_chw.shape != (3, 320, 320) or rgb_chw.dtype != torch.uint8:
            raise ValueError("expected a 3x320x320 planar uint8 RGB tensor")
        return run_heads(self.block, rgb_chw.cuda())

    @torch.inference_mode()
    def infer(self, rgb_chw):
        outputs = {}
        for scale, cls, reg, aux in self.heads(rgb_chw):
            for idx, tensor in zip(scale.output[:2], (cls, reg)):
                outputs[idx] = self.requantize_output(idx, tensor)
            # TP writes interleaved bytes despite the declared auxiliary dimensions.
            outputs[scale.output[2]] = sigmoid_hwc(aux)
        return outputs


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
    if args.action == "pack":
        pack_graph(args, MODEL_HASH, (74, 109), COMMANDS, SCALES)
    else:
        export_inference(args, Slot10, output_specs)


if __name__ == "__main__":
    main()
