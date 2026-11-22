"""Portable forward for the Sony ILCE-1M2 slot-18 pose network."""

import torch
from torch import nn
import torch.nn.functional as F


INPUT_SHAPE = (3, 288, 416)
MODEL_HASH = "84a35252118fa92ed7a5e36e7f16d5f8b00f0ab6b901d5ba3d83cbc5a4a10e75"
OUTPUT_COMMANDS = (49, 51, 61, 67, 57, 63, 53, 65, 59, 69, 55, 47)


def pose_requantize(accum, scale, zero_point, cuda_rounding):
    if accum.is_cuda:
        from nbg_cuda_quantization import round_scaled_u8

        return round_scaled_u8(accum, scale, zero_point, match_trace=cuda_rounding == "simulator")
    return (accum.double() * scale.double()).round().add(zero_point).clamp(0, 255).byte()


def stem_raster(rgb, pad=127):
    # Four TP stripes write the same planar, channel-first pixel-unshuffle.
    return F.pad(F.pixel_unshuffle(rgb[None], 2), (0, 1, 0, 1), value=pad)


class Slot18Block(nn.Module):
    def __init__(self, command, tensors, *, unsigned_weights=False):
        super().__init__()
        w = tensors["weight_kzyx"]
        if w.dtype == torch.uint8 and not unsigned_weights:
            w = w.view(torch.int8)
        bias = tensors["bias_channel_lo_hi"]
        q = tensors["postprocess_records"]
        channels = torch.arange(w.shape[0], dtype=torch.int32)
        if not (torch.equal(bias[:, 0], channels) and torch.equal(q[:, 0], channels) and
                q[:, 5].unique().numel() == 1 and not torch.any(q[:, 6])):
            raise ValueError(f"unexpected bias or quantization at NN {command}")
        scale = q[:, 1].contiguous().view(torch.float32)
        bound = 128 * w.to(torch.int64).abs().sum((1, 2, 3)) + bias[:, 1].to(torch.int64).abs()
        if not torch.all(scale.isfinite() & (scale > 0)) or torch.any(bound >= 2**24):
            raise ValueError(f"unsupported arithmetic at NN {command}")
        self.register_buffer("weight", w)
        self.register_buffer("bias", bias[:, 1].contiguous()[None, :, None, None])
        self.register_buffer("scale", scale[None, :, None, None])
        self.zero_point = int(q[0, 5])
        self.padding = 1 if w.shape[-2:] == (3, 3) else 0
        self.pool = command in (2, 5)
        self.cuda_rounding = "simulator"

    def forward(self, activation):
        x = activation.float() - 128
        if self.padding:
            x = F.pad(x, (1, 1, 1, 1), value=-128)
        x = F.conv2d(x, self.weight.float())
        if self.pool:
            x = F.max_pool2d(x, 2)
        return pose_requantize(x + self.bias, self.scale, self.zero_point, self.cuda_rounding)


class Slot18(nn.Module):
    def __init__(self, package):
        super().__init__()
        src = torch.load(package, weights_only=True, map_location="cpu")
        if (src["metadata"]["source_nbg_sha256"] != MODEL_HASH or
                set(src["commands"]) != set(range(1, 70))):
            raise ValueError("unexpected slot-18 parameter package")
        self.layers = nn.ModuleList(Slot18Block(i, src["commands"][i]) for i in range(1, 70))

    def activations(self, rgb):
        x = stem_raster(rgb)
        for i, layer in enumerate(self.layers[:12], 1):
            if i in (4, 7, 9, 11):
                x = x.flatten(2).reshape(1, x.shape[1], 117, -1)
            if i == 5:
                x = x.reshape(1, 32, 72, 104)
            if i in (8, 10, 12):
                x = x.reshape(1, 64, 36, 52)
            x = layer(x)
            yield x
        for i in range(12, 45, 3):
            # Each triplet writes adjacent SRAM banks consumed by the next stage.
            branches = []
            for layer in self.layers[i:i + 3]:
                x = layer(x)
                branches.append(x)
                yield x
            x = torch.cat(branches, dim=1)
        shared = x.reshape(1, 144, 117, 16)
        for i in range(45, 69, 2):
            x = self.layers[i](shared)
            yield x
            x = self.layers[i + 1](x)
            yield x

    def forward(self, rgb):
        values = tuple(self.activations(rgb))
        return tuple(values[i - 1][0].reshape(-1, 36, 52) for i in OUTPUT_COMMANDS)
