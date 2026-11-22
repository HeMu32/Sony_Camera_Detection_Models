"""Reconstruct the tiled front of the R5 cat and shared human pose NBGs."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sony_slot18_portable import Slot18Block, pose_requantize, stem_raster


# First command for each logical stage; subsequent placements reuse its weights.
BACKBONES = {
    "cat-r5": (1, 2, 3, 7, 10, 13, 17, 21, 26, 57),
    "human": (1, 2, 3, 7, 10, 13, 17, 44, 50, 86),
}


class Depthwise(nn.Module):
    def __init__(self, command, group, order, stride, *, pad_code=0):
        super().__init__()
        count = len(order)
        weights = group["weight_kzyx"]
        if weights.shape != (count, 1, 3, 3) or sorted(order) != list(range(count)):
            raise ValueError(f"unexpected kernel map for NN {command}")
        if weights.dtype == torch.uint8:
            weights = weights.view(torch.int8)
        weights = weights[order]
        bias, q = group["bias_channel_lo_hi"], group["postprocess_records"]
        channels = torch.arange(count, dtype=torch.int32)
        if not (torch.equal(bias[:, 0], channels) and torch.equal(q[:, 0], channels)
                and not torch.any(q[:, 5:7])):
            raise ValueError(f"unexpected depthwise quantization at NN {command}")
        scale = q[:, 1].contiguous().view(torch.float32)
        bound = 128 * weights.to(torch.int64).abs().sum((1, 2, 3)) + bias[:, 1].to(torch.int64).abs()
        if not torch.all(scale.isfinite() & (scale > 0)) or torch.any(bound >= 2**24):
            raise ValueError(f"unsupported depthwise arithmetic at NN {command}")
        self.register_buffer("weight", weights)
        self.register_buffer("bias", bias[:, 1].contiguous()[None, :, None, None])
        self.register_buffer("scale", scale[None, :, None, None])
        self.groups = count
        self.stride = stride
        self.padding = (1, 1, 1, 1) if stride == 1 else (0, 1, 0, 1)
        self.pad_value = pad_code - 128
        self.cuda_rounding = "simulator"

    def forward(self, x):
        padded = F.pad(x.float() - 128, self.padding, value=self.pad_value)
        acc = F.conv2d(padded, self.weight.float(), groups=self.groups, stride=self.stride)
        return pose_requantize(acc + self.bias, self.scale, 0, self.cuda_rounding)


class LegacyBackbone(nn.Module):
    def __init__(self, name, groups, maps, instructions):
        super().__init__()
        self.commands = BACKBONES[name]
        self.layers = nn.ModuleList()
        for command in self.commands:
            if command not in maps:
                self.layers.append(Slot18Block(command, groups[command]))
            else:
                step = 2 if int(instructions[command - 1, 0]) >> 24 == 0xd8 else 1
                self.layers.append(Depthwise(command, groups[command], maps[command], step))

    def activations(self, rgb):
        x = stem_raster(rgb, pad=0)
        for command, layer in zip(self.commands, self.layers):
            x = layer(x)
            yield command, x


def read_instructions(path, commands):
    with path.open("rb") as stream:
        if stream.read(8) != b"NBNNC01\0":
            raise ValueError("invalid instruction capture")
        data = np.fromfile(stream, dtype="<u4").reshape(-1, 34)
    if len(data) != commands or not np.array_equal(data[:, 0], np.arange(1, commands + 1)):
        raise ValueError("unexpected instruction capture order")
    return data[:, 2:]
