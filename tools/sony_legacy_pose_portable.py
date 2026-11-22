"""Recover the two legacy pose tails from the recorded NN and TP command schedule."""

from collections import defaultdict

import torch
from torch import nn

from nbg_integer_quantization import integer_formula
from sony_legacy_pose_backbone import BACKBONES, Depthwise, LegacyBackbone
from sony_slot18_portable import Slot18Block


GRAPHS = {
    # First untiled NN, output grid (height, width), and output addresses / channels.
    "cat-r5": (97, (36, 52), ((0x82597c0, 40), (0x826be80, 19))),
    "human": (274, (42, 64), ((0x8292580, 40), (0x82acbc0, 19))),
}


class PlaneBindings:
    """Track the latest writer of each CHW plane at its original byte address."""

    def __init__(self, plane_bytes):
        self.plane = plane_bytes
        self.writers = {}

    def publish(self, address, channels, producer):
        for i in range(channels):
            self.writers[address + i * self.plane] = (producer, i)

    def resolve(self, address, channels):
        runs = []
        for i in range(channels):
            location = address + i * self.plane
            if location not in self.writers:
                raise ValueError(f"no tensor producer for address 0x{location:08x}")
            producer, channel = self.writers[location]
            if runs and (producer, channel) == (runs[-1][0], runs[-1][2]):
                runs[-1] = (producer, runs[-1][1], channel + 1)
            else:
                runs.append((producer, channel, channel + 1))
        return tuple(runs)


def consume(values, edges):
    tensors = [values[producer][:, start:end] for producer, start, end in edges]
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)


class TPRequantize(nn.Module):
    def __init__(self, words):
        super().__init__()
        mode, input_params, output_params = (words[i] for i in (24, 29, 30))
        if mode not in (0x40a, 0x80a) or input_params & 0xffffff or output_params >> 23:
            raise ValueError("unsupported legacy TP requantization")
        izp, ozp = input_params >> 24, output_params & 255
        mult = 0x8000 | ((output_params >> 8) & 0x7fff)
        shift = 15 + (mode >> 10)
        lut = [integer_formula(i - izp, ozp, mult, shift) for i in range(256)]
        self.register_buffer("lut", torch.tensor(lut, dtype=torch.uint8))

    def forward(self, x):
        return self.lut[x.long()]


def tp_stripes(invocations, height, width):
    """Turn four adjacent 9-row DMA stripes back into one planar operation."""
    if len(invocations) % 4 or height != 36:
        raise ValueError("unsupported tail TP stripes")
    for offset in range(0, len(invocations), 4):
        rows = [x.words for x in invocations[offset:offset + 4]]
        first = rows[0]
        channels = first[1] >> 16
        if channels not in (40, 128):
            raise ValueError("unsupported tail TP channel count")
        for j, row in enumerate(rows):
            if (row[:4] != (width, channels << 16 | 9, width, height * width) or
                    row[10] != first[10] + j * 9 * width or row[13] != first[13] + j * 9 * width or
                    any(row[i] != first[i] for i in (24, 29, 30))):
                raise ValueError("noncontiguous TP stripes")
        yield first, channels


class LegacyPose(nn.Module):
    def __init__(self, name, groups, maps, instructions, tp):
        super().__init__()
        self.backbone = LegacyBackbone(name, groups, maps, instructions)
        self.layers = nn.ModuleList()
        self.schedule = []
        first, (height, width), outputs = GRAPHS[name]
        memory = PlaneBindings(height * width)
        seed = BACKBONES[name][-1]
        memory.publish(int(instructions[seed - 1, 7]), 128, seed)
        after = defaultdict(list)
        for call in tp:
            if call.after_nn:
                after[call.after_nn].append(call)
        if any(not first <= i <= len(groups) for i in after):
            raise ValueError("unexpected TP operation inside legacy backbone")

        tp_producer = len(groups)
        for cmd in range(first, len(groups) + 1):
            words = [int(v) for v in instructions[cmd - 1]]
            group = groups[cmd]
            co, ci, ky, kx = group["weight_kzyx"].shape
            if any(((words[i] >> 6) & 8191, words[i] >> 19) != (width, height) for i in (1, 3)):
                raise ValueError(f"unexpected grid in NN {cmd}")
            if cmd in maps:
                if ci != 1 or ky != kx or kx != 3:
                    raise ValueError(f"unsupported depthwise NN {cmd}")
                ci = co
                # GetNnInst decodes the border mode and byte from word 15.
                if words[15] & 3:
                    raise ValueError(f"unsupported border mode in NN {cmd}")
                pad_code = (words[15] >> 2) & 255
                layer = Depthwise(cmd, group, maps[cmd], stride=1, pad_code=pad_code)
            else:
                if ky != kx or kx != 1:
                    raise ValueError(f"unsupported pointwise NN {cmd}")
                w = group["weight_kzyx"]
                unsigned = w.dtype == torch.uint8
                if unsigned and (
                    (co, ci) != (128, 256)
                    or not torch.equal(128 * w.to(torch.int32).sum((1, 2, 3)),
                                       group["bias_channel_lo_hi"][:, 1])
                ):
                    raise ValueError(f"unrecognized unsigned branch fusion in NN {cmd}")
                layer = Slot18Block(cmd, group, unsigned_weights=unsigned)
            self.schedule.append((cmd, memory.resolve(words[6], ci)))
            self.layers.append(layer)
            memory.publish(words[7], co, cmd)
            for words, channels in tp_stripes(after[cmd], height, width) if after[cmd] else ():
                tp_producer += 1
                self.schedule.append((tp_producer, memory.resolve(words[10], channels)))
                self.layers.append(TPRequantize(words))
                memory.publish(words[13], channels, tp_producer)
        self.outputs = tuple(memory.resolve(address, c) for address, c in outputs)
        self.seed = seed

    def activations(self, rgb):
        values = {}
        for command, tree in self.backbone.activations(rgb):
            yield command, tree
        values[self.seed] = tree
        for (producer, edges), layer in zip(self.schedule, self.layers):
            tree = layer(consume(values, edges))
            values[producer] = tree
            yield producer, tree

    def forward(self, rgb):
        values = dict(self.activations(rgb))
        return tuple(consume(values, edges)[0] for edges in self.outputs)
