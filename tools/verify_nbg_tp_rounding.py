"""Separate observed simulator parity from exhaustive CUDA/formula parity."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from inspect_nbg_tp_trace import read_trace
from nbg_cuda_quantization import tp_requantize_u8
from nbg_integer_quantization import integer_formula
from slot10_cuda import SCALES as SLOT10
from verify_nbg_cuda_multiscale import SCALES as SLOT08, interval
from verify_nbg_cuda_tp40 import decode_quantization


def verify(slot, paths):
    scales, schedule = {"08": (SLOT08, (69, 92)), "10": (SLOT10, (74, 109))}[slot]
    commands = [i for scale in scales for bounds in (scale.class_tp, scale.bbox_tp)
                for i in interval(bounds)]
    observations = defaultdict(dict)
    samples = 0
    for path in paths:
        nn_count, calls = read_trace(path)
        if (nn_count, len(calls)) != schedule:
            raise ValueError(f"unexpected slot-{slot} schedule in {path}")
        for index in commands:
            call = calls[index - 1]
            quant = decode_quantization(call.words)
            samples += call.samples
            for encoded, output in call.pairs:
                centered = (encoded ^ (1 << 31)) - (1 << 31)
                if not -quant[0] <= centered <= 255 - quant[0] or not 0 <= output <= 255:
                    raise ValueError("trace value outside the supported uint8 conversion")
                if observations[quant].setdefault(centered, output) != output:
                    raise ValueError(f"conflicting simulator outputs for {quant}, {centered}")
    observed_count = 0
    for (iz, oz, multiplier, shift), observed in observations.items():
        inputs = range(-iz, 256 - iz)
        cuda = tp_requantize_u8(torch.tensor(list(inputs), device="cuda", dtype=torch.int32),
                                 oz, multiplier, shift).tolist()
        exact = [integer_formula(a, oz, multiplier, shift) for a in inputs]
        if cuda != exact:
            raise ValueError(f"CUDA differs from exact integer formula for {(iz, oz, multiplier, shift)}")
        if any(cuda[a + iz] != expected for a, expected in observed.items()):
            raise ValueError(f"CUDA differs from simulator observations for {(iz, oz, multiplier, shift)}")
        observed_count += len(observed)
    total = 256 * len(observations)
    print(f"slot {slot}: {observed_count}/{total} distinct cases in {samples} simulator samples; "
          f"{total}/{total} CUDA vs integer formula, all matches")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("slot", choices=("08", "10"))
    cli.add_argument("traces", nargs="+", type=Path)
    args = cli.parse_args()
    verify(args.slot, args.traces)
