"""Run the two packaged CUDA graphs against bundled CPU simulator tensors."""

from pathlib import Path

import numpy as np
import torch

from slot08_cuda import PLACEHOLDERS, Slot08, output_specs as specs08
from slot10_cuda import Slot10, output_specs as specs10


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "debug/cpu_raw"
INPUT = ROOT / "debug/cpu_inputs/320x320_rgb"
EMPTY = "Empty_DSC03284"


def verify(model, specs, placeholders, input_name, prefix):
    image = np.fromfile(INPUT / f"{input_name}.raw", dtype=np.uint8).reshape(3, 320, 320)
    outputs = model.infer(torch.from_numpy(image))
    if set(outputs) != specs.keys() - set(placeholders):
        raise AssertionError(f"unexpected output indices: {prefix}")
    compared = 0
    for i, (shape, _, _) in sorted(specs.items()):
        data = bytes(int(np.prod(shape))) if i in placeholders else outputs[i].cpu().numpy().tobytes()
        reference = Path(f"{prefix}.{i}.raw").read_bytes()
        if data != reference:
            raise AssertionError(f"output {i} differs: {prefix}")
        compared += len(data)
    print(f"{model.__class__.__name__} {input_name}: {len(specs)} outputs, {compared} bytes, exact")


def main():
    bird = RAW / "coeff_probe_01"
    plane = RAW / "r5_s10"
    model = Slot08(bird / "sony-slot08-cuda.npz")
    for name, prefix in (("Bird_DSC02510", "NoTrace"), (EMPTY, "EmptyLayer")):
        verify(model, specs08(model), PLACEHOLDERS, name, bird / prefix)
    model = Slot10(plane / "sony-slot10-cuda.npz")
    for name, prefix in (("Flight_DSC02983", "Flight"), (EMPTY, "Empty")):
        verify(model, specs10(model), (), name, plane / prefix)


if __name__ == "__main__":
    main()
