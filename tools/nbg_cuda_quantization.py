"""Fused uint8 requantization with no FP64 GPU arithmetic."""

import cupy
import numpy as np
import torch


ROUNDING_MODES = ("simulator", "simple")


_REQUANTIZE = cupy.ElementwiseKernel(
    "float32 accum, float32 scale, int32 zero_point, bool relu, bool match_trace",
    "uint8 output",
    """
    const float value = relu ? fmaxf(accum, 0.0f) : accum;
    const float product = __fmul_rn(value, scale);
    float rounded = rintf(product);
    const float distance = rounded - product;
    if (match_trace && fabsf(distance) == 0.5f) {
        // Only a rounded half-integer can hide which side of the tie we were on.
        const float error = __fmaf_rn(value, scale, -product);
        rounded += (distance < 0.0f && error > 0.0f)
                 - (distance > 0.0f && error < 0.0f);
    }
    output = (unsigned char)fminf(255.0f, fmaxf(0.0f, rounded + zero_point));
    """,
    "nbg_requantize_u8_f32",
)


_TP_REQUANTIZE = cupy.ElementwiseKernel(
    "int32 centered, int32 output_zp, int32 multiplier, int32 shift, bool match_trace",
    "uint8 output",
    """
    float value = ldexpf((float)(centered * multiplier), -shift);
    if (match_trace) {
        // SE8M15 is rounded before and after the zero-point add.
        unsigned bits = __float_as_uint(value);
        bits = (bits + 0x7fu + ((bits >> 8) & 1u)) & 0xffffff00u;
        value = __uint_as_float(bits) + output_zp;
        bits = __float_as_uint(value);
        bits = (bits + 0x7fu + ((bits >> 8) & 1u)) & 0xffffff00u;
        value = rintf(__uint_as_float(bits));
    } else {
        value = rintf(value + output_zp);
    }
    output = (unsigned char)fminf(255.0f, fmaxf(0.0f, value));
    """,
    "nbg_tp_requantize_u8_se8m15",
)


def round_scaled_u8(accum, scales, zero_point=0, *, relu=False, match_trace=True):
    if (not accum.is_cuda or accum.device != scales.device or
            accum.dtype != torch.float32 or scales.dtype != torch.float32 or
            not 0 <= zero_point <= 255):
        raise ValueError("expected CUDA FP32 accumulators/scales and a uint8 zero point")
    device = accum.device.index
    stream = torch.cuda.current_stream(accum.device).cuda_stream
    with cupy.cuda.Device(device), cupy.cuda.ExternalStream(stream, device_id=device):
        raw = _REQUANTIZE(cupy.from_dlpack(accum), cupy.from_dlpack(scales),
                          np.int32(zero_point), np.bool_(relu), np.bool_(match_trace))
        return torch.from_dlpack(raw)


def tp_requantize_u8(centered, output_zp, multiplier, shift, *, match_trace=True):
    if (not centered.is_cuda or centered.dtype != torch.int32 or
            not 0 <= output_zp <= 255 or not 0 < multiplier < 65536 or not 1 <= shift <= 30):
        raise ValueError("expected centered CUDA int32 bytes and a uint8 TP conversion")
    device = centered.device.index
    stream = torch.cuda.current_stream(centered.device).cuda_stream
    with cupy.cuda.Device(device), cupy.cuda.ExternalStream(stream, device_id=device):
        raw = _TP_REQUANTIZE(cupy.from_dlpack(centered), np.int32(output_zp),
                             np.int32(multiplier), np.int32(shift), np.bool_(match_trace))
        return torch.from_dlpack(raw)
