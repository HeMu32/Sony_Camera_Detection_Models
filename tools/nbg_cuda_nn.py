"""Integer-valued NBG convolution using FP32 GEMM and fused requantization."""

import torch
import torch.nn.functional as F

from nbg_cuda_quantization import round_scaled_u8


def nn_forward(activation, matrix, bias, scale, kernel, *, relu, zero_point,
               padding=None, pad_code=0, pool=False, match_trace=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    image = activation.to(dtype=torch.float32)[None] - 128
    padding = (kernel // 2,) * 4 if padding is None else padding
    padded = F.pad(image, padding, value=pad_code - 128)
    # Direct GEMM avoids transform-based convolution's fractional intermediates.
    patches = F.unfold(padded, kernel)[0]
    height, width = (dim - kernel + 1 for dim in padded.shape[-2:])
    accum = (matrix @ patches).reshape(matrix.shape[0], height, width)
    if pool:
        accum = F.max_pool2d(accum, 2)
    accum = accum + bias
    return accum, round_scaled_u8(accum, scale, zero_point, relu=relu, match_trace=match_trace)
