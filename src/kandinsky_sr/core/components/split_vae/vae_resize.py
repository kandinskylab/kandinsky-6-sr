"""Nearest upsampling with a single channels-last destination."""

import torch
from torch._inductor.inductor_prims import force_stride_order


def require_channels_last_3d(x):
    if torch.compiler.is_compiling():
        n, c, t, h, w = x.shape
        return force_stride_order(x, (c * t * h * w, 1, c * h * w, c * w, c))
    return x.contiguous(memory_format=torch.channels_last_3d)


def spatial_nearest_2x(x):
    n, c, t, h, w = x.shape
    output = torch.empty((n, c, t, 2 * h, 2 * w), device=x.device, dtype=x.dtype,
                         memory_format=torch.channels_last_3d)
    output.view(n, c, t, h, 2, w, 2).copy_(x[:, :, :, :, None, :, None])
    return output


def temporal_nearest_2x(x):
    n, c, t, h, w = x.shape
    output = torch.empty((n, c, 2 * t - 1, h, w), device=x.device, dtype=x.dtype,
                         memory_format=torch.channels_last_3d)
    output[:, :, :1].copy_(x[:, :, :1])
    if t > 1:
        output[:, :, 1:].view(n, c, t - 1, 2, h, w).copy_(x[:, :, 1:, None])
    return output
