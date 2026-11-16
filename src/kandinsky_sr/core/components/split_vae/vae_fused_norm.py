"""Inference-only spatial RMS normalization and latent modulation."""

import torch
from torch.nn import functional as F

from .vae_conditioning import modulate_


def fused_spatial_norm(x, gamma, bias, projected_y, projected_b, origin, full_hw,
                       *, eps, scale, activation="none"):
    if activation not in ("none", "silu", "native"):
        raise ValueError(f"Unknown fused activation: {activation}")
    if torch.is_grad_enabled() and (any(value.requires_grad for value in
                                      (x, gamma, projected_y, projected_b))
                                    or (bias is not None and bias.requires_grad)):
        raise RuntimeError("Fused spatial normalization is inference-only")
    if x.is_cuda:
        from .vae_fused_norm_kernel import launch_fused_norm
        return launch_fused_norm(x, gamma, bias, projected_y, projected_b, origin, full_hw,
                                 eps, scale, activation)
    output = F.normalize(x, dim=1, eps=eps)
    output.mul_(scale).mul_(gamma[None])
    if bias is not None:
        output.add_(bias[None])
    modulate_(output, projected_y, origin, full_hw, multiply=True)
    modulate_(output, projected_b, origin, full_hw, multiply=False)
    if activation == "silu":
        output = F.silu(output, inplace=True)
    elif activation == "native":
        output = output * torch.sigmoid(output)
    return output.contiguous(memory_format=torch.channels_last_3d)
