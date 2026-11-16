"""Apply latent conditioning without expanded modulation tensors."""

import torch


def modulate_(output, projected, origin, full_hw, *, multiply):
    if torch.is_grad_enabled() and (output.requires_grad or projected.requires_grad):
        raise RuntimeError("Direct conditioning is inference-only")
    if output.is_cuda:
        from .vae_conditioning_kernel import launch_modulation
        launch_modulation(output, projected, origin, full_hw, multiply)
        return output

    # Bounded reference path for CPU checks.
    n, c, t, h, w = output.shape
    st, sh, sw = projected.shape[-3:]
    for start in range(0, output.numel(), 32768):
        index = torch.arange(start, min(start + 32768, output.numel()), device=output.device)
        ix, iy = index % w, index // w % h
        it, ic, batch = index // (w * h) % t, index // (w * h * t) % c, index // (w * h * t * c)
        source_t = torch.where(it == 0, 0, (it - 1) * (st - 1) // max(t - 1, 1) + 1) if st > 1 else torch.zeros_like(it)
        source = projected[batch, ic, source_t,
                           (iy + origin[0]) * sh // full_hw[0],
                           (ix + origin[1]) * sw // full_hw[1]]
        value = output[batch, ic, it, iy, ix]
        output[batch, ic, it, iy, ix] = value * source if multiply else value + source
    return output
