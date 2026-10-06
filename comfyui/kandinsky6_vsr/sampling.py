"""Kandinsky 6 SR sampling on ComfyUI's model management and samplers.

The released checkpoints denoise one tile batch at a time, starting at sigma 1
from the degraded upscaled latent rather than from Gaussian noise (instruct
type "noise"). The flow-matching checkpoint runs ComfyUI's Euler sampler on
the canonical shifted schedule; the distilled checkpoint runs the canonical
pi-Flow segment loop on the native multi-grid DiT. Tensors handed in and out
use the canonical ``(N, H, W, C)`` tile layout, ``N = tiles * frames``.
"""

import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import torch

from .runtime.core.algo.piflow_math import DXPolicy, policy_rollout_fm, shift_timesteps
from .sr_contract import EULER_SAMPLING, PIFLOW_SAMPLING


def is_piflow(model_patcher):
    return int(getattr(model_patcher.model.diffusion_model, "n_grid", 1)) > 1


def sampling_settings(model_patcher):
    return PIFLOW_SAMPLING if is_piflow(model_patcher) else EULER_SAMPLING


def steps_per_tile(model_patcher, num_steps):
    """Network evaluations per tile batch (``num_steps`` grid points = N - 1 steps)."""
    return int(PIFLOW_SAMPLING["nfe"]) if is_piflow(model_patcher) else int(num_steps) - 1


def degraded_start(lq_latent, *, seed, noise_scale, noise_type):
    """Canonical ``degrade_lq_latent``: RNG on the latent's device, layout and dtype."""
    generator = torch.Generator(device=lq_latent.device).manual_seed(int(seed))
    eps = torch.randn(lq_latent.shape, device=lq_latent.device, dtype=lq_latent.dtype, generator=generator)
    if noise_type == "ddpm":
        return (1 - noise_scale**2) ** 0.5 * lq_latent + noise_scale * eps
    return (1 - noise_scale) * lq_latent + noise_scale * eps


def to_comfy(x, batch):
    """``(B*T, H, W, C)`` -> ``(B, C, T, H, W)``."""
    frames = x.shape[0] // batch
    return x.reshape(batch, frames, *x.shape[1:]).permute(0, 4, 1, 2, 3)


def to_canonical(x):
    """``(B, C, T, H, W)`` -> ``(B*T, H, W, C)``."""
    return x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[3], x.shape[4], x.shape[1])


def euler_sigmas(num_steps, shift):
    """Canonical schedule: ``num_steps`` grid points from 1 to 0, time-shifted."""
    t = torch.linspace(1.0, 0.0, int(num_steps))
    return shift * t / (1 + (shift - 1) * t)


def sample_euler(model_patcher, start, batch, num_steps, progress=None):
    """Run ComfyUI's Euler sampler from the degraded latent at sigma 1.

    ``noise_scaling`` of a flow model returns the noise at sigma 1, so the
    degraded latent is passed as the noise and the clean latent is zero.
    """
    noise = to_comfy(start, batch).float()
    conditioning = [[torch.zeros((1, 1, 1)), {}]]
    sigmas = euler_sigmas(num_steps, float(EULER_SAMPLING["scheduler_scale"]))

    def callback(step, x0, x, total_steps):
        if progress is not None:
            progress(1)

    samples = comfy.sample.sample_custom(
        model_patcher,
        noise,
        1.0,
        comfy.samplers.sampler_object("euler"),
        sigmas,
        conditioning,
        conditioning,
        torch.zeros_like(noise),
        callback=callback,
        disable_pbar=True,
    )
    return to_canonical(samples.to(start.device))


def sample_piflow(model_patcher, start, batch, progress=None, memory_required=0):
    """Canonical pi-Flow segment loop: ``nfe`` multi-grid DiT calls, policy rollout between.

    The state keeps the canonical dtypes: it starts in the model dtype and is
    fp32 after the first policy rollout; ``t * 1000`` is cast to the state
    dtype as in ``piflow_generate``.
    """
    mm.load_models_gpu([model_patcher], memory_required=memory_required)
    model = model_patcher.model
    dit = model.diffusion_model
    dtype = model.get_dtype_inference()
    device = model_patcher.load_device
    transformer_options = dict(model_patcher.model_options.get("transformer_options", {}))

    params = PIFLOW_SAMPLING
    nfe, shift, eps = int(params["nfe"]), float(params["shift"]), float(params["eps"])
    final_step = max(float(params["final_step_size_scale"]), eps)
    one_minus_final = 1.0 - final_step
    base_segment = 1.0 / (nfe - one_minus_final)
    final_segment = final_step * base_segment
    channels = int(dit.out_visual_dim)

    x = start.to(device)
    tokens = x.shape[0]
    for step_index in range(nfe):
        mm.throw_exception_if_processing_interrupted()
        index = nfe - step_index
        raw_src = min(max((index - one_minus_final) * base_segment, eps), 1.0)
        segment = final_segment if index == 1 else base_segment
        raw_dst = max(raw_src - segment, 0.0)
        sigma_src = shift_timesteps(torch.tensor(raw_src, device=device), shift).item()

        timestep = torch.full((batch,), sigma_src * 1000.0, device=device, dtype=x.dtype).float()
        model_input = to_comfy(x, batch)
        zeros = torch.zeros((batch, channels + 1, *model_input.shape[2:]), device=device, dtype=model_input.dtype)
        model_input = torch.cat([model_input, zeros], dim=1).to(dtype)
        velocity = dit(model_input, timestep, transformer_options=transformer_options)
        grids = velocity.shape[1] // channels
        velocity = velocity.reshape(batch, grids, channels, *velocity.shape[2:]).permute(0, 3, 1, 4, 5, 2)
        velocity = velocity.reshape(tokens, grids, *velocity.shape[3:])

        sigma = torch.full((tokens, 1, 1, 1), sigma_src, device=device)
        policy = DXPolicy(
            velocity, x, sigma, torch.full((tokens,), segment, device=device), shift=shift, mode="grid", eps=eps
        )
        x, _, _ = policy_rollout_fm(
            x,
            sigma,
            torch.full((tokens,), raw_src, device=device),
            torch.full((tokens,), raw_dst, device=device),
            int(params["num_policy_substeps"]),
            policy,
        )
        if progress is not None:
            progress(1)
    return x


def sample(model_patcher, start, batch, num_steps, progress=None, memory_required=0):
    if is_piflow(model_patcher):
        return sample_piflow(model_patcher, start, batch, progress, memory_required)
    return sample_euler(model_patcher, start, batch, num_steps, progress)
