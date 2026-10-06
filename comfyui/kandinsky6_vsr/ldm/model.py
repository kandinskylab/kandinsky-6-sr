"""Kandinsky 6 video super-resolution transformer for ComfyUI.

The released SR DiT is the text-free member of the Kandinsky 5 transformer
family: no text encoder or cross-attention, a learned ``pooled_bias`` in place
of the pooled text embedding, a ``2C+1`` channel input (latent, conditioning
latent, mask) and an optional multi-grid head for pi-Flow checkpoints. It is
built from ComfyUI's Kandinsky 5 layers on ComfyUI ``operations`` and keeps the
canonical numeric contract: time embeddings, AdaLN modulation, norms, RoPE and
residual sums run in fp32, projections and attention in the compute dtype.
"""

import contextlib
import math

import comfy.ops
import comfy.patcher_extension
import torch
import torch.nn.functional as F
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.kandinsky5.model import (
    FeedForward,
    Modulation,
    OutLayer,
    SelfAttention,
    TimeEmbeddings,
    VisualEmbeddings,
    attention,
    get_shift_scale_gate,
)
from torch import nn

from ..nabla import attention as nabla_attention
from ..nabla import fractal_flatten_batch, fractal_flatten_grid, fractal_unflatten_batch
from ..sr_contract import ATTENTION_CONFIG, DIT_CONFIG, SR_COMMON


@contextlib.contextmanager
def _cast_bias_weight_fp32(layer, x):
    """Use Comfy's patched/offloadable weights across stable API versions."""
    kwargs = {"device": x.device, "dtype": torch.float32, "bias_dtype": torch.float32, "offloadable": True}
    context_type = getattr(comfy.ops, "CastBiasWeightContext", None)
    if context_type is not None:
        with context_type(layer, **kwargs) as weights:
            yield weights
        return
    state = comfy.ops.cast_bias_weight(layer, **kwargs)
    try:
        yield state[:2]
    finally:
        comfy.ops.uncast_bias_weight(layer, *state)


def _linear_fp32(layer, x):
    """Run a Comfy-managed Linear in fp32 without bypassing weight patches."""
    with _cast_bias_weight_fp32(layer, x) as (weight, bias):
        return F.linear(x.float(), weight, bias)


def _rms_norm_fp32(layer, x):
    """Canonical query/key RMSNorm: fp32 math, result in the input dtype."""
    with _cast_bias_weight_fp32(layer, x) as (weight, _bias):
        return F.rms_norm(x.float(), layer.normalized_shape, weight, layer.eps).to(dtype=x.dtype)


def _time_embeddings_fp32(module, timestep):
    """Canonical sinusoidal embedding and MLP, evaluated in fp32."""
    freqs = module.freqs.to(device=timestep.device, dtype=torch.float32)
    args = torch.outer(timestep.float(), freqs)
    embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = _linear_fp32(module.in_layer, embed)
    return _linear_fp32(module.out_layer, module.activation(hidden))


def _modulation_fp32(module, x):
    """Canonical AdaLN parameter projection, evaluated in fp32."""
    return _linear_fp32(module.out_layer, module.activation(x.float()))


def _scale_shift_norm_fp32(norm, x, scale, shift):
    """AdaLN affine math in fp32 (the canonical autocast-fp32 region)."""
    return norm(x.float()) * (scale.float() + 1.0) + shift.float()


def _gate_sum_fp32(x, out, gate):
    """Gated residual in fp32; the residual stream stays fp32 afterwards."""
    return x.float() + gate.float() * out.float()


def _self_attention(module, x, freqs, transformer_options):
    """Canonical self-attention: fp32 q/k RMSNorm and RoPE, compute-dtype attention."""
    shape = x.shape[:-1]
    query = module.to_query(x).view(*shape, module.num_heads, -1)
    key = module.to_key(x).view(*shape, module.num_heads, -1)
    value = module.to_value(x).view(*shape, module.num_heads, -1)
    query = apply_rope1(_rms_norm_fp32(module.query_norm, query), freqs)
    key = apply_rope1(_rms_norm_fp32(module.key_norm, key), freqs)
    sparse = transformer_options.get("_k6_vsr_nabla")
    if sparse is None:
        out = attention(query, key, value, module.num_heads, transformer_options=transformer_options)
    else:
        out = nabla_attention(query, key, value, sparse["shape"], sparse["config"])
    return module.out_layer(out)


class TransformerBlock(nn.Module):
    """Text-free decoder block: modulated self-attention and feed-forward."""

    def __init__(self, model_dim, time_dim, ff_dim, head_dim, operation_settings=None):
        super().__init__()
        operations = operation_settings.get("operations")
        settings = {"device": operation_settings.get("device"), "dtype": operation_settings.get("dtype")}
        self.visual_modulation = Modulation(time_dim, model_dim, 6, operation_settings=operation_settings)
        self.self_attention_norm = operations.LayerNorm(model_dim, elementwise_affine=False, **settings)
        self.self_attention = SelfAttention(model_dim, head_dim, operation_settings=operation_settings)
        self.feed_forward_norm = operations.LayerNorm(model_dim, elementwise_affine=False, **settings)
        self.feed_forward = FeedForward(model_dim, ff_dim, operation_settings=operation_settings)

    def forward(self, x, time_embed, freqs, compute_dtype, transformer_options=None):
        transformer_options = transformer_options or {}
        self_attn_params, ff_params = torch.chunk(_modulation_fp32(self.visual_modulation, time_embed), 2, dim=-1)

        shift, scale, gate = get_shift_scale_gate(self_attn_params)
        out = _scale_shift_norm_fp32(self.self_attention_norm, x, scale, shift).to(compute_dtype)
        out = _self_attention(self.self_attention, out, freqs, transformer_options)
        x = _gate_sum_fp32(x, out, gate)

        shift, scale, gate = get_shift_scale_gate(ff_params)
        out = _scale_shift_norm_fp32(self.feed_forward_norm, x, scale, shift).to(compute_dtype)
        out = self.feed_forward(out)
        return _gate_sum_fp32(x, out, gate)


class Kandinsky6SR(nn.Module):
    """Text-free Kandinsky 6 SR DiT.

    ``n_grid > 1`` widens the output head for pi-Flow checkpoints; the forward
    then returns ``(B, n_grid * out_visual_dim, T, H, W)`` with channel index
    ``grid * out_visual_dim + channel``.
    """

    def __init__(
        self,
        in_visual_dim=int(DIT_CONFIG["in_visual_dim"]),
        out_visual_dim=int(DIT_CONFIG["out_visual_dim"]),
        time_dim=int(DIT_CONFIG["time_dim"]),
        model_dim=int(DIT_CONFIG["model_dim"]),
        ff_dim=int(DIT_CONFIG["ff_dim"]),
        patch_size=tuple(int(value) for value in DIT_CONFIG["patch_size"]),
        num_visual_blocks=int(DIT_CONFIG["num_visual_blocks"]),
        axes_dims=tuple(int(value) for value in DIT_CONFIG["axes_dims"]),
        rope_scale_factor=tuple(float(value) for value in SR_COMMON["scale_factor"]),
        n_grid=1,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.out_visual_dim = out_visual_dim
        self.n_grid = int(n_grid)
        self.patch_size = patch_size
        self.rope_scale_factor = rope_scale_factor
        self.attention_config = dict(ATTENTION_CONFIG)
        self.num_heads = model_dim // head_dim
        self.head_dim = head_dim
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # The learned constant that replaces the pooled text embedding; it is a
        # bare parameter, so it keeps its checkpoint dtype under fp8 weights.
        self.pooled_bias = nn.Parameter(torch.zeros(time_dim, device=device, dtype=dtype))
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim, operation_settings=operation_settings)
        self.visual_embeddings = VisualEmbeddings(
            2 * in_visual_dim + 1, model_dim, patch_size, operation_settings=operation_settings
        )
        self.visual_transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(model_dim, time_dim, ff_dim, head_dim, operation_settings=operation_settings)
                for _ in range(num_visual_blocks)
            ]
        )
        self.out_layer = OutLayer(
            model_dim, time_dim, out_visual_dim * self.n_grid, patch_size, operation_settings=operation_settings
        )
        self.rope_embedder_3d = EmbedND(dim=head_dim, theta=10000.0, axes_dim=list(axes_dims))

    def rope_encode_3d(self, t, h, w, device=None, *, fractal=False):
        """Canonical RoPE3D positions ``index / scale_factor``, computed in fp32."""
        steps = [size // patch for size, patch in zip((t, h, w), self.patch_size, strict=True)]
        ids = torch.zeros((*steps, 3), device=device, dtype=torch.float32)
        for axis, (count, scale) in enumerate(zip(steps, self.rope_scale_factor, strict=True)):
            positions = torch.arange(count, device=device, dtype=torch.float32) / scale
            view = [1, 1, 1]
            view[axis] = count
            ids[..., axis] += positions.view(*view)
        if fractal:
            ids = fractal_flatten_grid(ids, tuple(steps))
        return self.rope_embedder_3d(ids.reshape(1, -1, 3)).movedim(1, 2)

    def forward(self, *args, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, kwargs.get("transformer_options", {})
            ),
        ).execute(*args, **kwargs)

    def _forward(self, x, timestep, context=None, transformer_options=None, **kwargs):
        transformer_options = dict(transformer_options or {})
        compute_dtype = x.dtype
        _, _, t_len, height, width = x.shape
        time_embed = _time_embeddings_fp32(self.time_embeddings, timestep) + self.pooled_bias.float()

        visual_embed = self.visual_embeddings(x)
        visual_shape = visual_embed.shape[:-1]
        token_shape = tuple(int(value) for value in visual_embed.shape[1:4])
        use_nabla = (
            x.device.type == "cuda"
            and self.attention_config.get("type") == "nabla"
            and not transformer_options.get("k6_vsr_disable_nabla", False)
        )
        if use_nabla:
            visual_embed = fractal_flatten_batch(visual_embed, token_shape)
            transformer_options["_k6_vsr_nabla"] = {
                "shape": token_shape,
                "config": self.attention_config,
            }
        else:
            visual_embed = visual_embed.flatten(1, -2)
        freqs = self.rope_encode_3d(t_len, height, width, device=x.device, fractal=use_nabla)

        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        transformer_options["total_blocks"] = len(self.visual_transformer_blocks)
        transformer_options["block_type"] = "double"
        for index, block in enumerate(self.visual_transformer_blocks):
            transformer_options["block_index"] = index
            if ("double_block", index) in blocks_replace:

                def block_wrap(args, _block=block):
                    return {
                        "x": _block(
                            args["x"],
                            args["time_embed"],
                            args["freqs"],
                            compute_dtype,
                            transformer_options=args.get("transformer_options"),
                        )
                    }

                visual_embed = blocks_replace[("double_block", index)](
                    {
                        "x": visual_embed,
                        "time_embed": time_embed,
                        "freqs": freqs,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )["x"]
            else:
                visual_embed = block(
                    visual_embed, time_embed, freqs, compute_dtype, transformer_options=transformer_options
                )

        if use_nabla:
            visual_embed = fractal_unflatten_batch(visual_embed, token_shape)
        else:
            visual_embed = visual_embed.reshape(*visual_shape, -1)
        shift, scale = torch.chunk(_modulation_fp32(self.out_layer.modulation, time_embed), 2, dim=-1)
        out = _scale_shift_norm_fp32(
            self.out_layer.norm, visual_embed, scale[:, None, None, None, :], shift[:, None, None, None, :]
        ).to(compute_dtype)
        out = self.out_layer.out_layer(out)
        out_dim = out.shape[-1] // math.prod(self.patch_size)
        return (
            out.view(*out.shape[:4], out_dim, *self.patch_size)
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .flatten(2, 3)
            .flatten(3, 4)
            .flatten(4, 5)
        )
