"""Stateless decoder with locally tiled expansion/reduction pieces."""

import math
from functools import lru_cache
from typing import Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .vae_fused_norm import fused_spatial_norm
from .vae_resize import spatial_nearest_2x, temporal_nearest_2x, require_channels_last_3d


class Piece3Plan(NamedTuple):
    input_slice: Tuple[int, int, int, int]
    steps: tuple

def cast_tuple(t: Union[int, Tuple[int]], length: int = 1) -> Tuple[int]:
    return t if isinstance(t, tuple) else ((t,) * length)


def get_context(x: torch.Tensor) -> Dict:
    return {"dtype": x.dtype, "device": x.device, "memory_format": torch.channels_last_3d}


def configure_decoder_layout(decoder, enabled):
    for module in decoder.modules():
        if isinstance(module, (CausalConv3D, SplitPXSUpsample)):
            module.preserve_layout = enabled
        elif isinstance(module, (RMSNorm, SplitSpatialNorm3D, SplitResnetBlock3D, SplitDecoder3D)):
            module.preserve_layout = False
    if enabled:
        nn.utils.convert_conv3d_weight_memory_format(decoder, torch.channels_last_3d)


class CausalConv3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_size: Union[int, Tuple[int, int, int]], 
                 stride: Tuple[int, int, int] = (1,1,1), dilation: Tuple[int, int, int] = (1,1,1), 
                 padding_mode: str = "replicate", **kwargs):
        super().__init__()
        self.preserve_layout = False
        k_size = cast_tuple(k_size, 3)
        tks, hks, wks = k_size
        self.t_pad = tks - 1
        self.t_stride = stride[0]
        self.offset_out = math.ceil(self.t_pad / self.t_stride)        # forward on `padding_poisoned` should take exactly this range
        self.offset_in = self.offset_out * self.t_stride - self.t_pad  # to make forward on `input_parallel` take slice starting with this index 
        self.offset_padding = self.offset_in + tks - self.t_stride

        self.conv = nn.Conv3d(
            c_in, c_out, k_size, padding=(0, hks // 2, wks // 2), stride=stride,
            padding_mode=padding_mode or "replicate", dilation=dilation, **kwargs
        )

    @torch.compile(dynamic=True, fullgraph=False)
    def pad(self, x: Tensor) -> Tensor:
        output_shape = list(x.shape)
        output_shape[2] += self.t_pad
        output = torch.empty(output_shape, **get_context(x))
        output[:, :, :self.t_pad] = x[:, :, :1]
        output[:, :, self.t_pad:] = x
        return require_channels_last_3d(output) if self.preserve_layout else output

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, x: Tensor) -> Tensor:
        if self.preserve_layout:
            x = require_channels_last_3d(x)
        out_size = list(x.shape)
        out_size[1] = self.conv.out_channels
        if self.t_stride == 2:
            out_size[2] = (out_size[2] + 1) // 2
        output = torch.empty(out_size, **get_context(x))

        if self.offset_out > 0:
            padding_poisoned = self.pad(x[:, :, :self.offset_padding])
            output[:, :, :self.offset_out] = self.conv(padding_poisoned)

        if self.offset_out < output.size(2):
            output[:, :, self.offset_out:] = self.conv(x[:, :, self.offset_in:])

        return require_channels_last_3d(output) if self.preserve_layout else output


class RMSNorm(nn.Module):  # SplittedRMSNorm
    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = False, device=None, dtype=None):
        super().__init__()
        self.preserve_layout = False
        self.scale = num_channels ** 0.5
        self.eps = eps
        shape = (num_channels, 1, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
    
    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, x: Tensor, *, inplace=False) -> Tensor:
        if self.preserve_layout:
            x = require_channels_last_3d(x)
        x = F.normalize(x, dim=1, eps=self.eps)
        x.mul_(self.scale).mul_(self.gamma[None])
        if self.bias is not None:
            x.add_(self.bias[None])
        return require_channels_last_3d(x) if self.preserve_layout else x
    

class SplitSpatialNorm3D(nn.Module):
    def __init__(
        self, f_channels, zq_channels, normalization=RMSNorm, **norm_layer_params,
):
        super().__init__()
        self.preserve_layout = False
        self.norm_layer = normalization(f_channels, **norm_layer_params)
        self.conv_y = nn.Conv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = nn.Conv3d(zq_channels, f_channels, kernel_size=1)

    def forward(self, x, z, origin=(0, 0), full_hw=None, *, inplace=False, activation="none"):
        if self.preserve_layout:
            x, z = require_channels_last_3d(x), require_channels_last_3d(z)
        full_hw = (x.shape[-2], x.shape[-1]) if full_hw is None else (full_hw[0], full_hw[1])
        return fused_spatial_norm(
            x, self.norm_layer.gamma, self.norm_layer.bias, self.conv_y(z), self.conv_b(z),
            origin, full_hw, eps=self.norm_layer.eps, scale=self.norm_layer.scale,
            activation=activation)


class SplitResnetBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: Optional[int] = None, zq_ch=None, normalization=RMSNorm, padding_mode="replicate"):
        super().__init__()
        self.preserve_layout = False
        if c_out is None:
            c_out = c_in
        
        self.norm1 = SplitSpatialNorm3D(c_in, zq_ch, normalization)
        self.norm2 = SplitSpatialNorm3D(c_out, zq_ch, normalization)
        
        self.conv1 = CausalConv3D(c_in=c_in,  c_out=c_out, k_size=3, padding_mode=padding_mode)
        self.conv2 = CausalConv3D(c_in=c_out, c_out=c_out, k_size=3, padding_mode=padding_mode)
        
        if c_in != c_out:
            self.nin_shortcut = nn.Conv3d(c_in, c_out, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: Tensor, z, origin=(0, 0), full_hw=None) -> Tensor:
        if self.preserve_layout:
            x, z = require_channels_last_3d(x), require_channels_last_3d(z)
        full_hw = (x.shape[-2], x.shape[-1]) if full_hw is None else (full_hw[0], full_hw[1])
        shortcut = self.nin_shortcut(x)
        if self.preserve_layout:
            shortcut = require_channels_last_3d(shortcut)
        h = self.norm1(x, z, origin, full_hw, activation="silu")
        h = self.conv1(h)
        h = self.norm2(h, z, origin, full_hw, activation="silu")
        h = self.conv2(h)
        output = h + shortcut
        return require_channels_last_3d(output) if self.preserve_layout else output


class SplitPXSUpsample(nn.Module):
    def __init__(self, c_in: int, compress_time: bool, padding_mode=None):
        super().__init__()
        self.preserve_layout = False
        self.temporal_compress = compress_time
        self.spatial_conv = nn.Conv3d(c_in, c_in, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                                      padding_mode=padding_mode or 'reflect')
        self.linear = nn.Conv3d(c_in, c_in, kernel_size=1)
        if self.temporal_compress:
            self.temporal_conv = CausalConv3D(c_in, c_in, k_size=(3, 1, 1), padding_mode="zeros")

    def temporal(self, x):
        x = temporal_nearest_2x(x) if self.preserve_layout else x.repeat_interleave(2, dim=2)[:, :, 1:]
        if self.preserve_layout:
            x = require_channels_last_3d(x)
        return self.temporal_conv(x).add_(x)

    def spatial(self, x):
        x = spatial_nearest_2x(x)
        if self.preserve_layout:
            x = require_channels_last_3d(x)
        
        x.add_(self.spatial_conv(x))
        x.copy_(self.linear(x))
        return x

    def forward(self, x):
        if self.temporal_compress:
            x = self.temporal(x)
        x = self.spatial(x)
        return require_channels_last_3d(x) if self.preserve_layout else x


class SplitDecoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=None,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        resamp_with_conv=True,
        resolution=0,
        z_channels=16,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        padding_mode=None,
        temporal_compress_times=4,
        gather_norm=False,
        norm_type="group_norm",
        temporal_compress_start_level=0,
        **kwargs,
    ):
        super().__init__()
        self.preserve_layout = False
        if norm_type != "rms_norm":
            raise ValueError("SplitDecoder3D requires RMSNorm for spatial tiling")
        if add_conv or give_pre_end:
            raise ValueError("SplitDecoder3D does not support add_conv or give_pre_end")
        if len(ch_mult) != 5:
            raise ValueError("The fixed decoder plan requires five resolution levels")
        self.temporal_compress_times = temporal_compress_times
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # log2 of temporal_compress_times
        temporal_compress_level = int(math.log2(temporal_compress_times)) + temporal_compress_start_level

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = round(ch * ch_mult[self.num_resolutions - 1])
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = CausalConv3D(z_channels, block_in, k_size=3, padding_mode=padding_mode)
        modulated_norm = RMSNorm

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SplitResnetBlock3D(
            block_in, block_in, zq_ch=zq_ch, normalization=modulated_norm, padding_mode=padding_mode
        )

        self.mid.block_2 = SplitResnetBlock3D(
            block_in, block_in, zq_ch=zq_ch, normalization=modulated_norm, padding_mode=padding_mode
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = round(ch * ch_mult[i_level])
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    SplitResnetBlock3D(
                        block_in, block_out, zq_ch=zq_ch, normalization=modulated_norm, padding_mode=padding_mode
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if self.num_resolutions - temporal_compress_start_level > i_level >= self.num_resolutions - temporal_compress_level:
                    up.upsample = SplitPXSUpsample(block_in, compress_time=True, padding_mode=padding_mode) 
                else:
                    up.upsample = SplitPXSUpsample(block_in, compress_time=False, padding_mode=padding_mode)
            self.up.insert(0, up)

        self.norm_out = SplitSpatialNorm3D(block_in, zq_ch, normalization=modulated_norm)
        self.conv_out = CausalConv3D(block_in, out_ch, k_size=3, padding_mode=padding_mode)
        configure_decoder_layout(self, True)


    @torch.compile(dynamic=True, fullgraph=False)
    def piece_012(self, z):
        if self.preserve_layout:
            z = require_channels_last_3d(z)
        h = self.conv_in(z)
        h = self.mid.block_1(h, z)
        h = self.mid.block_2(h, z)

        h = self.up[4].block[0](h, z)
        h = self.up[4].block[1](h, z)
        h = self.up[4].block[2](h, z)
        h = self.up[4].upsample(h)

        h = self.up[3].block[0](h, z)
        h = self.up[3].block[1](h, z)
        h = self.up[3].block[2](h, z)
        h = self.up[3].upsample(h)

        h = self.up[2].block[0](h, z)
        h = self.up[2].block[1](h, z)
        h = self.up[2].block[2](h, z)

        return h
    
    def piece_3(self, x, z):
        N, _, T, H, W = x.shape
        offset = 5
        for level in (2, 1):
            if self.up[level].upsample.temporal_compress:
                T = 2 * T - 1
        channels = self.up[0].block[0].conv2.conv.out_channels
        output = torch.empty((N, channels, T, 4 * H, 4 * W), **get_context(x))
        for i in range(2):
            y0, y1 = i * H // 2, (i + 1) * H // 2
            if y0 == y1:
                continue
            sy0, sy1 = max(0, y0 - offset), min(H, y1 + offset)
            for j in range(4):
                x0, x1 = j * W // 4, (j + 1) * W // 4
                if x0 == x1:
                    continue
                sx0, sx1 = max(0, x0 - offset), min(W, x1 + offset)
                tile = x[:, :, :, sy0:sy1, sx0:sx1]
                tile = self.piece_3a(tile, z, (sy0, sx0), (H, W),
                                     core=(y0, y1, x0, x1))
                output[:, :, :, 4 * y0:4 * y1, 4 * x0:4 * x1] = tile
                del tile
        return output

    @staticmethod
    @lru_cache(maxsize=128)
    def _block_plan(input_hw, origin, full_hw, core):
        H, W = full_hw
        oy, ox = origin
        if core is None:
            core = (oy, min(H, oy + input_hw[0]), ox, min(W, ox + input_hw[1]))
        y0, y1, x0, x1 = core
        if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
            raise ValueError(f"Core {core} lies outside full size {full_hw}")

        def relative(box, offset, shape):
            a, b, c, d = box
            y, x = offset
            if not (y <= a < b <= y + shape[0] and x <= c < d <= x + shape[1]):
                raise ValueError(f"Crop {box} lies outside tile at {offset} with shape {shape}")
            return a - y, b - y, c - x, d - x

        initial = (max(0, y0 - 5), min(H, y1 + 5), max(0, x0 - 5), min(W, x1 + 5))
        selected = relative(initial, origin, input_hw)
        size = (2 * (initial[1] - initial[0]), 2 * (initial[3] - initial[2]))
        offset = (2 * initial[0], 2 * initial[2])
        steps = []
        # Remaining spatial radii after the three up.1 residual blocks.
        for radius in (6, 4, 2):
            box = (max(0, 2 * y0 - radius), min(2 * H, 2 * y1 + radius),
                   max(0, 2 * x0 - radius), min(2 * W, 2 * x1 + radius))
            crop = relative(box, offset, size)
            steps.append((offset, (2 * H, 2 * W), crop))
            offset = (box[0], box[2])
            size = (box[1] - box[0], box[3] - box[2])
        offset, size = (2 * offset[0], 2 * offset[1]), (2 * size[0], 2 * size[1])
        final = relative((4 * y0, 4 * y1, 4 * x0, 4 * x1), offset, size)
        steps.append((offset, (4 * H, 4 * W), final))
        return Piece3Plan(selected, tuple(steps))

    def piece_3a(self, h, z, origin, full_hw, core=None):
        plan = self._block_plan(
            (int(h.shape[-2]), int(h.shape[-1])), tuple(map(int, origin)),
            tuple(map(int, full_hw)), None if core is None else tuple(map(int, core)))
        y0, y1, x0, x1 = plan.input_slice
        h = h[..., y0:y1, x0:x1]
        for dimension in (0, 1, 2):
            torch._dynamo.mark_static(h, dimension)
            torch._dynamo.mark_static(z, dimension)
        return self._piece_3a_tensor(h, z, plan.steps)

    @torch.compile(dynamic=True, fullgraph=False)
    def _piece_3a_tensor(self, h, z, steps):
        if self.preserve_layout:
            h, z = require_channels_last_3d(h), require_channels_last_3d(z)
        h = self.up[2].upsample(h)
        origin, full_hw, crop = steps[0]
        h = self.up[1].block[0](h, z, origin, full_hw)
        h = h[..., crop[0]:crop[1], crop[2]:crop[3]]
        origin, full_hw, crop = steps[1]
        h = self.up[1].block[1](h, z, origin, full_hw)
        h = h[..., crop[0]:crop[1], crop[2]:crop[3]]
        origin, full_hw, crop = steps[2]
        h = self.up[1].block[2](h, z, origin, full_hw)
        h = h[..., crop[0]:crop[1], crop[2]:crop[3]]
        h = self.up[1].upsample(h)
        origin, full_hw, crop = steps[3]
        h = self.up[0].block[0](h, z, origin, full_hw)
        return h[..., crop[0]:crop[1], crop[2]:crop[3]]

    @torch.compile(dynamic=True, fullgraph=False)
    def piece_4(self, h, z):
        if self.preserve_layout:
            h, z = require_channels_last_3d(h), require_channels_last_3d(z)
        h = self.up[0].block[1](h, z)
        h = self.up[0].block[2](h, z)
        h = self.norm_out(h, z, activation="native")
        h = self.conv_out(h)
        
        return h

    def forward(self, z):
        self.last_z_shape = tuple(z.shape)
        if self.preserve_layout:
            z = require_channels_last_3d(z)
        for dimension in (0, 1, 2):
            torch._dynamo.mark_static(z, dimension)

        h = self.piece_012(z)
        h = self.piece_3(h, z)
        for dimension in (0, 1, 2):
            torch._dynamo.mark_static(h, dimension)
        h = self.piece_4(h, z)

        return h
    
