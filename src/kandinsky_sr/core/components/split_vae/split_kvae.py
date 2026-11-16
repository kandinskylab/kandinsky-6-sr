import math
import os
import hashlib
import json
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from safetensors.torch import load_file as safetensors_load_file

import torch._inductor.config

DEFAULT_TEMPORAL_ASSETS = Path(__file__).resolve().parent / "assets"

def configure_split_compile():
    torch._inductor.config.combo_kernels = True
    # torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"
    # torch._inductor.config.max_autotune = True
    # torch._inductor.config.memory_planning = True
    # if hasattr(torch._inductor.config, "force_same_precision"):
    #     torch._inductor.config.force_same_precision = True
    # else:
    #     torch._inductor.config.emulate_precision_casts = True
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True


def cast_tuple(t: Union[int, Tuple[int]], length: int = 1) -> Tuple[int]:
    return t if isinstance(t, tuple) else ((t,) * length)


def get_n_tile_log(t_h_w: Tuple[int, int, int], max_channels: int, exact: bool) -> float:

    SPLIT_INFO = {1: (1, 2, 1, 2, 6, 0), 2: (2, 2, 2, 2, 12, 6)}
    T, H, W = t_h_w
    n_tiles_log = -33 + math.log2(T) + math.log2(H) + math.log2(W) + math.log2(max_channels)
    if n_tiles_log < 0:
        return n_tiles_log

    if W > H:
        H, W = W, H

    for _ in range(2):
   
        t_scale, h_scale, w_scale, ch_mult, offset_1, offset_2 = SPLIT_INFO[int(math.ceil(n_tiles_log))]
        hw_scale = h_scale * 2
        ch = max_channels * ch_mult
        if not exact:
            offset_1, offset_2 = 0, 0

        n1 = (
            math.log2((T - 1) // t_scale + 1) + math.log2(ch) 
            + math.log2(H) + math.log2(W) - 2 * math.log2(h_scale)
        )
        n2 = (
            math.log2(T) + math.log2(max_channels) + 2 * math.log2(hw_scale)
            + math.log2(math.ceil((H + 2 * offset_1 * (h_scale - 1)) / h_scale / hw_scale))
            + math.log2(math.ceil((W + 2 * offset_2 * (w_scale - 1)) / w_scale / hw_scale))
        )
        n_tiles_log = -33 + max((n1, n2)) + math.log2(h_scale) + math.log2(w_scale)

    return n_tiles_log


def get_offset(offset: int, idx: int, N: int, downscale: int = 1) -> Tuple[int, int, int]:
    offset_1l = -offset if idx > 0 else 0
    offset_1r = offset if idx < N-1 else 0
    offset_2 = (offset // downscale) if idx > 0 else 0
    return offset_1l, offset_1r, offset_2


def get_context(x: torch.Tensor) -> Dict:
    return {"dtype": x.dtype, "device": x.device, "memory_format": torch.channels_last_3d}


class CausalConv3D(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_size: Union[int, Tuple[int, int, int]], 
                 stride: Tuple[int, int, int] = (1,1,1), dilation: Tuple[int, int, int] = (1,1,1), 
                 padding_mode: str = "replicate", **kwargs):
        super().__init__()
        k_size = cast_tuple(k_size, 3)
        tks, hks, wks = k_size
        self.t_pad = tks - 1
        self.t_stride = stride[0]
        self.offset_out = math.ceil(self.t_pad / self.t_stride)        # forward on `padding_poisoned` should take exactly this range
        self.offset_in = self.offset_out * self.t_stride - self.t_pad  # to make forward on `input_parallel` take slice starting with this index 
        self.offset_padding = self.offset_in + tks - self.t_stride

        self.conv = nn.Conv3d(
            c_in, c_out, k_size, padding=(0, hks // 2, wks // 2), stride=stride,
            padding_mode=padding_mode, dilation=dilation, **kwargs
        )

    @torch.compile(dynamic=True, fullgraph=False)
    def pad(self, x: Tensor) -> Tensor:
        output_shape = list(x.shape)
        output_shape[2] += self.t_pad
        output = torch.empty(output_shape, **get_context(x))
        output[:, :, :self.t_pad] = x[:, :, :1]
        output[:, :, self.t_pad:] = x
        return output

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(self, x: Tensor) -> Tensor:

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

        return output


class SplittedRMSNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, bias: bool = False, device=None, dtype=None):
        super().__init__()
        self.scale = num_channels ** 0.5
        self.eps = eps
        shape = (1, num_channels, 1, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
    
    @torch.compile(dynamic=True, fullgraph=False)
    def normalize(self, x: Tensor) -> Tensor:
        normalized = F.normalize(x, dim=1, eps=self.eps) * self.scale
        normalized = normalized * self.gamma
        if self.bias is not None:
            normalized += self.bias
        return F.silu(normalized, inplace=True).to(dtype=x.dtype)  # normalized


class EffDownsampleV2(nn.Module):
    def __init__(self, c_in: int, compress_time: bool, factor: int = 2, padding_mode: str = "reflect"):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        c_out = 2 * c_in

        self.spatial_conv = nn.Conv3d(c_in, c_out, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                      padding=(0, 1, 1), padding_mode=padding_mode)

        if compress_time:
            self.temporal_conv = CausalConv3D(c_out, c_out, k_size=(3, 1, 1), stride=(2, 1, 1),
                                              padding_mode="zeros")

        self.linear = nn.Conv3d(c_out, c_out, kernel_size=1)

    def spatial(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        pxs_out = (
            x.reshape(b, c, t, h // self.factor, self.factor, w // self.factor, self.factor)
            .mean(dim=-1)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(b, c * self.factor, t, h // self.factor, w // self.factor)
        )

        conv_out = self.spatial_conv(x)
        return conv_out + pxs_out

    def temporal(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        if t == 1:
            return x

        full_interp = torch.empty((b, c, 1 + t // self.factor, h, w), **get_context(x))
        full_interp[:, :, :1] = x[:, :, :1]
        full_interp[:, :, 1:] = (
            x[:, :, 1:].view(b, c, t // self.factor, self.factor, h, w).mean(dim=3)
        )
        conv_out = self.temporal_conv(x)
        return conv_out + full_interp


class EffEncoderResnetBlock3D(nn.Module):
    def __init__(self, c_in: int, c_out: Optional[int] = None, padding_mode="replicate"):
        super().__init__()
        if c_out is None:
            c_out = c_in
        
        self.norm1 = SplittedRMSNorm(num_channels=c_in)
        self.norm2 = SplittedRMSNorm(num_channels=c_out)
        self.conv1 = CausalConv3D(c_in=c_in,  c_out=c_out, k_size=3, padding_mode=padding_mode)
        self.conv2 = CausalConv3D(c_in=c_out, c_out=c_out, k_size=3, padding_mode=padding_mode)
        
        if c_in != c_out:
            self.nin_shortcut = nn.Conv3d(c_in, c_out, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.norm1.normalize(x)
        y = self.conv1(y)
        y = self.norm2.normalize(y)
        y = self.conv2(y)

        return y + self.nin_shortcut(x)


class SplitEncoder3D(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: Tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: int = 3,
        z_channels: int = 48,
        resolution: int = 0,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
        fix_pxs: bool = True,
        norm_type: str = "rms_norm",
        downsample_version: int = 2,
        temporal_compress_start_level: int = 0,
        skip_last_resolution: bool = False,
        padding_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.skip_last_resolution = skip_last_resolution
        assert fix_pxs, "Currently doesn't support fix_pxs=False"

        self.downsample_version = downsample_version
        
        self.temporal_compress_level = int(np.log2(temporal_compress_times)) + temporal_compress_start_level

        in_ch_mult = (ch_mult[0],) + tuple(ch_mult)
        conv_padding = padding_mode or "replicate"
        self.conv_in = CausalConv3D(c_in=4, c_out=round(in_ch_mult[0] * self.ch), k_size=3, padding_mode=conv_padding)

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()

            block_in = round(ch * in_ch_mult[i_level])
            block_out = round(ch * ch_mult[i_level])

            if downsample_version > 1 and i_level > 0:
                block_in *= 2
            
            for i_block in range(self.num_res_blocks):
                block.append(EffEncoderResnetBlock3D(c_in=block_in, c_out=block_out, padding_mode=conv_padding))
                block_in = block_out
            
            down = nn.Module()
            down.block = block

            if i_level != self.num_resolutions - 1:
                compress_time = temporal_compress_start_level <= i_level < self.temporal_compress_level
                down.downsample = EffDownsampleV2(block_in, compress_time=compress_time, padding_mode=padding_mode)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in, padding_mode=conv_padding)
        self.mid.block_2 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in, padding_mode=conv_padding)
        self.norm_out = SplittedRMSNorm(num_channels=block_in)
        self.conv_out = CausalConv3D(
            c_in=block_in,
            c_out=2 * z_channels if double_z else z_channels,
            k_size=3,
            padding_mode=conv_padding,
        )
        self.z_channels = z_channels

    @torch.compile(dynamic=True, fullgraph=False)
    def piece_0(self, x: Tensor) -> Tensor:
        """conv_in + down[0] blocks + downsample"""
        x = x.to(memory_format=torch.channels_last_3d)
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))   # M / 10
        x = self.conv_in(x)                      # M / 2.5 (output numel)
        x = self.down[0].block[0](x)             # M / 2.5
        x = self.down[0].block[1](x)             # M / 2.5
        x = self.down[0].downsample.spatial(x)   # M / 5
        return x  # 256, T, H//2, W//2
    
    @torch.compile(dynamic=True, fullgraph=False)
    def piece_1(self, x: Tensor) -> Tensor:
        """down[1] blocks + spatial downsample → 512 channels"""
        x = self.down[0].downsample.linear(x)    # M / 5
        x = self.down[1].block[0](x)             # M (160 ch, HW / 2)
        x = self.down[1].block[1](x)             # M
        x = self.down[1].downsample.spatial(x)   # M / 2
        return x  # 512, (T-1)//2+1, H//4, W//4
    
    @torch.compile(dynamic=True, fullgraph=False)
    def piece_1a(self, x: Tensor) -> Tensor:
        """down[1] temporal"""
        x = self.down[1].downsample.temporal(x)  # M / 4
        return x  # 512, (T-1)//4+1, H//4, W//4    
    
    @torch.compile(dynamic=True, fullgraph=False)
    def piece_2(self, x: Tensor) -> Tensor:
        """Remaining levels + mid + output"""
        
        x = self.down[1].downsample.linear(x)    # M / 4
        x = self.down[2].block[0](x)             # M / 4
        x = self.down[2].block[1](x)             # M / 4
        x = self.down[2].downsample.spatial(x)   # M / 8

        x = self.down[2].downsample.temporal(x)  # M / 16
        
        x = self.down[2].downsample.linear(x)    # M / 16
        x = self.down[3].block[0](x)             # M / 16
        x = self.down[3].block[1](x)             # M / 16
        
        if self.num_resolutions > 4:
            x = self.down[3].downsample.spatial(x)   # M / 32
            x = self.down[3].downsample.linear(x)    # M / 32
        
            x = self.down[4].block[0](x)             # M / 32
            x = self.down[4].block[1](x)             # M / 32
        
        # middle
        x = self.mid.block_1(x)                  # M / 32
        x = self.mid.block_2(x)                  # M / 32
        
        # output
        x = self.norm_out.normalize(x)           # M / 32
        x = self.conv_out(x)[:, :self.z_channels]  # M / 320
        return x.contiguous()

    def forward_1(self, x: Tensor) -> Tensor:
        """No tiling."""
        x = self.piece_0(x)
        x = self.piece_1(x)
        x = self.piece_1a(x)
        x = self.piece_2(x)
        return x

    def forward_2(self, x: Tensor, h_split: int = 1, w_split: int = 2, offset_flag=False) -> Tensor:
        """2 tiles"""
        x = self.piece_0(x)
        _, _, T, H, W = x.shape
        offset = 6 if offset_flag else 0
        z = torch.empty((1, 2 * self.ch, T, H // 2, W // 2), **get_context(x))
        Hi, Wj = H // h_split, W // w_split
        Hi2, Wj2 = Hi // 2, Wj // 2
        for i in range(h_split):
            offset_l, offset_r, offset_h = get_offset(offset, i, h_split, 2)
            Hil, Hir, Hi2l, Hi2r = Hi * i + offset_l, Hi * i + Hi + offset_r, Hi2 * i, Hi2 * i + Hi2
            for j in range(w_split):
                offset_l, offset_r, offset_w = get_offset(offset, j, w_split, 2)
                Wjl, Wjr, Wj2l, Wj2r = Wj * j + offset_l, Wj * j + Wj + offset_r, Wj2 * j, Wj2 * j + Wj2
                y = self.piece_1(x[[0], :, :, Hil:Hir, Wjl:Wjr])
                z[0, :, :, Hi2l:Hi2r, Wj2l:Wj2r] = y[:, :, :, offset_h:Hi2 + offset_h, offset_w:Wj2 + offset_w]
        x = self.piece_1a(z)
        x = self.piece_2(x)
        return x

    def forward_4(self, x: Tensor, offset_flag=False) -> Tensor:
        """4 tiles (2hx2w grid)."""
        ## NOTE: offset_1, offset_2 = (12, 6) or (16, 6)
        _, _, T, H, W = x.shape
        offset_1, offset_2 = (12, 6) if offset_flag else (0, 0)
        z = torch.empty((1, 2 * self.ch, (T-1) // 2 + 1, H // 4, W // 4), **get_context(x))
        H2, H8, W4, W8 = H // 2, H // 8, W // 4, W // 8
        for i in range(2):
            offset_l, offset_r, offset_h = get_offset(offset_1, i, 2, 4)
            h2l, h2r, h8l, h8r = H2 * i + offset_l, H2 * i + H2 + offset_r, H8 * i, H8 * i + H8
            y0 = self.piece_0(x[[0], :, :, h2l:h2r, :])
            for j in range(2):
                offset_l, offset_r, offset_w = get_offset(offset_2, j, 2, 2)
                w4l, w4r, w8l, w8r = W4 * j + offset_l, W4 * j + W4 + offset_r, W8 * j, W8 * j + W8
                y = self.piece_1(y0[:, :, :, :, w4l:w4r])
                y = y[:, :, :, offset_h:H8 + offset_h, offset_w:W8 + offset_w]
                z[0, :, :, h8l:h8r, w8l:w8r] = self.piece_1a(y)
        x = self.piece_2(z)
        return x
    

    def forward_4w(self, x: Tensor, offset_flag=False) -> Tensor:
        """4 tiles (2wx2h grid)."""
        ## NOTE: offset_1, offset_2 = (12, 6) or (16, 6)
        _, _, T, H, W = x.shape
        offset_1, offset_2 = (12, 6) if offset_flag else (0, 0)
        z = torch.empty((1, 2 * self.ch, (T-1) // 2 + 1, H // 4, W // 4), **get_context(x))
        H4, H8, W2, W8 = H // 4, H // 8, W // 2, W // 8
        for j in range(2):
            offset_l, offset_r, offset_w = get_offset(offset_1, j, 2, 4)
            w2l, w2r, w8l, w8r = W2 * j + offset_l, W2 * j + W2 + offset_r, W8 * j, W8 * j + W8
            y0 = self.piece_0(x[[0], :, :, :, w2l:w2r])
            for i in range(2):
                offset_l, offset_r, offset_h = get_offset(offset_2, i, 2, 2)
                h4l, h4r, h8l, h8r = H4 * i + offset_l, H4 * i + H4 + offset_r, H8 * i, H8 * i + H8
                y = self.piece_1(y0[:, :, :, h4l:h4r, :])
                y = y[:, :, :, offset_h:H8 + offset_h, offset_w:W8 + offset_w]
                z[0, :, :, h8l:h8r, w8l:w8r] = self.piece_1a(y)
        x = self.piece_2(z)
        return x


class SplitVAE(nn.Module):
    """Efficient VAE for 4x16x16 compression with tiled encoding."""
    
    def __init__(self, encoder_conf, decoder_conf, ckpt_path=None, temporal_weights_dir=None, 
                 decoder_max_tensor_elements=None, decoder_tile_grid=None,
                 decoder_conditioning_order="project-first", decoder_piece1_grid=None, decoder_backend="torch"):
        super().__init__()
        self.decoder_backend = decoder_backend
        self.decoder_max_tensor_elements = decoder_max_tensor_elements
        self.decoder_tile_grid = tuple(decoder_tile_grid) if decoder_tile_grid is not None else None
        self.decoder_piece1_grid = tuple(decoder_piece1_grid) if decoder_piece1_grid is not None else None
        self.temporal_weights_dir = Path(temporal_weights_dir) if temporal_weights_dir is not None else DEFAULT_TEMPORAL_ASSETS
        self.encoder = SplitEncoder3D(**encoder_conf)
        if decoder_backend == "magi":
            from .split_decoder_magi import SplitDecoder3D
        else:
            from .split_decoder import SplitDecoder3D
        self.decoder = SplitDecoder3D(**decoder_conf)
        self.temporal_compress_times = encoder_conf.get('temporal_compress_times', 4)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.encoder = nn.utils.convert_conv3d_weight_memory_format(self.encoder, torch.channels_last_3d)
        self.model_size = sum(p.numel() for p in self.parameters()) * 2

    def init_from_ckpt(self, path):
        """Load checkpoint weights and fitted temporal kernels."""
        source = safetensors_load_file(str(path), device="cpu")
        target = self.state_dict()
        sd = {}
        for key, value in source.items():
            if key.startswith("loss."):
                continue
            if key.startswith("encoder.") and "temporal_conv" in key:
                continue
            if key.startswith("decoder.") and key not in target:
                key = key.replace(".conv_b.conv", ".conv_b").replace(".conv_y.conv", ".conv_y")
            if key == "encoder.conv_in.conv.weight" and value.shape[1] == 3:
                value = F.pad(value, (0, 0, 0, 0, 0, 0, 0, 1))
            if key.startswith("encoder.") and "norm" in key and value.ndim == 4:
                value = value.unsqueeze(0)
            elif key.startswith("decoder.") and ".norm_layer." in key and value.ndim == 1:
                value = value.view(-1, 1, 1, 1)
            sd[key] = value
        fitted_paths = []
        for level in (1, 2):
            prefix = f"encoder.down.{level}.downsample.temporal_conv."
            channels = self.encoder.down[level].downsample.temporal_conv.conv.out_channels
            fitted_path = self.temporal_weights_dir / f"t4hw16ch{channels}.pt"
            metadata_path = fitted_path.with_suffix(".json")
            if metadata_path.is_file():
                digest = hashlib.sha256()
                for key in sorted(key for key in source if key.startswith(prefix)):
                    digest.update(key.removeprefix(prefix).encode())
                    digest.update(source[key].detach().contiguous().view(torch.uint8).numpy().tobytes())
                metadata = json.loads(metadata_path.read_text())
                if metadata["source_weights_sha256"] != digest.hexdigest():
                    raise ValueError(f"Fitted temporal weights do not match checkpoint: {fitted_path}")
            weights = torch.load(fitted_path, map_location="cpu", weights_only=True)
            for key, value in weights.items():
                sd[prefix + key] = value
            fitted_paths.append(str(fitted_path))
        self.load_state_dict(sd, strict=True, assign=True)
        self.encoder.to(memory_format=torch.channels_last_3d)
        if self.decoder_backend == "magi":
            from .split_decoder_magi import configure_decoder_layout
            checkpoint = Path(path).resolve()
            stat = checkpoint.stat()
            self.decoder.magi_checkpoint_identity = (str(checkpoint), stat.st_size, stat.st_mtime_ns)
        else:
            from .split_decoder import configure_decoder_layout
        configure_decoder_layout(self.decoder, True)
        self.weight_load_report = {
            "checkpoint": str(path), "strict": True, "loaded_tensors": len(sd),
            "temporal_weights": "fitted_assets",
            "temporal_version": "v2", "fitted_asset_paths": fitted_paths,
        }

    @staticmethod
    def normalize_data(data):
        return data / 128 - 1.0

    @staticmethod
    def denormalize_data(data):
        return (data + 1) * 128

    def make_empty_cache(self, block: str):
        def make_dict(name):
            if name == 'conv':
                return {'padding': None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean': None, 'var': None}
                else:
                    return {'norm': make_dict('norm_enc'), 'add_conv': make_dict('conv')}
            elif layer == 'resblock':
                return {
                    'norm1': make_dict(f'norm_{module}'),
                    'norm2': make_dict(f'norm_{module}'),
                    'conv1': make_dict('conv'),
                    'conv2': make_dict('conv'),
                    'conv_shortcut': make_dict('conv')
                }
            elif layer.isdigit():
                return {
                    0: make_dict(f'resblock_{module}'),
                    1: make_dict(f'resblock_{module}'),
                    2: make_dict(f'resblock_{module}'),
                    'down': [make_dict('conv'), make_dict('conv')],
                    'up': make_dict('conv')
                }

        cache = {
            'conv_in': make_dict('conv'),
            'mid_1': make_dict(f'resblock_{block}'),
            'mid_2': make_dict(f'resblock_{block}'),
            'norm_out': make_dict(f'norm_{block}'),
            'conv_out': make_dict('conv'),
            0: make_dict(f'0_{block}'),
            1: make_dict(f'1_{block}'),
            2: make_dict(f'2_{block}'),
            3: make_dict(f'3_{block}'),
            4: make_dict(f'4_{block}'),
        }

        return cache

    def encode(self, x: Tensor, exact: bool = True) -> Tensor:
        N, _, T, H, W = x.shape
        if N > 1:
            raise Exception("Batch size greater than 1 is not supported.")
        # For 4x16x16 with downsample_version=2, channels double at each level
        n_tiles_log = get_n_tile_log((T, H, W), self.encoder.ch, exact)

        with torch._dynamo.utils.disable_cache_limit():
            if n_tiles_log <= 0:
                out = self.encoder.forward_1(x)

            elif n_tiles_log <= 1:
                h_split, w_split = (2, 1) if H >= W else (1, 2)
                out = self.encoder.forward_2(x, h_split, w_split, offset_flag=exact)

            elif n_tiles_log <= 2:
                if H >= W:
                    out = self.encoder.forward_4(x, offset_flag=exact)
                else:
                    out = self.encoder.forward_4w(x, offset_flag=exact)

            else:
                raise NotImplementedError("forward_8 is not implemented yet")
            
            return out

    def decode(self, z: Tensor, seg_len=16) -> Tensor:
        return self.decoder(z)


    def forward(self, x: Tensor, exact: bool = False) -> Tensor:
        latent = self.encode(x, exact)
        recs = self.decode(latent)
        return recs


def inference_dtype(conf):
    name = getattr(conf, "dtype", "bfloat16")
    choices = {"float32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
    if name not in choices:
        raise ValueError(f"Unsupported VAE inference dtype: {name}")
    return choices[name]


def build_split_kvae(conf):
    """Build the split VAE from a native KVAE sidecar prefix."""
    from omegaconf import OmegaConf
    from kandinsky_sr.core.components.model.vae_io import kvae_weights_path

    configure_split_compile()
    config = OmegaConf.load(f"{conf.checkpoint_path}.yaml")
    architecture = config if "encoder_params" in config else config.model
    with torch.device("meta"):
        vae = SplitVAE(
            encoder_conf=architecture.encoder_params,
            decoder_conf=architecture.decoder_params,
            ckpt_path=kvae_weights_path(str(conf.checkpoint_path)),
            temporal_weights_dir=getattr(conf, "split_weights_dir", None),
            decoder_backend=getattr(conf, "decoder_backend", "torch"),
            decoder_max_tensor_elements=getattr(conf, "decoder_max_tensor_elements", None),
            decoder_tile_grid=getattr(conf, "decoder_tile_grid", None),
            decoder_piece1_grid=getattr(conf, "decoder_piece1_grid", None)
        )
    vae = vae.eval().to(dtype=inference_dtype(conf))
    vae.config = config
    return vae
