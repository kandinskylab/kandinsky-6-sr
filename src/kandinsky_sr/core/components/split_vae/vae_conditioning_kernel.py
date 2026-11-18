"""Triton gather and pointwise modulation; no full-size intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _modulate(X, Z, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
              ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
              OY: tl.constexpr, OX: tl.constexpr, FH: tl.constexpr, FW: tl.constexpr,
              XSN: tl.constexpr, XSC: tl.constexpr, XST: tl.constexpr,
              XSH: tl.constexpr, XSW: tl.constexpr,
              ZSN: tl.constexpr, ZSC: tl.constexpr, ZST: tl.constexpr,
              ZSH: tl.constexpr, ZSW: tl.constexpr, COUNT: tl.constexpr,
              CL: tl.constexpr, MULTIPLY: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if CL:
        c = i % C
        x = i // C % W
        y = i // (C * W) % H
        t = i // (C * W * H) % T
    else:
        x = i % W
        y = i // W % H
        t = i // (W * H) % T
        c = i // (W * H * T) % C
    n = i // (C * T * H * W)
    if ST > 1 and T > 1:
        zt = tl.where(t == 0, 0, (t - 1) * (ST - 1) // (T - 1) + 1)
    else:
        zt = tl.full((BLOCK,), 0, tl.int32)
    zy = (y + OY) * SH // FH
    zx = (x + OX) * SW // FW
    xp = n * XSN + c * XSC + t * XST + y * XSH + x * XSW
    zp = n * ZSN + c * ZSC + zt * ZST + zy * ZSH + zx * ZSW
    value = tl.load(X + xp, i < COUNT, other=0).to(tl.float32)
    conditioning = tl.load(Z + zp, i < COUNT, other=0).to(tl.float32)
    if MULTIPLY:
        result = value * conditioning
    else:
        result = value + conditioning
    tl.store(X + xp, result, i < COUNT)


def launch_modulation(output, projected, origin, full_hw, multiply):
    _, c, t, h, w = output.shape
    with torch.cuda.device(output.device):
        _modulate[(triton.cdiv(output.numel(), 256),)](
            output, projected, c, t, h, w, *projected.shape[-3:],
            *origin, *full_hw, *output.stride(), *projected.stride(), output.numel(),
            output.stride(1) == 1, multiply, 256, enable_fp_fusion=False)
