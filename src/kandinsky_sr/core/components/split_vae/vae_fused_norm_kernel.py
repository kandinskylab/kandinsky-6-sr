"""Channel-reduction and gather/modulation without expanded intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm(X, G, B, Y, Z, O,
                C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                OY: tl.constexpr, OX: tl.constexpr, FH: tl.constexpr, FW: tl.constexpr,
                XN: tl.constexpr, XC: tl.constexpr, XT: tl.constexpr,
                XH: tl.constexpr, XW: tl.constexpr,
                YN: tl.constexpr, YC: tl.constexpr, YT: tl.constexpr,
                YH: tl.constexpr, YW: tl.constexpr,
                ZN: tl.constexpr, ZC: tl.constexpr, ZT: tl.constexpr,
                ZH: tl.constexpr, ZW: tl.constexpr,
                PIXELS: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr,
                HAS_BIAS: tl.constexpr, ACTIVATION: tl.constexpr,
                BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    c = tl.arange(0, BLOCK_C)
    ix, iy = p % W, p // W % H
    it, batch = p // (W * H) % T, p // (W * H * T)
    mask = (p[:, None] < PIXELS) & (c[None, :] < C)
    offsets = batch * XN + it * XT + iy * XH + ix * XW
    value = tl.load(X + offsets[:, None] + c[None, :] * XC, mask, other=0).to(tl.float32)
    dtype = O.dtype.element_ty
    denominator = tl.sqrt(tl.sum(value * value, axis=1)).to(dtype).to(tl.float32)
    denominator = tl.maximum(denominator, EPS).to(dtype).to(tl.float32)
    norm = (value / denominator[:, None]).to(dtype).to(tl.float32)
    norm = (norm * SCALE).to(dtype).to(tl.float32)
    gamma = tl.load(G + c, c < C, other=0).to(tl.float32)
    norm = (norm * gamma[None, :]).to(dtype).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(B + c, c < C, other=0).to(tl.float32)
        norm = (norm + bias[None, :]).to(dtype).to(tl.float32)
    if ST > 1 and T > 1:
        zt = tl.where(it == 0, 0, (it - 1) * (ST - 1) // (T - 1) + 1)
    else:
        zt = tl.full((BLOCK_P,), 0, tl.int32)
    zy, zx = (iy + OY) * SH // FH, (ix + OX) * SW // FW
    yo = batch * YN + zt * YT + zy * YH + zx * YW
    zo = batch * ZN + zt * ZT + zy * ZH + zx * ZW
    scale = tl.load(Y + yo[:, None] + c[None, :] * YC, mask, other=0).to(tl.float32)
    shift = tl.load(Z + zo[:, None] + c[None, :] * ZC, mask, other=0).to(tl.float32)
    norm = (norm * scale).to(dtype).to(tl.float32)
    norm = (norm + shift).to(dtype).to(tl.float32)
    if ACTIVATION != 0:
        sigmoid = 1.0 / (1.0 + tl.exp(-norm))
        if ACTIVATION == 2:
            sigmoid = sigmoid.to(dtype).to(tl.float32)
        norm = norm * sigmoid
    tl.store(O + p[:, None] * C + c[None, :], norm, mask)


def launch_fused_norm(x, gamma, bias, projected_y, projected_b, origin, full_hw,
                      eps, scale, activation):
    n, c, t, h, w = x.shape
    output = torch.empty((n, c, t, h, w), device=x.device, dtype=x.dtype,
                         memory_format=torch.channels_last_3d)
    block_c = triton.next_power_of_2(c)
    block_p = min(32, max(1, 2048 // block_c))
    with torch.cuda.device(x.device):
        _fused_norm[(triton.cdiv(n * t * h * w, block_p),)](
            x, gamma, gamma if bias is None else bias, projected_y, projected_b, output,
            c, t, h, w, *projected_y.shape[-3:], *origin, *full_hw,
            *x.stride(), *projected_y.stride(), *projected_b.stride(),
            n * t * h * w, eps, scale, bias is not None,
            0 if activation == "none" else (1 if activation == "silu" else 2),
            block_p, block_c, num_warps=4, enable_fp_fusion=False)
    return output
