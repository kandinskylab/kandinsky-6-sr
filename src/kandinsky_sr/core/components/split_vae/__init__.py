"""Tiled KVAE used by ``vae_backend: split``.

Encode and decode run through a spatially tiled, ``torch.compile``d KVAE.
Fitted temporal kernels live in ``assets/``.
"""

from kandinsky_sr.core.components.split_vae.adapter import SplitSRVAE, build_split_sr_vae

__all__ = ("SplitSRVAE", "build_split_sr_vae")
