"""ComfyUI model type of the Kandinsky 6 SR DiT."""

import comfy.latent_formats
import comfy.model_base
import torch

from .ldm.model import Kandinsky6SR
from .sr_contract import DIT_CONFIG


class Kandinsky6SRLatent(comfy.latent_formats.LatentFormat):
    """KVAE latent; the SR nodes apply the KVAE scaling factor themselves."""

    latent_channels = int(DIT_CONFIG["in_visual_dim"])
    latent_dimensions = 3


class Kandinsky6SRModel(comfy.model_base.BaseModel):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=Kandinsky6SR)

    def process_timestep(self, timestep, **kwargs):
        # The canonical Euler sampler keeps its state in the model dtype and
        # passes ``t * 1000`` cast to it; reproduce that rounding.
        return timestep.to(self.get_dtype_inference()).float()

    def concat_cond(self, **kwargs):
        # The released checkpoints run with instruct type "noise": the
        # conditioning latent and its mask channel are zero.
        noise = kwargs["noise"]
        return torch.zeros(
            (noise.shape[0], noise.shape[1] + 1, *noise.shape[2:]), device=kwargs["device"], dtype=noise.dtype
        )
