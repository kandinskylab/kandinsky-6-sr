"""ComfyUI supported-model registration of the Kandinsky 6 SR DiT."""

import comfy.supported_models_base
import torch

from . import model_base
from .checkpoint_keys import native_dit_state
from .sr_contract import EULER_SAMPLING, PIFLOW_SAMPLING

SR_IMAGE_MODEL = "kandinsky6_sr"


class Kandinsky6SR(comfy.supported_models_base.BASE):
    unet_config = {"image_model": SR_IMAGE_MODEL}
    unet_extra_config = {}
    latent_format = model_base.Kandinsky6SRLatent
    supported_inference_dtypes = [torch.bfloat16, torch.float32]
    memory_usage_factor = 1.0
    sampling_settings = {"shift": float(EULER_SAMPLING["scheduler_scale"])}

    def __init__(self, unet_config):
        super().__init__(unet_config)
        if int(unet_config.get("n_grid", 1)) > 1:
            self.sampling_settings = {"shift": float(PIFLOW_SAMPLING["shift"])}

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Kandinsky6SRModel(self, device=device)

    def process_unet_state_dict(self, state_dict):
        return native_dit_state(state_dict)
