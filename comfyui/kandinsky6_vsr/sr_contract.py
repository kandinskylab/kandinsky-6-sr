"""Kandinsky 6 SR contract for ComfyUI.

Release constants for the standalone ComfyUI package. Maintain them alongside
the checkpoint configs and native adapter tests.
"""

# Architecture shared by every released SR DiT checkpoint.
DIT_CONFIG = {'axes_dims': (16, 24, 24),
 'ff_dim': 7168,
 'in_visual_dim': 64,
 'model_dim': 1792,
 'num_visual_blocks': 32,
 'out_visual_dim': 64,
 'patch_size': (1, 1, 1),
 'time_dim': 512,
 'use_text': False,
 'visual_cond': True}

# Released 512px inference attention. Only flex attention is regionally compiled.
ATTENTION_CONFIG = {'P': 0.8, 'add_sta': True, 'method': 'topcdf', 'type': 'nabla', 'wH': 7, 'wT': 11, 'wW': 7}

# ``load_sr_components`` replaces the trained instruct type: denoising starts
# from the degraded upscaled latent, the conditioning channels stay zero.
INSTRUCT_TYPE = 'noise'

SR_COMMON = {'fps': 24, 'scale_factor': (1.0, 2.0, 2.0), 'visual_size': 512}

# Sampling of the flow-matching checkpoint (Kandinsky-6.0-VSR-5s).
EULER_SAMPLING = {'lq_noise_scale': 0.7, 'lq_noise_type': 'ddpm', 'scheduler_scale': 5.0}

# Sampling of the distilled pi-Flow checkpoint (Kandinsky-6.0-VSR-distilled2steps-5s).
PIFLOW_SAMPLING = {'eps': 1e-06,
 'final_step_size_scale': 0.5,
 'lq_noise_scale': 0.7,
 'lq_noise_type': 'ddpm',
 'n_grid': 10,
 'nfe': 2,
 'num_policy_substeps': 128,
 'scheduler_scale': 3.5,
 'shift': 3.5}

VAE = {'name': 'video-kvae', 'spatial_factor': 16, 'temporal_factor': 4}

RUN_DEFAULTS = {'num_steps': 5, 'overlap': 0.2, 'seed': 42, 'tiles_batch_size': 1}

RESOLUTIONS = {512: ((512, 512), (512, 768), (768, 512))}
TARGET_FPS = 24
MAX_NUM_FRAMES = 121
RESAMPLE_FPS_TOLERANCE = 1.5
RESIZE_FRAME_CHUNK = 8
TARGET_RESOLUTIONS = {'2k': ((1440, 2560), (1440, 1440), (2560, 1440)),
 'fullhd': ((1080, 1920), (1080, 1080), (1920, 1080)),
 'hd': ((720, 1280), (720, 720), (1280, 720))}

# Deliberate target choices, not claims about the canonical runtime.
COMFY_FEATURES = {'magi_vae': False, 'sparse_attention': True, 'torch_compile': False}
