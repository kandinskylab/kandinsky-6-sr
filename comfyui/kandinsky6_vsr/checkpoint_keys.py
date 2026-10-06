"""Map published Diffusers parameter names to the ComfyUI-native layers."""


def _remap(state_dict, replacements):
    result = {}
    for original, tensor in state_dict.items():
        name = original
        for source, target in replacements:
            name = name.replace(source, target)
        if name in result:
            raise ValueError(f"Duplicate SR checkpoint parameter after remapping: {name}")
        result[name] = tensor
    return result


def native_dit_state(state_dict):
    """Rename keys without copying tensors or changing their dtype/device."""
    return _remap(
        state_dict,
        (
            ("time_embeddings.timestep_embedder.linear_1.", "time_embeddings.in_layer."),
            ("time_embeddings.timestep_embedder.linear_2.", "time_embeddings.out_layer."),
            (".feed_forward.net.0.proj.", ".feed_forward.in_layer."),
            (".feed_forward.net.2.", ".feed_forward.out_layer."),
        ),
    )


def native_lu_state(state_dict):
    """Retain the eager codec architecture while accepting published LU names."""
    return _remap(
        state_dict,
        (
            ("x2_branch.mid_blocks.", "x2_branch.private_mid_blocks."),
            ("x2_branch.upsample.", "x2_branch.private_upsample."),
            ("x2_branch.blocks.", "x2_branch.private_blocks."),
            ("x2_branch.output_proj.", "x2_branch.private_output_proj."),
            ("output_proj.norm.", "output_proj.0."),
            ("output_proj.conv.", "output_proj.2."),
        ),
    )
