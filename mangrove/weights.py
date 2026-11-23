import torch

def load_checkpoint(path):

    return torch.load(str(path), map_location="cpu")


def load_s2_weights(encoder, path, key="teacher"):
    import math
    from torch.nn import functional as F
    state = load_checkpoint(path)
    if key and key in state:
        state = state[key]
    elif "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Expected a checkpoint state dictionary")
    cleaned = {}
    expected = encoder.state_dict()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        while name.startswith("module.") or name.startswith("backbone."):
            name = name.split(".", 1)[1]
        if name in expected:
            cleaned[name] = value

    if "pos_embed" in cleaned and cleaned["pos_embed"].shape != expected["pos_embed"].shape:
        pos = cleaned["pos_embed"]
        old_n, new_n = pos.shape[1] - 1, expected["pos_embed"].shape[1] - 1
        old_side, new_side = int(math.sqrt(old_n)), int(math.sqrt(new_n))
        if old_side ** 2 != old_n or new_side ** 2 != new_n or pos.shape[2] != encoder.embed_dim:
            raise ValueError("Incompatible positional embeddings")
        patch = pos[:, 1:].reshape(1, old_side, old_side, -1).permute(0, 3, 1, 2).float()
        patch = F.interpolate(patch, size=(new_side, new_side), mode="bicubic", align_corners=False)
        cleaned["pos_embed"] = torch.cat([pos[:, :1], patch.flatten(2).transpose(1, 2)], dim=1)
    missing = sorted(set(expected) - set(cleaned))
    if missing:
        raise ValueError(f"Missing Sentinel-2 backbone parameters: {missing[:12]}")
    patch_key = "patch_embed.proj.weight"
    if patch_key in cleaned and cleaned[patch_key].shape != expected[patch_key].shape:
        raise ValueError("Incompatible teacher patch embedding: "
                         "input channels and encoder dimensions must match")
    encoder.load_state_dict(cleaned, strict=True)
