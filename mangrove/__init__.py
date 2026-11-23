from . import backbone as vits

ARCH = {"small": (384, 12, 6, [2, 5, 8, 11]),
        "base": (768, 12, 12, [2, 5, 8, 11]),
        "large": (1024, 24, 16, [4, 11, 17, 23])}


def build_dpt_encoder(arch, channels, patch_size, registers):
    dim, depth, heads, indices = ARCH[arch]
    encoder = vits.DinoVisionTransformer(
        img_size=512, patch_size=patch_size, in_chans=channels,
        embed_dim=dim, depth=depth, num_heads=heads, mlp_ratio=4,
        init_values=1e-5, 
        num_register_tokens=registers, embed_layer=vits.PatchEmbed,
        block_fn=vits.partial(vits.Block, attn_class=vits.MemEffAttention),
    )
    return encoder, indices
