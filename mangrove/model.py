import torch
from torch import nn
from torch.nn import functional as F
from mangrove import build_dpt_encoder
from .decoder import DPTHead1

class DPTHead(DPTHead1):
    
    pass

class DualDPT(nn.Module):
    def __init__(self, s2_arch="large", dem_arch="large", s2_channels=10,
                 use_coords=False, patch_size=16, registers=0, width=128,
                 gradient_checkpointing=True, decoder_checkpointing=False):
        super().__init__()
        if s2_channels != 10:
            raise ValueError("This model requires 10 Sentinel-2 input bands")
        if 512 % patch_size:
            raise ValueError("Patch size must divide 512")
        self.s2_encoder, self.s2_indices = build_dpt_encoder(s2_arch, s2_channels, patch_size, registers)
        self.dem_encoder, self.dem_indices = build_dpt_encoder(dem_arch, 3 if use_coords else 1, patch_size, registers)
        self.s2_decoder = DPTHead(self.s2_encoder.embed_dim, width)
        self.dem_decoder = DPTHead1(self.dem_encoder.embed_dim, width)
        self.head = nn.Sequential(nn.Conv2d(width, width // 2, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(width // 2, 1, 1))


        self.dem_encoder.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing = gradient_checkpointing
        self.decoder_checkpointing = decoder_checkpointing
        self.s2_encoder.requires_grad_(False)
        self.s2_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.s2_encoder.eval()
        return self

    def _decode(self, decoder, features):
        if self.decoder_checkpointing and self.training and torch.is_grad_enabled():
            flat = [t for pair in features for t in pair]
            if not any(t.requires_grad for t in flat):
                flat[0] = flat[0].detach().requires_grad_(True)
            amp = torch.is_autocast_enabled()
            def run(*values):
                with torch.cuda.amp.autocast(enabled=amp):
                    return decoder(tuple((values[i], values[i+1]) for i in range(0,8,2)))
            return torch.utils.checkpoint.checkpoint(run, *flat)
        return decoder(features)

    def forward(self, s2, dem, return_aux=False):
        if s2.shape[-2:] != (512, 512) or dem.shape[-2:] != (512, 512):
            raise ValueError("Expected 512 x 512 inputs")
        with torch.no_grad():
            a = self.s2_encoder.get_intermediate_layers(s2, n=self.s2_indices, reshape=True,
                                                        return_class_token=True, norm=False)
        dem_input = dem
        b = self.dem_encoder.get_intermediate_layers(dem_input, n=self.dem_indices, reshape=True,
                                                     return_class_token=True, norm=False)
        features = self._decode(self.s2_decoder, a) + self._decode(self.dem_decoder, b)
        logits = self.head(features)
        logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)

        prediction = F.softplus(logits.float())
        if return_aux:
            return {"prediction": prediction}
        return prediction
