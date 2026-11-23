import torch
from torch import nn
from torch.nn import functional as F

class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.skip = ResidualConv(channels)
        self.out = ResidualConv(channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x, skip=None, size=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.skip(skip)
        x = self.out(x)
        x = F.interpolate(x, size=size, scale_factor=None if size is not None else 2,
                          mode="bilinear", align_corners=False)
        return self.project(x)


class DPTHead1(nn.Module):
    
    def __init__(self, embed_dim, width=128):
        super().__init__()
        self.readouts = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU()) for _ in range(4)])
        self.projects = nn.ModuleList([nn.Conv2d(embed_dim, width, 1) for _ in range(4)])
        self.resize = nn.ModuleList([
            nn.ConvTranspose2d(width, width, 4, stride=4),
            nn.ConvTranspose2d(width, width, 2, stride=2), nn.Identity(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
        ])
        self.fusions = nn.ModuleList([FusionBlock(width) for _ in range(4)])
        self.project = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Each DPT branch requires four feature levels")
        pyramid = []
        for i, (feature, cls) in enumerate(features):
            b, c, h, w = feature.shape
            tokens = feature.flatten(2).transpose(1, 2)
            tokens = self.readouts[i](torch.cat([tokens, cls[:, None].expand(-1, h * w, -1)], dim=-1))
            feature = tokens.transpose(1, 2).reshape(b, c, h, w)
            pyramid.append(self.resize[i](self.projects[i](feature)))
        x = self.fusions[0](pyramid[3], size=pyramid[2].shape[-2:])
        x = self.fusions[1](x, pyramid[2], size=pyramid[1].shape[-2:])
        x = self.fusions[2](x, pyramid[1], size=pyramid[0].shape[-2:])
        x = self.fusions[3](x, pyramid[0], size=(256, 256))
        return self.project(x)
