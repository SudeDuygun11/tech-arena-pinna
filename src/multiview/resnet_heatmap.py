"""ResNet18 encoder + heatmap decoder, for per-view 2D landmark localisation.

WHY RESNET18 IS WRITTEN OUT BY HAND
-----------------------------------
torchvision is not installed, and installing it would pull a torch version that
does not match this machine's `torch 2.9.0+cu128` CUDA build -- pip resolves to
torchvision 0.28, which pairs with torch ~2.13. Replacing a working CUDA install
mid-project is not worth the convenience, so the architecture is defined here
and the official ImageNet `state_dict` (weights/resnet18.pth) loads straight
into it. Layer names match torchvision's exactly, which is what makes that work.

WHY PRETRAINED AT ALL
---------------------
This is the main reason multi-view is worth building. Every model in this
project so far is trained from scratch on 400 ears, and the pipeline has been
underfit at every scale tested. ImageNet weights bring in edge, ridge and
shading-gradient filters learned from a million images -- outside data, which no
amount of squeezing the 400 ears can substitute for.

FIVE INPUT CHANNELS, NOT THREE
------------------------------
Input is depth + normal(3) + signed curvature. The pretrained stem expects RGB,
so conv1's weights are averaged across the colour axis and tiled to 5 channels,
rescaled by 3/5 to preserve activation magnitude. This is the standard way to
repurpose an ImageNet stem for non-RGB input.

The curvature channel carries the cue found empirically for landmark 0: measured
over 40 ears, smoothed curvature at the landmark is 0.042 +- 0.665 (a sign
change) against -2.085 at landmark 6, and the landmark sits 0.688mm from the
nearest zero crossing where random nearby surface sits 2.013mm away.

DECODER
-------
A small U-Net-style upsampling path with skip connections back to the encoder
stages, producing a heatmap at 1/4 input resolution. Position is decoded by
soft-argmax rather than taking the arg-max pixel, so the prediction is
continuous: at 256px the frame is 0.419mm/pixel, and hard pixel snapping would
throw away accuracy we measured we can keep (round-trip floor 0.285mm).


ROLE IN THE PIPELINE
--------------------
Reading order : 19 of 20   (multiview)
Duty          : ResNet18 encoder plus heatmap decoder for per-view 2D localisation.

Written out by hand so the official ImageNet state_dict loads without
installing torchvision, which would replace this machine's CUDA torch build.
conv1 is adapted from 3 to 5 channels by averaging the colour axis and
rescaling.

Known issues / status:
  ImageNet pretraining is the main justification for the multi-view direction
  and is still unablated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(cin, cout, stride)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cout, cout)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(out + idt)


class ResNet18Encoder(nn.Module):
    """Layer names deliberately mirror torchvision so the official
    state_dict loads without remapping."""

    def __init__(self, in_channels: int = 5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))       # 1/2
        c1 = x
        x = self.maxpool(x)
        c2 = self.layer1(x)                           # 1/4
        c3 = self.layer2(c2)                          # 1/8
        c4 = self.layer3(c3)                          # 1/16
        c5 = self.layer4(c4)                          # 1/32
        return c1, c2, c3, c4, c5


def load_pretrained(encoder: ResNet18Encoder, path: str, in_channels: int = 5,
                    verbose: bool = True):
    """Load ImageNet weights, adapting conv1 from 3 channels to `in_channels`.

    conv1's pretrained kernel is (64,3,7,7). Averaging over the colour axis
    gives the "mean filter" response, which is then tiled across the new
    channels and rescaled by 3/in_channels so the summed activation magnitude
    matches what the following BatchNorm statistics expect.
    """
    sd = torch.load(path, map_location="cpu")
    w = sd["conv1.weight"]
    if in_channels != 3:
        mean_w = w.mean(dim=1, keepdim=True)                       # (64,1,7,7)
        sd["conv1.weight"] = mean_w.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}   # drop classifier
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if verbose:
        print(f"  loaded ImageNet weights: {len(sd)} tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
    return encoder


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HeatmapNet(nn.Module):
    """Per-view heatmap for `n_landmarks` points, at 1/4 input resolution."""

    def __init__(self, n_landmarks: int = 1, in_channels: int = 5,
                 pretrained: str | None = None, width: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)
        if pretrained:
            load_pretrained(self.encoder, pretrained, in_channels)
        self.up4 = UpBlock(512, 256, width * 2)
        self.up3 = UpBlock(width * 2, 128, width)
        self.up2 = UpBlock(width, 64, width)
        self.head = nn.Conv2d(width, n_landmarks, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.encoder(x)
        d = self.up4(c5, c4)
        d = self.up3(d, c3)
        d = self.up2(d, c2)                # 1/4 of input
        return self.head(d)                # (B, L, H/4, W/4) logits


def soft_argmax(logits, temperature: float = 1.0):
    """Continuous (x, y) per landmark, in units of the HEATMAP grid.

    Soft-argmax rather than hard arg-max: at 256px the frame is 0.419mm per
    pixel and the heatmap is quarter resolution, so snapping to a grid cell
    would discard roughly 1.7mm of precision. The measured round-trip floor is
    0.285mm, and this is what keeps it.
    """
    B, L, H, W = logits.shape
    p = F.softmax(logits.reshape(B, L, -1) / temperature, dim=-1).reshape(B, L, H, W)
    xs = torch.arange(W, device=logits.device, dtype=logits.dtype)
    ys = torch.arange(H, device=logits.device, dtype=logits.dtype)
    x = (p.sum(dim=2) * xs).sum(dim=-1)
    y = (p.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1), p


def gaussian_target(px, py, H, W, sigma: float = 2.0, device=None):
    """Gaussian heatmap target centred on a sub-pixel location."""
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    d2 = (xs - px[..., None, None]) ** 2 + (ys - py[..., None, None]) ** 2
    g = torch.exp(-d2 / (2 * sigma ** 2))
    return g / (g.sum(dim=(-2, -1), keepdim=True) + 1e-12)
