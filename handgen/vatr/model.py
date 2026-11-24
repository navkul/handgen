"""Inference portion of VATr++ for the published IAM generator checkpoint.

Architecture and checkpoint names follow the published model (see README).
Training discriminators, recognizers and dataset loaders are unnecessary here.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from torchvision.models import resnet18

from ..paths import DEFAULT_DIFFBRUSH_ROOT


class GlyphDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, hidden, memory, query):
        # Unifont is a positional query on both attentions in each layer.
        residual = self.norm1(hidden)
        qk = residual + query
        hidden = hidden + self.self_attn(qk, qk, residual, need_weights=False)[0]
        hidden = hidden + self.multihead_attn(self.norm2(hidden) + query, memory, memory, need_weights=False)[0]
        return hidden + self.linear2(torch.relu(self.linear1(self.norm3(hidden))))


class InkConv(nn.Module):
    def __init__(self, channels, out, kernel, *, residual_end=False, output=False):
        super().__init__()
        self.conv = nn.Conv2d(channels, out, kernel, padding=kernel // 2, padding_mode="reflect")
        self.norm = nn.Identity() if output else nn.InstanceNorm2d(out)
        self.activation = nn.Tanh() if output else nn.Identity() if residual_end else nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class ResidualInk(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(InkConv(512, 512, 3), InkConv(512, 512, 3, residual_end=True))

    def forward(self, x):
        return x + self.model(x)


class InkStack(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WordGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.Feat_Encoder = resnet18(weights=None)
        self.Feat_Encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.Feat_Encoder.fc = self.Feat_Encoder.avgpool = nn.Identity()
        layer = nn.TransformerEncoderLayer(512, 8, 512, dropout=0, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(512), enable_nested_tensor=False)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([GlyphDecoderLayer(512, 8, 512, dropout=0) for _ in range(3)])
        self.decoder.norm = nn.LayerNorm(512)
        self.query_embed = nn.Module()
        self.query_embed.linear = nn.Linear(256, 512)
        self.linear_q = nn.Linear(512, 4096)

        blocks = [InkStack(ResidualInk(), ResidualInk())]
        for channels in (512, 256, 128):
            blocks += [nn.Upsample(scale_factor=2), InkConv(channels, channels // 2, 5)]
        blocks.append(InkConv(64, 1, 7, output=True))
        self.DEC = InkStack(*blocks)

    def encode_style(self, crops):
        features = self.Feat_Encoder(crops)
        # This reshape across reference examples matches the training code.
        return self.encoder(features.reshape(1, 512, -1).permute(2, 0, 1))

    def forward(self, memory, tiles):
        query = self.query_embed.linear(tiles)
        hidden = torch.zeros_like(query)
        for layer in self.decoder.layers:
            hidden = layer(hidden, memory, query)
        columns = self.linear_q(self.decoder.norm(hidden).transpose(0, 1))
        features = columns.reshape(columns.shape[0], columns.shape[1] * 2, 4, 512).permute(0, 3, 2, 1)
        return self.DEC(features)


class VATrModel:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device)
        self.net = WordGenerator()
        prefix = "model.netG."
        with safe_open(checkpoint, framework="pt") as weights:
            tensor_names = weights.keys()
            generator = {k[len(prefix):]: weights.get_tensor(k) for k in tensor_names
                         if k.startswith(prefix) and not k.startswith(prefix + "pos_encoder.")}
        self.net.load_state_dict(generator)
        self.net.to(self.device).eval().requires_grad_(False)
        with (DEFAULT_DIFFBRUSH_ROOT / "files" / "unifont.pickle").open("rb") as f:
            # Same bundled archetypes as DiffBrush. Keep just the prose alphabet.
            glyphs = {chr(g["idx"][0]): g["mat"].ravel() for g in pickle.load(f) if 32 <= g["idx"][0] <= 126}
        self.alphabet = glyphs.keys()
        self.tiles = {c: torch.as_tensor(g.copy(), device=self.device, dtype=torch.float32) for c, g in glyphs.items()}

    @torch.no_grad()
    def encode(self, words: list[Image.Image], seed: int):
        if not words:
            raise ValueError("VATr++ needs visible reference words. Register a natural handwriting line.")
        rng = np.random.default_rng(seed % (2**64))
        images = []
        for idx in rng.choice(len(words), 15, replace=len(words) < 15):
            source = words[idx].convert("L")
            w = max(1, round(source.width * 32 / source.height))
            word = source.resize((w, 32), Image.Resampling.BILINEAR)
            padded = Image.new("L", (192, 32), 255)
            padded.paste(word, (0, 0))
            images.append(np.asarray(padded))
        tensor = torch.from_numpy(np.stack(images)[:, None]).to(self.device, torch.float32) / 127.5 - 1
        return self.net.encode_style(tensor)

    def pixels(self, memory, word: str):
        unknown = sorted(set(word) - self.alphabet)
        if unknown:
            raise ValueError(f"VATr++ has no prose archetypes for {unknown}")
        query = torch.stack([self.tiles[c] for c in word])[:, None]
        return self.net(memory, query)[0, 0].clamp(-1, 1).add(1).mul(0.5)
