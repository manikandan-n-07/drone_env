"""
rl/model.py
PyTorch model definitions: MapEncoder CNN + PathQNet MLP.
"""
from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
CELL2IDX = {"road": 0, "building": 1, "tree": 2, "obstacle": 3, "delivery": 4, "done_del": 5, "drone": 6}


class MapEncoder(nn.Module):
    """Encodes a flat cell-type grid to a latent embedding using a mini-CNN."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(len(CELL2IDX) + 1, 8)  # +1 for unknown
        self.conv = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, cell_ids: torch.Tensor) -> torch.Tensor:
        B = cell_ids.size(0)
        x = self.embed(cell_ids)          # (B, H*W, 8)
        x = x.permute(0, 2, 1)           # (B, 8, H*W)
        side = int(math.ceil(math.sqrt(x.size(-1))))
        pad_len = side * side - x.size(-1)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        x = x.reshape(B, 8, side, side)
        x = self.conv(x)
        x = x.reshape(B, -1)
        return F.relu(self.fc(x))


class PathQNet(nn.Module):
    """
    Maps (map_embed, drone_xy_norm, battery, target_xy_norm) → Q(a) for 5 actions.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        in_dim = embed_dim + 5   # embed + 2(drone) + 1(battery) + 2(target)
        self.map_encoder = MapEncoder(embed_dim=embed_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, len(ACTIONS)),
        )

    def forward(
        self,
        cell_ids: torch.Tensor,    # (B, H*W)
        drone_xy: torch.Tensor,    # (B, 2) normalised
        battery: torch.Tensor,     # (B, 1)
        target_xy: torch.Tensor,   # (B, 2) normalised
    ) -> torch.Tensor:
        map_emb = self.map_encoder(cell_ids)
        state = torch.cat([map_emb, drone_xy, battery, target_xy], dim=-1)
        return self.net(state)
