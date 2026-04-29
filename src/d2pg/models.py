from __future__ import annotations

import torch
from torch import nn


class CorrectionMLP(nn.Module):
    def __init__(self, in_dim: int = 6, out_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiffusionCorrectionMLP(nn.Module):
    """Tiny conditional denoiser for correction vectors.

    This is intentionally small: the goal is a first feasibility signal, not a
    production diffusion model.
    """

    def __init__(self, cond_dim: int = 6, target_dim: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + target_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, noisy: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([noisy, t, cond], dim=-1))

