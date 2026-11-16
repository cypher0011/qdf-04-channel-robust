"""Light back-end on top of the frozen encoder.

Each of the 13 layers of the encoder captures something different. Lower
layers stay close to acoustics and vocoder artefacts, higher layers are closer
to phonetics and language. Instead of picking one layer by hand, the head
learns a softmax weight per layer (as in the SUPERB benchmark), then an MLP
classifies the weighted statistics. The learned weights also show which depth
of the network "knows" about deepfakes (README, Figure 3).

Inputs are standardised per dimension with training-set statistics first.
Without it the high-variance upper layers dominate the weighted sum
(ablation in the README: 3.0 % -> 0.7 % EER on ArA-DF dev).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerWeightedHead(nn.Module):
    def __init__(self, n_layers: int = 13, dim: int = 1536, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        # per-dimension standardisation, fitted on the training set (set_norm)
        self.register_buffer("mu", torch.zeros(1, n_layers, dim))
        self.register_buffer("sd", torch.ones(1, n_layers, dim))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def set_norm(self, X) -> "LayerWeightedHead":
        X = torch.as_tensor(X, dtype=torch.float32)
        self.mu.copy_(X.mean(0, keepdim=True)); self.sd.copy_(X.std(0, keepdim=True) + 1e-5)
        return self

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, D)
        x = (x - self.mu) / self.sd
        h = (self.layer_weights()[None, :, None] * x).sum(1)
        return self.mlp(self.norm(h)).squeeze(1)          # logit, higher = more fake
