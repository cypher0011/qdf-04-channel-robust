"""Frozen multilingual self-supervised speech encoder.

mHuBERT-147 (Boito et al., Interspeech 2024) is a 95M-parameter HuBERT model
pre-trained on 90k hours of speech in 147 languages, Arabic included. We keep
it frozen and use it as a feature extractor. That makes the detector cheap to
train (minutes on a laptop) and stops it from forgetting what multilingual
speech sounds like while it learns about fakes.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "utter-project/mHuBERT-147"
N_LAYERS = 13          # CNN output + 12 transformer layers
DIM = 768
FPS = 50               # 320-sample stride at 16 kHz -> one frame every 20 ms


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class Backbone:
    def __init__(self, dev: torch.device | None = None):
        self.dev = dev or device()
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.dev).eval()

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:  # Wav2Vec2FeatureExtractor(do_normalize=True)
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)

    @torch.no_grad()
    def hidden_states(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, T) float -> (B, 13, frames, 768)"""
        out = self.model(self._norm(wav.to(self.dev)), output_hidden_states=True)
        return torch.stack(out.hidden_states, dim=1)

    @torch.no_grad()
    def pooled(self, wav: torch.Tensor) -> torch.Tensor:
        """Mean + std over time for every layer -> (B, 13, 1536)."""
        hs = self.hidden_states(wav)
        return torch.cat([hs.mean(dim=2), hs.std(dim=2)], dim=-1)


def int16_to_float(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32) / 32767.0)
