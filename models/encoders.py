"""
Visual and language encoders for the Instruct-to-Act controller.

Visual encoder: 4-block CNN (channels [depth, 2d, 4d, 8d], stride 2, SiLU)
  matching DreamerV3 default with CLIP/DINOv2 optional features.

Language encoder: MiniLM-L6-H384 (frozen sentence-transformer, 384-dim).
  A learned null-embedding stands in when no instruction is active.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# CNN visual encoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """4-layer strided CNN as used in DreamerV3 / Instruct-to-Act.

    App A.3: "4 convolutional blocks; channels [32,64,128,256], stride 2, SiLU"
    """

    def __init__(self, obs_channels: int = 3, depth: int = 32, embed_size: int = 1024):
        super().__init__()
        channels = [obs_channels] + [depth * (2 ** i) for i in range(4)]  # [3,32,64,128,256]
        layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=0), nn.SiLU()]
        self.cnn = nn.Sequential(*layers)
        # Compute flattened size dynamically (assumes 64×64 input)
        dummy = torch.zeros(1, obs_channels, 64, 64)
        flat = self.cnn(dummy).view(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat, embed_size), nn.SiLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W) or (B, H, W, C)
        if obs.shape[-1] in (1, 3):
            obs = obs.permute(0, 3, 1, 2)
        obs = obs.float() / 255.0
        x = self.cnn(obs)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Language encoder (frozen MiniLM)
# ---------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """Frozen MiniLM-L6-H384 sentence encoder + learned null embedding.

    At train time the encoder is frozen; only null_embed is learnable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lang_size: int = 384,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.lang_size = lang_size
        self._device = device or torch.device("cpu")

        # Lazy-load to avoid import overhead at module import time
        self._model = None
        self._model_name = model_name

        # Learned null embedding (used when no instruction is available)
        self.null_embed = nn.Parameter(torch.zeros(lang_size))

    # ------------------------------------------------------------------
    # Lazy model initialization
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=str(self._device))
                # Freeze all parameters
                for p in self._model.parameters():
                    p.requires_grad_(False)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of strings → (N, lang_size) tensor (no gradient)."""
        self._load_model()
        embeddings = self._model.encode(
            texts, convert_to_tensor=True, device=str(self._device), show_progress_bar=False
        )
        return embeddings.to(self._device)

    def embed_instructions(self, texts: List[Optional[str]]) -> torch.Tensor:
        """Encode instructions; use null_embed where text is None/empty."""
        device = self.null_embed.device
        result = []
        non_null = [(i, t) for i, t in enumerate(texts) if t]
        if non_null:
            indices, valid_texts = zip(*non_null)
            encoded = self.encode(list(valid_texts)).to(device)
        else:
            indices, encoded = [], torch.zeros(0, self.lang_size, device=device)

        for i in range(len(texts)):
            if i in dict(zip(indices, range(len(indices)))):
                result.append(encoded[dict(zip(indices, range(len(indices))))[i]])
            else:
                result.append(self.null_embed)
        return torch.stack(result)

    def embed_batch(self, texts: List[Optional[str]]) -> torch.Tensor:
        """Batch encode; returns (B, lang_size). None/empty → null_embed."""
        device = self.null_embed.device
        embeds = []
        for t in texts:
            if t:
                e = self.encode([t])[0].to(device)
            else:
                e = self.null_embed
            embeds.append(e)
        return torch.stack(embeds)

    def embed_single(self, text: Optional[str]) -> torch.Tensor:
        """Encode a single string (or return null_embed)."""
        if text:
            return self.encode([text])[0].to(self.null_embed.device)
        return self.null_embed
