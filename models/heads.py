"""
Decoder and policy heads for the Instruct-to-Act controller.

Following App A.3:
  - Actor / value heads: 3-layer MLP, 512 units, SiLU
  - Decoders: 2-layer MLP, SiLU + transposed-CNN for observations
  - Stop head: small MLP → sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent, OneHotCategorical


def _mlp(in_size: int, units: int, depth: int, out_size: int) -> nn.Sequential:
    layers: list = []
    cur = in_size
    for _ in range(depth):
        layers += [nn.Linear(cur, units), nn.SiLU()]
        cur = units
    layers.append(nn.Linear(cur, out_size))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Policy / Actor
# ---------------------------------------------------------------------------

class ActorHead(nn.Module):
    """
    π(a | s_t, e_t) = Softmax(ℓ(s_t, e_t) / τ)_a   (discrete)
    or a diagonal Gaussian (continuous).

    App A.3: 3-layer MLP, 512 units, SiLU.
    """

    def __init__(
        self,
        state_size: int,
        lang_size: int,
        action_size: int,
        action_type: str = "discrete",
        units: int = 512,
        depth: int = 3,
        temperature: float = 1.0,
        entropy_scale: float = 1e-3,
    ):
        super().__init__()
        self.action_type = action_type
        self.temperature = temperature
        self.entropy_scale = entropy_scale

        out_size = action_size if action_type == "discrete" else 2 * action_size
        self.net = _mlp(state_size + lang_size, units, depth - 1, out_size)

    def forward(self, state_feat: torch.Tensor, lang_embed: torch.Tensor):
        logits = self.net(torch.cat([state_feat, lang_embed], dim=-1))
        if self.action_type == "discrete":
            return OneHotCategorical(logits=logits / self.temperature)
        else:
            mean, log_std = logits.chunk(2, dim=-1)
            std = F.softplus(log_std) + 0.1
            return Independent(Normal(mean, std), 1)

    def sample(self, state_feat: torch.Tensor, lang_embed: torch.Tensor, greedy: bool = False):
        dist = self(state_feat, lang_embed)
        if greedy:
            if self.action_type == "discrete":
                return dist.probs.argmax(-1)
            else:
                return dist.base_dist.mean
        return dist.sample()


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """V_ψ(s_t, e_t) scalar critic. Same MLP depth as actor."""

    def __init__(self, state_size: int, lang_size: int, units: int = 512, depth: int = 3):
        super().__init__()
        self.net = _mlp(state_size + lang_size, units, depth - 1, 1)

    def forward(self, state_feat: torch.Tensor, lang_embed: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state_feat, lang_embed], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Stop / completion head
# ---------------------------------------------------------------------------

class StopHead(nn.Module):
    """
    p^stop_t = σ(g(s_t, e_t))

    Predicts whether the current instruction has been completed.
    """

    def __init__(self, state_size: int, lang_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + lang_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state_feat: torch.Tensor, lang_embed: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probability (B,) in [0, 1]."""
        logit = self.net(torch.cat([state_feat, lang_embed], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Observation decoder  (transposed CNN)
# ---------------------------------------------------------------------------

class ObsDecoder(nn.Module):
    """Reconstruct observation from latent state.

    Uses a linear projection + transposed-CNN mirroring the encoder.
    """

    def __init__(
        self,
        state_size: int,
        obs_channels: int = 3,
        depth: int = 32,
        units: int = 512,
    ):
        super().__init__()
        # Project state to spatial seed
        self.proj = nn.Sequential(nn.Linear(state_size, units), nn.SiLU())
        # 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 (roughly)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(units, depth * 8, 5, 2),  nn.SiLU(),
            nn.ConvTranspose2d(depth * 8, depth * 4, 5, 2), nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 6, 2), nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth,     6, 2), nn.SiLU(),
            nn.ConvTranspose2d(depth, obs_channels,  2, 1),
        )

    def forward(self, state_feat: torch.Tensor) -> Independent:
        B = state_feat.shape[0]
        x = self.proj(state_feat).view(B, -1, 1, 1)
        mean = self.deconv(x)  # (B, C, H, W)
        return Independent(Normal(mean, torch.ones_like(mean)), 3)


# ---------------------------------------------------------------------------
# Reward decoder
# ---------------------------------------------------------------------------

class RewardDecoder(nn.Module):
    """p_θ(r_t | s_t) — scalar Gaussian reward."""

    def __init__(self, state_size: int, units: int = 512, depth: int = 2):
        super().__init__()
        self.net = _mlp(state_size, units, depth - 1, 1)

    def forward(self, state_feat: torch.Tensor) -> Normal:
        mean = self.net(state_feat).squeeze(-1)
        return Normal(mean, torch.ones_like(mean))


# ---------------------------------------------------------------------------
# Continuation / discount decoder
# ---------------------------------------------------------------------------

class ContinuationDecoder(nn.Module):
    """p_θ(c_t | s_t) — Bernoulli continuation (not done)."""

    def __init__(self, state_size: int, units: int = 512, depth: int = 2):
        super().__init__()
        self.net = _mlp(state_size, units, depth - 1, 1)

    def forward(self, state_feat: torch.Tensor) -> Bernoulli:
        logit = self.net(state_feat).squeeze(-1)
        return Bernoulli(logits=logit)


# ---------------------------------------------------------------------------
# Next-instruction predictor  (teacher-forced on x_{t+1})
# ---------------------------------------------------------------------------

class NextInstructionHead(nn.Module):
    """p_θ(x_{t+1} | s_t) — predict the embedding of the next instruction.

    Only trained when instruction label m_{t+1} = 1 (i.e., next step has a label).
    """

    def __init__(self, state_size: int, lang_size: int, units: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, units), nn.SiLU(),
            nn.Linear(units, lang_size),
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        """Returns predicted instruction embedding (B, lang_size)."""
        return self.net(state_feat)
