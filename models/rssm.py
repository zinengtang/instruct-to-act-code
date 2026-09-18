"""
Recurrent State Space Model (DreamerV3-style) with language conditioning.

State s_t = (h_t, z_t):
  h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1}, e_{t-1})   deterministic GRU
  p_θ(z_t | h_t)                                    prior (Gaussian)
  q_φ(z_t | h_t, o_t, e_t)                          posterior (Gaussian)

Language embedding e_t conditions both the GRU input and the posterior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class RSSMState:
    h: torch.Tensor   # deterministic state  (B, deter)
    z: torch.Tensor   # stochastic state     (B, stoch)
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None


class RSSM(nn.Module):
    """Language-conditioned RSSM matching the Instruct-to-Act controller architecture."""

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        hidden_size: int,
        embed_size: int,
        lang_size: int,
        action_size: int,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.lang_size = lang_size

        # Project language embedding to match GRU input expectations
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_size, hidden_size), nn.SiLU()
        )

        # GRU: input = [z_{t-1} || a_{t-1} || lang_proj(e_{t-1})]
        gru_input_size = stoch_size + action_size + hidden_size
        self.gru = nn.GRUCell(gru_input_size, deter_size)

        # Prior: p_θ(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )

        # Posterior: q_φ(z_t | h_t, obs_embed, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + embed_size + lang_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 2 * stoch_size),
        )

        self.min_std = min_std

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            h=torch.zeros(batch_size, self.deter_size, device=device),
            z=torch.zeros(batch_size, self.stoch_size, device=device),
        )

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        """Return s_t = cat(h_t, z_t) used by decoder/policy heads."""
        return torch.cat([state.h, state.z], dim=-1)

    @property
    def feat_size(self) -> int:
        return self.deter_size + self.stoch_size

    # ------------------------------------------------------------------
    # Core transitions
    # ------------------------------------------------------------------

    def _prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.prior_net(h)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def _posterior(
        self, h: torch.Tensor, obs_embed: torch.Tensor, lang_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([h, obs_embed, lang_embed], dim=-1)
        out = self.posterior_net(inp)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
        lang_embed: torch.Tensor,
    ) -> Tuple[RSSMState, Dict, Dict]:
        """One posterior step (uses real observation)."""
        e_proj = self.lang_proj(lang_embed)
        gru_in = torch.cat([prev_state.z, prev_action, e_proj], dim=-1)
        h = self.gru(gru_in, prev_state.h)

        z_prior, prior_mean, prior_std = self._prior(h)
        z_post, post_mean, post_std = self._posterior(h, obs_embed, lang_embed)

        state = RSSMState(h=h, z=z_post, mean=post_mean, std=post_std)
        prior_stats = {"mean": prior_mean, "std": prior_std}
        post_stats = {"mean": post_mean, "std": post_std}
        return state, prior_stats, post_stats

    def imagine_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        lang_embed: torch.Tensor,
    ) -> Tuple[RSSMState, Dict]:
        """One prior step (no observation — used in imagination)."""
        e_proj = self.lang_proj(lang_embed)
        gru_in = torch.cat([prev_state.z, prev_action, e_proj], dim=-1)
        h = self.gru(gru_in, prev_state.h)

        z, mean, std = self._prior(h)
        state = RSSMState(h=h, z=z, mean=mean, std=std)
        prior_stats = {"mean": mean, "std": std}
        return state, prior_stats

    # ------------------------------------------------------------------
    # Sequence processing
    # ------------------------------------------------------------------

    def observe_sequence(
        self,
        obs_embeds: torch.Tensor,       # (T, B, embed)
        actions: torch.Tensor,           # (T, B, action)
        lang_embeds: torch.Tensor,       # (T, B, lang)
        initial_state: Optional[RSSMState] = None,
    ) -> Tuple[list, list, list]:
        """Process a full sequence; returns lists of (state, prior_stats, post_stats)."""
        B = obs_embeds.shape[1]
        device = obs_embeds.device
        state = initial_state or self.initial_state(B, device)

        states, prior_stats_list, post_stats_list = [], [], []
        for t in range(obs_embeds.shape[0]):
            state, prior_stats, post_stats = self.observe_step(
                state, actions[t], obs_embeds[t], lang_embeds[t]
            )
            states.append(state)
            prior_stats_list.append(prior_stats)
            post_stats_list.append(post_stats)

        return states, prior_stats_list, post_stats_list

    def imagine_sequence(
        self,
        start_state: RSSMState,
        actions: torch.Tensor,          # (H, B, action)
        lang_embeds: torch.Tensor,      # (H, B, lang)
    ) -> Tuple[list, list]:
        state = start_state
        states, prior_stats_list = [], []
        for t in range(actions.shape[0]):
            state, prior_stats = self.imagine_step(state, actions[t], lang_embeds[t])
            states.append(state)
            prior_stats_list.append(prior_stats)
        return states, prior_stats_list

    # ------------------------------------------------------------------
    # KL loss  (Eq. from App A.1, with free-nats clipping)
    # ------------------------------------------------------------------

    @staticmethod
    def kl_loss(
        post_stats: Dict,
        prior_stats: Dict,
        free_nats: float = 1.5,
        beta: float = 1.0,
        balance: float = 0.8,
    ) -> torch.Tensor:
        """
        KL-balanced loss as in DreamerV3:
          L_kl = balance * KL(sg(post) || prior)
               + (1-balance) * KL(post || sg(prior))
        """
        post_dist = Normal(post_stats["mean"], post_stats["std"])
        prior_dist = Normal(prior_stats["mean"], prior_stats["std"])

        kl_lhs = torch.distributions.kl_divergence(
            Normal(post_stats["mean"].detach(), post_stats["std"].detach()), prior_dist
        ).sum(-1)
        kl_rhs = torch.distributions.kl_divergence(
            post_dist, Normal(prior_stats["mean"].detach(), prior_stats["std"].detach())
        ).sum(-1)

        kl_lhs = torch.clamp(kl_lhs, min=free_nats)
        kl_rhs = torch.clamp(kl_rhs, min=free_nats)

        return beta * (balance * kl_lhs + (1 - balance) * kl_rhs).mean()
