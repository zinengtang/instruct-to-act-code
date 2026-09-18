"""
LangCondAgent: extends the official DreamerV3 Agent with language conditioning.

Design (from Instruct-to-Act §3.2, §3.4, App A.1–A.2):
─────────────────────────────────────────────────────────
• Language embedding e_t (MiniLM-L6-H384, 384-dim) is included in the
  observation dict under the key 'lang_embed'.  DreamerV3's encoder
  creates a separate MLP branch for it automatically.

• The stop/completion head  p^stop_t = σ(g(s_t, e_t))  is a thin MLP
  that receives concat(feat_t, e_t).

• Two extra training losses added on top of standard DreamerV3 losses:

    L_BC   = −λ_BC Σ_{annotated t} log π(a_t | feat_t, e_t)
    L_stop = −Σ_t [complete_t·log(p^stop_t) + (1−complete_t)·log(1−p^stop_t)]

  Both are added to the dreamerv3 loss dict so they flow through the
  existing self.opt optimizer and get proper gradients.

• At inference, policy() returns p^stop in the extras dict so the
  inference algorithms (Algorithm 1 & 2) can detect instruction completion.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dreamerv3'))

import os
import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj
import optax
import embodied.jax as ejax
import embodied.jax.nets as nn

from dreamerv3.agent import Agent as DreamerV3Agent, sample, f32

# ── Stop head ─────────────────────────────────────────────────────────────────

class StopHead(nj.Module):
    """
    g(feat_t, e_t) → scalar logit → p^stop_t = σ(logit)
    Mirrors the actor/value head architecture (App A.3).
    """
    def __init__(self, units: int = 512):
        self._units = units

    def __call__(self, feat, lang_embed):
        inp = jnp.concatenate([feat, lang_embed], axis=-1)
        x   = self.sub('mlp',    nn.MLP,    layers=2, units=self._units)(inp)
        x   = self.sub('linear', nn.Linear, 1)(x)
        return jax.nn.sigmoid(x[..., 0])


# ── Language-conditioned Agent ────────────────────────────────────────────────

class LangCondAgent(DreamerV3Agent):
    """
    DreamerV3 Agent extended with:
      (1) lang_embed observation branch (the encoder handles it automatically
          if 'lang_embed' is present in obs_space)
      (2) StopHead for instruction-completion prediction
      (3) L_BC behavior-cloning loss on annotated replay segments
      (4) L_stop binary cross-entropy on instruction completion labels

    Both extra losses are added to the dreamerv3 `losses` dict inside loss()
    so they go through self.opt and receive proper gradients.
    """

    def __init__(self, obs_space, act_space, config):
        # Inject lang observation fields if missing
        lang_size = int(config.get('lang_size', 384))
        import elements
        if 'lang_embed' not in obs_space:
            obs_space = dict(obs_space)
            obs_space['lang_embed'] = elements.Space(np.float32, (lang_size,))
        for key in ('complete', 'annotated') + (('finish',) if ((config.get('finish_action', False) or config.get('wm_reward', False)) and config.get('finish_obs', True)) else ()):
            if key not in obs_space:
                obs_space = dict(obs_space)
                obs_space[key] = elements.Space(np.float32, ())

        super().__init__(obs_space, act_space, config)

        self._lang_size        = lang_size
        self._lang_bc_scale    = float(config.get('lang_bc_scale',    1.0))
        self._lang_stop_scale  = float(config.get('lang_stop_scale',  1.0))
        self._bc_separate      = bool(config.get('bc_separate_batch', False))
        self._finish_action    = bool(config.get('finish_action', False))   # ablation: last action index = [finish]
        self._wm_reward        = bool(config.get('wm_reward', False))       # option 4: instruction-conditioned reward head + imagination
        self._wm_reward_horizon= int(config.get('wm_reward_horizon', 10))
        self._lang_rew_scale   = float(config.get('lang_rew_scale', 1.0))

        self._stop_head = StopHead(
            units=int(config.get('stop_head_units', 512)),
            name='stop_head',
        )
        self.modules.append(self._stop_head)
        if self._wm_reward:
            self._rew_lang = StopHead(units=int(config.get('stop_head_units', 512)), name='rew_lang')   # p(event now | feat, instruction)
            self.modules.append(self._rew_lang)
        # BUG FIX (2026-09-14): the base optimizer captured self.modules before the stop head existed, so the
        # stop head's parameters were never updated (checkpoints show bit-identical init values after 300k steps;
        # only the RSSM was bent to satisfy a random projection). Rebuild the optimizer over the full module list.
        import embodied
        self.opt = embodied.jax.Optimizer(self.modules, self._make_opt(**config.opt), summary_depth=1, name='opt')
        # Do NOT add bc/stop to self.scales — dreamerv3 asserts scales==losses keys,
        # and we add our losses after super().loss() returns.

        if self._bc_separate:
            # Separate optimizer that only updates policy + stop_head.
            # This lets BC gradients flow exclusively to the action head
            # without disturbing the world model.
            import embodied.jax as ejax
            self.bc_opt = ejax.Optimizer(
                [self.pol, self._stop_head],
                self._make_opt(**config.opt),
                summary_depth=1,
                name='bc_opt',
            )

    @property
    def policy_keys(self):
        # Include stop_head so its params are available during policy() execution.
        return '^(enc|dyn|dec|pol|stop_head|rew_lang)/'

    def load(self, data, regex=None):
        """Strip bc_opt keys from checkpoint if bc_separate_batch is now off."""
        params = data.get('params', {})
        extra = {k for k in params if k.startswith('bc_opt/')}
        if extra and not self._bc_separate:
            # Checkpoint was saved with bc_separate=True; load the rest normally
            # using regex to skip the shape-equality assertion.
            print(f'[LangCondAgent] Dropping {len(extra)} bc_opt keys '
                  f'(bc_separate_batch is now False).')
            regex = regex or r'^(?!bc_opt/).*'
        super().load(data, regex=regex)

    # ── Override loss() to inject L_BC and L_stop ────────────────────────────

    def loss(self, carry, obs, prevact, training):
        # Extract annotation labels before the RSSM sees them.
        # `complete` and `annotated` are training-only metadata; passing them
        # through the encoder causes label leakage (stop head can trivially read
        # `complete` back from the RSSM state) and a train/test distribution gap
        # (at test time both are always 0). We zero them out so the RSSM never
        # encodes them, while the custom losses still use the real values below.
        _ref = obs.get('complete', obs.get('annotated', obs['reward']))
        zeros2d = jnp.zeros_like(f32(_ref))
        # Clip to [0, 1] to guard against any upstream corruption (inf/nan)
        # that would make stop_loss or bc_loss blow up to inf.
        complete  = jnp.clip(f32(obs.get('complete',  zeros2d)), 0.0, 1.0)
        finish    = jnp.clip(f32(obs.get('finish',    zeros2d)), 0.0, 1.0)
        annotated = jnp.clip(f32(obs.get('annotated', zeros2d)), 0.0, 1.0)
        obs_clean = {k: (zeros2d if k in ('complete', 'annotated', 'finish') else v)
                     for k, v in obs.items()}

        loss, (carry, entries, outs, metrics) = super().loss(
            carry, obs_clean, prevact, training)

        repfeat    = outs['repfeat']                         # RSSM state dict
        feat       = self.feat2tensor(repfeat)               # (B, T, feat_dim)
        lang_embed = f32(obs.get(
            'lang_embed',
            jnp.zeros((*feat.shape[:-1], self._lang_size))))  # (B, T, lang)

        # ── L_BC ─────────────────────────────────────────────────────────────
        # lang_embed is already encoded in feat via the obs encoder branch.
        # prevact[t] = action at t-1, so prevact[:, 1:] aligns with feat[:, :-1]
        # (action taken at state t paired with policy evaluated at state t).
        policy   = self.pol(feat[:, :-1], 2)
        log_prob = self._bc_logp(policy, prevact, finish)     # (B, T-1)
        annotated_bc = annotated[:, :-1]                     # (B, T-1)
        n_annot  = annotated_bc.sum() + 1e-8
        # Guard: use stop_gradient on the zero branch so XLA cannot propagate
        # NaN gradients through non-annotated or near-zero-probability steps.
        # jnp.where alone is insufficient: JAX still differentiates through both
        # branches, so 0 * NaN_grad = NaN in the backward pass.
        safe_log_prob = jnp.where(
            (annotated_bc > 0) & jnp.isfinite(log_prob),
            log_prob,
            jax.lax.stop_gradient(jnp.zeros_like(log_prob)),
        )
        bc_loss  = -(annotated_bc * safe_log_prob).sum() / n_annot

        # ── L_stop ───────────────────────────────────────────────────────────
        p_stop    = self._stop_head(feat, lang_embed)         # (B, T)
        # Cast to float32 and clip before log: bfloat16's log implementation
        # can return -inf for very small positive inputs (near the bf16 precision
        # floor), turning the BCE loss into +inf and freezing all parameters.
        p_stop_f32 = jnp.clip(p_stop.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
        # Mask to annotated timesteps only: unannotated steps have complete=0
        # by default (missing label, not "definitely not complete") and must
        # not contribute to the loss. Within annotated segments complete=1 only
        # at the final step (1 per 32-120 steps), giving ~147:1 class imbalance;
        # pos_weight corrects for this so positive gradients are not swamped.
        n_pos = (annotated * complete).sum() + 1e-8
        n_neg = (annotated * (1 - complete)).sum() + 1e-8
        pos_weight = n_neg / n_pos   # dynamic; ~76 at default segment lengths
        bce = -(
            pos_weight * complete * jnp.log(p_stop_f32) +
            (1 - complete) * jnp.log(1 - p_stop_f32)
        )
        stop_loss = (annotated * bce).sum() / (annotated.sum() + 1e-8)

        # Inject into losses dict (scales already registered in __init__)
        losses = dict(outs['losses'])
        losses['bc']   = bc_loss   * jnp.ones_like(losses['rew'])
        losses['stop'] = stop_loss * jnp.ones_like(losses['rew'])
        rew_lang_loss = jnp.float32(0.0)
        if self._wm_reward:
            # option 4: instruction-conditioned event reward r_lang(feat, e) trained on the ground-truth event pulse (finish=1 at the unlock step)
            r = jnp.clip(self._rew_lang(feat, lang_embed).astype(jnp.float32), 1e-6, 1.0 - 1e-6)
            n_ev = (annotated * finish).sum() + 1e-8; n_no = (annotated * (1 - finish)).sum() + 1e-8; pw = n_no / n_ev
            bce_r = -(pw * finish * jnp.log(r) + (1 - finish) * jnp.log(1 - r))
            rew_lang_loss = (annotated * bce_r).sum() / (annotated.sum() + 1e-8)
            losses['rew_lang'] = rew_lang_loss * jnp.ones_like(losses['rew'])
            metrics['loss/rew_lang'] = rew_lang_loss; metrics['rew_lang_pred_mean'] = r.mean()
        outs = dict(outs, losses=losses)

        # Only add BC/stop to the optimised loss during training;
        # during report() passes these terms are logged but not back-propagated.
        # When bc_separate_batch=True, BC is handled by a dedicated bc_opt step
        # so it is excluded here to avoid double-counting.
        metrics['loss/bc']        = bc_loss
        metrics['training_flag']  = jnp.float32(1.0 if training else 0.0)
        metrics['stop_grad_probe'] = jnp.abs(jax.grad(lambda z: (annotated * (-(pos_weight * complete * jnp.log(jnp.clip(z, 1e-6, 1 - 1e-6)) + (1 - complete) * jnp.log(1 - jnp.clip(z, 1e-6, 1 - 1e-6))))).sum())(p_stop_f32)).mean()
        metrics['loss/stop']      = stop_loss
        metrics['bc_frac']        = annotated.mean()
        metrics['stop_pred_mean'] = p_stop.mean()

        if training and not self._bc_separate:
            extra = self._lang_bc_scale * bc_loss + self._lang_stop_scale * stop_loss + self._lang_rew_scale * rew_lang_loss
        else:
            extra = jax.lax.stop_gradient(
                self._lang_bc_scale * bc_loss + self._lang_stop_scale * stop_loss + self._lang_rew_scale * rew_lang_loss)

        return loss + extra, (carry, entries, outs, metrics)

    def bc_only_loss(self, carry, obs, prevact, training):
        """Loss function for the separate BC optimizer step.
        Only computes L_BC + L_stop; world-model and RL terms are excluded.
        """
        _ref     = obs.get('complete', obs.get('annotated', obs['reward']))
        zeros2d  = jnp.zeros_like(f32(_ref))
        complete  = jnp.clip(f32(obs.get('complete',  zeros2d)), 0.0, 1.0)
        finish    = jnp.clip(f32(obs.get('finish',    zeros2d)), 0.0, 1.0)
        annotated = jnp.clip(f32(obs.get('annotated', zeros2d)), 0.0, 1.0)
        obs_clean = {k: (zeros2d if k in ('complete', 'annotated', 'finish') else v)
                     for k, v in obs.items()}

        # Run the world model forward to get features.
        # stop_gradient on carry: BC only updates pol+stop_head, not the WM state.
        _, (carry, _, outs, _) = super().loss(
            jax.lax.stop_gradient(carry), obs_clean, prevact, training=False)
        carry = jax.lax.stop_gradient(carry)

        feat       = self.feat2tensor(outs['repfeat'])
        lang_embed = f32(obs.get(
            'lang_embed',
            jnp.zeros((*feat.shape[:-1], self._lang_size))))

        # L_BC
        policy   = self.pol(feat[:, :-1], 2)
        log_prob = self._bc_logp(policy, prevact, finish)
        annotated_bc  = annotated[:, :-1]
        n_annot       = annotated_bc.sum() + 1e-8
        safe_log_prob = jnp.where(
            (annotated_bc > 0) & jnp.isfinite(log_prob),
            log_prob,
            jax.lax.stop_gradient(jnp.zeros_like(log_prob)),
        )
        bc_loss = -(annotated_bc * safe_log_prob).sum() / n_annot

        # L_stop (same masked weighted BCE as in loss())
        p_stop    = self._stop_head(feat, lang_embed)
        p_stop_f32 = jnp.clip(p_stop.astype(jnp.float32), 1e-6, 1.0 - 1e-6)
        n_pos_bc = (annotated * complete).sum() + 1e-8
        n_neg_bc = (annotated * (1 - complete)).sum() + 1e-8
        pos_weight_bc = n_neg_bc / n_pos_bc
        bce_bc   = -(pos_weight_bc * complete * jnp.log(p_stop_f32) +
                     (1 - complete) * jnp.log(1 - p_stop_f32))
        stop_loss = (annotated * bce_bc).sum() / (annotated.sum() + 1e-8)

        loss    = self._lang_bc_scale * bc_loss + self._lang_stop_scale * stop_loss
        metrics = {'bc_opt/loss/bc': bc_loss, 'bc_opt/loss/stop': stop_loss}
        return loss, (carry, {}, {}, metrics)

    def train(self, carry, data):
        """Override: BC step first, then RL step.

        Order matters: BC nudges pol toward expert behavior, then RL fine-tunes
        from that initialisation. Reversed order (RL first) causes RL to
        continuously overwrite the BC signal since RL gradient is larger.
        """
        if self._bc_separate:
            # _apply_replay_context strips prevact, returning the 3-tuple carry
            # that loss() expects.
            bc_carry, obs, prevact, _ = self._apply_replay_context(carry, data)
            bc_mets, (_, _, _, extra_mets) = self.bc_opt(
                self.bc_only_loss, bc_carry, obs, prevact,
                training=True, has_aux=True)
            # carry comes from the RL step below; BC only updates pol+stop_head params.

        carry, outs, metrics = super().train(carry, data)

        if self._bc_separate:
            metrics.update(bc_mets)
            metrics.update(extra_mets)

        return carry, outs, metrics

    # ── Override policy() to append p_stop ───────────────────────────────────

    @staticmethod
    def _act_logits(dist):
        for d in (dist, getattr(dist, 'dist', None)):
            if d is not None and hasattr(d, 'logits'): return d.logits
        raise AttributeError('no logits on action distribution')

    @staticmethod
    def _finish_event(dist, n, like):
        """Event tensor selecting the [finish] action (last index) in the format dist.logp expects:
        integer indices for Categorical, one-hot vectors for OneHot (wraps a Categorical in .dist)."""
        if hasattr(dist, 'dist') or (like.ndim > 0 and like.shape[-1] == n and like.dtype != jnp.int32):
            return jnp.broadcast_to(jax.nn.one_hot(n - 1, n, dtype=f32), (*like.shape[:-1], n) if like.shape[-1] == n else (*like.shape, n))
        return jnp.full(like.shape, n - 1, dtype=like.dtype if like.dtype in (jnp.int32, jnp.int64) else jnp.int32)

    def _bc_logp(self, policy, prevact, finish):
        """BC log-prob of the recorded action; with finish_action, steps labelled finish=1 use the [finish] action
        (last index) as the target instead of the recorded action."""
        log_prob = sum(v.logp(prevact[k][:, 1:]) for k, v in policy.items())
        if self._finish_action and 'action' in policy:
            dist = policy['action']; n = self._act_logits(dist).shape[-1]
            fin = self._finish_event(dist, n, prevact['action'][:, 1:])
            lp_fin = dist.logp(fin) + sum(v.logp(prevact[k][:, 1:]) for k, v in policy.items() if k != 'action')
            log_prob = jnp.where(finish[:, :-1] > 0, lp_fin, log_prob)
        return log_prob

    def policy(self, carry, obs, mode='train'):
        carry, act, out = super().policy(carry, obs, mode)

        # Recover feat from carry (dyn_carry after observe step)
        # carry = (enc_carry, dyn_carry, dec_carry, prevact)
        # We compute p_stop from the last dyn_carry hidden state.
        dyn_carry = carry[1]
        feat = self.feat2tensor(dyn_carry)                    # (B, feat_dim)
        lang_embed = f32(obs.get(
            'lang_embed',
            jnp.zeros((*feat.shape[:-1], self._lang_size))))
        if lang_embed.ndim == feat.ndim:
            pass  # already (B, lang)
        p_stop = self._stop_head(feat, lang_embed)
        # Use log/ prefix so the driver doesn't store p_stop in the replay.
        out['log/p_stop'] = p_stop
        if self._wm_reward:
            # option 4: judge completion with the world model: event probability now, and within H imagined steps under the current policy
            out['log/r_lang'] = self._rew_lang(feat, lang_embed)
            policyfn = lambda f: sample(self.pol(self.feat2tensor(f), 1))
            _, imgfeat, _ = self.dyn.imagine(nn.cast(dyn_carry), policyfn, self._wm_reward_horizon, training=False)   # (B, H, ...)
            imgfeat_t = self.feat2tensor(imgfeat)
            r_img = self._rew_lang(imgfeat_t, jnp.broadcast_to(lang_embed[:, None, :], (*imgfeat_t.shape[:-1], lang_embed.shape[-1])))
            out['log/p_soon'] = 1.0 - jnp.prod(1.0 - jnp.clip(r_img.astype(jnp.float32), 0.0, 1.0), axis=1)
        if self._finish_action:
            dist = self.pol(feat, 1)['action']; n = self._act_logits(dist).shape[-1]
            out['log/p_finish'] = jnp.exp(dist.logp(self._finish_event(dist, n, jnp.zeros(feat.shape[:-1], jnp.int32))))
        if os.environ.get('I2A_EXPORT_FEAT'):   # eval-only (stop-head relabel study); the train driver rejects non-scalar log/ keys
            out['log/feat'] = feat
        return carry, act, out
