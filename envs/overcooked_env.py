"""
Overcooked-AI wrapper for Instruct-to-Act.

Each OvercookedEnv instance is self-contained: it controls agent 0 with the
policy's action and uses a fixed STAY action for agent 1.  This design is
compatible with DreamerV3's multiprocess Driver (one env per worker process).

DreamerV3 interface: step(action_dict) → obs_dict (no gym-style tuple).

install:
  pip install git+https://github.com/HumanCompatibleAI/overcooked_ai.git
"""

import numpy as np

_IMG_H, _IMG_W = 64, 64
_N_ACTIONS = 6          # UP DOWN LEFT RIGHT STAY INTERACT
_STAY_ACTION = 4        # fixed action used for the background agent
_LANG_SIZE  = 384


# ── Observation rendering ─────────────────────────────────────────────────────

def _state_to_image(mdp, state, agent_idx, horizon):
    """Return a (64, 64, 3) uint8 image for one agent's view of the state."""
    try:
        enc = mdp.lossless_state_encoding(state, horizon)
        enc = np.array(enc, dtype=np.float32)
        if enc.ndim == 4:
            arr = enc[agent_idx]
        elif enc.ndim == 3:
            arr = enc
        else:
            raise ValueError(f"unexpected encoding shape {enc.shape}")
        c = arr.shape[2]
        arr = arr[:, :, [0, c // 2, min(c - 1, c // 2 + 1)]]
    except Exception:
        arr = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.float32)

    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
    arr = arr.astype(np.uint8)

    if arr.shape[:2] != (_IMG_H, _IMG_W):
        try:
            from PIL import Image
            arr = np.array(
                Image.fromarray(arr).resize((_IMG_W, _IMG_H), Image.NEAREST)
            )
        except ImportError:
            iy = (np.linspace(0, arr.shape[0] - 1, _IMG_H)).astype(int)
            ix = (np.linspace(0, arr.shape[1] - 1, _IMG_W)).astype(int)
            arr = arr[np.ix_(iy, ix, np.arange(arr.shape[2]))]

    return arr


# ── Self-contained single-agent env ──────────────────────────────────────────

class OvercookedEnv:
    """
    DreamerV3-compatible Overcooked env.

    Controls agent 0; agent 1 always takes STAY.
    step(action_dict) returns an obs dict — NOT a gym tuple.
    """

    def __init__(self, layout_name: str, horizon: int = 400):
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as _OEnv

        self._mdp     = OvercookedGridworld.from_layout_name(layout_name)
        self._env     = _OEnv.from_mdp(self._mdp, horizon=horizon)
        self._horizon = horizon
        self._done    = True   # triggers reset on first step
        self._trunc   = False
        self._obs     = None

    @property
    def obs_space(self):
        import elements
        return {
            'image':       elements.Space(np.uint8,   (_IMG_H, _IMG_W, 3)),
            'reward':      elements.Space(np.float32, ()),
            'is_first':    elements.Space(bool,       ()),
            'is_last':     elements.Space(bool,       ()),
            'is_terminal': elements.Space(bool,       ()),
            'lang_embed':  elements.Space(np.float32, (_LANG_SIZE,)),
            'annotated':   elements.Space(np.float32, ()),
            'complete':    elements.Space(np.float32, ()),
        }

    @property
    def act_space(self):
        import elements
        return {
            'action': elements.Space(np.int32, (), 0, _N_ACTIONS),
            'reset':  elements.Space(bool, ()),
        }

    def _make_obs(self, state, reward: float, is_first: bool) -> dict:
        img = _state_to_image(self._mdp, state, 0, self._horizon)
        return {
            'image':       img,
            'reward':      np.float32(reward),
            'is_first':    np.bool_(is_first),
            'is_last':     np.bool_(self._done or self._trunc),
            'is_terminal': np.bool_(self._done),
            'lang_embed':  np.zeros(_LANG_SIZE, dtype=np.float32),
            'annotated':   np.float32(0.0),
            'complete':    np.float32(0.0),
        }

    def _reset(self) -> dict:
        result = self._env.reset()
        state  = result[0] if isinstance(result, tuple) else result
        self._done  = False
        self._trunc = False
        return self._make_obs(state, 0.0, is_first=True)

    def step(self, action) -> dict:
        reset = False
        if isinstance(action, dict):
            reset  = bool(action.get('reset', False))
            action = action.get('action', _STAY_ACTION)
        a0 = int(np.asarray(action).flat[0]) % _N_ACTIONS

        if reset or self._done or self._trunc:
            return self._reset()

        from overcooked_ai_py.mdp.actions import Action
        joint  = (Action.INDEX_TO_ACTION[a0], Action.INDEX_TO_ACTION[_STAY_ACTION])
        result = self._env.step(joint)
        if len(result) == 5:
            state, rew, done, trunc, info = result
        else:
            state, rew, done, info = result
            trunc = False
        self._done  = bool(done)
        self._trunc = bool(trunc)
        shaped = info.get('shaped_r_by_agent', [float(rew) / 2])
        reward = float(shaped[0])
        return self._make_obs(state, reward, is_first=False)

    def close(self):
        pass


# ── Factory ───────────────────────────────────────────────────────────────────

def make_overcooked_env(layout_name: str, horizon: int = 400, lang_size: int = _LANG_SIZE):
    return OvercookedEnv(layout_name, horizon)
