"""
Environment factories for Instruct-to-Act.

Wraps dreamerv3's own environment loading so we get the same
wrappers and obs/act spaces, then injects the lang_embed fields.
"""

import importlib, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dreamerv3'))

import numpy as np
import elements
import embodied  # noqa: embodied.wrappers is an attribute, not a submodule


# ── Language observation wrapper ──────────────────────────────────────────────

class LangObsWrapper(embodied.Wrapper):
    """
    Adds lang_embed / annotated / complete fields to obs_space.
    Actual embeddings are written by PostHocAnnotator; this wrapper
    just ensures the fields exist with the right shapes for the agent.
    """

    def __init__(self, env, lang_size: int = 384, finish: bool = False):
        super().__init__(env)
        self._lang_size = lang_size
        self._finish = bool(finish)   # only when the [finish]-action ablation is on (keeps old checkpoints loadable)

    @property
    def obs_space(self):
        spaces = dict(self.env.obs_space)
        spaces['lang_embed'] = elements.Space(np.float32, (self._lang_size,))
        spaces['annotated']  = elements.Space(np.float32, ())
        spaces['complete']   = elements.Space(np.float32, ())
        if self._finish: spaces['finish'] = elements.Space(np.float32, ())
        return spaces

    def step(self, action):
        obs = self.env.step(action)
        obs = dict(obs)
        obs['lang_embed'] = np.zeros(self._lang_size, dtype=np.float32)
        obs['annotated']  = np.float32(0.0)
        obs['complete']   = np.float32(0.0)
        if self._finish: obs['finish'] = np.float32(0.0)
        return obs

    def reset(self):
        obs = self.env.reset()
        obs = dict(obs)
        obs['lang_embed'] = np.zeros(self._lang_size, dtype=np.float32)
        obs['annotated']  = np.float32(0.0)
        obs['complete']   = np.float32(0.0)
        if self._finish: obs['finish'] = np.float32(0.0)
        return obs


# ── dreamerv3-style env loading ───────────────────────────────────────────────

_SUITE_MAP = {
    'crafter':   ('embodied.envs.crafter',   'Crafter'),
    'minecraft': ('embodied.envs.minecraft',  'Minecraft'),
    'atari':     ('embodied.envs.atari',      'Atari'),
    'dmlab':     ('embodied.envs.dmlab',      'DMLab'),
}

_SUITE_TASK = {
    'crafter':   'reward',
    'minecraft': 'diamond',
    'atari':     'pong',
    'dmlab':     'explore_goal_locations_small',
}

_SUITE_KWARGS = {
    'crafter':   {'size': (64, 64), 'logs': False},
    'minecraft': {'size': (64, 64), 'logs': False},
    'atari':     {'size': (96, 96), 'repeat': 4, 'gray': True,
                  'actions': 'all', 'lives': 'unused', 'noops': 30},
    'dmlab':     {'size': (64, 64), 'repeat': 4, 'episodic': True},
}


class FinishAction(embodied.wrappers.Wrapper if hasattr(embodied, 'wrappers') and hasattr(embodied.wrappers, 'Wrapper') else object):
    """Adds one extra discrete action ([finish], last index) that the env executes as a no-op (index 0)."""
    def __init__(self, env):
        self.env = env
        sp = env.act_space['action']
        self._n = int(getattr(sp, 'classes', None) or sp.high)
        import elements
        self._act_space = dict(env.act_space); self._act_space['action'] = elements.Space(np.int32, (), 0, self._n + 1)
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        return getattr(self.env, name)
    def __len__(self): return len(self.env)
    @property
    def act_space(self): return self._act_space
    @property
    def obs_space(self): return self.env.obs_space
    def step(self, action):
        a = dict(action); v = np.asarray(a['action'])
        a['action'] = np.where(v >= self._n, 0, v).astype(v.dtype) if v.ndim else (v.dtype.type(0) if int(v) >= self._n else v)
        return self.env.step(a)
    def close(self): return self.env.close()


def _wrap_env(env):
    """Apply dreamerv3's standard wrappers."""
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def make_env(env_name: str, config, lang_size: int = 384):
    """
    Create a language-observation-wrapped environment.

    env_name: crafter | minecraft | atari | dmlab | overcooked
    config:   flat dict or object with .get()
    """
    suite = env_name.split('_')[0].lower()

    def _get(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    if suite == 'overcooked':
        from envs.overcooked_env import make_overcooked_env
        layout  = _get('overcooked_layout', 'asymmetric_advantages')
        horizon = _get('episode_length', 400)
        return make_overcooked_env(layout, horizon, lang_size)

    if suite not in _SUITE_MAP:
        raise ValueError(f"Unknown env {env_name!r}. Choose from {list(_SUITE_MAP) + ['overcooked']}")

    module_path, cls_name = _SUITE_MAP[suite]
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import {module_path} for env '{env_name}'. "
            f"Install the required package and try again.\n  {e}"
        )
    ctor = getattr(module, cls_name)

    # Task string (after the suite prefix)
    if '_' in env_name:
        task = env_name.split('_', 1)[1]
    else:
        task = _SUITE_TASK.get(suite, suite)

    kwargs = dict(_SUITE_KWARGS.get(suite, {}))
    if 'logs' in kwargs and _get('env_logs', None) is not None:
        kwargs['logs'] = bool(_get('env_logs'))   # log/achievement_* needed by the ground-truth completion annotator

    if suite == 'atari':
        game = _get('atari_game', task)
        kwargs['size'] = tuple(_get('atari_size', (96, 96)))
    if suite == 'dmlab':
        task = _get('dmlab_task', task)

    env = ctor(task, **kwargs)
    if _get('finish_action', False):
        env = FinishAction(env)   # ablation: [finish] as an extra no-op action
    env = _wrap_env(env)
    return LangObsWrapper(env, lang_size=lang_size, finish=bool((_get('finish_action', False) or _get('wm_reward', False)) and _get('finish_obs', True)))
