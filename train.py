"""
train.py — Instruct-to-Act training entry point.

Usage
─────
  python train.py \
    --config configs/minecraft.yaml configs/base.yaml \
    --logdir /data/terran/instruct_to_act/minecraft/seed0 \
    --seed 0

Our YAML files are loaded as a flat project config dict.
DreamerV3's nested config is built separately from dreamerv3/configs.yaml.
"""

import sys, os, threading, time, argparse
from functools import partial as bind
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / 'dreamerv3'))

import numpy as np
import ruamel.yaml as yaml
import elements
import embodied
import embodied.run

from agent     import LangCondAgent
from training  import PostHocAnnotator, LangReplayBuffer
from envs      import make_env as _make_env


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_yaml(path):
    with open(path) as f:
        return yaml.YAML(typ='safe').load(f) or {}


def load_project_config(yaml_paths, overrides=None):
    """
    Merge our flat YAML files into one dict.
    Files are applied in the order given; later files override earlier ones.
    Convention: pass base.yaml first, then env-specific config.
    """
    cfg = {}
    for p in yaml_paths:
        cfg.update(_load_yaml(p))
    if overrides:
        cfg.update(overrides)
    return cfg


def build_dv3_config(proj):
    """
    Start from dreamerv3 defaults and apply the subset of our project keys
    that have direct dreamerv3 equivalents.
    """
    dv3_yaml = SCRIPT_DIR / 'dreamerv3' / 'dreamerv3' / 'configs.yaml'
    dv3_cfgs = yaml.YAML(typ='safe').load(dv3_yaml.read_text())
    config   = elements.Config(dv3_cfgs['defaults'])

    # Keys that map 1-to-1 from our flat config to dreamerv3's flat top-level
    FLAT_PASSTHROUGH = ('batch_size', 'batch_length', 'seed')
    flat = {k: proj[k] for k in FLAT_PASSTHROUGH if k in proj}
    if flat:
        config = config.update(flat)

    # run.* sub-keys
    run_map = {
        'steps':      'steps',
        'log_every':  'log_every',
        'save_every': 'save_every',
        'eval_every': 'report_every',
        'run_envs':   'envs',
    }
    run_updates = {dv3k: proj[pk] for pk, dv3k in run_map.items() if pk in proj}
    if run_updates:
        config = config.update({'run': run_updates})

    # Architecture params: map our flat keys to dreamerv3's nested agent config.
    if 'rssm_deter' in proj:
        config = config.update({'agent': {'dyn': {'rssm': {'deter': proj['rssm_deter']}}}})
    if 'rssm_hidden' in proj:
        config = config.update({'agent': {'dyn': {'rssm': {'hidden': proj['rssm_hidden']}}}})
    if 'rssm_classes' in proj:
        config = config.update({'agent': {'dyn': {'rssm': {'classes': proj['rssm_classes']}}}})
    if 'units' in proj:
        config = config.update({'agent': {
            'policy': {'units': proj['units']},
            'value':  {'units': proj['units']},
            'rewhead': {'units': proj['units']},
            'conhead': {'units': proj['units']},
        }})
    if 'depth' in proj:
        config = config.update({'agent': {
            'enc': {'simple': {'depth': proj['depth']}},
            'dec': {'simple': {'depth': proj['depth']}},
        }})

    # Hyperparameters from base.yaml that don't have a flat dreamerv3 equivalent.
    # dreamerv3 defaults: opt.lr=4e-5, opt.agc=0.3, imag_loss.actent=0.0003, rssm.free_nats=1.0
    opt_updates = {}
    if 'lr' in proj:
        opt_updates['lr'] = proj['lr']
    if 'grad_clip' in proj:
        opt_updates['agc'] = proj['grad_clip']
    if opt_updates:
        config = config.update({'agent': {'opt': opt_updates}})
    if 'actor_entropy' in proj:
        config = config.update({'agent': {'imag_loss': {'actent': proj['actor_entropy']}}})
    if 'free_nats' in proj:
        config = config.update({'agent': {'dyn': {'rssm': {'free_nats': proj['free_nats']}}}})

    return config


# ── Factory callables ─────────────────────────────────────────────────────────

def make_agent_fn(proj, dv3, obs_space, act_space):
    obs_space = {k: v for k, v in obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in act_space.items() if k != 'reset'}
    # dreamerv3 Agent receives config.agent + logdir/seed/jax sub-configs.
    # Merge our project lang keys at construction time (elements.Config is strict
    # about adding new keys, so they must be present from the start).
    lang_defaults = {
        'lang_size':          384,
        'lang_bc_scale':      1.0,
        'lang_stop_scale':    1.0,
        'stop_head_units':    512,
        'bc_separate_batch':  False,
        'finish_action':      False,   # [finish]-as-action ablation
        'wm_reward':          False,   # option 4: instruction-conditioned reward head + imagination-based completion
        'wm_reward_horizon':  10,
        'lang_rew_scale':     1.0,
        'finish_obs':         True,    # False: keep the event-pulse label out of the encoder input (needed to fine-tune old checkpoints)
    }
    lang_keys = {k: proj.get(k, v) for k, v in lang_defaults.items()}
    agent_cfg = elements.Config(
        **dv3.agent,
        logdir=dv3.logdir,
        seed=dv3.seed,
        jax=dv3.jax,
        batch_size=dv3.batch_size,
        batch_length=dv3.batch_length,
        replay_context=dv3.replay_context,
        report_length=dv3.report_length,
        replica=dv3.replica,
        replicas=dv3.replicas,
        **lang_keys,
    )
    return LangCondAgent(obs_space, act_space, agent_cfg)


def make_replay_fn(proj, dv3, logdir):
    # dreamerv3 expects sequences of length batch_length + replay_context
    seq_len = int(proj.get('batch_length', 64)) + int(dv3.replay_context)
    buf = LangReplayBuffer(
        directory  = elements.Path(logdir) / 'replay',
        capacity   = int(proj.get('replay_size', 1_000_000)),
        min_length = seq_len,
        max_length = seq_len,
        lang_size  = int(proj.get('lang_size', 384)),
        finish     = bool(proj.get('finish_action', False) or proj.get('wm_reward', False)),
        obs_key    = 'image',
    )
    buf._batch_size = int(dv3.batch_size)
    # Report uses a shorter sequence length than training; store it so
    # make_stream_fn can return the right length per mode.
    buf._report_seq_len = int(dv3.report_length) + int(dv3.replay_context)
    return buf


def make_env_fn(proj, idx):
    return _make_env(
        proj.get('env_name', 'crafter'),
        proj,
        lang_size=int(proj.get('lang_size', 384)),
    )


def make_logger_fn(logdir, step, dv3):
    logdir = elements.Path(logdir)
    outputs = [
        elements.logger.TerminalOutput(dv3.logger.filter),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    ]
    try:
        outputs.append(elements.logger.TensorBoardOutput(logdir))
    except Exception:
        pass  # tensorflow not installed
    return elements.Logger(step, outputs)


def make_stream_fn(proj, dv3, replay, mode='train'):
    batch = getattr(replay, '_batch_size', 16)
    gen = replay.dataset(batch=batch, length=replay._maxlen)
    if mode == 'report':
        T = int(dv3.report_length) + int(dv3.replay_context)
        def _truncated(g):
            for data in g:
                yield {k: v[:, :T] for k, v in data.items()}
        return _truncated(gen)
    return gen


# ── Annotation thread ─────────────────────────────────────────────────────────

def start_annotator(proj, replay, logdir):
    ann = PostHocAnnotator(
        annotator_type    = proj.get('annotator_type', 'gpt4o'),
        task_guidance     = proj.get('task_guidance', ''),
        annotate_fraction = float(proj.get('annotate_fraction', 0.50)),
        lang_size         = int(proj.get('lang_size', 384)),
        log_dir           = str(elements.Path(logdir) / 'annotations'),
        segment_len_range = (int(proj.get('segment_min_len', 32)),
                             int(proj.get('segment_max_len', 128))),
        annotate_once     = bool(proj.get('annotate_once', False)),
        vlm_kwargs        = dict(done_label=str(proj.get('done_label', 'window')), finish_action=bool(proj.get('finish_action', False) or proj.get('wm_reward', False)), segment_chunk=int(proj.get('segment_chunk', 192)), segment_done_len=int(proj.get('segment_done_len', 4)), segment_frames=int(proj.get('segment_frames', 8)), segment_upscale=int(proj.get('segment_upscale', 1))),
        max_calls         = (int(proj['annotate_max_calls'])
                             if proj.get('annotate_max_calls') else None),
    )
    interval = int(proj.get('annotate_interval', 5000))
    env_name = proj.get('env_name', 'unknown')

    def _worker():
        import os, traceback
        while True:
            time.sleep(10)
            if len(replay) >= interval:
                try:
                    n = ann.annotate_buffer(replay, env_name=env_name)
                except RuntimeError as e:
                    if 'interpreter shutdown' in str(e): return   # training finished while a pass was running
                    traceback.print_exc(); print('[Annotator] FATAL: annotator thread crashed; aborting training', flush=True); os._exit(3)
                except Exception:
                    # A dead annotator thread must not let training continue silently without labels.
                    traceback.print_exc(); print('[Annotator] FATAL: annotator thread crashed; aborting training', flush=True)
                    os._exit(3)
                print(f'[Annotator] annotated {n} timesteps')

    threading.Thread(target=_worker, daemon=True).start()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  nargs='+',
                   default=['configs/base.yaml'],
                   help='One or more project YAML config files (merged left→right)')
    p.add_argument('--logdir',  default='/data/terran/instruct_to_act')
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--annotator_type', default=None,
                   choices=['gpt4o','qwen','gemma','llava',
                            'template','cluster','random'])
    p.add_argument('--batch_size',   type=int, default=None)
    p.add_argument('--batch_length', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Build project config from our YAML files
    overrides = {}
    if args.annotator_type:
        overrides['annotator_type'] = args.annotator_type
    overrides['seed']   = args.seed
    overrides['logdir'] = args.logdir
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.batch_length is not None:
        overrides['batch_length'] = args.batch_length

    proj = load_project_config(args.config, overrides)

    # Build dreamerv3 config from its own defaults
    dv3 = build_dv3_config(proj)
    dv3 = dv3.update({'logdir': args.logdir, 'seed': args.seed})

    logdir = elements.Path(args.logdir)
    logdir.mkdir()

    # Build shared objects
    env0   = _make_env(proj.get('env_name', 'crafter'), proj,
                       lang_size=int(proj.get('lang_size', 384)))
    replay = make_replay_fn(proj, dv3, logdir)

    if proj.get('annotator_type') in ('inventory', 'hybrid') or (proj.get('annotator_type') == 'segment' and 'minecraft' in str(proj.get('env_name', ''))):
        # ground-truth completion for minecraft: track the scripted-planner items in the env's inventory vector
        from training.annotator import INVENTORY_ITEMS
        def _find(obj, name, depth=0):
            if depth > 8 or obj is None: return None
            if hasattr(obj, name): return getattr(obj, name)
            for a in ('_env', 'env', '_gymenv', 'envs'):
                sub = getattr(obj, a, None); sub = sub[0] if isinstance(sub, (list, tuple)) and sub else sub
                r = _find(sub, name, depth + 1)
                if r is not None: return r
        keys = [k.replace('inventory/', '') for k in (_find(env0, '_inv_keys') or [])]
        idx = [keys.index(it) for it in INVENTORY_ITEMS]
        replay.set_inventory_tracking(idx); print(f'[train] inventory tracking indices for {INVENTORY_ITEMS}: {idx}', flush=True)
    start_annotator(proj, replay, logdir)

    step   = elements.Counter()
    logger = make_logger_fn(logdir, step, dv3)

    args_run = elements.Config(
        **dv3.run,
        replica=dv3.replica,
        replicas=dv3.replicas,
        logdir=args.logdir,
        batch_size=dv3.batch_size,
        batch_length=dv3.batch_length,
        report_length=dv3.report_length,
        consec_train=dv3.consec_train,
        consec_report=dv3.consec_report,
        replay_context=dv3.replay_context,
    )

    embodied.run.train(
        bind(make_agent_fn, proj, dv3, env0.obs_space, env0.act_space),
        lambda: replay,
        bind(make_env_fn, proj),
        bind(make_stream_fn, proj, dv3),
        lambda: logger,
        args_run,
    )


if __name__ == '__main__':
    main()
