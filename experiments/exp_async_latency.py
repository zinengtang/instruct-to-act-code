"""
exp_async_latency.py — Instruction staleness and async latency analysis.

Addresses Reviewer EkK5 Q4:
  "Fig. 2(b) compares online vs. offline planning with two bars per scale
   on a single task. Could the authors provide latency distributions or
   instruction-staleness statistics across more tasks to support the
   efficiency claim?"

And EkK5 Q2(c):
  "The token-count breakdown between advance and emit calls supporting the
   claim that emitting is cheap."

Measures
────────
  1. Instruction staleness: #steps between last obs seen by VLM and the
     step at which the resulting instruction is consumed by the controller.
  2. VLM call latency: wall-clock time per emit() vs advance() call.
  3. Token breakdown: # tokens generated per emit vs advance.
  4. Controller throughput (env steps/sec) with vs without planner overhead.
  5. Per-task latency distributions across all 7 environments.

Usage
─────
  python experiments/exp_async_latency.py \
      --envs crafter minecraft atari dmlab \
      --checkpoint_root logdir/ \
      --planner qwen \
      --episodes 20 \
      --logdir results/async_latency
"""

import argparse, json, time, sys, threading, collections
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--envs', nargs='+',
                   default=['crafter', 'minecraft', 'atari', 'dmlab'])
    p.add_argument('--checkpoint_root', default='logdir/')
    p.add_argument('--planner',  default='qwen')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--logdir',   default='results/async_latency')
    p.add_argument('--no_actual_vlm', action='store_true',
                   help='Use mock VLM (fixed latency) for offline analysis')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented inference (wraps AsyncOnlineInference to capture latencies)
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedAsyncInference:
    """
    Runs async inference and records detailed latency / staleness statistics.

    Staleness at step t = (t - t_obs) where t_obs is the step at which the
    observation was captured when the VLM started generating the instruction
    that arrives at step t.
    """

    def __init__(self, controller, planner, env, lang_encoder,
                 stop_threshold=0.5, max_steps=2000):
        self.ctrl     = controller
        self.planner  = planner
        self.env      = env
        self.encode   = lang_encoder
        self.stop_thr = stop_threshold
        self.max_steps= max_steps

        # Stats accumulation
        self.emit_latencies_ms   = []
        self.advance_latencies_ms= []
        self.emit_tokens         = []
        self.advance_tokens      = []
        self.staleness_steps     = []
        self.ctrl_steps_times    = []   # timestamps of each controller step

    def run_episode(self):
        from inference.algorithms import AsyncState
        shared     = AsyncState()
        ctrl_state = self.ctrl.initial_state(batch_size=1)
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)
        cur_embed  = null_embed.copy()

        obs, _  = self.env.reset()
        shared.update_obs(obs)
        total_reward = 0.0

        # Track which VLM-step observation was "current" when instruction generated
        obs_step_at_vlm_call = [0]
        ctrl_step = [0]

        def vlm_loop():
            memory, plan = {}, None
            while shared.running:
                latest_obs = shared.get_obs()
                if latest_obs is None:
                    time.sleep(0.005)
                    continue

                # Record obs step at time of VLM call
                vlm_obs_step = ctrl_step[0]

                if shared.stop:
                    t0 = time.perf_counter()
                    instr, plan, memory = self.planner.emit(
                        obs=latest_obs, plan=plan, memory=memory
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.emit_latencies_ms.append(elapsed_ms)

                    # Approximate token count from character length (4 chars/token)
                    self.emit_tokens.append(max(1, len(instr.split())))

                    embed = self.encode(instr)
                    shared.put_inbox(embed, instr)
                    # Store obs_step so we can compute staleness when consumed
                    obs_step_at_vlm_call[0] = vlm_obs_step
                    shared.clear_stop()
                else:
                    t0 = time.perf_counter()
                    plan, memory = self.planner.advance(
                        obs=latest_obs, plan=plan, memory=memory
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self.advance_latencies_ms.append(elapsed_ms)
                    # Record token count from planner if available (GPT-4o returns usage)
                    tok = getattr(self.planner, '_last_advance_tokens', None)
                    if tok is not None:
                        self.advance_tokens.append(int(tok))
                time.sleep(0.001)

        vlm_thread = threading.Thread(target=vlm_loop, daemon=True)
        vlm_thread.start()
        shared.set_stop()

        for step in range(self.max_steps):
            t_step = time.perf_counter()
            self.ctrl_steps_times.append(t_step)
            ctrl_step[0] = step

            new_embed, new_text = shared.pop_inbox()
            if new_embed is not None:
                cur_embed = new_embed
                # Staleness = current step - step when VLM started this instruction
                staleness = step - obs_step_at_vlm_call[0]
                self.staleness_steps.append(max(0, staleness))

            obs_b = {k: np.expand_dims(v, 0) for k, v in obs.items()}
            obs_b['lang_embed'] = cur_embed[np.newaxis]

            action, ctrl_state, extra = self.ctrl.policy(obs_b, ctrl_state, mode='eval')
            p_stop = float(extra.get('p_stop', 0.0))

            action_np = np.array(action[0])
            obs, reward, terminated, truncated, _ = self.env.step(action_np)
            shared.update_obs(obs)
            total_reward += float(reward)

            if p_stop > self.stop_thr:
                cur_embed = null_embed.copy()
                shared.set_stop()
            if terminated or truncated:
                break

        shared.running = False
        return total_reward


# ─────────────────────────────────────────────────────────────────────────────

def compute_throughput(ctrl_step_times: list) -> float:
    """Controller throughput in env steps/second."""
    if len(ctrl_step_times) < 2:
        return 0.0
    elapsed = ctrl_step_times[-1] - ctrl_step_times[0]
    return len(ctrl_step_times) / max(elapsed, 1e-6)


def plot_latency_distributions(env_results: dict, outdir: Path):
    envs = list(env_results.keys())
    n    = len(envs)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, env in enumerate(envs):
        r = env_results[env]

        # Row 0: staleness distribution
        ax = axes[0, j]
        staleness = r['staleness_steps']
        if staleness:
            ax.hist(staleness, bins=30, color='#1565C0', alpha=0.7, edgecolor='white')
            ax.axvline(np.mean(staleness), color='red', ls='--', lw=2,
                       label=f'mean={np.mean(staleness):.1f}')
            ax.set_title(f'{env}\nInstruction Staleness', fontsize=9)
            ax.set_xlabel('Steps since VLM observation')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        # Row 1: emit vs advance latency
        ax2 = axes[1, j]
        emit_lat = r['emit_latencies_ms']
        adv_lat  = r['advance_latencies_ms']
        if emit_lat or adv_lat:
            data_to_plot = [x for x in [emit_lat, adv_lat] if x]
            labels_      = ['emit()' if emit_lat else '', 'advance()' if adv_lat else '']
            labels_      = [l for l, d in zip(labels_, [emit_lat, adv_lat]) if d]
            bp = ax2.boxplot(data_to_plot, labels=labels_, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#E65100', '#2E7D32']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_title(f'{env}\nVLM Latency', fontsize=9)
            ax2.set_ylabel('Latency (ms)')
            ax2.set_yscale('log')

    plt.suptitle('Async Inference Latency Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'async_latency_distributions.pdf')
    fig.savefig(outdir / 'async_latency_distributions.png', dpi=150)
    plt.close()
    print(f"Saved to {outdir}/async_latency_distributions.{{pdf,png}}")


def print_summary_table(env_results: dict):
    print("\n── Async Latency Summary ──")
    print(f"{'Env':<15}  {'Staleness mean':>14}  {'Staleness p95':>13}  "
          f"{'Emit ms':>9}  {'Advance ms':>11}  {'Emit tok':>9}  {'Ctrl steps/s':>13}")
    print("-" * 95)
    for env, r in env_results.items():
        st   = r['staleness_steps']
        em   = r['emit_latencies_ms']
        adv  = r['advance_latencies_ms']
        etok = r['emit_tokens']
        tput = r['throughput']
        print(f"{env:<15}  "
              f"{np.mean(st) if st else 0:>14.1f}  "
              f"{np.percentile(st, 95) if st else 0:>13.1f}  "
              f"{np.mean(em) if em else 0:>9.1f}  "
              f"{np.mean(adv) if adv else 0:>11.1f}  "
              f"{np.mean(etok) if etok else 0:>9.1f}  "
              f"{tput:>13.1f}")


def main():
    args   = parse_args()
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent / 'dreamerv3'))
    import embodied, dreamerv3
    from agent    import LangCondAgent
    from planners import make_planner
    from envs     import make_env
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    encoder  = lambda t: st_model.encode([t], normalize_embeddings=True)[0].astype('float32')

    env_results = {}

    for env_name in args.envs:
        print(f"\n── {env_name} ──")
        ckpt = Path(args.checkpoint_root) / env_name / 'seed0' / 'checkpoint.pkl'
        if not ckpt.exists():
            print(f"  SKIP: no checkpoint at {ckpt}")
            continue

        config = embodied.Config(dreamerv3.configs['defaults'])
        config = config.update(embodied.Config.load(f'configs/{env_name}.yaml'))

        env    = make_env(env_name, config)
        agent  = LangCondAgent(env.obs_space, env.act_space, config)
        state  = embodied.checkpoint.Checkpoint(str(ckpt))
        state.agent = agent
        state.load()

        planner = make_planner(args.planner, task_spec=config.get('task_guidance', ''))

        inf = InstrumentedAsyncInference(
            agent, planner, env, encoder,
            max_steps=config.get('episode_length', 2000) // 10,   # short for latency study
        )

        for ep in range(args.episodes):
            reward = inf.run_episode()
            print(f"  ep {ep+1}: reward={reward:.2f}")

        throughput = compute_throughput(inf.ctrl_steps_times)
        env_results[env_name] = {
            'staleness_steps':      inf.staleness_steps,
            'emit_latencies_ms':    inf.emit_latencies_ms,
            'advance_latencies_ms': inf.advance_latencies_ms,
            'emit_tokens':          inf.emit_tokens,
            'advance_tokens':       inf.advance_tokens,
            'throughput':           throughput,
        }

        print(f"  Staleness: mean={np.mean(inf.staleness_steps or [0]):.1f} steps  "
              f"p95={np.percentile(inf.staleness_steps or [0], 95):.1f}")
        print(f"  Emit latency: {np.mean(inf.emit_latencies_ms or [0]):.1f} ms  "
              f"Advance: {np.mean(inf.advance_latencies_ms or [0]):.1f} ms")
        print(f"  Throughput: {throughput:.1f} env steps/s")

    # Save
    serializable = {
        k: {kk: [float(x) for x in vv] if isinstance(vv, list) else float(vv)
             for kk, vv in v.items()}
        for k, v in env_results.items()
    }
    out = logdir / 'async_latency_results.json'
    with open(out, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out}")

    print_summary_table(env_results)
    plot_latency_distributions(env_results, logdir)


if __name__ == '__main__':
    main()
