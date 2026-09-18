"""
Inference algorithms from Instruct-to-Act §3.1.

Algorithm 1 — Asynchronous Online Inference
────────────────────────────────────────────
The VLM planner and controller run as independent threads.

  Controller thread:
    while running:
      (a_t, s_t) ← π_c.step(o_t, x_t)
      env.step(a_t)
      if s_t (stop signal): Stop←1; x←∅
      if Inbox has message: x ← Inbox.pop()

  VLM thread:
    while running:
      if Stop:
          Inbox ← π_p.emit(o_t, Plan)   # emit next instruction (few tokens)
          Stop←0; Plan←∅
      else:
          Plan ← π_p.advance(o_t, Plan) # advance background reasoning

The key advantage: the VLM can reason continuously while the controller
executes, so plan latency does not block control throughput.

Algorithm 2 — Offline (Synchronous) Planning
──────────────────────────────────────────────
The controller waits for the VLM to finish planning before starting to
execute.  Simpler but every VLM call adds latency directly to the
instruction-boundary.

  while running:
      (x_t, Pn_{t+1}) ← π_p.step(o_t, Pn_t, Stop)
      if x_t: Stop←0
      (a_t, s_t) ← π_c.step(o_t, x_t)
      env.step(a_t)
      if s_t: Stop←1
"""

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared state between controller and VLM threads (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AsyncState:
    """Shared mutable state for the async planner–controller interface."""
    inbox:     Optional[np.ndarray] = None   # lang_embed of next instruction
    inbox_txt: Optional[str]        = None   # raw text (for logging)
    stop:      bool                 = False  # controller signals completion
    stop_cause: str                 = ''     # head | timer | stuck | anomaly
    inbox_meta: Optional[dict]      = None   # audit: S/F text the planner saw + its raw output
    plan:      Optional[str]        = None   # VLM partial plan (Pn_t)
    obs:       Optional[Any]        = None   # latest observation (updated by ctrl)
    running:   bool                 = True
    _lock:     threading.Lock       = field(default_factory=threading.Lock)

    def set_stop(self, cause: str = ''):
        with self._lock:
            self.stop = True; self.stop_cause = cause
    def put_meta(self, meta: dict):
        with self._lock:
            self.inbox_meta = meta
    def pop_meta(self) -> Optional[dict]:
        with self._lock:
            m, self.inbox_meta = self.inbox_meta, None
            return m

    def clear_stop(self):
        with self._lock:
            self.stop = False

    def put_inbox(self, embed: np.ndarray, text: str):
        with self._lock:
            self.inbox     = embed
            self.inbox_txt = text

    def pop_inbox(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        with self._lock:
            embed, txt = self.inbox, self.inbox_txt
            self.inbox     = None
            self.inbox_txt = None
            return embed, txt

    def update_obs(self, obs):
        with self._lock:
            self.obs = obs

    def get_obs(self):
        with self._lock:
            return self.obs


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 1: Asynchronous online inference
# ─────────────────────────────────────────────────────────────────────────────

class AsyncOnlineInference:
    """
    Runs π_c (controller) and π_p (VLM planner) as concurrent threads.

    The controller runs at the environment's control frequency; the VLM
    reasons in the background and emits instructions whenever Stop fires.
    Both threads communicate through AsyncState.

    Parameters
    ----------
    controller : agent.LangCondAgent (or any policy with .policy() / encode_instruction())
    planner    : planners.VLMPlanner
    env        : gymnasium-compatible environment
    lang_encoder : callable(str) -> np.ndarray  (instruction → embedding)
    stop_threshold : float  p_stop > threshold triggers Stop signal
    max_steps  : int  episode horizon
    """

    def __init__(
        self,
        controller,
        planner,
        env,
        lang_encoder,
        stop_threshold: float = 0.5,
        max_steps: int = 18_000,
        record: bool = False,
        stop_patience: int = 1,
        min_instr_steps: int = 1,
        max_instr_steps: Optional[int] = None,
        stop_probe: Optional[str] = None,
        stop_signal: str = 'p_stop',
        stop_window: int = 1,
        state_fmt=None,
        feedback: bool = True,
        stuck_steps: int = 120,
        anomaly=None,
        anomaly_patience: int = 5,
    ):
        # Planner harness: text state (StateFormatter), controller feedback (done / stuck / anomaly / working, with
        # state-based completion verification), optional feature-space anomaly detector (planners.anomaly.FeatAnomaly).
        self.state_fmt = state_fmt; self.feedback = bool(feedback); self.stuck_steps = int(stuck_steps)
        self.anomaly = anomaly; self.anomaly_patience = int(anomaly_patience)
        # Which exported signal drives hand-off: 'p_stop' (stop head), 'r_lang' (event-reward head, option 4), 'p_soon' (imagination).
        # stop_window K > 1 uses the running max of the signal over the last K steps (event pulses are momentary).
        self.stop_signal = stop_signal; self.stop_window = max(1, int(stop_window))
        self.ctrl       = controller
        self.planner    = planner
        self.env        = env
        self.encode     = lang_encoder
        self.stop_thr   = stop_threshold
        self.max_steps  = max_steps
        self.record     = record
        # Optional external stop probe (fit offline on recorded rollouts with ground-truth completion labels):
        # pickle with {'model': sklearn classifier, 'instructions': [str], 'feat_dim': int}; input = [feat, onehot(instruction)].
        self.stop_probe = None
        if stop_probe:
            import joblib
            self.stop_probe = joblib.load(stop_probe)
            print(f"[stop_probe] loaded {stop_probe}: {type(self.stop_probe['model']).__name__}, {len(self.stop_probe['instructions'])} instructions")
        # Stop-signal gating (see run_episode for semantics).
        self.stop_patience   = max(1, int(stop_patience))     # consecutive steps p_stop>thr to fire
        self.min_instr_steps = max(1, int(min_instr_steps))   # min dwell before stop can fire
        self.max_instr_steps = max_instr_steps                # force replan after K steps (None=off)

    def run_episode(self) -> Dict:
        """
        Run one full episode with async planning.
        Returns a dict with reward, trajectory, instruction log, latency stats.
        """
        shared     = AsyncState()
        ctrl_state = self.ctrl.init_policy(batch_size=1)
        trajectory = []; sig_hist = []
        issue_inv = None; last_change = 0; stuck_fired = False; anom_run = 0; anom_fired = False; status = 'working'; prev_inv = None
        metrics    = {
            'total_reward': 0.0,
            'episode_steps': 0,
            'instructions_issued': 0,
            'instruction_log': [],
            'p_stop_log': [],
            'feedback_stuck': 0, 'feedback_anomaly': 0, 'verified_yes': 0, 'verified_no': 0, 'verified_unknown': 0,
            'handoff_head': 0, 'handoff_timer': 0, 'handoff_stuck': 0, 'handoff_anomaly': 0,
            'vlm_latencies_ms': [],   # time the VLM took per call
            'staleness_steps': [],    # steps since last obs when instr arrived
        }

        obs, _ = self.env.reset()
        shared.update_obs(obs)

        # Current instruction embedding (zeros = no instruction)
        null_embed = np.zeros(self.ctrl._lang_size if hasattr(self.ctrl, '_lang_size')
                              else 384, dtype=np.float32)
        cur_embed  = null_embed.copy()
        cur_text   = ''

        # Planner reasoning state (shared between the synchronous warm-up emit
        # below and the background thread).
        pstate = {'memory': {}, 'plan': None}

        # Seed the FIRST instruction synchronously so the controller never starts
        # the episode acting uninstructed (cold-start) — it's a one-time cost.
        t0 = time.perf_counter()
        instr0, pstate['plan'], pstate['memory'] = self.planner.emit(
            obs=obs, plan=pstate['plan'], memory=pstate['memory']
        )
        metrics['vlm_latencies_ms'].append((time.perf_counter() - t0) * 1000)
        cur_embed = self.encode(instr0)
        cur_text  = instr0
        metrics['instructions_issued'] += 1
        metrics['instruction_log'].append({'step': 0, 'instruction': cur_text})

        # ── VLM background thread ─────────────────────────────────────────────
        def vlm_loop():
            while shared.running:
                latest_obs = shared.get_obs()
                if latest_obs is None:
                    time.sleep(0.01)
                    continue

                t0 = time.perf_counter()

                if shared.stop:
                    # Emit next instruction (fast path: draft plan available)
                    cause = shared.stop_cause
                    instr_text, pstate['plan'], pstate['memory'] = self.planner.emit(
                        obs=latest_obs, plan=pstate['plan'], memory=pstate['memory']
                    )
                    embed = self.encode(instr_text)
                    shared.put_meta({'cause': cause,
                                     'S': latest_obs.get('_state_text') if isinstance(latest_obs, dict) else None,
                                     'F': latest_obs.get('_feedback') if isinstance(latest_obs, dict) else None,
                                     'planner': getattr(self.planner, '_last_result', None)})
                    shared.put_inbox(embed, instr_text)
                    shared.clear_stop()
                    metrics['vlm_latencies_ms'].append(
                        (time.perf_counter() - t0) * 1000
                    )
                else:
                    # Advance background reasoning (slow path: no new instruction)
                    pstate['plan'], pstate['memory'] = self.planner.advance(
                        obs=latest_obs, plan=pstate['plan'], memory=pstate['memory']
                    )
                # Yield CPU so the controller thread isn't starved
                time.sleep(0.001)

        vlm_thread = threading.Thread(target=vlm_loop, daemon=True)
        vlm_thread.start()

        # Stop-signal gating state.
        #   awaiting     — a stop was requested; we keep executing the *current*
        #                  instruction until the replacement arrives, so the
        #                  controller is never run uninstructed during VLM latency.
        #   instr_steps  — steps spent on the current instruction (dwell).
        #   stop_run     — consecutive steps with p_stop above threshold (debounce).
        awaiting    = False
        instr_steps = 0
        stop_run    = 0
        issue_step  = 0       # control step at which the current instruction was issued

        # ── Controller main loop ──────────────────────────────────────────────
        for step in range(self.max_steps):
            # Check if a new instruction arrived (replaces whatever is active).
            new_embed, new_text = shared.pop_inbox()
            if new_embed is not None:
                cur_embed = new_embed
                cur_text  = new_text
                metrics['staleness_steps'].append(int(step - issue_step))
                metrics['instructions_issued'] += 1
                metrics['instruction_log'].append({
                    'step': step, 'instruction': cur_text, **(shared.pop_meta() or {})
                })
                awaiting    = False
                instr_steps = 0
                stop_run    = 0
                issue_step  = step
                sig_hist    = []
                issue_inv   = self.state_fmt.inventory(obs) if self.state_fmt else None
                last_change = step; stuck_fired = False; anom_run = 0; anom_fired = False; status = 'working'

            # Controller step
            obs_batch  = {k: np.expand_dims(v, 0) for k, v in obs.items()
                          if not k.startswith('log/')}
            obs_batch['lang_embed'] = cur_embed[np.newaxis]

            inv_pre = np.array(obs.get('inventory', []), np.float32)   # relabel study
            ctrl_state, action, extra = self.ctrl.policy(
                ctrl_state, obs_batch, mode='eval'
            )
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
            if self.stop_signal != 'p_stop':
                sig_hist.append(float(extra.get(f'log/{self.stop_signal}', 0.0))); del sig_hist[:-self.stop_window]
                p_stop = max(sig_hist)
            if self.stop_probe is not None:
                sp = self.stop_probe; oh = np.zeros(len(sp['instructions']), np.float32)
                if cur_text in sp['instructions']: oh[sp['instructions'].index(cur_text)] = 1.0
                x = np.concatenate([np.asarray(extra['log/feat'][0], np.float32).ravel(), oh])[None]
                p_stop = float(sp['model'].predict_proba(x)[0, 1])
            metrics['p_stop_log'].append(p_stop)

            # Execute action
            action_np = np.array(action['action'][0])
            try:
                obs, reward, terminated, truncated, info = self.env.step(action_np)
            except (ConnectionRefusedError, TimeoutError, OSError):
                # Java server died mid-episode — end episode cleanly
                break
            metrics['total_reward']   += float(reward)
            metrics['episode_steps']  += 1
            instr_steps += 1

            # ── Controller feedback: done / stuck / anomaly / working, plus state-verified completion ──
            if self.state_fmt is not None:
                inv_now = self.state_fmt.inventory(obs)
                if prev_inv is not None and inv_now != prev_inv: last_change = step
                prev_inv = inv_now
            anom_z = None
            if self.anomaly is not None and 'log/feat' in extra:
                anom_z = self.anomaly.score(np.asarray(extra['log/feat'][0], np.float32))
                anom_run = anom_run + 1 if anom_z > self.anomaly.thr else 0
            stuck = instr_steps >= self.stuck_steps and (step - last_change) >= self.stuck_steps
            anomalous = anom_run >= self.anomaly_patience
            status = 'stuck' if stuck else ('anomaly' if anomalous else 'working')
            if self.feedback and self.state_fmt is not None:
                ver = self.state_fmt.verify(cur_text, obs, issue_inv)
                feedback_txt = (f"status={status}; steps on current instruction={instr_steps}; completion signal={p_stop:.2f}; "
                                f"completion verified by state={ver}")   # raw anomaly score deliberately not shown: status carries it
                obs_p = dict(obs); obs_p['_state_text'] = self.state_fmt.text(obs, issue_inv); obs_p['_feedback'] = feedback_txt
                shared.update_obs(obs_p)
            else:
                shared.update_obs(obs)
            if self.feedback and not awaiting:
                if stuck and not stuck_fired:
                    stuck_fired = True; metrics['feedback_stuck'] += 1; metrics['handoff_stuck'] += 1; awaiting = True
                    if self.state_fmt is not None: metrics['verified_' + self.state_fmt.verify(cur_text, obs, issue_inv)] += 1
                    shared.set_stop('stuck')
                elif anomalous and not anom_fired:
                    anom_fired = True; metrics['feedback_anomaly'] += 1; metrics['handoff_anomaly'] += 1; awaiting = True
                    if self.state_fmt is not None: metrics['verified_' + self.state_fmt.verify(cur_text, obs, issue_inv)] += 1
                    shared.set_stop('anomaly')

            if self.record:
                trajectory.append({
                    'action': action_np, 'reward': reward,
                    'p_stop': p_stop, 'instruction': cur_text, 'step': step,
                    'p_stop_head': float(extra.get('log/p_stop', 0.0)), 'r_lang': float(extra.get('log/r_lang', 0.0)), 'p_soon': float(extra.get('log/p_soon', 0.0)),
                    'status': status, 'anomaly_z': float(anom_z) if anom_z is not None else -1.0,
                    'feat': np.asarray(extra['log/feat'][0], np.float16), 'inv': inv_pre,
                    'inv_post': np.array(obs.get('inventory', []), np.float32),
                })

            # ── Decide whether the current instruction is complete ───────────
            # Debounced stop: require p_stop above threshold for `stop_patience`
            # consecutive steps, and at least `min_instr_steps` of dwell.
            stop_run = stop_run + 1 if p_stop > self.stop_thr else 0
            fire_stop = (instr_steps >= self.min_instr_steps
                         and stop_run >= self.stop_patience)
            # Anti-stuck: force a replan if the controller never signals done.
            force_stop = (self.max_instr_steps is not None
                          and instr_steps >= self.max_instr_steps)

            if (fire_stop or force_stop) and not awaiting:
                # Keep `cur_embed` active (do NOT null it) so the controller
                # keeps pursuing the last subgoal until the next instruction
                # lands — avoids an uninstructed window during VLM latency.
                awaiting = True
                stop_run = 0
                cause = 'head' if fire_stop else 'timer'; metrics['handoff_' + cause] += 1
                if self.state_fmt is not None:
                    metrics['verified_' + self.state_fmt.verify(cur_text, obs, issue_inv)] += 1
                shared.set_stop(cause)   # wake VLM to emit next instruction

            if terminated or truncated:
                break

        shared.running = False
        if self.record:
            metrics['trajectory'] = trajectory
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 2: Offline (synchronous) planning
# ─────────────────────────────────────────────────────────────────────────────

class OfflineInference:
    """
    Synchronous planning: the VLM waits for the controller to finish each
    instruction before generating the next one.  Throughput is limited by
    VLM inference latency at every instruction boundary.

    Corresponds to Algorithm 2 in the paper.
    """

    def __init__(
        self,
        controller,
        planner,
        env,
        lang_encoder,
        stop_threshold: float = 0.5,
        max_steps: int = 18_000,
        record: bool = False,
        stop_patience: int = 1,
        min_instr_steps: int = 1,
        max_instr_steps: Optional[int] = None,
        **_unused,
    ):
        self.ctrl      = controller
        self.planner   = planner
        self.env       = env
        self.encode    = lang_encoder
        self.stop_thr  = stop_threshold
        self.max_steps = max_steps
        self.record    = record
        self.stop_patience   = max(1, int(stop_patience))
        self.min_instr_steps = max(1, int(min_instr_steps))
        self.max_instr_steps = max_instr_steps

    def run_episode(self) -> Dict:
        ctrl_state = self.ctrl.init_policy(batch_size=1)
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)

        obs, _    = self.env.reset()
        plan      = None
        memory    = {}
        stop_flag = True    # True = request new instruction from planner

        cur_embed   = null_embed.copy()
        cur_text    = ''
        instr_steps = 0
        stop_run    = 0
        trajectory  = []
        metrics    = {
            'total_reward': 0.0,
            'episode_steps': 0,
            'instructions_issued': 0,
            'instruction_log': [],
            'p_stop_log': [],
            'feedback_stuck': 0, 'feedback_anomaly': 0, 'verified_yes': 0, 'verified_no': 0, 'verified_unknown': 0,
            'handoff_head': 0, 'handoff_timer': 0, 'handoff_stuck': 0, 'handoff_anomaly': 0,
            'vlm_latencies_ms': [],
        }

        for step in range(self.max_steps):

            # ── Planner step (synchronous; blocks controller) ─────────────────
            if stop_flag:
                t0 = time.perf_counter()
                instr_text, plan, memory = self.planner.step(
                    obs=obs, plan=plan, memory=memory, p_stop=float(stop_flag)
                )
                metrics['vlm_latencies_ms'].append(
                    (time.perf_counter() - t0) * 1000
                )
                if instr_text:
                    # Only swap in a new instruction when the planner emits one;
                    # otherwise keep executing the current one (no null window).
                    cur_embed   = self.encode(instr_text)
                    cur_text    = instr_text
                    stop_flag   = False
                    instr_steps = 0
                    stop_run    = 0
                    metrics['instructions_issued'] += 1
                    metrics['instruction_log'].append(
                        {'step': step, 'instruction': cur_text}
                    )

            # ── Controller step ───────────────────────────────────────────────
            obs_batch = {k: np.expand_dims(v, 0) for k, v in obs.items()
                         if not k.startswith('log/')}
            obs_batch['lang_embed'] = cur_embed[np.newaxis]

            inv_pre = np.array(obs.get('inventory', []), np.float32)   # relabel study
            ctrl_state, action, extra = self.ctrl.policy(
                ctrl_state, obs_batch, mode='eval'
            )
            p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
            metrics['p_stop_log'].append(p_stop)

            action_np = np.array(action['action'][0])
            try:
                obs, reward, terminated, truncated, info = self.env.step(action_np)
            except (ConnectionRefusedError, TimeoutError, OSError):
                break
            metrics['total_reward']  += float(reward)
            metrics['episode_steps'] += 1
            instr_steps += 1

            if self.record:
                trajectory.append({
                    'action': action_np, 'reward': reward,
                    'p_stop': p_stop, 'instruction': cur_text, 'step': step,
                    'p_stop_head': float(extra.get('log/p_stop', 0.0)), 'r_lang': float(extra.get('log/r_lang', 0.0)), 'p_soon': float(extra.get('log/p_soon', 0.0)),
                    'feat': np.asarray(extra['log/feat'][0], np.float16), 'inv': inv_pre,
                    'inv_post': np.array(obs.get('inventory', []), np.float32),
                })

            # Debounced + dwell-gated stop, with an anti-stuck forced replan.
            stop_run = stop_run + 1 if p_stop > self.stop_thr else 0
            fire_stop = (instr_steps >= self.min_instr_steps
                         and stop_run >= self.stop_patience)
            force_stop = (self.max_instr_steps is not None
                          and instr_steps >= self.max_instr_steps)
            if fire_stop or force_stop:
                stop_flag = True

            if terminated or truncated:
                break

        if self.record:
            metrics['trajectory'] = trajectory
        return metrics
