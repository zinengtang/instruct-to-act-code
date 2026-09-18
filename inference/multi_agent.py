"""
Multi-agent inference with shared controller and VLM-based coordination (§3.1).

Architecture
────────────
• n agents share controller parameters θ_c but maintain independent
  recurrent states and execute distinct action streams.
• Each agent i has its own VLM planner π_p^(i).
• Planners coordinate via a Chatroom[t], a shared message buffer that
  each planner can read and (optionally) write to at each step.
• Two communication modes (§3.1, App A.7):
    Decentralized — any agent can message any other (m_{ij})
    Centralized   — hub agent h broadcasts role assignments; others reply sparsely.

Synchronization with async planning
─────────────────────────────────────
Agents communicate "in discrete rounds aligned with the control step" (paper §3.1).
Messages are buffered and flushed at instruction boundaries, so that the chatroom
state is consistent across agents at every new instruction.

The multi-agent inference runs n (planner, controller) pairs in parallel threads,
with a Chatroom mediating communication between the planner threads.
"""

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from .algorithms import AsyncState


# ─────────────────────────────────────────────────────────────────────────────
# Chatroom
# ─────────────────────────────────────────────────────────────────────────────

class Chatroom:
    """
    Shared message buffer for multi-agent communication.

    Chatroom[t] = list of (sender_idx, recipient_idx_or_'all', message_str)
    at control step t.

    Messages are committed to the shared log at each control step so all
    agents see a consistent view when they next call read().
    """

    def __init__(self, n_agents: int, mode: str = 'decentralized', hub: int = 0):
        assert mode in ('decentralized', 'centralized', 'no_comm')
        self.n_agents = n_agents
        self.mode     = mode
        self.hub      = hub
        self._log: List[Dict] = []
        self._pending: List[Dict] = []
        self._lock = threading.Lock()
        self._step = 0

    def send(self, sender: int, recipient, message: str, step: int):
        """Queue a message. In no_comm mode all messages are silently dropped."""
        if self.mode == 'no_comm':
            return
        if self.mode == 'centralized':
            if sender != self.hub and recipient != self.hub:
                return
        with self._lock:
            self._pending.append({
                'step': step, 'from': sender,
                'to': recipient, 'text': message,
            })

    def flush(self, step: int):
        """Commit pending messages for this control step."""
        with self._lock:
            self._log.extend(self._pending)
            self._pending = []
            self._step = step

    def read(self, agent: int, since_step: int = 0) -> List[Dict]:
        """Return messages addressed to `agent` or 'all' since `since_step`."""
        with self._lock:
            return [m for m in self._log
                    if m['step'] >= since_step
                    and (m['to'] == 'all' or m['to'] == agent)]

    def format_for_planner(self, agent: int, since_step: int = 0) -> str:
        """Format relevant messages as text for the planner's context."""
        msgs = self.read(agent, since_step)
        if not msgs:
            return ''
        lines = []
        for m in msgs[-10:]:   # last 10 messages
            to_str = '@all' if m['to'] == 'all' else f'@agent{m["to"]}'
            lines.append(f"[step {m['step']}] @agent{m['from']} → {to_str}: {m['text']}")
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent inference
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentInference:
    """
    Runs n (planner, controller) pairs with a shared Chatroom.

    The shared controller parameters are used by all agents; each agent
    maintains its own recurrent state (distinct history / obs stream).

    Parameters
    ----------
    controller    : LangCondAgent  (shared, parameters frozen)
    planners      : List[VLMPlanner]  — one per agent
    envs          : List  — per-agent environment handles
    lang_encoder  : callable(str) -> np.ndarray
    mode          : 'decentralized' | 'centralized'
    hub           : int  — hub agent index (centralized only)
    stop_threshold: float
    max_steps     : int
    """

    def __init__(
        self,
        controller,           # single shared controller OR list of per-agent controllers
        planners: List,
        envs: List,
        lang_encoder,
        mode: str = 'decentralized',
        hub: int = 0,
        stop_threshold: float = 0.5,
        max_steps: int = 18_000,
        record: bool = False,
    ):
        assert len(planners) == len(envs), "Need one env per agent"
        n = len(planners)

        # Support both a single shared controller and per-agent separate controllers.
        if isinstance(controller, list):
            assert len(controller) == n, "Need one controller per agent for separate_ctrl"
            self.ctrls = controller
        else:
            self.ctrls = [controller] * n   # shared parameters

        self.planners   = planners
        self.envs       = envs
        self.n_agents   = n
        self.encode     = lang_encoder
        self.mode       = mode
        self.hub        = hub
        self.stop_thr   = stop_threshold
        self.max_steps  = max_steps
        self.record     = record
        self.chatroom   = Chatroom(self.n_agents, mode, hub)

        # Keep a single-controller reference for backward compat with callers
        # that access .ctrl directly.
        self.ctrl = self.ctrls[0]

    def run_episode(self) -> Dict:
        """
        Run one cooperative multi-agent episode.
        Returns per-agent metrics plus team reward.
        """
        lang_size  = getattr(self.ctrl, '_lang_size', 384)
        null_embed = np.zeros(lang_size, dtype=np.float32)

        # Per-agent state (each agent uses its own controller instance)
        ctrl_states   = [self.ctrls[i].init_policy(batch_size=1)
                         for i in range(self.n_agents)]
        cur_embeds    = [null_embed.copy() for _ in range(self.n_agents)]
        cur_texts     = ['' for _ in range(self.n_agents)]
        stop_flags    = [True] * self.n_agents    # request instruction at start
        memories      = [{} for _ in range(self.n_agents)]
        plans         = [None] * self.n_agents

        # Reset envs
        obs_list   = [env.reset()[0] for env in self.envs]
        done_flags = [False] * self.n_agents

        metrics = {
            'agent_rewards': [0.0] * self.n_agents,
            'team_reward':   0.0,
            'episode_steps': 0,
            'instruction_logs': [[] for _ in range(self.n_agents)],
            'message_logs': [],
            'p_stop_logs':  [[] for _ in range(self.n_agents)],
        }

        for step in range(self.max_steps):

            # ── Flush chatroom for this step ──────────────────────────────────
            self.chatroom.flush(step)

            # ── Planner steps (parallel, one per agent) ───────────────────────
            new_instrs = [None] * self.n_agents

            def planner_step(i):
                if not stop_flags[i]:
                    return
                chat_ctx = self.chatroom.format_for_planner(i, since_step=max(0, step - 5))
                instr_text, plans[i], memories[i] = self.planners[i].step(
                    obs=obs_list[i], plan=plans[i], memory=memories[i],
                    p_stop=1.0, chat_context=chat_ctx,
                )
                if instr_text:
                    new_instrs[i] = instr_text
                # Extract optional chat message from planner output
                out_msg = memories[i].pop('_outgoing_message', None)
                if out_msg:
                    recipient = out_msg.get('to', 'all')
                    self.chatroom.send(i, recipient, out_msg['text'], step)
                    metrics['message_logs'].append(
                        {'step': step, 'from': i, 'to': recipient, 'text': out_msg['text']}
                    )

            threads = [threading.Thread(target=planner_step, args=(i,))
                       for i in range(self.n_agents)]
            for t in threads: t.start()
            for t in threads: t.join()

            # Update instructions
            for i in range(self.n_agents):
                if new_instrs[i] is not None:
                    cur_embeds[i] = self.encode(new_instrs[i])
                    cur_texts[i]  = new_instrs[i]
                    stop_flags[i] = False
                    metrics['instruction_logs'][i].append(
                        {'step': step, 'instruction': new_instrs[i]}
                    )

            # ── Controller steps (independent per agent, shared params) ───────
            for i in range(self.n_agents):
                if done_flags[i]:
                    continue

                obs_b = {k: np.expand_dims(v, 0) for k, v in obs_list[i].items()}
                obs_b['lang_embed'] = cur_embeds[i][np.newaxis]

                ctrl_states[i], action, extra = self.ctrls[i].policy(
                    ctrl_states[i], obs_b, mode='eval'
                )
                p_stop = float(extra.get('log/p_stop', extra.get('p_stop', 0.0)))
                metrics['p_stop_logs'][i].append(p_stop)

                action_np = np.array(action['action'][0])
                obs_new, rew, term, trunc, _ = self.envs[i].step(action_np)
                obs_list[i] = obs_new

                metrics['agent_rewards'][i] += float(rew)
                metrics['team_reward']      += float(rew)

                if p_stop > self.stop_thr:
                    cur_embeds[i] = null_embed.copy()
                    stop_flags[i] = True

                if term or trunc:
                    done_flags[i] = True

            metrics['episode_steps'] += 1

            if all(done_flags):
                break

        return metrics
