"""
LangReplayBuffer: extends the dreamerv3 / embodied replay buffer with
language-annotation fields required by Instruct-to-Act (§3.4).

Extra fields stored per timestep (on top of standard dreamerv3 fields):
  lang_embed  : float32 (lang_size,)  — MiniLM embedding of the instruction
                                         covering this timestep (zeros = no label)
  annotated   : float32 scalar       — 1 if this timestep belongs to an annotated
                                         segment, else 0
  complete    : float32 scalar       — 1 if the action at this step completed the
                                         current instruction (i.e. stop signal fired)

The buffer wraps dreamerv3's embodied.replay.Replay so that all existing
sampling / streaming / chunking logic is reused.

Public API used by the training loop (train_lang.py):
  buffer.get_frames(u, v)           → list of obs images for annotator
  buffer.get_actions(u, v)          → list of raw actions
  buffer.write_lang_embed(u, v, e, text)  → write embedding + set annotated=1
  buffer.write_complete(t, value)   → write stop/completion label
  buffer.dataset(batch, seq_len)    → identical to embodied.replay.Replay.dataset()
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dreamerv3'))

import threading
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import embodied


class LangReplayBuffer:
    """
    Thin wrapper around embodied.replay.Replay that manages language-annotation
    arrays in a parallel in-memory store (to avoid touching dreamerv3 internals).

    The in-memory lang arrays are synced from the circular buffer state so
    that indexing is always consistent with what embodied.Replay returns.

    Parameters
    ----------
    directory : str | Path
        Root directory for episode storage (passed to embodied.Replay).
    capacity : int
        Maximum number of timesteps stored (default 1_000_000 = ~1M as in paper).
    min_length : int
        Minimum episode length for sampling (dreamerv3 default: 64).
    max_length : int
        Maximum chunk length for sampling (dreamerv3 default: 64).
    lang_size : int
        Dimension of the language embedding (default 384).
    obs_key : str
        Observation key containing visual frames (used by annotator).
    """

    def __init__(
        self,
        directory: str,
        capacity: int = 1_000_000,
        min_length: int = 64,
        max_length: int = 64,
        lang_size: int = 384,
        obs_key: str = 'image',
        finish: bool = False,
    ):
        self._dir     = Path(directory)
        self._finish_enabled = bool(finish)
        self._cap     = capacity
        self._minlen  = min_length
        self._maxlen  = max_length
        self._lang_size = lang_size
        self._obs_key   = obs_key

        # Underlying dreamerv3 replay (handles all episode I/O)
        self._replay = embodied.replay.Replay(
            length=max_length,
            capacity=capacity,
            directory=self._dir,
        )

        # Parallel in-memory circular arrays for language annotations
        self._lang_embed = np.zeros((capacity, lang_size), dtype=np.float32)
        self._annotated  = np.zeros(capacity, dtype=np.float32)
        self._complete   = np.zeros(capacity, dtype=np.float32)
        self._finish     = np.zeros(capacity, dtype=np.float32)   # [finish]-action BC target (ablation)
        self._frames     = [None] * capacity   # raw obs for annotator
        self._ach        = [None] * capacity   # log/achievement_* counts (crafter) for ground-truth completion
        self._is_first   = np.zeros(capacity, dtype=np.bool_)
        self._worker     = np.full(capacity, -1, dtype=np.int32)   # env index per slot (slots interleave envs)
        self._inv        = [None] * capacity   # tracked inventory counts (minecraft) for ground-truth completion
        self._inv_idx    = None
        self._actions    = [None] * capacity   # raw actions for annotator

        self._ptr  = 0      # write pointer (matches circular buffer)
        self._size = 0      # number of valid timesteps
        self._total_added = 0   # monotone count of add() calls (never wraps)
        self._lock = threading.Lock()

    # ── Step-level write ──────────────────────────────────────────────────────

    def add(self, transition: dict, worker: int = 0):
        """
        Add one transition to the buffer.

        We tag every transition with '_buf_idx' (our internal circular-buffer
        slot index) before passing it to embodied.replay.  When dataset() later
        samples a batch, each timestep carries its '_buf_idx', which lets us
        look up the correct lang_embed / annotated / complete values from our
        in-memory arrays — no more random misalignment.
        """
        with self._lock:
            idx = self._ptr % self._cap

            # Cache frame and action for the annotator
            if self._obs_key in transition:
                self._frames[idx] = transition[self._obs_key]
            if 'action' in transition:
                self._actions[idx] = transition['action']
            ach = [transition[k] for k in sorted(transition) if k.startswith('log/achievement_')]
            self._ach[idx] = np.asarray(ach, dtype=np.int32) if ach else None
            self._is_first[idx] = bool(transition.get('is_first', False))
            self._worker[idx] = int(worker)
            if self._inv_idx is not None and 'inventory' in transition:
                self._inv[idx] = np.asarray(transition['inventory'])[self._inv_idx].astype(np.int32)
            # Completion label (may be 0 until inference annotates it)
            self._complete[idx]   = float(transition.get('complete', 0.0))
            self._finish[idx]     = 0.0
            self._lang_embed[idx] = 0.0
            self._annotated[idx]  = 0.0

            self._ptr  = (self._ptr + 1) % self._cap
            self._size = min(self._size + 1, self._cap)
            self._total_added += 1

        # Embed our circular-buffer slot so dataset() can recover alignment.
        full = dict(transition)
        full['_buf_idx'] = np.int32(idx)
        self._replay.add(full, worker)

    # ── Annotator write-back ──────────────────────────────────────────────────

    def write_lang_embed(
        self,
        u: int,
        v: int,
        embed: np.ndarray,
        instruction: str = '',
    ):
        """Write a language embedding into every slot [u, v] (inclusive)."""
        indices = np.arange(u, v + 1) % self._cap
        with self._lock:
            self._lang_embed[indices] = embed
            self._annotated[indices]  = 1.0

    def write_complete(self, t: int, value: float = 1.0):
        with self._lock:
            self._complete[t % self._cap] = value

    def write_complete_range(self, u: int, v: int, value: float = 1.0):
        """done-state label over [u, v] inclusive (ground-truth completion annotator)."""
        with self._lock:
            self._complete[np.arange(u, v + 1) % self._cap] = value

    def set_inventory_tracking(self, indices):
        """indices into the env's inventory vector for the items whose increments define completion events"""
        self._inv_idx = np.asarray(indices, dtype=np.int64)

    def get_inventory(self, u: int, v: int):
        with self._lock:
            return [self._inv[i % self._cap] for i in range(u, v + 1)]

    def get_workers(self, u: int, v: int):
        with self._lock:
            return [int(self._worker[i % self._cap]) for i in range(u, v + 1)]

    def write_lang_embed_idx(self, indices, embed: np.ndarray):
        idx = np.asarray(indices) % self._cap
        with self._lock:
            self._lang_embed[idx] = embed; self._annotated[idx] = 1.0

    def write_finish_idx(self, indices, value: float):
        with self._lock:
            for t in indices: self._finish[t % self._cap] = value

    def write_complete_idx(self, indices, value: float):
        idx = np.asarray(indices) % self._cap
        with self._lock:
            self._complete[idx] = value

    def get_achievements(self, u: int, v: int):
        with self._lock:
            return [self._ach[i % self._cap] for i in range(u, v + 1)]

    def get_is_first(self, u: int, v: int):
        with self._lock:
            return [bool(self._is_first[i % self._cap]) for i in range(u, v + 1)]

    # ── Annotator read helpers ─────────────────────────────────────────────────

    def get_frames(self, u: int, v: int) -> List[np.ndarray]:
        with self._lock:
            return [self._frames[i % self._cap]
                    for i in range(u, v + 1)
                    if self._frames[i % self._cap] is not None]

    def get_frames_idx(self, indices) -> List[np.ndarray]:
        with self._lock:
            fr = [self._frames[i % self._cap] for i in indices]
        return [f for f in fr if f is not None]

    def get_actions_idx(self, indices) -> list:
        with self._lock:
            return [self._actions[i % self._cap] for i in indices]

    def get_actions(self, u: int, v: int) -> list:
        with self._lock:
            return [self._actions[i % self._cap]
                    for i in range(u, v + 1)
                    if self._actions[i % self._cap] is not None]

    # ── Dataset / sampling ────────────────────────────────────────────────────

    def dataset(self, batch: int, length: int):
        """
        Yield batches with correctly-aligned language fields.

        Each transition stored by add() carries a '_buf_idx' field whose value
        is the exact slot in our in-memory lang arrays.  We use it here for a
        precise, zero-misalignment lookup — no random sampling needed.
        """
        while True:
            # dreamerv3's Replay uses sample() not dataset()
            batch_data = self._replay.sample(batch)
            batch_data = dict(batch_data)

            if '_buf_idx' in batch_data:
                # Shape: (B, T) int32 array of circular-buffer slots.
                indices = np.asarray(batch_data.pop('_buf_idx'))   # remove internal field
                with self._lock:
                    # Advanced indexing: shape (B, T, lang_size) and (B, T)
                    lang_embeds = self._lang_embed[indices]
                    annotated   = self._annotated[indices]
                    complete    = self._complete[indices]
                    finish      = self._finish[indices]
            else:
                # Fallback if the field is missing (e.g. episodes pre-dating this fix).
                B, T = batch_data['reward'].shape[:2]
                lang_embeds = np.zeros((B, T, self._lang_size), dtype=np.float32)
                annotated   = np.zeros((B, T), dtype=np.float32)
                complete    = np.zeros((B, T), dtype=np.float32)
                finish      = np.zeros((B, T), dtype=np.float32)

            batch_data['lang_embed'] = lang_embeds
            batch_data['annotated']  = annotated
            batch_data['complete']   = complete
            if self._finish_enabled: batch_data['finish'] = finish
            # dreamerv3 expects 'consec' (consecutive chunk index) in every batch.
            # sample() doesn't add it, so we default to 0.
            if 'consec' not in batch_data:
                B, T = batch_data['reward'].shape[:2]
                batch_data['consec'] = np.zeros((B, T), dtype=np.int32)
            yield batch_data

    # ── Misc / delegation ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._replay)

    @property
    def n_timesteps(self) -> int:
        """
        Actual number of valid timestep slots in the buffer.

        After a fresh start, _size grows with each add() call.
        After loading a checkpoint, _size=0 but _replay already has chunks;
        use len(_replay) * _maxlen as the estimate so annotation isn't skipped.
        """
        loaded_estimate = len(self._replay) * self._maxlen
        return min(max(self._size, loaded_estimate), self._cap)

    @property
    def total_added(self) -> int:
        """Monotone number of timesteps ever added (does not wrap at capacity).
        Used by the annotate-once cursor so data past one full buffer rotation
        still gets annotated exactly once."""
        return max(self._total_added, self.n_timesteps)

    def stats(self):
        return self._replay.stats()

    def update(self, data):
        return self._replay.update(data)

    def save(self):
        return self._replay.save()

    def load(self, data):
        return self._replay.load(data)

    @property
    def replay(self):
        return self._replay
