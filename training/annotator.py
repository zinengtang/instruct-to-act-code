"""
Post-hoc instruction annotation pipeline (§3.4, App A.7).

During training, the controller collects rollouts WITHOUT language conditioning.
These rollouts are stored in the replay buffer.  A VLM (default: GPT-4o) then
summarizes randomly-selected contiguous segments into natural-language
instructions.  Those instructions are encoded with MiniLM and stored back in
the replay buffer as e_t embeddings.

Algorithm:
  1. Sample non-overlapping intervals I = {[u_k, v_k]}^K whose total length
     is ≈ annotate_fraction × |buffer| (default 50%).
  2. For each interval, call VLM with the training summarisation prompt
     (Fig. 3) to get one instruction string.
  3. Encode the string with MiniLM → 384-dim embedding.
  4. Write the embedding to every timestep in [u_k, v_k] in the buffer.

The annotation runs asynchronously alongside training so it adds ~17% GPU
hours and a ~12% training slowdown (App A.4).

Supported annotator types
──────────────────────────
  'gpt4o'        → OpenAI GPT-4o  (paper default)
  'qwen'         → Qwen-VL-2.5-72B (via API or local)
  'gemma'        → Gemma-3-27B (via API or local)
  'llava'        → LLaVA-v1.6-34b (local)
  'template'     → Rule-based templates (ablation)
  'cluster'      → k-means cluster labels (ablation)
  'random'       → Random strings (ablation / lower bound)
"""

import os
import base64
import random
import json
import threading
import sys
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_openai_key

# ── Lazy imports ──────────────────────────────────────────────────────────────

def _sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def _openai_client():
    import openai, os
    base = os.environ.get('I2A_VLM_BASE_URL')   # set to a local vLLM server (OpenAI-compatible) to annotate with an open VLM
    if base: return openai.OpenAI(api_key='EMPTY', base_url=base, timeout=600)
    return openai.OpenAI(api_key=load_openai_key())
_LOCAL_FLAGS = ('done_label', 'finish_action', 'done_pre_max', 'done_post_max', 'segment_chunk', 'segment_done_len', 'segment_frames', 'segment_upscale')   # annotator flags that must not reach the API


def _hf_vlm_pipeline(model_name: str, device: str = 'cuda'):
    """Load a HuggingFace vision-language pipeline for open-source annotators."""
    from transformers import pipeline
    return pipeline(
        'image-text-to-text',
        model=model_name,
        device=0 if device == 'cuda' else -1,
        torch_dtype='auto',
    )


# ── Prompt template (Fig. 3 verbatim) ────────────────────────────────────────

SUMMARIZER_PROMPT = """\
You are a concise action summarizer for training labels. Read inputs IN THIS ORDER:

1. TaskGuidance G (what succeeds in this environment) and a short Message M (any extra note),
2. sampled visual Frames F[t : t+L-1] with timestamps,
3. the executed Actions A[t : t+L-1] with timestamps.

Goal
produce ONE short imperative instruction x that, if given before t, would make a competent \
controller reproduce A[t : t+L-1]. Keep it atomic and task-grounded.

Rules
• Focus on intent, not low-level joystick/button spam.
• Refer only to what is visible/achieved in Frames and consistent with Actions.
• No hallucinations; if ambiguous, pick the MINIMAL instruction that explains A.
• 6–18 tokens; one clause; present tense verb first word.
• If the segment ends at completion, align x to that subgoal's completion boundary.

Output JSON ONLY
{"instruction": "<single-line imperative>"}
"""

SEGMENT_PROMPT = """\
You are an action segmenter for training labels. You see F evenly spaced visual frames (with their step indices) and the \
executed actions of one contiguous rollout chunk of a single agent, plus TaskGuidance G.

Goal
Split the chunk into consecutive subgoal segments. For each segment give the step index where the subgoal is COMPLETED \
(the step at which the effect becomes visible / the agent switches to a different subgoal) and ONE short imperative \
instruction (6-18 tokens, present-tense verb first) that would make a competent controller reproduce that segment.

Rules
- Segments must be in order, non-overlapping, cover the chunk; 1 to 6 segments; each at least 8 steps long.
- 'end' is the completion step of the segment (inclusive); the next segment starts at end+1.
- Refer only to what is visible or achieved; if nothing identifiable happens, use a single navigation/exploration segment.

Output JSON ONLY
{"segments": [{"end": <int step index>, "instruction": "<imperative>"}, ...]}
"""

# Template-based instruction patterns (ablation)
TEMPLATE_PATTERNS = [
    "move {direction}",
    "collect {resource}",
    "attack {enemy}",
    "craft {item}",
    "place {block}",
    "explore {area}",
    "interact with {object}",
    "navigate to {location}",
    "avoid {hazard}",
    "pick up {item}",
]

TEMPLATE_SLOTS = {
    "direction": ["left", "right", "forward", "toward the goal"],
    "resource": ["wood", "stone", "food", "water", "items"],
    "enemy": ["the enemy", "obstacles"],
    "item": ["tools", "weapons", "supplies"],
    "block": ["a block", "material"],
    "area": ["the area", "surroundings", "nearby region"],
    "object": ["the object", "lever", "switch"],
    "location": ["the target", "next objective"],
    "hazard": ["danger", "enemies", "obstacles"],
}


# ─────────────────────────────────────────────────────────────────────────────

ACHIEVEMENT_INSTRUCTIONS = {   # crafter achievement -> instruction text (ground-truth completion annotator)
    'collect_wood': 'collect wood from a tree', 'place_table': 'place a crafting table',
    'make_wood_pickaxe': 'make a wood pickaxe', 'make_wood_sword': 'make a wood sword',
    'collect_stone': 'collect stone', 'place_stone': 'place a stone block', 'make_stone_pickaxe': 'make a stone pickaxe',
    'make_stone_sword': 'make a stone sword', 'collect_coal': 'collect coal', 'place_furnace': 'place a furnace',
    'collect_iron': 'collect iron', 'make_iron_pickaxe': 'make an iron pickaxe', 'make_iron_sword': 'make an iron sword',
    'collect_diamond': 'collect a diamond', 'collect_drink': 'drink water', 'collect_sapling': 'collect a sapling',
    'place_plant': 'plant a sapling', 'eat_plant': 'eat the plant', 'eat_cow': 'eat a cow',
    'defeat_zombie': 'defeat the zombie', 'defeat_skeleton': 'defeat the skeleton', 'wake_up': 'sleep and wake up',
}
ACHIEVEMENT_NAMES = sorted(ACHIEVEMENT_INSTRUCTIONS)
# minecraft: inventory item -> instruction (the scripted planner's sentences); events = the item count increasing
INVENTORY_INSTRUCTIONS = {
    'log': 'collect wood from nearby trees', 'planks': 'craft planks from collected wood', 'crafting_table': 'craft a crafting table',
    'stick': 'craft sticks for tool recipes', 'wooden_pickaxe': 'craft a wooden pickaxe', 'cobblestone': 'mine stone to gather cobblestone',
    'stone_pickaxe': 'craft a stone pickaxe', 'iron_ore': 'mine iron ore with the stone pickaxe', 'furnace': 'craft a furnace near your position',
    'iron_ingot': 'smelt iron ore into ingots', 'iron_pickaxe': 'craft an iron pickaxe', 'diamond': 'mine diamond ore with the iron pickaxe',
}
INVENTORY_ITEMS = list(INVENTORY_INSTRUCTIONS)   # tracking order == index order in the replay cache   # == sorted(log/achievement_*) key order in the replay cache


class PostHocAnnotator:
    """
    Annotates replay buffer segments with natural-language instructions.

    Parameters
    ----------
    annotator_type : str
        One of: 'gpt4o', 'qwen', 'gemma', 'llava', 'template', 'cluster', 'random'
    task_guidance : str
        Environment-specific task description passed to the VLM as context G.
    annotate_fraction : float
        Fraction of replay buffer timesteps to annotate per pass (default 0.50).
    segment_len_range : tuple
        (min_L, max_L) for randomly sampled segment length (default (1, 20)).
    lang_size : int
        Dimension of language embedding (default 384 for MiniLM-L6-H384).
    vlm_kwargs : dict
        Extra kwargs forwarded to the VLM API call (e.g. temperature, max_tokens).
    """

    def __init__(
        self,
        annotator_type: str = 'gpt4o',
        task_guidance: str = '',
        annotate_fraction: float = 0.50,
        segment_len_range: Tuple[int, int] = (1, 20),
        lang_size: int = 384,
        vlm_kwargs: Optional[Dict] = None,
        log_annotations: bool = True,
        log_dir: str = 'logs/annotations',
        annotate_once: bool = False,
        max_calls: Optional[int] = None,
    ):
        self.annotator_type = annotator_type
        self.annotate_once = annotate_once
        self.max_calls = max_calls
        self._cursor = 0        # absolute timestep up to which we've annotated
        self._n_calls = 0       # cumulative VLM API calls (budget guard)
        self._cap_warned = False
        self.task_guidance = task_guidance
        self.annotate_fraction = annotate_fraction
        self.seg_min, self.seg_max = segment_len_range
        self.lang_size = lang_size
        self.vlm_kwargs = vlm_kwargs or {}
        self.log_annotations = log_annotations
        self.log_dir = Path(log_dir)
        if log_annotations:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._encoder = None    # lazy-loaded SentenceTransformer
        self._vlm = None        # lazy-loaded VLM client
        self._lock = threading.Lock()
        self._annotation_log: List[Dict] = []

    # ── Encoding ──────────────────────────────────────────────────────────────

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = _sentence_transformer()
        return self._encoder

    def encode(self, text: str) -> np.ndarray:
        """Encode one instruction string → (lang_size,) float32 array."""
        emb = self.encoder.encode([text], normalize_embeddings=True)
        return emb[0].astype(np.float32)

    def null_embed(self) -> np.ndarray:
        return np.zeros(self.lang_size, dtype=np.float32)

    # ── Interval sampling (§3.4) ──────────────────────────────────────────────

    def sample_non_overlapping_intervals(
        self, buffer_len: int
    ) -> List[Tuple[int, int]]:
        """
        Sample non-overlapping intervals [u_k, v_k] such that their total
        length ≈ annotate_fraction × buffer_len.

        Starts from a random offset so that repeated calls cover different
        regions of the buffer rather than always annotating slots [0, buf/2].
        """
        target = int(self.annotate_fraction * buffer_len)
        intervals = []
        # Random start so each annotation pass covers a different region.
        start_offset = random.randint(0, buffer_len - 1)
        ptr = start_offset
        covered = 0
        while covered < target:
            gap   = random.randint(0, 3)
            start = (ptr + gap) % buffer_len
            max_len = min(self.seg_max, buffer_len - start)
            if max_len < self.seg_min:
                ptr = (start + 1) % buffer_len
                continue
            length = random.randint(self.seg_min, max_len)
            end    = start + length - 1
            if end >= buffer_len:
                # Don't wrap segments across the buffer boundary — skip.
                ptr = 0
                continue
            intervals.append((start, end))
            covered += length
            ptr = end + 1
            if ptr >= buffer_len:
                ptr = 0
                # Avoid infinite loop if we can't fit more non-wrapping segments.
                if covered < target and len(intervals) > buffer_len // self.seg_min:
                    break
        return intervals

    def _incremental_intervals(self, total_added: int) -> List[Tuple[int, int]]:
        """Carve the new region [cursor, total_added) into segments covering
        ~annotate_fraction of it; advance the cursor. Each timestep is visited
        at most once across the whole run."""
        intervals = []
        ptr, end = self._cursor, total_added
        if end - ptr < self.seg_min:
            return intervals
        frac = max(self.annotate_fraction, 1e-6)
        while ptr + self.seg_min <= end:
            length = random.randint(self.seg_min, self.seg_max)
            v = min(ptr + length - 1, end - 1)
            if v - ptr + 1 < self.seg_min:
                break
            intervals.append((ptr, v))
            gap = int((v - ptr + 1) * (1.0 - frac) / frac)
            ptr = v + 1 + gap
        self._cursor = end
        return intervals

    # ── VLM-based annotation ──────────────────────────────────────────────────

    def annotate_segment(
        self,
        frames: List[np.ndarray],
        actions: List[Any],
        t_start: int,
        extra_message: str = '',
    ) -> str:
        """Call the configured VLM to summarise one segment → instruction string."""
        atype = self.annotator_type

        if atype == 'template':
            return self._template_instruction()
        elif atype == 'random':
            return self._random_instruction()
        elif atype == 'cluster':
            return self._cluster_instruction(actions)
        elif atype in ('gpt4o', 'hybrid', 'segment', 'qwen', 'gemma', 'llava'):
            return self._vlm_instruction(frames, actions, t_start, extra_message)
        else:
            raise ValueError(f"Unknown annotator_type: {atype!r}")

    def _vlm_instruction(self, frames, actions, t_start, extra_message='') -> str:
        """Call the configured VLM with the Fig. 3 annotation prompt."""
        action_text = ", ".join(str(a) for a in actions)
        user_text = (
            f"TaskGuidance (G): {self.task_guidance}\n"
            f"Message (M): {extra_message}\n"
            f"Frames F[{t_start}:{t_start+len(frames)-1}]: see images above\n"
            f"Actions A[{t_start}:{t_start+len(frames)-1}]: {action_text}"
        )

        self._n_calls += 1
        if self._n_calls % 500 == 0:
            print(f'[Annotator] cumulative VLM calls: {self._n_calls}')
        if self.annotator_type in ('gpt4o', 'hybrid'):
            raw = self._gpt4o_call(frames, user_text)
        else:
            raw = self._hf_call(frames, user_text)

        try:
            instruction = json.loads(raw)["instruction"]
        except (json.JSONDecodeError, KeyError):
            # Strip markdown fences and retry parse
            stripped = raw.strip()
            if '```' in stripped:
                stripped = stripped.split('```')[1].lstrip('json').strip()
            try:
                instruction = json.loads(stripped)["instruction"]
            except (json.JSONDecodeError, KeyError):
                instruction = " ".join(stripped.split()[:18])

        # Discard refusals and empty instructions — they poison BC training
        _REFUSAL_PREFIXES = ("i'm sorry", "i cannot", "i can't", "as an ai",
                             "i apologize", "it seems there is an error")
        if (not instruction.strip() or
                any(instruction.lower().startswith(p) for p in _REFUSAL_PREFIXES) or
                instruction.startswith('```')):
            return None
        return instruction

    # ── GPT-4o annotation call ────────────────────────────────────────────────

    def _gpt4o_call(self, frames, user_text: str) -> str:
        if self._vlm is None:
            self._vlm = _openai_client()

        import imageio, io as _io
        n_frames = min(2, len(frames))
        indices  = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        content  = []
        for idx in indices:
            frame = frames[idx]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            buf = _io.BytesIO()
            imageio.imwrite(buf, frame, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })
        content.append({"type": "text", "text": user_text})

        import os
        resp = self._vlm.chat.completions.create(
            model=os.environ.get('I2A_VLM_MODEL', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": SUMMARIZER_PROMPT},
                {"role": "user",   "content": content},
            ],
            max_tokens=64, temperature=0.2,
            **({'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}} if os.environ.get('I2A_VLM_BASE_URL') else {}),
            **{k: v for k, v in self.vlm_kwargs.items() if k not in _LOCAL_FLAGS},
        )
        return resp.choices[0].message.content.strip()

    # ── HuggingFace annotation call (qwen / gemma / llava) ───────────────────

    def _hf_call(self, frames, user_text: str) -> str:
        """
        Use a local HuggingFace pipeline for open-source annotators.
        The pipeline is loaded lazily and cached in self._vlm.
        """
        model_map = {
            'qwen':  'Qwen/Qwen2.5-VL-72B-Instruct',
            'gemma': 'google/gemma-3-27b-it',
            'llava': 'llava-hf/llava-v1.6-34b-hf',
        }
        if self._vlm is None:
            model_name = model_map[self.annotator_type]
            self._vlm  = _hf_vlm_pipeline(model_name)

        from PIL import Image as PILImage
        n_frames = min(4, len(frames))
        indices  = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        pil_images = []
        for idx in indices:
            frame = frames[idx]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            pil_images.append(PILImage.fromarray(frame))

        prompt = f"{SUMMARIZER_PROMPT}\n\nUSER: {user_text}\nASSISTANT:"
        result = self._vlm(
            pil_images[0] if pil_images else None,
            prompt=prompt, max_new_tokens=128,
        )
        return result[0]['generated_text'].split('ASSISTANT:')[-1].strip()

    def _template_instruction(self) -> str:
        pattern = random.choice(TEMPLATE_PATTERNS)
        for slot_key, options in TEMPLATE_SLOTS.items():
            placeholder = "{" + slot_key + "}"
            if placeholder in pattern:
                pattern = pattern.replace(placeholder, random.choice(options))
        return pattern

    def _random_instruction(self) -> str:
        words = ["move", "go", "collect", "explore", "find", "interact",
                 "navigate", "attack", "defend", "wait", "craft", "build"]
        return " ".join(random.choices(words, k=random.randint(3, 6)))

    def _cluster_instruction(self, actions) -> str:
        """Use clustered action-sequence label as instruction (ablation)."""
        action_str = "_".join(str(int(a) if hasattr(a, '__int__') else a)
                               for a in actions[:5])
        return f"cluster_{hash(action_str) % 64}"

    # ── Full-buffer annotation pass ───────────────────────────────────────────

    def annotate_buffer(
        self,
        buffer,       # LangReplayBuffer instance
        env_name: str = '',
    ) -> int:
        """
        Annotate approximately annotate_fraction of the replay buffer.
        Returns the number of annotated timesteps.

        This is called asynchronously from a background thread during training
        (see training/train_lang.py).
        """
        buf_len = buffer.n_timesteps
        if buf_len == 0:
            return 0
        if self.annotator_type in ('achievement', 'inventory'):
            return self._annotate_achievements(buffer, env_name)
        if self.annotator_type == 'segment':
            total = int(getattr(buffer, '_total_added', getattr(buffer, 'total_added', buf_len)))
            n1 = self._annotate_segments_per_worker(buffer, env_name, total)
            cur = self._cursor; self._cursor = getattr(self, '_cursor_gt', 0)
            n2 = self._annotate_achievements(buffer, env_name)
            self._cursor_gt = self._cursor; self._cursor = cur
            return n1 + n2
        if self.annotator_type == 'hybrid' and not getattr(self, '_in_hybrid', False):
            # dense VLM instructions (no boundary completion labels) + ground-truth completion labels from inventory events
            self._in_hybrid = True
            try:
                n1 = self.annotate_buffer(buffer, env_name)
            finally:
                self._in_hybrid = False
            cur = self._cursor; self._cursor = getattr(self, '_cursor_gt', 0)
            n2 = self._annotate_achievements(buffer, env_name)
            self._cursor_gt = self._cursor; self._cursor = cur
            return n1 + n2

        if self.annotate_once:
            # Incremental mode: annotate only data added since the last pass,
            # exactly once, at ~annotate_fraction coverage. Cursor runs over
            # the monotone total_added counter so it survives buffer wrap.
            if (self.max_calls is not None and self._n_calls >= self.max_calls
                    and self.annotator_type in ('gpt4o', 'hybrid', 'qwen', 'gemma',
                                                'llava')):
                if not self._cap_warned:
                    print(f'[Annotator] max_calls={self.max_calls} reached — '
                          'no further VLM annotation (budget guard).')
                    self._cap_warned = True
                return 0
            total = getattr(buffer, 'total_added', buf_len)
            if self.annotator_type == 'hybrid':
                return self._annotate_dense_per_worker(buffer, env_name, int(getattr(buffer, '_total_added', total)))
            intervals = self._incremental_intervals(total)
        else:
            intervals = self.sample_non_overlapping_intervals(buf_len)

        n_annotated = 0
        for (u, v) in intervals:
            frames  = buffer.get_frames(u, v)
            actions = buffer.get_actions(u, v)
            if not frames:
                # After a resume the in-memory frame cache is empty for data
                # loaded from disk; annotating without frames would produce
                # garbage instructions. Skip (cursor already advanced).
                continue

            instruction = self.annotate_segment(frames, actions, t_start=u)
            if instruction is None:   # refusal or malformed — skip
                continue
            embed = self.encode(instruction)

            with self._lock:
                buffer.write_lang_embed(u, v, embed, instruction)
                if self.annotator_type != 'hybrid':
                    buffer.write_complete(v, 1.0)   # segment boundary = completion signal (hybrid: ground truth only)
                n_annotated += (v - u + 1)

            if self.log_annotations:
                self._annotation_log.append({
                    'env': env_name,
                    'u': u, 'v': v,
                    'instruction': instruction,
                })

        # NB: guard on n_new — `list[-0:]` would re-append the ENTIRE history
        # on every empty pass (this bloated logs to millions of lines once).
        n_new = len(self._annotation_log) - getattr(self, '_log_flushed', 0)
        if self.log_annotations and n_new > 0:
            import json
            log_path = self.log_dir / f'annotations_{env_name}.jsonl'
            with open(log_path, 'a') as f:
                for entry in self._annotation_log[-n_new:]:
                    f.write(json.dumps(entry) + '\n')
            self._log_flushed = len(self._annotation_log)

        return n_annotated

    def _segment_call(self, frames, steps, actions, n_frames: int = 8) -> list:
        """Ask the VLM to segment one per-worker chunk. frames: list of images; steps: their step indices (relative);
        returns list of (end_rel, instruction) or [] on failure."""
        import imageio, io as _io, os
        if self._vlm is None:
            self._vlm = _openai_client()
        idx = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)), dtype=int)
        content = []
        for i in idx:
            fr = frames[i]
            if fr.dtype != np.uint8: fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
            up = int(self.vlm_kwargs.get('segment_upscale', 1))
            if up > 1: fr = np.repeat(np.repeat(fr, up, axis=0), up, axis=1)   # nearest-neighbour upscale so a 64x64 frame is legible
            buf = _io.BytesIO(); imageio.imwrite(buf, fr, format='PNG')
            content.append({"type": "text", "text": f"frame at step {int(steps[i])}:"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), "detail": "low"}})
        acts = " ".join(str(int(a) if np.ndim(a) == 0 else int(np.argmax(a))) for a in actions)
        content.append({"type": "text", "text": f"TaskGuidance G: {self.task_guidance}\nChunk steps 0..{len(frames)-1}\nActions: {acts}"})
        try:
            resp = self._vlm.chat.completions.create(
                model=os.environ.get('I2A_VLM_MODEL', 'gpt-4o-mini'),
                messages=[{"role": "system", "content": SEGMENT_PROMPT}, {"role": "user", "content": content}],
                max_tokens=256, temperature=0.2,
                **({'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}} if os.environ.get('I2A_VLM_BASE_URL') else {}))
            raw = resp.choices[0].message.content.strip()
            if raw.startswith('```'): raw = raw.strip('`'); raw = raw[raw.find('{'):]
            segs = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])["segments"]
            out = []; last = -1
            for sg in segs:
                e = int(sg["end"]); ins = str(sg["instruction"]).strip()
                if e <= last or not ins: continue
                out.append((min(e, len(frames) - 1), ins)); last = e
            return out
        except Exception as ex:
            self._n_fail = getattr(self, '_n_fail', 0) + 1
            if self._n_fail <= 5: print(f'[SegmentAnnotator] parse/call failure: {str(ex)[:120]}', flush=True)
            return []

    def _annotate_segments_per_worker(self, buffer, env_name: str, total: int) -> int:
        """Segment-and-label: per worker, cut the new region into chunks of `segment_chunk` steps (not crossing episode
        starts), let the VLM split each chunk into subgoal segments with completion steps, write instruction embeddings
        over each segment and complete=1 on its last `segment_done_len` steps (0 before)."""
        import os, time as _time, concurrent.futures as cf
        _t0 = _time.time(); CH = int(self.vlm_kwargs.get('segment_chunk', 192)); DL = int(self.vlm_kwargs.get('segment_done_len', 4))
        start, end = self._cursor, total
        if end - start < CH:
            return 0
        wk = buffer.get_workers(start, end - 1); first = buffer.get_is_first(start, end - 1)
        by_w = {}
        for i, w in enumerate(wk): by_w.setdefault(w, []).append(i)
        chunks = []
        for w, rel in by_w.items():
            ptr = 0
            while ptr + 16 <= len(rel):
                v = min(ptr + CH - 1, len(rel) - 1)
                cut = next((q for q in range(ptr + 1, v + 1) if first[rel[q]]), None)
                if cut is not None: v = cut - 1
                if v - ptr + 1 >= 16: chunks.append((w, [start + rel[q] for q in range(ptr, v + 1)]))
                ptr = v + 1
        self._cursor = end
        if not chunks:
            return 0
        def work(item):
            w, seg = item
            frames = buffer.get_frames_idx(seg); actions = buffer.get_actions_idx(seg)
            if len(frames) < 16: return None
            return (w, seg, self._segment_call(frames, list(range(len(frames))), actions, n_frames=int(self.vlm_kwargs.get('segment_frames', 8))))
        conc = int(os.environ.get('I2A_VLM_CONC', 8))
        with cf.ThreadPoolExecutor(conc) as ex: results = [r for r in ex.map(work, chunks) if r and r[2]]
        n_annotated = 0; n_segs = 0
        for w, seg, parsed in results:
            s0 = 0
            for e_rel, instr in parsed:
                idxs = seg[s0:e_rel + 1]
                if len(idxs) < 8: s0 = e_rel + 1; continue
                embed = self.encode(instr)
                with self._lock:
                    buffer.write_lang_embed_idx(idxs, embed); buffer.write_complete_idx(idxs[:-DL], 0.0); buffer.write_complete_idx(idxs[-DL:], 1.0)
                n_annotated += len(idxs); n_segs += 1
                if self.log_annotations:
                    self._annotation_log.append({'env': env_name, 'worker': int(w), 'src': 'vlm_seg', 'u': idxs[0], 'v': idxs[-1], 'instruction': instr})
                s0 = e_rel + 1
        n_new = len(self._annotation_log) - getattr(self, '_log_flushed', 0)
        if self.log_annotations and n_new > 0:
            with open(self.log_dir / f'annotations_{env_name}.jsonl', 'a') as f:
                for entry in self._annotation_log[-n_new:]: f.write(json.dumps(entry) + '\n')
            self._log_flushed = len(self._annotation_log)
        print(f'[SegmentAnnotator] {len(chunks)} chunks over {len(by_w)} workers, {len(results)} parsed, {n_segs} segments, {n_annotated} timesteps, {_time.time()-_t0:.0f}s', flush=True)
        return n_annotated

    def _annotate_dense_per_worker(self, buffer, env_name: str, total: int) -> int:
        """Dense VLM instruction windows carved PER WORKER along each env's own slot sequence (replay slots interleave the
        parallel envs, so a range window would mix frames from different episodes), never crossing an episode start,
        at ~annotate_fraction coverage; VLM calls run in parallel (I2A_VLM_CONC threads) against the local server."""
        import os, time as _time, concurrent.futures as cf
        _t0 = _time.time()
        start, end = self._cursor, total
        if end - start < self.seg_min:
            return 0
        wk = buffer.get_workers(start, end - 1); first = buffer.get_is_first(start, end - 1)
        by_w = {}
        for i, w in enumerate(wk): by_w.setdefault(w, []).append(i)
        frac = max(self.annotate_fraction, 1e-6); segments = []
        for w, rel in by_w.items():
            ptr = 0
            while ptr + self.seg_min <= len(rel):
                length = random.randint(self.seg_min, self.seg_max); v = min(ptr + length - 1, len(rel) - 1)
                cut = next((q for q in range(ptr + 1, v + 1) if first[rel[q]]), None)
                if cut is not None: v = cut - 1
                if v - ptr + 1 >= self.seg_min: segments.append([start + rel[q] for q in range(ptr, v + 1)])
                gap = int(max(v - ptr + 1, self.seg_min) * (1.0 - frac) / frac)
                ptr = v + 1 + gap
        self._cursor = end
        if not segments:
            return 0
        def work(seg):
            frames = buffer.get_frames_idx(seg); actions = buffer.get_actions_idx(seg)
            if len(frames) < 2: return None
            instr = self.annotate_segment(frames, actions, t_start=seg[0])
            return (seg, instr) if instr else None
        conc = int(os.environ.get('I2A_VLM_CONC', 8))
        with cf.ThreadPoolExecutor(conc) as ex: results = [r for r in ex.map(work, segments) if r]
        n_annotated = 0
        for seg, instr in results:
            embed = self.encode(instr)
            with self._lock:
                buffer.write_lang_embed_idx(seg, embed)
            n_annotated += len(seg)
            if self.log_annotations:
                self._annotation_log.append({'env': env_name, 'u': seg[0], 'v': seg[-1], 'n': len(seg), 'instruction': instr})
        n_new = len(self._annotation_log) - getattr(self, '_log_flushed', 0)
        if self.log_annotations and n_new > 0:
            import json
            with open(self.log_dir / f'annotations_{env_name}.jsonl', 'a') as f:
                for entry in self._annotation_log[-n_new:]: f.write(json.dumps(entry) + '\n')
            self._log_flushed = len(self._annotation_log)
        print(f'[DenseAnnotator] {len(segments)} windows over {len(by_w)} workers, {len(results)} annotated, {n_annotated} timesteps, {_time.time()-_t0:.0f}s', flush=True)
        return n_annotated

    def _annotate_achievements(self, buffer, env_name: str = '') -> int:
        """Ground-truth hindsight annotation (crafter). Replay slots interleave the parallel envs, so everything is
        done PER WORKER on that worker's own slot sequence. For every achievement unlock at (worker) time t_e:
        segment [t_e - pre, t_e + post], pre ~ U[8, pre_max], post ~ U[8, post_max] (randomised: onset time carries
        no information), instruction = achievement sentence, annotated = 1, complete = 1 for t >= t_e; plus one
        matched NEGATIVE segment (same instruction/length, no unlock of that achievement inside, complete = 0)."""
        pre_max = int(self.vlm_kwargs.get('done_pre_max', 96)); post_max = int(self.vlm_kwargs.get('done_post_max', 32))
        total = int(getattr(buffer, '_total_added', buffer.total_added))
        start, end = self._cursor, total
        if end - start < 2 * (pre_max + post_max + 2):
            return 0
        if self.annotator_type in ('inventory', 'hybrid') or (self.annotator_type == 'segment' and 'minecraft' in env_name):
            ach = buffer.get_inventory(start, end - 1); names, sentences = INVENTORY_ITEMS, INVENTORY_INSTRUCTIONS
        else:
            ach = buffer.get_achievements(start, end - 1); names, sentences = ACHIEVEMENT_NAMES, ACHIEVEMENT_INSTRUCTIONS
        first = buffer.get_is_first(start, end - 1); wk = buffer.get_workers(start, end - 1)
        by_w = {}
        for i, w in enumerate(wk): by_w.setdefault(w, []).append(i)
        n_annotated = 0; n_events = 0
        for w, slots in by_w.items():
            A = [ach[i] for i in slots]; Fst = [first[i] for i in slots]; n = len(slots); used = np.zeros(n, dtype=bool)
            events = []
            for j in range(1, n):
                if A[j] is None or A[j - 1] is None or Fst[j]: continue
                delta = A[j] - A[j - 1]
                if (delta > 0).any(): events.append((j, int(np.argmax(delta))))
            n_events += len(events)
            for j, k in events:
                name = names[k]; pre = random.randint(8, pre_max); post = random.randint(8, post_max)
                lo, hi = max(j - pre, 0), min(j + post, n - 1)
                for q in range(j - 1, lo - 1, -1):
                    if Fst[q]: lo = q; break
                if used[lo:hi + 1].any() or j - lo < 4 or hi - j < 4: continue
                used[lo:hi + 1] = True
                instruction = sentences[name]; embed = self.encode(instruction)
                seg = [start + slots[q] for q in range(lo, hi + 1)]; neg_part = [start + slots[q] for q in range(lo, j)]; pos_part = [start + slots[q] for q in range(j, hi + 1)]
                buffer.write_lang_embed_idx(seg, embed); buffer.write_complete_idx(neg_part, 0.0)
                if str(self.vlm_kwargs.get('done_label', 'window')) == 'event':   # ablation: completion label only at the event step
                    buffer.write_complete_idx(pos_part, 0.0); buffer.write_complete_idx([start + slots[j]], 1.0)
                else:
                    buffer.write_complete_idx(pos_part, 1.0)
                if self.vlm_kwargs.get('finish_action'):   # ablation: [finish] as an extra action, BC target at the event step
                    buffer.write_finish_idx([start + slots[j]], 1.0)
                n_annotated += len(seg)
                L = hi - lo + 1
                for _ in range(20):
                    a0 = random.randint(0, n - L); a1 = a0 + L - 1
                    if used[a0:a1 + 1].any() or any(Fst[a0 + 1:a1 + 1]): continue
                    if A[a0] is None or A[a1] is None or A[a1][k] > A[a0][k]: continue
                    used[a0:a1 + 1] = True
                    negseg = [start + slots[q] for q in range(a0, a1 + 1)]
                    buffer.write_lang_embed_idx(negseg, embed); buffer.write_complete_idx(negseg, 0.0); n_annotated += L; break
                if self.log_annotations:
                    self._annotation_log.append({'env': env_name, 'worker': int(w), 'u': start + slots[lo], 'v': start + slots[hi], 't_e': start + slots[j], 'achievement': name, 'instruction': instruction})
        self._cursor = end
        print(f'[AchievementAnnotator v3] scanned {end - start} slots / {len(by_w)} workers, {n_events} unlock events, '
              f'wrote {n_annotated} annotated steps, {len(self._annotation_log)} positive segments so far', flush=True)
        return n_annotated

    def get_annotation_log(self) -> List[Dict]:
        return list(self._annotation_log)
