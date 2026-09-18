"""
VLM Planner (π_p) — maps observations to natural-language instructions (§3.3).

The planner maintains:
  M_r  : reasoning memory (brief scratch notes, ≤40 tokens delta updates)
  P_n  : partial plan (bullet list of upcoming micro-steps)

At each VLM step t', the planner takes:
  input  = (S_p, M_{r,t'-1}, P_{n,t'-1}, O_{<t'}, p^stop_{t'-1})

and produces:
  (M_{r,t'}, P_{n,t'}, optional instruction x)

Two operation modes (Algorithm 1):
  emit()    — generate next instruction from draft plan (fast: few tokens)
  advance() — continue reasoning without emitting (slow path)

Step mode (Algorithm 2):
  step()    — decide whether to emit and generate instruction

Multi-agent extension (§3.1):
  The planner can additionally output a short message for the Chatroom.
  This is parsed from the JSON output's "comm" field (Fig. 5).

Supported backends: GPT-4o, Qwen-VL-2.5-72B, Gemma-3-27B, LLaVA-v1.6-34b.
"""

import os
import base64
import json
import io
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_openai_key


# ── Prompt templates (Figs. 4–5) ─────────────────────────────────────────────

SINGLE_AGENT_SYSTEM = """\
You are the planner module π_p running ASYNCHRONOUSLY with a low-latency controller π_c.
The controller executes ONE short imperative instruction at a time and raises a
stop signal (p_stop) when it judges the instruction done. Your job is to keep it
working on the RIGHT next subgoal and never leave it idle.

Maintain
• Reasoning memory M_r: a compact running state of what is DONE and what is NEXT
  (e.g. "have: planks,table; next: wooden pickaxe"). Update it every step.
• Partial plan P_n: an ordered bullet list of upcoming subgoals → micro-steps.

You receive a live observation stream O_{<t'} (oldest→newest images; the last is
current) and the controller's stop probability p_stop_{t'-1}.

Read the frames carefully — especially the inventory/HUD and what changed between
frames — to infer which milestones are already achieved. Track progress through
the task's tech tree in order; do NOT re-issue a subgoal whose result you can
already see in the inventory.

Emit a NEW instruction x when either (a) p_stop_{t'-1} is high (≥ 0.5) — the
current subgoal is finished, advance to the next one — or (b) the observation
shows the current instruction is impossible, complete, or off-track and needs
revision. Otherwise set emit=false and ADVANCE the background plan silently.
If you have made no visible progress for a while, CHANGE strategy (relocate,
explore, or pick a different subgoal) rather than repeating the same command.

Instruction style (must match what the controller was trained to follow)
• 4–12 tokens, single-clause, imperative, concrete, verifiable from observations.
• Use direct action verbs grounded in the scene: "collect wood from the tree",
  "craft a stone pickaxe", "dig down to find iron ore", "place a furnace".
• One subgoal per instruction; never chain multiple goals with "and".
• Never block control; if unsure, issue the safest useful next step.

State text S(t') and controller feedback F(t') (when provided)
• S lists the inventory, equipped item, vitals, and the inventory change since the current instruction was issued. \
Trust S over what you can read from the small frames.
• F reports the controller's status: working | stuck (no state change for many steps) | anomaly (its world model \
finds the situation unfamiliar), the steps spent on the current instruction, its completion signal, and whether \
completion is VERIFIED by state (the item named in the instruction increased).
• Before advancing on a high completion signal, check verification: if F says verified=no, the subgoal is NOT done — \
re-issue it or fix the approach. If verified=unknown (navigation-type instruction), use S and the frames.
• If F says stuck: change strategy (relocate, explore, or pick a different subgoal), never repeat the same command.
• If F says anomaly: issue a short, safe re-orienting instruction (e.g. "move back to open ground", "look around").

Examples (Minecraft) — ILLUSTRATIVE ONLY: never copy their wording; every instruction you emit must be grounded in the \
current S and frames, and must differ from the previous instruction unless verified=no forces a retry.
S: inventory: logx3 | change since current instruction: +3 log ; F: status=working; completion signal=0.91; verified=yes
→ {"emit": true, "instruction": "craft planks from the logs", "reasoning_update": "have logs; next planks, table", \
"plan_update": ["craft planks", "craft a crafting table", "craft sticks"], "confidence": 0.9}
S: inventory: logx3 | change since current instruction: none ; F: status=working; completion signal=0.86; verified=no
→ {"emit": true, "instruction": "collect wood from the tree", "reasoning_update": "signal fired but no planks yet; retry", \
"plan_update": ["craft planks", "craft a crafting table"], "confidence": 0.6}
S: inventory: planksx4, sticksx2 ; F: status=stuck; steps on current instruction=140; verified=no
→ {"emit": true, "instruction": "walk to the nearest tree", "reasoning_update": "stuck crafting; relocate first", \
"plan_update": ["collect wood", "craft a crafting table", "craft a wooden pickaxe"], "confidence": 0.7}
S: inventory: cobblestonex5 ; F: status=working; completion signal=0.12; verified=no
→ {"emit": false, "instruction": "", "reasoning_update": "mining stone in progress", "plan_update": ["craft a stone pickaxe"], "confidence": 0.8}

Output JSON ONLY
{
  "emit": <true|false>,
  "instruction": "<if emit=true, the next single imperative subgoal; else empty>",
  "reasoning_update": "<=40 token delta for M_r: done/next state",
  "plan_update": ["<=12 token subgoal 1>", "<subgoal 2>", "..."],
  "confidence": <0.0-1.0>
}"""

MULTI_AGENT_ADDON = """\
You are planner {agent_id} in a cooperative team of {n_agents} agents. \
You may send AT MOST ONE short message per step.

Inbox: {inbox}

Messaging rules
• Keep messages ≤ 25 tokens.
• Prefer structured intents: {{claim:<subgoal>}}, {{status:<brief>}}, \
{{block:<issue>}}, {{request:<help on X>}}, {{handoff:<asset/role>}}.
• Addressing: use @all for broadcast or @{j} for a specific agent j.
• Do not restate observations verbatim; send actionable deltas that change teammates' plans.

Augmented OUTPUT (append to (2)'s JSON)
"comm": {{
  "send": <true|false>,
  "to": "@all" | "@{j}",
  "message": "<one line following the intent tags above>"
}}"""


def _encode_frame(frame: np.ndarray) -> str:
    """Encode an observation frame to a base64 PNG string."""
    import imageio
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    imageio.imwrite(buf, frame, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _obs_to_frames(obs) -> List[np.ndarray]:
    """Extract frame(s) from an observation dict or array."""
    if isinstance(obs, dict):
        for key in ('image', 'rgb', 'pixels', 'obs'):
            if key in obs:
                frame = obs[key]
                if frame.ndim == 3:
                    return [frame]
                elif frame.ndim == 4:
                    return [frame[0]]
        return []
    elif isinstance(obs, np.ndarray) and obs.ndim >= 3:
        return [obs if obs.ndim == 3 else obs[0]]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class VLMPlanner(ABC):
    """Abstract base class for VLM planners."""

    def __init__(self, system_prompt: str, task_spec: str = '', max_tokens: int = 256):
        self.system_prompt = system_prompt
        self.task_spec     = task_spec
        self.max_tokens    = max_tokens

    # ── Core API ──────────────────────────────────────────────────────────────

    def step(
        self,
        obs,
        plan: Optional[str],
        memory: Dict,
        p_stop: float = 0.0,
        chat_context: str = '',
    ) -> Tuple[Optional[str], Optional[str], Dict]:
        """
        Algorithm 2 synchronous step.
        Returns (instruction_or_None, updated_plan, updated_memory).
        """
        result = self._call_vlm(obs, plan, memory, p_stop, chat_context)
        instruction = result.get('instruction', '') if result.get('emit') else None
        new_plan    = self._update_plan(plan, result.get('plan_update', []))
        new_memory  = self._update_memory(memory, result.get('reasoning_update', ''))
        new_memory  = self._record_frame(new_memory, obs)
        new_memory['_outgoing_message'] = result.get('comm', None)
        return instruction or None, new_plan, new_memory

    def emit(
        self,
        obs,
        plan: Optional[str],
        memory: Dict,
        chat_context: str = '',
    ) -> Tuple[str, Optional[str], Dict]:
        """Algorithm 1: forced emit — must return a non-empty instruction."""
        result      = self._call_vlm(obs, plan, memory, p_stop=1.0, chat_context=chat_context)
        instruction = result.get('instruction', '')
        if not instruction:
            # Fallback: fall through to the next drafted plan step rather than a
            # meaningless 'continue', so the controller still gets a real subgoal.
            instruction = self._first_plan_step(plan) or 'explore and gather the next resource'
        new_plan    = self._update_plan(plan, result.get('plan_update', []))
        new_memory  = self._update_memory(memory, result.get('reasoning_update', ''))
        new_memory  = self._record_frame(new_memory, obs)
        new_memory['_outgoing_message'] = result.get('comm', None)
        return instruction, new_plan, new_memory

    def advance(
        self,
        obs,
        plan: Optional[str],
        memory: Dict,
    ) -> Tuple[Optional[str], Dict]:
        """Algorithm 1: background advance — returns (updated_plan, updated_memory)."""
        result     = self._call_vlm(obs, plan, memory, p_stop=0.0, chat_context='')
        new_plan   = self._update_plan(plan, result.get('plan_update', []))
        new_memory = self._update_memory(memory, result.get('reasoning_update', ''))
        new_memory = self._record_frame(new_memory, obs)
        return new_plan, new_memory

    # ── Subclass interface ─────────────────────────────────────────────────────

    @abstractmethod
    def _call_vlm(
        self, obs, plan, memory, p_stop, chat_context
    ) -> Dict[str, Any]:
        """Make the VLM API call; return parsed JSON dict."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_user_message(self, obs, plan, memory, p_stop, chat_context) -> List:
        """Build the multimodal user message for the VLM API.

        Images are ordered oldest→newest so the model can read progress (motion,
        inventory/HUD changes) across steps. Earlier frames are sent at low
        detail to save tokens; the current frame is sent at high detail so the
        inventory/health HUD is legible.
        """
        prev_frames = memory.get('_frame_history', []) if isinstance(memory, dict) else []
        cur_frames  = _obs_to_frames(obs)
        content = []
        for frame in list(prev_frames)[-2:]:        # temporal context (older)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_frame(frame)}",
                    "detail": "low",
                }
            })
        for frame in cur_frames[:1]:                 # current frame (HUD-legible)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{_encode_frame(frame)}",
                    "detail": "high",
                }
            })
        n_prev = len(list(prev_frames)[-2:])
        frame_note = (f"{n_prev} earlier frame(s) then the CURRENT frame (last image)"
                      if n_prev else "the CURRENT frame")
        user_text_parts = [
            f"- M_r(t'-1): {memory.get('reasoning', '(empty)')}",
            f"- P_n(t'-1): {plan or '(empty)'}",
            f"- p_stop_{{t'-1}}: {p_stop:.2f}",
            f"- O_{{<t'}}: images above are {frame_note}. "
            f"Read the inventory/HUD to track milestones already achieved.",
        ]
        if isinstance(obs, dict) and obs.get('_state_text'):
            user_text_parts.append(f"- S(t'): {obs['_state_text']}")
        if isinstance(obs, dict) and obs.get('_feedback'):
            user_text_parts.append(f"- F(t'): {obs['_feedback']}")
        if self.task_spec:
            user_text_parts.append(f"- Task spec (if any): {self.task_spec}")
        if chat_context:
            user_text_parts.append(f"- Inbox messages:\n{chat_context}")
        content.append({"type": "text", "text": "\n".join(user_text_parts)})
        if os.environ.get('I2A_DEBUG_PROMPT') and not getattr(self, '_prompt_shown', False) and isinstance(obs, dict) and obs.get('_state_text'):
            self._prompt_shown = True; print('[planner prompt sample]\n' + content[-1]['text'], flush=True)
        return content

    def _record_frame(self, memory: dict, obs, keep: int = 2) -> dict:
        """Append the current frame to a short rolling history kept in memory.

        Used to give the planner temporal context on the next call without
        changing the controller↔planner interface.
        """
        frames = _obs_to_frames(obs)
        if not frames:
            return memory
        mem  = dict(memory)
        hist = list(mem.get('_frame_history', []))
        hist.append(frames[0])
        mem['_frame_history'] = hist[-keep:]
        return mem

    def _update_plan(self, old_plan, plan_update: list) -> str:
        if not plan_update:
            return old_plan or ''
        return '\n'.join(f'• {s}' for s in plan_update if s)

    def _first_plan_step(self, plan) -> str:
        """Return the first bullet of the current plan as a bare instruction."""
        if not plan:
            return ''
        first = plan.split('\n')[0].strip()
        return first.lstrip('•').strip()

    def _update_memory(self, memory: dict, reasoning_update: str) -> dict:
        mem = dict(memory)
        if reasoning_update:
            existing = mem.get('reasoning', '')
            mem['reasoning'] = (existing + ' ' + reasoning_update).strip()[-200:]
        return mem

    def _parse_json(self, raw: str) -> Dict:
        """Extract JSON from the VLM response (strips markdown fences)."""
        raw = raw.strip()
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Best-effort: look for {…}
            start = raw.find('{')
            end   = raw.rfind('}')
            if start != -1 and end != -1:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError:
                    pass
            return {'emit': False, 'instruction': '', 'reasoning_update': '', 'plan_update': []}


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o planner
# ─────────────────────────────────────────────────────────────────────────────

class GPT4oPlanner(VLMPlanner):
    """Uses OpenAI GPT-4o as the VLM planner (paper default)."""

    def __init__(self, task_spec: str = '', max_tokens: int = 256, temperature: float = 0.3):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec, max_tokens)
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            base = os.environ.get('I2A_VLM_BASE_URL')   # local vLLM server (OpenAI-compatible) instead of OpenAI
            self._client = openai.OpenAI(api_key='EMPTY', base_url=base, timeout=600) if base else openai.OpenAI(api_key=load_openai_key())
        return self._client

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict:
        content  = self._build_user_message(obs, plan, memory, p_stop, chat_context)
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user',   'content': content},
        ]
        # response_format guarantees syntactically valid JSON, removing most
        # parse failures; retry transient API/JSON errors a few times before
        # falling back to a no-op result (so control is never blocked).
        last_exc = None
        for attempt in range(3):
            try:
                local = bool(os.environ.get('I2A_VLM_BASE_URL'))
                resp = self.client.chat.completions.create(
                    model=os.environ.get('I2A_VLM_MODEL', 'gpt-4o'),
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={'type': 'json_object'},
                    **({'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}} if local else {}),
                )
                result = self._parse_json(resp.choices[0].message.content)
                self._last_result = result   # audit
                return result
            except Exception as e:                       # noqa: BLE001 (API/network/JSON)
                last_exc = e
                import time as _t
                _t.sleep(0.5 * (attempt + 1))
        print(f"[GPT4oPlanner] VLM call failed after retries: {last_exc}")
        return {'emit': False, 'instruction': '', 'reasoning_update': '', 'plan_update': []}


# ─────────────────────────────────────────────────────────────────────────────
# Qwen-VL planner (local or API)
# ─────────────────────────────────────────────────────────────────────────────

def _hf_user_content(obs, plan, memory, p_stop, chat_context, task_spec='') -> List:
    """
    Build the multimodal user content list in HuggingFace chat format.

    HuggingFace vision models expect content items with:
      {"type": "image", "image": <PIL.Image>}   (for vision)
      {"type": "text",  "text":  "..."}

    This is different from the OpenAI {"type": "image_url", ...} format used
    by GPT4oPlanner._build_user_message().
    """
    from PIL import Image as PILImage
    prev_frames = memory.get('_frame_history', []) if isinstance(memory, dict) else []
    frames      = list(prev_frames)[-2:] + _obs_to_frames(obs)[:1]   # oldest→current
    content: List[Dict] = []

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        content.append({"type": "image", "image": PILImage.fromarray(frame)})

    text_parts = [
        f"- M_r(t'-1): {memory.get('reasoning', '(empty)')}",
        f"- P_n(t'-1): {plan or '(empty)'}",
        f"- p_stop_{{t'-1}}: {p_stop:.2f}",
        f"- O_{{<t'}}: images are oldest→current (last is current); "
        f"read the inventory/HUD to track milestones already achieved",
    ]
    if isinstance(obs, dict) and obs.get('_state_text'): text_parts.append(f"- S(t'): {obs['_state_text']}")
    if isinstance(obs, dict) and obs.get('_feedback'): text_parts.append(f"- F(t'): {obs['_feedback']}")
    if task_spec:
        text_parts.append(f"- Task spec: {task_spec}")
    if chat_context:
        text_parts.append(f"- Inbox:\n{chat_context}")
    content.append({"type": "text", "text": "\n".join(text_parts)})
    return content


class QwenPlanner(VLMPlanner):
    """Qwen-VL-2.5-72B planner via HuggingFace transformers (local)."""

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        task_spec: str = '',
        max_tokens: int = 256,
        device: str = 'cuda',
        temperature: float = 0.3,
    ):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec, max_tokens)
        self.model_name  = model_name
        self.device      = device
        self.temperature = temperature
        self._model      = None
        self._processor  = None

    def _load(self):
        if self._model is None:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            try:
                import accelerate  # noqa: F401
                kwargs = {'device_map': self.device}
            except ImportError:
                kwargs = {}
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype='auto', **kwargs
            )
            if 'device_map' not in kwargs and self.device:
                self._model = self._model.to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict:
        self._load()
        import torch
        # Use HF-native content format (not OpenAI image_url)
        content  = _hf_user_content(obs, plan, memory, p_stop, chat_context,
                                     self.task_spec)
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user',   'content': content},
        ]
        # Extract PIL images for the processor
        images = [item['image'] for item in content if item['type'] == 'image']
        text   = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=images if images else None, return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_tokens,
                temperature=self.temperature, do_sample=True,
            )
        raw = self._processor.decode(out[0][inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)
        return self._parse_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Gemma-3 planner
# ─────────────────────────────────────────────────────────────────────────────

class GemmaPlanner(VLMPlanner):
    """Gemma-3-27B planner via HuggingFace transformers (local)."""

    def __init__(
        self,
        model_name: str = 'google/gemma-3-27b-it',
        task_spec: str = '',
        max_tokens: int = 256,
        device: str = 'cuda',
    ):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec, max_tokens)
        self.model_name = model_name
        self.device     = device
        self._model     = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self._model     = AutoModelForImageTextToText.from_pretrained(
                self.model_name, device_map=self.device, torch_dtype='bfloat16'
            )
            self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict:
        self._load()
        import torch
        content  = _hf_user_content(obs, plan, memory, p_stop, chat_context,
                                     self.task_spec)
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user',   'content': content},
        ]
        images = [item['image'] for item in content if item['type'] == 'image']
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors='pt', images=images if images else None,
        ).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_tokens)
        raw = self._processor.decode(out[0][inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)
        return self._parse_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# LLaVA planner
# ─────────────────────────────────────────────────────────────────────────────

class LlavaPlanner(VLMPlanner):
    """LLaVA-v1.6-34b planner (local HuggingFace)."""

    def __init__(
        self,
        model_name: str = 'llava-hf/llava-v1.6-34b-hf',
        task_spec: str = '',
        max_tokens: int = 256,
        device: str = 'cuda',
    ):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec, max_tokens)
        self.model_name = model_name
        self.device     = device
        self._pipe      = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline(
                'image-to-text', model=self.model_name,
                device=0 if self.device == 'cuda' else -1,
            )

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict:
        self._load()
        # Build HF-compatible content and extract PIL image for the pipeline
        content  = _hf_user_content(obs, plan, memory, p_stop, chat_context,
                                     self.task_spec)
        images   = [item['image'] for item in content if item['type'] == 'image']
        text_parts = [item['text'] for item in content if item['type'] == 'text']
        user_txt = "\n".join(text_parts)
        prompt   = f"{self.system_prompt}\n\nUSER: {user_txt}\nASSISTANT:"
        result   = self._pipe(images[0] if images else None,
                              prompt=prompt, max_new_tokens=self.max_tokens)
        raw = result[0]['generated_text'].split('ASSISTANT:')[-1]
        return self._parse_json(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Scripted planner for no-API/no-VLM ablations
# ─────────────────────────────────────────────────────────────────────────────

class ScriptedPlanner(VLMPlanner):
    """
    Deterministic milestone planner for controlled ablations.

    This avoids API calls and local VLM dependencies while still exercising the
    language-conditioned controller through realistic Minecraft-style commands.
    It is useful for robustness tests where the perturbation, not planner
    visual reasoning, is the experimental variable.
    """

    MINECRAFT_PLAN = [
        "collect wood from nearby trees",
        "craft planks from collected wood",
        "craft a crafting table",
        "craft sticks for tool recipes",
        "craft a wooden pickaxe",
        "mine stone to gather cobblestone",
        "craft a stone pickaxe",
        "search underground for iron ore",
        "mine iron ore with the stone pickaxe",
        "craft a furnace near your position",
        "smelt iron ore into ingots",
        "craft an iron pickaxe",
        "search deep underground for diamond ore",
        "mine diamond ore with the iron pickaxe",
    ]

    def __init__(self, task_spec: str = '', max_tokens: int = 256, **kwargs):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec, max_tokens)

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict[str, Any]:
        idx = int(memory.get('script_idx', 0))
        if p_stop >= 0.5 or not plan:
            instruction = self.MINECRAFT_PLAN[idx % len(self.MINECRAFT_PLAN)]
            memory['script_idx'] = idx + 1
            return {
                'emit': True,
                'instruction': instruction,
                'reasoning_update': f'script step {idx + 1}',
                'plan_update': self.MINECRAFT_PLAN[idx:idx + 3],
                'confidence': 1.0,
            }
        return {
            'emit': False,
            'instruction': '',
            'reasoning_update': 'continue current scripted subgoal',
            'plan_update': [],
            'confidence': 1.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_planner(name: str, task_spec: str = '', **kwargs) -> VLMPlanner:
    """
    Instantiate a planner by name.

    name ∈ {'gpt4o', 'qwen', 'gemma', 'llava', 'scripted'}
    """
    mapping = {
        'gpt4o': GPT4oPlanner,
        'qwen':  QwenPlanner,
        'gemma': GemmaPlanner,
        'llava': LlavaPlanner,
        'scripted': ScriptedPlanner,
    }
    if name not in mapping:
        raise ValueError(f"Unknown planner {name!r}. Choose from {list(mapping)}")
    return mapping[name](task_spec=task_spec, **kwargs)
