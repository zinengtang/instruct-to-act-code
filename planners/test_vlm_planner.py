"""
Comprehensive test harness for planners/vlm_planner.py.

Covers:
  - _parse_json          (clean, fenced, partial, malformed, empty)
  - _encode_frame        (uint8, float, shape variants)
  - _obs_to_frames       (all obs formats)
  - _update_plan / _update_memory (including truncation)
  - _build_user_message  (text construction, frame injection, optional fields)
  - step / emit / advance (control flow, emit flag, fallback, comm propagation)
  - make_planner         (valid names, invalid name)
  - multi-agent prompt   (MULTI_AGENT_ADDON substitution)

No real VLM calls are made; a StubPlanner captures inputs and returns
a configurable response dict.

Run with:
    pytest planners/test_vlm_planner.py -v
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from vlm_planner import (
    MULTI_AGENT_ADDON,
    SINGLE_AGENT_SYSTEM,
    GPT4oPlanner,
    GemmaPlanner,
    LlavaPlanner,
    QwenPlanner,
    VLMPlanner,
    _encode_frame,
    _obs_to_frames,
    make_planner,
)


# ── StubPlanner ───────────────────────────────────────────────────────────────

class StubPlanner(VLMPlanner):
    """Concrete VLMPlanner whose _call_vlm returns a pre-set response."""

    def __init__(self, response: Optional[Dict] = None, task_spec: str = ''):
        super().__init__(SINGLE_AGENT_SYSTEM, task_spec=task_spec)
        self._response: Dict = response or {
            'emit': True,
            'instruction': 'chop a tree',
            'reasoning_update': 'saw oak nearby',
            'plan_update': ['move to tree', 'chop trunk'],
            'confidence': 0.9,
        }
        self.last_call: Dict[str, Any] = {}

    def set_response(self, response: Dict):
        self._response = response

    def _call_vlm(self, obs, plan, memory, p_stop, chat_context) -> Dict:
        self.last_call = dict(obs=obs, plan=plan, memory=memory,
                              p_stop=p_stop, chat_context=chat_context)
        return dict(self._response)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def planner():
    return StubPlanner()

@pytest.fixture
def rgb_frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

@pytest.fixture
def float_frame():
    rng = np.random.default_rng(1)
    return rng.random((64, 64, 3)).astype(np.float32)

@pytest.fixture
def obs_dict(rgb_frame):
    return {'image': rgb_frame}

@pytest.fixture
def default_memory():
    return {'reasoning': 'start of run'}

@pytest.fixture
def default_plan():
    return '• gather wood\n• craft pickaxe'


# ── _encode_frame ─────────────────────────────────────────────────────────────

class TestEncodeFrame:
    def test_uint8_returns_nonempty_string(self, rgb_frame):
        result = _encode_frame(rgb_frame)
        assert isinstance(result, str) and len(result) > 0

    def test_float_frame_encodes_without_error(self, float_frame):
        result = _encode_frame(float_frame)
        assert isinstance(result, str) and len(result) > 0

    def test_output_is_valid_base64(self, rgb_frame):
        import base64
        result = _encode_frame(rgb_frame)
        decoded = base64.b64decode(result)
        # PNG magic bytes
        assert decoded[:4] == b'\x89PNG'

    def test_float_clipped_to_uint8_range(self):
        frame = np.full((4, 4, 3), 2.0, dtype=np.float32)   # all > 1.0 → clips to 255
        result = _encode_frame(frame)
        assert isinstance(result, str) and len(result) > 0


# ── _obs_to_frames ────────────────────────────────────────────────────────────

class TestObsToFrames:
    def test_image_key(self, rgb_frame):
        frames = _obs_to_frames({'image': rgb_frame})
        assert len(frames) == 1
        assert frames[0].shape == rgb_frame.shape

    def test_rgb_key(self, rgb_frame):
        frames = _obs_to_frames({'rgb': rgb_frame})
        assert len(frames) == 1

    def test_pixels_key(self, rgb_frame):
        frames = _obs_to_frames({'pixels': rgb_frame})
        assert len(frames) == 1

    def test_obs_key(self, rgb_frame):
        frames = _obs_to_frames({'obs': rgb_frame})
        assert len(frames) == 1

    def test_4d_frame_returns_first_slice(self, rgb_frame):
        batch = rgb_frame[np.newaxis]   # (1, 64, 64, 3)
        frames = _obs_to_frames({'image': batch})
        assert len(frames) == 1
        assert frames[0].shape == (64, 64, 3)

    def test_bare_ndarray_3d(self, rgb_frame):
        frames = _obs_to_frames(rgb_frame)
        assert len(frames) == 1

    def test_bare_ndarray_4d(self, rgb_frame):
        batch = rgb_frame[np.newaxis]
        frames = _obs_to_frames(batch)
        assert len(frames) == 1

    def test_empty_dict_returns_empty(self):
        assert _obs_to_frames({}) == []

    def test_unknown_key_returns_empty(self):
        assert _obs_to_frames({'depth': np.zeros((4, 4, 1))}) == []

    def test_0d_array_returns_empty(self):
        assert _obs_to_frames(np.array(42)) == []


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_clean_json(self, planner):
        raw = json.dumps({'emit': True, 'instruction': 'go left'})
        result = planner._parse_json(raw)
        assert result['emit'] is True
        assert result['instruction'] == 'go left'

    def test_markdown_fenced_json(self, planner):
        raw = '```json\n{"emit": false, "instruction": ""}\n```'
        result = planner._parse_json(raw)
        assert result['emit'] is False

    def test_fenced_without_language_tag(self, planner):
        raw = '```\n{"emit": true, "instruction": "build"}\n```'
        result = planner._parse_json(raw)
        assert result['emit'] is True

    def test_json_embedded_in_text(self, planner):
        raw = 'Here is my response:\n{"emit": true, "instruction": "craft table"}'
        result = planner._parse_json(raw)
        assert result['instruction'] == 'craft table'

    def test_malformed_returns_safe_default(self, planner):
        result = planner._parse_json('this is not json at all')
        assert result['emit'] is False
        assert result['instruction'] == ''
        assert result['plan_update'] == []

    def test_empty_string_returns_safe_default(self, planner):
        result = planner._parse_json('')
        assert result['emit'] is False

    def test_extra_fields_preserved(self, planner):
        raw = json.dumps({'emit': True, 'instruction': 'x', 'confidence': 0.7})
        result = planner._parse_json(raw)
        assert result['confidence'] == pytest.approx(0.7)

    def test_partial_json_with_braces(self, planner):
        raw = 'junk before {"emit": false} junk after'
        result = planner._parse_json(raw)
        assert result['emit'] is False


# ── _update_plan ──────────────────────────────────────────────────────────────

class TestUpdatePlan:
    def test_plan_update_replaces(self, planner):
        new = planner._update_plan('old', ['step A', 'step B'])
        assert '• step A' in new
        assert '• step B' in new

    def test_empty_update_keeps_old(self, planner):
        assert planner._update_plan('old plan', []) == 'old plan'

    def test_none_old_plan_with_empty_update(self, planner):
        assert planner._update_plan(None, []) == ''

    def test_filters_empty_strings(self, planner):
        new = planner._update_plan('x', ['step A', '', 'step B'])
        assert '• ' in new
        lines = [l for l in new.split('\n') if l]
        assert len(lines) == 2

    def test_single_step(self, planner):
        new = planner._update_plan(None, ['mine stone'])
        assert new == '• mine stone'


# ── _update_memory ────────────────────────────────────────────────────────────

class TestUpdateMemory:
    def test_appends_reasoning(self, planner):
        mem = {'reasoning': 'initial'}
        new = planner._update_memory(mem, 'saw iron')
        assert 'initial' in new['reasoning']
        assert 'saw iron' in new['reasoning']

    def test_empty_update_leaves_memory_unchanged(self, planner):
        mem = {'reasoning': 'initial'}
        new = planner._update_memory(mem, '')
        assert new['reasoning'] == 'initial'

    def test_memory_truncated_to_200_chars(self, planner):
        long_existing = 'x' * 190
        new = planner._update_memory({'reasoning': long_existing}, 'extra tokens here')
        assert len(new['reasoning']) <= 200

    def test_does_not_mutate_original(self, planner):
        mem = {'reasoning': 'original'}
        planner._update_memory(mem, 'delta')
        assert mem['reasoning'] == 'original'

    def test_missing_reasoning_key(self, planner):
        new = planner._update_memory({}, 'first note')
        assert 'first note' in new['reasoning']


# ── _build_user_message ───────────────────────────────────────────────────────

class TestBuildUserMessage:
    def test_text_block_present(self, planner, obs_dict, default_memory, default_plan):
        content = planner._build_user_message(obs_dict, default_plan, default_memory,
                                               p_stop=0.3, chat_context='')
        texts = [c['text'] for c in content if c.get('type') == 'text']
        assert texts, 'no text block found'
        combined = '\n'.join(texts)
        assert 'M_r' in combined
        assert 'P_n' in combined
        assert '0.30' in combined

    def test_image_block_present_when_obs_has_image(self, planner, obs_dict, default_memory):
        content = planner._build_user_message(obs_dict, None, default_memory, 0.0, '')
        image_blocks = [c for c in content if c.get('type') == 'image_url']
        assert len(image_blocks) == 1

    def test_no_image_when_obs_empty(self, planner, default_memory):
        content = planner._build_user_message({}, None, default_memory, 0.0, '')
        assert not any(c.get('type') == 'image_url' for c in content)

    def test_only_current_frame_without_history(self, planner, rgb_frame, default_memory):
        # No frame history in memory → exactly one (current) image is sent.
        content = planner._build_user_message({'image': rgb_frame}, None,
                                              default_memory, 0.0, '')
        image_blocks = [c for c in content if c.get('type') == 'image_url']
        assert len(image_blocks) == 1
        assert image_blocks[0]['image_url']['detail'] == 'high'   # current = HUD-legible

    def test_frame_history_prepended(self, planner, rgb_frame):
        # Up to 2 history frames are sent before the current one (oldest→current).
        mem = {'_frame_history': [rgb_frame, rgb_frame, rgb_frame]}  # keep last 2
        content = planner._build_user_message({'image': rgb_frame}, None, mem, 0.0, '')
        image_blocks = [c for c in content if c.get('type') == 'image_url']
        assert len(image_blocks) == 3                      # 2 history + 1 current
        assert image_blocks[-1]['image_url']['detail'] == 'high'   # last = current

    def test_task_spec_included_when_set(self, rgb_frame, default_memory):
        p = StubPlanner(task_spec='obtain diamond')
        content = p._build_user_message({'image': rgb_frame}, None, default_memory, 0.0, '')
        texts = ' '.join(c['text'] for c in content if c.get('type') == 'text')
        assert 'obtain diamond' in texts

    def test_task_spec_absent_when_empty(self, planner, obs_dict, default_memory):
        content = planner._build_user_message(obs_dict, None, default_memory, 0.0, '')
        texts = ' '.join(c['text'] for c in content if c.get('type') == 'text')
        assert 'Task spec' not in texts

    def test_chat_context_included(self, planner, obs_dict, default_memory):
        content = planner._build_user_message(obs_dict, None, default_memory, 0.0,
                                               chat_context='@agent1: claim wood')
        texts = ' '.join(c['text'] for c in content if c.get('type') == 'text')
        assert 'claim wood' in texts

    def test_chat_context_absent_when_empty(self, planner, obs_dict, default_memory):
        content = planner._build_user_message(obs_dict, None, default_memory, 0.0, '')
        texts = ' '.join(c['text'] for c in content if c.get('type') == 'text')
        assert 'Inbox' not in texts


# ── step() ────────────────────────────────────────────────────────────────────

class TestStep:
    def test_emit_true_returns_instruction(self, planner, obs_dict, default_memory, default_plan):
        planner.set_response({'emit': True, 'instruction': 'chop tree',
                               'reasoning_update': 'r', 'plan_update': ['p']})
        instr, _, _ = planner.step(obs_dict, default_plan, default_memory)
        assert instr == 'chop tree'

    def test_emit_false_returns_none(self, planner, obs_dict, default_memory, default_plan):
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        instr, _, _ = planner.step(obs_dict, default_plan, default_memory)
        assert instr is None

    def test_plan_updated(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': ['new step']})
        _, plan, _ = planner.step(obs_dict, None, default_memory)
        assert '• new step' in plan

    def test_memory_updated(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'x',
                               'reasoning_update': 'found ore', 'plan_update': []})
        _, _, mem = planner.step(obs_dict, None, default_memory)
        assert 'found ore' in mem['reasoning']

    def test_comm_propagated(self, planner, obs_dict, default_memory):
        comm = {'send': True, 'to': '@all', 'message': 'claim:wood'}
        planner.set_response({'emit': True, 'instruction': 'x',
                               'reasoning_update': '', 'plan_update': [], 'comm': comm})
        _, _, mem = planner.step(obs_dict, None, default_memory)
        assert mem['_outgoing_message'] == comm

    def test_comm_none_when_absent(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'x',
                               'reasoning_update': '', 'plan_update': []})
        _, _, mem = planner.step(obs_dict, None, default_memory)
        assert mem['_outgoing_message'] is None

    def test_p_stop_forwarded_to_call(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        planner.step(obs_dict, None, default_memory, p_stop=0.77)
        assert planner.last_call['p_stop'] == pytest.approx(0.77)

    def test_chat_context_forwarded(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        planner.step(obs_dict, None, default_memory, chat_context='hello')
        assert planner.last_call['chat_context'] == 'hello'

    def test_empty_instruction_with_emit_true_returns_none(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        instr, _, _ = planner.step(obs_dict, None, default_memory)
        assert instr is None


# ── emit() ────────────────────────────────────────────────────────────────────

class TestEmit:
    def test_always_returns_instruction(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': False, 'instruction': 'craft table',
                               'reasoning_update': '', 'plan_update': []})
        instr, _, _ = planner.emit(obs_dict, None, default_memory)
        assert instr == 'craft table'

    def test_fallback_when_instruction_empty(self, planner, obs_dict, default_memory):
        # With no plan, the fallback is a generic-but-actionable subgoal.
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        instr, _, _ = planner.emit(obs_dict, None, default_memory)
        assert instr == 'explore and gather the next resource'

    def test_fallback_uses_first_plan_step(self, planner, obs_dict, default_memory):
        # When a plan exists, the fallback emits its first bullet as a real subgoal.
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        instr, _, _ = planner.emit(obs_dict, '• craft a stone pickaxe\n• find iron',
                                   default_memory)
        assert instr == 'craft a stone pickaxe'

    def test_p_stop_forced_to_1(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'x',
                               'reasoning_update': '', 'plan_update': []})
        planner.emit(obs_dict, None, default_memory)
        assert planner.last_call['p_stop'] == pytest.approx(1.0)

    def test_plan_updated(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'go',
                               'reasoning_update': '', 'plan_update': ['step 1']})
        _, plan, _ = planner.emit(obs_dict, None, default_memory)
        assert '• step 1' in plan

    def test_comm_propagated(self, planner, obs_dict, default_memory):
        comm = {'send': False}
        planner.set_response({'emit': True, 'instruction': 'go',
                               'reasoning_update': '', 'plan_update': [], 'comm': comm})
        _, _, mem = planner.emit(obs_dict, None, default_memory)
        assert mem['_outgoing_message'] == comm


# ── advance() ────────────────────────────────────────────────────────────────

class TestAdvance:
    def test_returns_plan_and_memory(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'ignored',
                               'reasoning_update': 'update', 'plan_update': ['advance step']})
        plan, mem = planner.advance(obs_dict, None, default_memory)
        assert '• advance step' in plan
        assert 'update' in mem['reasoning']

    def test_p_stop_forced_to_0(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': True, 'instruction': 'x',
                               'reasoning_update': '', 'plan_update': []})
        planner.advance(obs_dict, None, default_memory)
        assert planner.last_call['p_stop'] == pytest.approx(0.0)

    def test_chat_context_empty(self, planner, obs_dict, default_memory):
        planner.set_response({'emit': False, 'instruction': '',
                               'reasoning_update': '', 'plan_update': []})
        planner.advance(obs_dict, None, default_memory)
        assert planner.last_call['chat_context'] == ''


# ── make_planner ──────────────────────────────────────────────────────────────

class TestMakePlanner:
    def test_gpt4o(self):
        p = make_planner('gpt4o', task_spec='test')
        assert isinstance(p, GPT4oPlanner)

    def test_qwen(self):
        p = make_planner('qwen', task_spec='test')
        assert isinstance(p, QwenPlanner)

    def test_gemma(self):
        p = make_planner('gemma', task_spec='test')
        assert isinstance(p, GemmaPlanner)

    def test_llava(self):
        p = make_planner('llava', task_spec='test')
        assert isinstance(p, LlavaPlanner)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match='Unknown planner'):
            make_planner('gpt3', task_spec='test')

    def test_task_spec_forwarded(self):
        p = make_planner('gpt4o', task_spec='obtain diamond')
        assert p.task_spec == 'obtain diamond'

    def test_kwargs_forwarded(self):
        p = make_planner('gpt4o', task_spec='', temperature=0.9)
        assert p.temperature == pytest.approx(0.9)


# ── multi-agent prompt ────────────────────────────────────────────────────────

class TestMultiAgentPrompt:
    def test_format_substitution(self):
        result = MULTI_AGENT_ADDON.format(
            agent_id='A', n_agents=3, inbox='msg1', j='B'
        )
        assert 'planner A' in result
        assert '3 agents' in result
        assert 'msg1' in result

    def test_comm_field_mentioned(self):
        assert '"comm"' in MULTI_AGENT_ADDON

    def test_single_agent_system_no_placeholders(self):
        # SINGLE_AGENT_SYSTEM should be usable as-is (no .format() required)
        assert '{agent_id}' not in SINGLE_AGENT_SYSTEM


# ── Round-trip: step → state accumulation ────────────────────────────────────

class TestStateAccumulation:
    """Multi-step scenario: verify plan and memory grow correctly over turns."""

    def test_memory_accumulates_across_steps(self, obs_dict):
        p = StubPlanner()
        mem = {}
        plan = None
        updates = ['saw oak', 'found stone', 'iron nearby']
        for u in updates:
            p.set_response({'emit': False, 'instruction': '',
                             'reasoning_update': u, 'plan_update': []})
            _, plan, mem = p.step(obs_dict, plan, mem)
        for u in updates:
            assert u in mem['reasoning']

    def test_plan_replaced_not_appended(self, obs_dict):
        p = StubPlanner()
        mem = {}
        p.set_response({'emit': False, 'instruction': '',
                         'reasoning_update': '', 'plan_update': ['step 1']})
        _, plan, mem = p.step(obs_dict, None, mem)
        p.set_response({'emit': False, 'instruction': '',
                         'reasoning_update': '', 'plan_update': ['step 2']})
        _, plan, mem = p.step(obs_dict, plan, mem)
        assert '• step 2' in plan
        assert '• step 1' not in plan   # replaced, not accumulated

    def test_emit_interspersed_with_no_emit(self, obs_dict):
        p = StubPlanner()
        mem = {}
        plan = None
        instructions = []
        responses = [
            {'emit': False, 'instruction': '', 'reasoning_update': '', 'plan_update': []},
            {'emit': True,  'instruction': 'chop oak', 'reasoning_update': '', 'plan_update': []},
            {'emit': False, 'instruction': '', 'reasoning_update': '', 'plan_update': []},
            {'emit': True,  'instruction': 'craft planks', 'reasoning_update': '', 'plan_update': []},
        ]
        for r in responses:
            p.set_response(r)
            instr, plan, mem = p.step(obs_dict, plan, mem)
            instructions.append(instr)
        assert instructions == [None, 'chop oak', None, 'craft planks']
