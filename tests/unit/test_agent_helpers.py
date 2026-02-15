"""Unit tests for small Agent helper methods.

Tests added:
- _get_last_tool_call_id: returns correct id or None for various histories
- get_history: respects HISTORY_TRUNCATE_CHARS
"""

import os
import sys

# Ensure src is importable when running tests from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from kodeagent.kodeagent import Agent
from kodeagent.models import ChatMessage


class DummyAgent(Agent):
    """Minimal concrete Agent for testing helpers."""

    def parse_text_response(self, text: str) -> ChatMessage:
        """Return a trivial ChatMessage for testing."""
        return ChatMessage(role='assistant', content=text)

    async def run(
        self,
        task: str,
        files: list[str] | None = None,
        task_id: str | None = None,
        max_iterations: int | None = None,
        recurrent_mode: bool = False,
        summarize_progress_on_failure: bool = True,
    ):
        """No-op run implementation for tests (async generator)."""
        if False:  # make this an async generator
            yield


def test_get_last_tool_call_id_none_when_no_history() -> None:
    """Test that _get_last_tool_call_id returns None when there is no chat history."""
    agent = DummyAgent(name='d', model_name='m', tools=[])
    assert agent._get_last_tool_call_id() is None


def test_get_last_tool_call_id_none_when_last_has_no_tool_calls() -> None:
    """Test that _get_last_tool_call_id returns None when the last assistant message
     has no tool_calls."""
    agent = DummyAgent(name='d', model_name='m', tools=[])
    agent.chat_history = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'assistant', 'content': 'hello'},
    ]
    assert agent._get_last_tool_call_id() is None


def test_get_last_tool_call_id_returns_id_when_present() -> None:
    """Test that _get_last_tool_call_id returns the correct id when the last assistant message."""
    agent = DummyAgent(name='d', model_name='m', tools=[])
    agent.chat_history = [
        {'role': 'system', 'content': 'sys'},
        {
            'role': 'assistant',
            'content': 'called tool',
            'tool_calls': [{'id': 'tool-call-123', 'name': 'fake_tool'}],
        },
    ]

    assert agent._get_last_tool_call_id() == 'tool-call-123'


def test_get_last_tool_call_id_none_when_tool_calls_empty() -> None:
    """Test that _get_last_tool_call_id returns None when the last assistant message
     has an empty tool_calls list."""
    agent = DummyAgent(name='d', model_name='m', tools=[])
    agent.chat_history = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'assistant', 'content': 'called tool', 'tool_calls': []},
    ]
    assert agent._get_last_tool_call_id() is None


def test_get_history_respects_truncation_constant() -> None:
    """Test that get_history truncates the history according to HISTORY_TRUNCATE_CHARS."""
    agent = DummyAgent(name='d', model_name='m', tools=[])
    long_text = 'x' * 50
    agent.chat_history = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': long_text},
    ]

    original = Agent.HISTORY_TRUNCATE_CHARS
    try:
        Agent.HISTORY_TRUNCATE_CHARS = 10
        hist = agent.get_history()
        assert '... [TRUNCATED]' in hist
        assert len(hist) < 100  # sanity check
    finally:
        Agent.HISTORY_TRUNCATE_CHARS = original
