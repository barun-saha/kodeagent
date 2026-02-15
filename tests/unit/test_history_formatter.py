"""Unit tests for history formatter classes."""

import json
from types import SimpleNamespace

import pytest

from kodeagent.history_formatter import (
    CODE_ACT_PSEUDO_TOOL_NAME,
    CodeActHistoryFormatter,
    HistoryFormatter,
    ReActHistoryFormatter,
)
from kodeagent.models import ChatMessage, CodeActChatMessage, ReActChatMessage


class TestHistoryFormatterABC:
    """Test HistoryFormatter abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that HistoryFormatter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            HistoryFormatter()  # type: ignore


class TestReActHistoryFormatter:
    """Test ReActHistoryFormatter implementation."""

    @pytest.fixture
    def formatter(self) -> ReActHistoryFormatter:
        """Create a ReActHistoryFormatter instance."""
        return ReActHistoryFormatter()

    def test_should_format_as_tool_call_with_action(self, formatter: ReActHistoryFormatter):
        """Test that messages with action are identified as tool calls."""
        msg = ReActChatMessage(
            role='assistant', thought='I will search', action='search_web', args='{"query": "test"}'
        )
        assert formatter.should_format_as_tool_call(msg) is True

    def test_should_format_as_tool_call_with_final_answer(self, formatter: ReActHistoryFormatter):
        """Test that messages with final_answer are not tool calls."""
        msg = ReActChatMessage(
            role='assistant',
            thought='Done',
            action='FINISH',
            args=None,
            final_answer='The answer is 42',
            task_successful=True,
        )
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_without_action(self, formatter: ReActHistoryFormatter):
        """Test that messages without action are not tool calls."""
        msg = ChatMessage(role='assistant', content='Just thinking...')
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_user_message(self, formatter: ReActHistoryFormatter):
        """Test that user messages are not tool calls."""
        msg = ChatMessage(role='user', content='Do something')
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_with_none_action(self, formatter: ReActHistoryFormatter):
        """Test that messages with None action are not tool calls."""
        # Mocking a message that behaves like ChatMessage (no action)
        msg = SimpleNamespace(
            role='assistant', content='Thinking', model_dump=lambda: {'role': 'assistant'}
        )
        assert formatter.should_format_as_tool_call(msg) is False

    def test_format_tool_call_structure(self, formatter: ReActHistoryFormatter):
        """Test that format_tool_call creates correct structure."""
        msg = ReActChatMessage(
            role='assistant',
            thought='I will search',
            action='search_web',
            args='{"query": "test"}',
        )
        state = {'last_tool_call_id': None, 'pending_tool_call': False}
        result = formatter.format_tool_call(msg, state)

        assert result['role'] == 'assistant'
        assert result['content'] is None
        assert 'tool_calls' in result
        assert len(result['tool_calls']) == 1

        tool_call = result['tool_calls'][0]
        assert tool_call['type'] == 'function'
        assert 'id' in tool_call
        assert tool_call['id'].startswith('call_')
        assert tool_call['function']['name'] == 'search_web'
        assert tool_call['function']['arguments'] == '{"query": "test"}'

    def test_format_tool_call_updates_state(self, formatter: ReActHistoryFormatter):
        """Test that format_tool_call updates state correctly."""
        msg = ReActChatMessage(
            role='assistant', thought='I will search', action='search_web', args='{}'
        )
        state = {'last_tool_call_id': None, 'pending_tool_call': False}
        result = formatter.format_tool_call(msg, state)

        # State should be updated
        assert state['last_tool_call_id'] is not None
        assert state['last_tool_call_id'] == result['tool_calls'][0]['id']
        assert state['pending_tool_call'] is True

    def test_format_tool_call_with_default_args(self, formatter: ReActHistoryFormatter):
        """Test format_tool_call with message that has no args attribute."""
        # ReActChatMessage requires args, so we use a simple object to simulate missing args
        msg = SimpleNamespace(
            role='assistant',
            thought='Thinking',
            action='test_tool',
            model_dump=lambda: {'role': 'assistant'},
        )
        state = {'last_tool_call_id': None, 'pending_tool_call': False}
        result = formatter.format_tool_call(msg, state)  # type: ignore

        assert result['tool_calls'][0]['function']['arguments'] == '{}'

    def test_pydantic_to_dict_tool_call(self, formatter: ReActHistoryFormatter):
        """
        Test pydantic_to_dict for tool call message.

        Args:
            formatter: The ReActHistoryFormatter instance to test.
        """
        msg = ReActChatMessage(
            role='assistant',
            thought='I will search',
            action='search_web',
            args='{"query": "test"}',
        )
        result = formatter.pydantic_to_dict(msg)

        assert result['role'] == 'assistant'
        assert result['content'] is None
        assert 'tool_calls' in result
        assert len(result['tool_calls']) == 1
        assert result['tool_calls'][0]['function']['name'] == 'search_web'
        assert result['_thought'] == 'I will search'
        assert result['_action'] == 'search_web'
        assert result['_args'] == '{"query": "test"}'

    def test_pydantic_to_dict_final_answer(self, formatter: ReActHistoryFormatter):
        """Test pydantic_to_dict for final answer."""
        msg = ReActChatMessage(
            role='assistant',
            thought='Done',
            action='FINISH',
            args=None,
            final_answer='The answer is 42',
            task_successful=True,
        )

        result = formatter.pydantic_to_dict(msg)

        assert result['role'] == 'assistant'
        assert result['content'] == 'The answer is 42'
        assert 'tool_calls' not in result
        assert result['_thought'] == 'Done'
        assert result['_task_successful'] is True

    def test_pydantic_to_dict_generic(self, formatter: ReActHistoryFormatter):
        """Test pydantic_to_dict for generic message."""
        # Generic message doesn't have ReAct specific fields usually, but we check basic preservation
        msg = ChatMessage(role='user', content='Hello')
        result = formatter.pydantic_to_dict(msg)

        assert result['role'] == 'user'
        content = result['content']
        if isinstance(content, list):
            assert content[0]['text'] == 'Hello'
        else:
            assert content == 'Hello'
        assert result['_thought'] is None


class TestCodeActHistoryFormatter:
    """Test CodeActHistoryFormatter implementation."""

    @pytest.fixture
    def formatter(self) -> CodeActHistoryFormatter:
        """Create a CodeActHistoryFormatter instance."""
        return CodeActHistoryFormatter()

    def test_should_format_as_tool_call_with_code(self, formatter: CodeActHistoryFormatter):
        """Test that messages with code are identified as tool calls."""
        msg = CodeActChatMessage(
            role='assistant', thought='I will calculate', code='result = 2 + 2\nprint(result)'
        )
        assert formatter.should_format_as_tool_call(msg) is True

    def test_should_format_as_tool_call_with_final_answer(self, formatter: CodeActHistoryFormatter):
        """Test that messages with final_answer are not tool calls."""
        msg = SimpleNamespace(
            role='assistant',
            thought='Done',
            code=None,
            final_answer='The answer is 4',
            task_successful=True,
            model_dump=lambda: {'role': 'assistant'},
        )
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_without_code(self, formatter: CodeActHistoryFormatter):
        """Test that messages without code are not tool calls."""
        msg = ChatMessage(role='assistant', content='Just thinking...')
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_user_message(self, formatter: CodeActHistoryFormatter):
        """Test that user messages are not tool calls."""
        msg = ChatMessage(role='user', content='Calculate 2+2')
        assert formatter.should_format_as_tool_call(msg) is False

    def test_should_format_as_tool_call_with_none_code(self, formatter: CodeActHistoryFormatter):
        """Test that messages with None code are not tool calls."""
        # Mocking a message that behaves like CodeActChatMessage (no code)
        msg = SimpleNamespace(
            role='assistant',
            thought='Thinking',
            code=None,
            final_answer='Answer',
            model_dump=lambda: {'role': 'assistant'},
        )
        assert formatter.should_format_as_tool_call(msg) is False

    def test_format_tool_call_structure(self, formatter: CodeActHistoryFormatter):
        """Test that format_tool_call creates correct pseudo tool call structure."""
        msg = CodeActChatMessage(
            role='assistant', thought='I will calculate', code='result = 2 + 2\nprint(result)'
        )
        state = {'last_tool_call_id': None, 'pending_tool_call': False}

        result = formatter.format_tool_call(msg, state)

        assert result['role'] == 'assistant'
        assert result['content'] is None
        assert 'tool_calls' in result
        assert len(result['tool_calls']) == 1

        tool_call = result['tool_calls'][0]
        assert tool_call['type'] == 'function'
        assert 'id' in tool_call
        assert tool_call['id'].startswith('call_')
        assert tool_call['function']['name'] == CODE_ACT_PSEUDO_TOOL_NAME

        # Arguments should be JSON-wrapped code
        args = json.loads(tool_call['function']['arguments'])
        assert 'code' in args
        assert args['code'] == 'result = 2 + 2\nprint(result)'

    def test_format_tool_call_updates_state(self, formatter: CodeActHistoryFormatter):
        """Test that format_tool_call updates state correctly."""
        msg = CodeActChatMessage(
            role='assistant', thought='I will calculate', code='print("hello")'
        )
        state = {'last_tool_call_id': None, 'pending_tool_call': False}

        result = formatter.format_tool_call(msg, state)

        # State should be updated with tool_call_id but NOT pending_tool_call
        assert state['last_tool_call_id'] is not None
        assert state['last_tool_call_id'] == result['tool_calls'][0]['id']
        # CodeAct does NOT set pending_tool_call
        assert state['pending_tool_call'] is False

    def test_format_tool_call_with_special_characters(self, formatter: CodeActHistoryFormatter):
        """Test format_tool_call with code containing special characters."""
        code_with_quotes = 'print("Hello \'world\'")\nresult = {"key": "value"}'
        msg = CodeActChatMessage(role='assistant', thought='Testing', code=code_with_quotes)
        state = {'last_tool_call_id': None, 'pending_tool_call': False}
        result = formatter.format_tool_call(msg, state)

        args = json.loads(result['tool_calls'][0]['function']['arguments'])
        assert args['code'] == code_with_quotes

    def test_pydantic_to_dict_code(self, formatter: CodeActHistoryFormatter):
        """Test pydantic_to_dict for code execution message."""
        msg = CodeActChatMessage(
            role='assistant',
            thought='I will calculate',
            code='print(2+2)',
        )
        result = formatter.pydantic_to_dict(msg)

        assert result['role'] == 'assistant'
        assert 'tool_calls' in result
        assert result['tool_calls'][0]['function']['name'] == CODE_ACT_PSEUDO_TOOL_NAME
        assert result['_thought'] == 'I will calculate'
        assert result['_code'] == 'print(2+2)'

    def test_pydantic_to_dict_final_answer(self, formatter: CodeActHistoryFormatter):
        """Test pydantic_to_dict for final answer."""
        msg = CodeActChatMessage(
            role='assistant',
            thought='Done',
            final_answer='4',
            task_successful=True,
            code=None,
        )
        result = formatter.pydantic_to_dict(msg)

        assert result['role'] == 'assistant'
        assert result['content'] == '4'
        assert 'tool_calls' not in result
        assert result['_thought'] == 'Done'
        assert result['_task_successful'] is True
