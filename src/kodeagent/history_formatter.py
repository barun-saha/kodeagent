"""History formatting strategies for agent message history.

This module provides the Strategy pattern implementation for formatting agent
message history for LLM consumption. Different agent types (ReAct, CodeAct) have
different formatting requirements, particularly for tool calls.
"""

import json
import uuid
from abc import ABC, abstractmethod

from .models import ChatMessage

CODE_ACT_PSEUDO_TOOL_NAME = 'code_execution'


class HistoryFormatter(ABC):
    """Abstract base class for formatting agent message history for LLM calls.

    This class defines the interface for formatting strategies. Subclasses implement
    agent-specific logic for detecting and formatting tool calls.
    """

    @abstractmethod
    def should_format_as_tool_call(self, msg: ChatMessage) -> bool:
        """Determine if a message should be formatted as a tool call.

        Args:
            msg: The message to check.

        Returns:
            True if the message should be formatted as a tool call.
        """

    @abstractmethod
    def format_tool_call(self, msg: ChatMessage, state: dict) -> dict:
        """Format a message as a tool call.

        Args:
            msg: The message containing tool call information.
            state: Mutable state dict with keys:
                - last_tool_call_id: The most recent tool call ID
                - pending_tool_call: Whether a tool call is awaiting response

        Returns:
            Formatted tool call dictionary with role, content, and tool_calls.
        """

    @abstractmethod
    def should_add_pending_placeholder(self, state: dict) -> bool:
        """Determine if a placeholder should be added for pending tool calls.

        Args:
            state: State dict with keys:
                - last_tool_call_id: The most recent tool call ID
                - pending_tool_call: Whether a tool call is awaiting response

        Returns:
            True if a placeholder should be added.
        """


class ReActHistoryFormatter(HistoryFormatter):
    """Formats ReAct agent history with action-based tool calls.

    ReAct agents use an action/args structure for tool calls and track pending
    tool calls to add placeholders for interrupted executions.
    """

    def should_format_as_tool_call(self, msg: ChatMessage) -> bool:
        """Check if message has action field for tool call.

        Args:
            msg: The message to check.

        Returns:
            True if message is an assistant message with an action and no final answer.
        """
        d = msg.model_dump()
        return bool(
            d.get('role') == 'assistant'
            and getattr(msg, 'action', None)
            and not getattr(msg, 'final_answer', None)
        )

    def format_tool_call(self, msg: ChatMessage, state: dict) -> dict:
        """Format ReAct action as tool call.

        Args:
            msg: Message containing action and args.
            state: Mutable state dict to update with tool call ID.

        Returns:
            Formatted tool call dictionary.
        """
        tool_call_id = f'call_{uuid.uuid4().hex[:8]}'
        state['last_tool_call_id'] = tool_call_id
        state['pending_tool_call'] = True
        return {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': tool_call_id,
                    'type': 'function',
                    'function': {
                        'name': msg.action,
                        'arguments': getattr(msg, 'args', '{}'),
                    },
                }
            ],
        }

    def should_add_pending_placeholder(self, state: dict) -> bool:
        """Check if placeholder needed for interrupted tool call.

        Args:
            state: State dict with pending_tool_call and last_tool_call_id.

        Returns:
            True if there's a pending tool call without a response.
        """
        return bool(state.get('pending_tool_call', False) and state.get('last_tool_call_id'))


class CodeActHistoryFormatter(HistoryFormatter):
    """Formats CodeAct agent history with code-based pseudo tool calls.

    CodeAct agents use a code field and format it as a pseudo tool call with
    the name 'code_execution'. They do not track pending tool calls.
    """

    def should_format_as_tool_call(self, msg: ChatMessage) -> bool:
        """Check if message has code field for tool call.

        Args:
            msg: The message to check.

        Returns:
            True if message is an assistant message with code and no final answer.
        """
        d = msg.model_dump()
        return bool(
            d.get('role') == 'assistant'
            and getattr(msg, 'code', None)
            and not getattr(msg, 'final_answer', None)
        )

    def format_tool_call(self, msg: ChatMessage, state: dict) -> dict:
        """Format CodeAct code as pseudo tool call.

        Args:
            msg: Message containing code.
            state: Mutable state dict to update with tool call ID.

        Returns:
            Formatted pseudo tool call dictionary.
        """
        tool_call_id = f'call_{uuid.uuid4().hex[:8]}'
        state['last_tool_call_id'] = tool_call_id
        return {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': tool_call_id,
                    'type': 'function',
                    'function': {
                        'name': CODE_ACT_PSEUDO_TOOL_NAME,
                        'arguments': json.dumps({'code': msg.code}),
                    },
                }
            ],
        }

    def should_add_pending_placeholder(self, _state: dict) -> bool:
        """CodeAct does not use placeholders for interrupted code execution.

        Args:
            _state: State dict (unused).

        Returns:
            Always False.
        """
        return False
