"""History formatting strategies for agent message history.

This module provides the Strategy pattern implementation for formatting agent
message history for LLM consumption. Different agent types (ReAct, CodeAct) have
different formatting requirements, particularly for tool calls.
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from . import kutils as ku
from .models import ChatMessage

CODE_ACT_PSEUDO_TOOL_NAME = 'code_execution'


class LLMMessage(dict):
    """A dictionary subclass that hides internal metadata from LLM APIs.

    Standard keys (role, content, tool_calls, tool_call_id, name) are stored
    in the dictionary itself. Non-standard keys (starting with '_' or 'files')
    are stored as instance attributes. This ensures that only compliant keys
    are serialized to JSON or iterated over when passed to LLM APIs like LiteLLM.
    """

    _ALLOWED_KEYS: ClassVar[set[str]] = {
        'role',
        'content',
        'tool_calls',
        'tool_call_id',
        'name',
    }

    def __init__(self, **kwargs: Any):
        """Create an LLM-compliant message with hidden metadata.

        Args:
            **kwargs: Message fields, including standard (role, content, etc.)
                      and non-standard (prefixed with _ or 'files').
        """
        standard = {k: v for k, v in kwargs.items() if k in self._ALLOWED_KEYS}
        metadata = {k: v for k, v in kwargs.items() if k not in self._ALLOWED_KEYS}
        super().__init__(**standard)
        # Store metadata as attributes so they don't appear in dict.keys() or JSON
        for k, v in metadata.items():
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        """Get a value from standard keys or metadata attributes."""
        # Check standard keys first
        if key in self._ALLOWED_KEYS:
            return super().__getitem__(key)
        # Fallback to attributes (metadata keys like _thought, files)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with an optional default, checking both storage and attributes."""
        if key in self._ALLOWED_KEYS:
            return super().get(key, default)
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in either standard storage or metadata attributes."""
        return super().__contains__(key) or (isinstance(key, str) and hasattr(self, key))

    def update(self, *args, **kwargs):
        """Update standard keys or metadata."""
        if args:
            if len(args) > 1:
                raise TypeError(f'update expected at most 1 arguments, got {len(args)}')
            other = args[0]
            if isinstance(other, dict):
                for k, v in other.items():
                    self[k] = v
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: Any):
        """Set a standard key or a metadata attribute."""
        if key in self._ALLOWED_KEYS:
            super().__setitem__(key, value)
        else:
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Return a string representation including both standard and metadata keys."""
        # Get standard dict representation
        std_dict = dict(self)
        # Collect metadata attributes
        metadata = {
            k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != '_ALLOWED_KEYS'
        }
        # Add internal metadata if any (starting with _)
        internal_metadata = {k: v for k, v in self.__dict__.items() if k.startswith('_')}
        metadata.update(internal_metadata)

        full_dict = {**std_dict, **metadata}
        return f'{self.__class__.__name__}({full_dict})'


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
    def pydantic_to_dict(self, msg: ChatMessage) -> dict:
        """Convert Pydantic message to dict with proper formatting and metadata.

        Args:
            msg: The Pydantic ChatMessage (ReActChatMessage or CodeActChatMessage).

        Returns:
            A dictionary representation compliant with OpenAPI/LLM specs,
            including metadata fields prefixed with '_'.
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

    def pydantic_to_dict(self, msg: ChatMessage) -> dict:
        """Convert ReActChatMessage to dict.

        Args:
            msg: The ReActChatMessage to convert.

        Returns:
            A dictionary representation of the message, formatted for LLM consumption,
            with special handling for tool calls and final answers, and metadata fields
            prefixed with '_'.
        """
        d = msg.model_dump()

        # Tool call request (Assistant -> Tool)
        if self.should_format_as_tool_call(msg):
            tool_call_id = f'call_{uuid.uuid4().hex[:8]}'
            return LLMMessage(
                role='assistant',
                content=None,
                tool_calls=[
                    {
                        'id': tool_call_id,
                        'type': 'function',
                        'function': {
                            'name': msg.action,
                            'arguments': getattr(msg, 'args', '{}'),
                        },
                    }
                ],
                files=d.get('files'),
                # Metadata for internal use
                _thought=getattr(msg, 'thought', None),
                _action=msg.action,
                _args=getattr(msg, 'args', None),
                _task_successful=getattr(msg, 'task_successful', False),
            )

        # Final answer (Assistant -> User)
        if hasattr(msg, 'final_answer') and getattr(msg, 'final_answer', None):
            return LLMMessage(
                role='assistant',
                content=msg.final_answer,
                files=d.get('files'),
                _thought=getattr(msg, 'thought', None),
                _task_successful=getattr(msg, 'task_successful', True),
                _action=getattr(msg, 'action', 'FINISH'),
            )

        # Generic assistant/user/system message
        if d.get('role') == 'user':
            # Use utility to handle files (images/text)
            usr_msgs = ku.make_user_message(d.get('content', ''), d.get('files'))
            if usr_msgs:
                usr_msg = usr_msgs[0]  # ku returns a list of one message dict
                # Standard metadata preserve
                usr_msg.update(
                    {
                        '_thought': getattr(msg, 'thought', None),
                        '_action': getattr(msg, 'action', None),
                        '_args': getattr(msg, 'args', None),
                        '_task_successful': getattr(msg, 'task_successful', True),
                        'files': d.get('files'),
                    }
                )
                return LLMMessage(**usr_msg)

        return LLMMessage(
            role=d.get('role'),
            content=d.get('content', ''),
            files=d.get('files'),
            _thought=getattr(msg, 'thought', None),
            _action=getattr(msg, 'action', None),
            _args=getattr(msg, 'args', None),
            _task_successful=getattr(msg, 'task_successful', True),
        )


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

    def pydantic_to_dict(self, msg: ChatMessage) -> dict:
        """Convert CodeActChatMessage to dict.

        Args:
            msg: The CodeActChatMessage to convert.

        Returns:
            A dictionary representation of the message, formatted for LLM consumption,
            with special handling for code execution tool calls and final answers.
        """
        d = msg.model_dump()

        # Code execution request (Assistant -> Pseudo-Tool)
        if self.should_format_as_tool_call(msg):
            tool_call_id = f'call_{uuid.uuid4().hex[:8]}'
            return LLMMessage(
                role='assistant',
                content=None,
                tool_calls=[
                    {
                        'id': tool_call_id,
                        'type': 'function',
                        'function': {
                            'name': CODE_ACT_PSEUDO_TOOL_NAME,
                            'arguments': json.dumps({'code': msg.code}),
                        },
                    }
                ],
                files=d.get('files'),
                # Metadata
                _thought=getattr(msg, 'thought', None),
                _code=msg.code,
                _task_successful=getattr(msg, 'task_successful', False),
            )

        # Final answer
        if hasattr(msg, 'final_answer') and getattr(msg, 'final_answer', None):
            return LLMMessage(
                role='assistant',
                content=msg.final_answer,
                files=d.get('files'),
                _thought=getattr(msg, 'thought', None),
                _task_successful=getattr(msg, 'task_successful', True),
            )

        # Generic
        if d.get('role') == 'user':
            usr_msgs = ku.make_user_message(d.get('content', ''), d.get('files'))
            if usr_msgs:
                usr_msg = usr_msgs[0]
                usr_msg.update(
                    {
                        '_thought': getattr(msg, 'thought', None),
                        '_code': getattr(msg, 'code', None),
                        '_task_successful': getattr(msg, 'task_successful', True),
                        'files': d.get('files'),
                    }
                )
                return LLMMessage(**usr_msg)

        return LLMMessage(
            role=d.get('role'),
            content=d.get('content', ''),
            files=d.get('files'),
            _thought=getattr(msg, 'thought', None),
            _code=getattr(msg, 'code', None),
            _task_successful=getattr(msg, 'task_successful', True),
        )
