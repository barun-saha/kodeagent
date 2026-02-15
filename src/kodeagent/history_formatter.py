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
    This is used during converstion of Pydantic structured reponse to a dict for LLM APIs.

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

    def pydantic_to_dict(self, msg: ChatMessage) -> dict:
        """Convert Pydantic message to dict with proper formatting and metadata.

        Args:
            msg: The Pydantic ChatMessage (ReActChatMessage or CodeActChatMessage).

        Returns:
            A dictionary representation compliant with OpenAPI/LLM specs,
            including metadata fields prefixed with '_'.
        """
        # Tool call request (Assistant -> Tool)
        if self.should_format_as_tool_call(msg):
            # Delegating to format_tool_call to avoid duplication
            # ReAct tracks pending calls, so we use dummy state here for pydantic_to_dict
            tc_state = {'last_tool_call_id': None, 'pending_tool_call': False}
            tool_call_dict = self.format_tool_call(msg, tc_state)

            metadata = self._get_common_metadata(msg)
            metadata['_task_successful'] = getattr(msg, 'task_successful', False)
            metadata.update(self._get_tool_call_metadata(msg))
            return LLMMessage(**tool_call_dict, **metadata)

        # Final answer (Assistant -> User)
        final_answer = getattr(msg, 'final_answer', None)
        if final_answer:
            metadata = self._get_common_metadata(msg)
            metadata['_task_successful'] = getattr(msg, 'task_successful', True)
            metadata.update(self._get_final_answer_metadata(msg))
            return LLMMessage(role='assistant', content=final_answer, **metadata)

        # Generic assistant/user/system message
        return self._format_user_or_generic(msg, extra_metadata=self._get_extra_metadata(msg))

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f'call_{uuid.uuid4().hex[:8]}'

    def _get_common_metadata(self, msg: ChatMessage) -> dict:
        """Extract common metadata fields from message.

        Args:
            msg: The message to extract metadata from.

        Returns:
            Dict containing _thought, _task_successful, and files.
        """
        return {
            '_thought': getattr(msg, 'thought', None),
            'files': getattr(msg, 'files', None),
        }

    @abstractmethod
    def _get_tool_call_metadata(self, msg: ChatMessage) -> dict:
        """Hook for strategy-specific tool call metadata."""

    @abstractmethod
    def _get_final_answer_metadata(self, msg: ChatMessage) -> dict:
        """Hook for strategy-specific final answer metadata."""

    @abstractmethod
    def _get_extra_metadata(self, msg: ChatMessage) -> dict:
        """Hook for strategy-specific generic metadata."""

    def _format_user_or_generic(
        self, msg: ChatMessage, extra_metadata: dict | None = None
    ) -> LLMMessage:
        """Handle user/system/generic messages and metadata preservation.

        Args:
            msg: The message to format.
            extra_metadata: Additional metadata to merge.

        Returns:
            Formatted LLMMessage.
        """
        d = msg.model_dump()
        role = d.get('role')

        metadata = self._get_common_metadata(msg)
        metadata['_task_successful'] = getattr(msg, 'task_successful', True)
        if extra_metadata:
            metadata.update(extra_metadata)

        if role == 'user':
            usr_msgs = ku.make_user_message(d.get('content', ''), d.get('files'))
            if usr_msgs:
                usr_msg = usr_msgs[0]
                usr_msg.update(metadata)
                return LLMMessage(**usr_msg)

        return LLMMessage(role=role, content=d.get('content', ''), **metadata)


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
        return bool(
            msg.role == 'assistant'
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
        tool_call_id = self._generate_tool_call_id()
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

    def _get_tool_call_metadata(self, msg: ChatMessage) -> dict:
        """Provide ReAct-specific tool call metadata."""
        return {
            '_action': getattr(msg, 'action', None),
            '_args': getattr(msg, 'args', None),
        }

    def _get_final_answer_metadata(self, msg: ChatMessage) -> dict:
        """Provide ReAct-specific final answer metadata."""
        return {'_action': getattr(msg, 'action', 'FINISH')}

    def _get_extra_metadata(self, msg: ChatMessage) -> dict:
        """Provide ReAct-specific generic metadata."""
        return {
            '_action': getattr(msg, 'action', None),
            '_args': getattr(msg, 'args', None),
        }


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
        return bool(
            msg.role == 'assistant'
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
        tool_call_id = self._generate_tool_call_id()
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
                        'arguments': json.dumps({'code': getattr(msg, 'code', '')}),
                    },
                }
            ],
        }

    def _get_tool_call_metadata(self, msg: ChatMessage) -> dict:
        """Provide CodeAct-specific tool call metadata."""
        return {'_code': getattr(msg, 'code', None)}

    def _get_final_answer_metadata(self, msg: ChatMessage) -> dict:
        """Provide CodeAct-specific final answer metadata."""
        return {}

    def _get_extra_metadata(self, msg: ChatMessage) -> dict:
        """Provide CodeAct-specific generic metadata."""
        return {'_code': getattr(msg, 'code', None)}
