"""Data models for tasks, agent plans, chat messages, and agent responses.
Uses Pydantic for data validation and serialization. This module defines various models
(task, plan, observation, chat response) to structure the interactions and behaviors of agents.
"""

import json
import uuid
from json import JSONDecodeError
from typing import Any, Literal, TypedDict

import json_repair
import pydantic as pyd

from . import kutils as ku

MESSAGE_ROLES = Literal['user', 'assistant', 'system', 'tool']
"""Defined roles for chat messages."""

AGENT_RESPONSE_TYPES = Literal['step', 'final', 'log']
"""Defined types for agent responses."""


class Task(pyd.BaseModel):
    """Task to be solved by an agent."""

    id: str = pyd.Field(
        description='Auto-generated task ID', default_factory=lambda: str(uuid.uuid4())
    )
    """Auto-generated task ID."""
    description: str = pyd.Field(description='Task description')
    """Task description."""
    files: list[str] | None = pyd.Field(description='A list of file paths or URLs', default=None)
    """A list of file paths or URLs."""
    result: Any | None = pyd.Field(description='Task result', default=None)
    """Task result."""
    is_finished: bool = pyd.Field(
        description='Whether the task has finished running', default=False
    )
    """Whether the task has finished running."""
    is_error: bool = pyd.Field(
        description='Whether the task execution resulted in any error', default=False
    )
    """Whether the task execution resulted in any error."""
    output_files: list[str] = pyd.Field(
        description='List of file paths generated during task execution', default_factory=list
    )
    """List of file paths generated during task execution."""
    steps_taken: int = pyd.Field(
        description='Number of steps taken by the agent for this task; skip this field', default=0
    )
    """Number of steps/iterations taken by the agent for this task."""
    total_llm_calls: int = pyd.Field(
        description='Total number of LLM calls made during task execution', default=0
    )
    """Total number of LLM calls made during task execution."""
    total_prompt_tokens: int = pyd.Field(description='Total prompt tokens used', default=0)
    """Total prompt tokens used."""
    total_completion_tokens: int = pyd.Field(description='Total completion tokens used', default=0)
    """Total completion tokens used."""
    total_tokens: int = pyd.Field(description='Total tokens used (prompt + completion)', default=0)
    """Total tokens used (prompt + completion)."""
    total_cost: float = pyd.Field(description='Total cost in USD for all LLM calls', default=0.0)
    """Total cost in USD for all LLM calls."""
    usage_by_component: dict[str, dict] | None = pyd.Field(
        description='Breakdown of usage by component (Planner, Observer, Agent)', default=None
    )
    """Breakdown of usage by component (Planner, Observer, Agent)."""


class PlanStep(pyd.BaseModel):
    """A single step in an agent's plan."""

    description: str = pyd.Field(description='A brief description of the step')
    """A brief description of the step."""
    is_done: bool = pyd.Field(description='Whether the step has been completed', default=False)
    """Whether the step has been completed."""


class AgentPlan(pyd.BaseModel):
    """A structured plan for an agent to follow."""

    steps: list[PlanStep] = pyd.Field(description='List of steps to accomplish the task')
    """List of steps to accomplish the task."""


class ObserverResponse(pyd.BaseModel):
    """The response from the observer after analyzing the agent's behavior."""

    is_progressing: bool = pyd.Field(
        description='True if the agent is making meaningful progress on the plan'
    )
    """True if the agent is making meaningful progress on the plan."""
    is_in_loop: bool = pyd.Field(description='True if the agent is stuck in a repetitive loop')
    """True if the agent is stuck in a repetitive loop."""
    reasoning: str = pyd.Field(description='A short reason for the assessment (max 15--20 words)')
    """A short reason for the assessment (max 15--20 words)."""
    correction_message: str | None = pyd.Field(
        description='A specific, actionable feedback to help the agent self-correct'
    )
    """A specific, actionable feedback to help the agent self-correct."""


class ChatMessage(pyd.BaseModel):
    """Generic chat message."""

    role: MESSAGE_ROLES = pyd.Field(description='Role of the message sender')
    """Role of the message sender."""
    content: Any = pyd.Field(description='Content of the message')
    """Content of the message."""
    files: list[str] | None = pyd.Field(
        description='Optional list of file paths or URLs associated with the message', default=None
    )
    """Optional list of file paths or URLs associated with the message."""

    def __str__(self) -> str:
        """Return proper string representation of the message."""
        return str(self.content) if self.content is not None else ''


class ReActChatMessage(ChatMessage):
    """Messages for the ReAct agent with built-in validation.
    Combines functionality of ReActAgentResponse and ReActChatMessage.
    """

    role: MESSAGE_ROLES = pyd.Field(
        description='Role of the message sender',
        default='assistant',  # Add default
    )
    """Role of the message sender. Defaults to 'assistant'."""
    content: str | None = pyd.Field(description='ALWAYS `None`', exclude=True, default=None)
    """Content of the message. Always `None` for ReAct messages."""
    thought: str = pyd.Field(description='Thoughts behind the tool use')
    """Thoughts behind the tool use."""
    action: str = pyd.Field(
        description=(
            "Name of the tool to use from available tools, or 'FINISH' to provide final answer. "
            'Must be exact tool name as listed in Available Tools section.'
        )
    )
    """Name of the tool to use from available tools, or 'FINISH' to provide final answer."""
    args: str | None = pyd.Field(
        default=None,
        description=(
            'Tool arguments as JSON string; `None` when `final_answer` is available. '
            'Must be valid JSON with double quotes.'
        ),
    )
    """Tool arguments as JSON string; `None` when `final_answer` is available."""
    final_answer: str | None = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    """Final answer for the task; set only in the final step."""
    task_successful: bool = pyd.Field(
        description='Task completed or failed? False when `args` is set.', default=False
    )
    """Task completed or failed? False when `args` is set."""

    @pyd.field_validator('args')
    @classmethod
    def validate_args_json(cls, v: str | None) -> str | None:
        """Validate that args is valid JSON and normalize it.

        Args:
            v: The args string to validate.

        Returns:
            The normalized JSON string if valid, else None.
        """
        if v is None:
            return None

        v = v.strip()
        if not v:
            return None

        # Clean and validate JSON
        v = ku.clean_json_string(v)
        try:
            # Validate it's parseable
            parsed = json.loads(v)
            # Ensure it's a dict
            if not isinstance(parsed, dict):
                raise ValueError(f'args must be a JSON object (dict), got {type(parsed)}')
            # Return normalized JSON string
            return json.dumps(parsed)
        except JSONDecodeError as e:
            try:
                v = json_repair.repair_json(v)
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError(f'args must be a JSON object (dict), got {type(parsed)}')
                return json.dumps(parsed)
            except Exception:
                raise ValueError(
                    f'args must be valid JSON object string. Received: {v[:100]}... Error: {e}'
                )

    @pyd.model_validator(mode='after')
    def validate_mutual_exclusivity(self) -> 'ReActChatMessage':
        """Ensure tool call and final answer are mutually exclusive.

        Raises:
            ValueError: If both action and final_answer are provided, or if neither is valid.
        """
        is_finish = self.action == 'FINISH'
        has_final_answer = self.final_answer is not None
        has_args = self.args is not None

        if self.action != 'FINISH':
            if has_final_answer:
                raise ValueError(
                    'Cannot have both action (tool call) and final_answer. '
                    "Use action='FINISH' when providing final_answer."
                )
            if self.task_successful:
                raise ValueError('task_successful must be False for intermediate tool calls')
            if not has_args:
                raise ValueError(f"args must be provided when using tool '{self.action}'")
        elif is_finish:
            if not has_final_answer:
                raise ValueError("final_answer must be provided when action is 'FINISH'")
            if has_args:
                raise ValueError("args must be None when action is 'FINISH'")
        return self

    @property
    def is_final(self) -> bool:
        """Check if this is a final answer."""
        return self.action == 'FINISH' and self.final_answer is not None

    def __str__(self) -> str:
        """Return a string representation of the message."""
        if self.is_final:
            return f'{self.final_answer}'

        parts = []
        if self.thought:
            parts.append(f'Thought: {self.thought}')
        if self.action:
            parts.append(f'Action: {self.action}')
        if self.args:
            parts.append(f'Args: {self.args}')

        return '\n'.join(parts)


class CodeActChatMessage(ChatMessage):
    """Messages for the CodeActAgent with built-in validation.
    Combines functionality of CodeActAgentResponse and CodeActChatMessage.
    """

    role: MESSAGE_ROLES = pyd.Field(description='Role of the message sender', default='assistant')
    """Role of the message sender. Defaults to 'assistant'."""
    content: str | None = pyd.Field(description='ALWAYS `None`', exclude=True, default=None)
    """Content of the message. Always `None` for CodeAct messages."""
    thought: str = pyd.Field(description='Thoughts behind the code')
    """Thoughts behind the code."""
    code: str | None = pyd.Field(
        default=None,
        description='Python code with tool use to run; `None` when providing final answer',
    )
    """Python code with tool use to run; `None` when providing final answer."""
    final_answer: str | None = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    """Final answer for the task; set only in the final step."""
    task_successful: bool = pyd.Field(
        description='Task completed or failed? False when `code` is set.', default=False
    )
    """Task completed or failed? False when `code` is set."""

    @pyd.model_validator(mode='after')
    def validate_mutual_exclusivity(self) -> 'CodeActChatMessage':
        """Ensure code execution and final answer are mutually exclusive.

        Raises:
            ValueError: If both code and final_answer are provided, or if neither is valid.
        """
        has_code = self.code is not None and self.code.strip()
        has_final_answer = self.final_answer is not None

        if has_code and has_final_answer:
            raise ValueError(
                'Cannot have both code and final_answer. '
                'Provide either code for execution or final_answer to conclude.'
            )
        if not has_code and not has_final_answer:
            raise ValueError('Must provide either code for execution or final_answer to conclude')
        if has_code and self.task_successful:
            raise ValueError(
                'task_successful must be False when executing code (intermediate step)'
            )
        return self

    @property
    def is_final(self) -> bool:
        """Check if this is a final answer."""
        return self.final_answer is not None

    def __str__(self) -> str:
        """Return a string representation of the message."""
        if self.is_final:
            return f'{self.final_answer}'

        parts = []
        if self.thought:
            parts.append(f'Thought: {self.thought}')
        if self.code:
            parts.append(f'Code:\n```python\n{self.code}\n```')

        return '\n'.join(parts)


class AgentResponse(TypedDict):
    """Streaming response sent by an agent in the course of solving a task."""

    type: AGENT_RESPONSE_TYPES
    """Type of the response: 'step', 'final', or 'log'."""
    channel: str | None
    """Optional channel name for the response."""
    value: Any
    """Value of the response, varies by type."""
    metadata: dict[str, Any] | None
    """Optional metadata associated with the response."""


class CodeReview(pyd.BaseModel):
    """Code review decision for CodeActAgent."""

    is_secure: bool = pyd.Field(description='Is the code safe & secure for execution?')
    """Is the code safe & secure for execution?"""
    reason: str = pyd.Field(description='A brief explanation of the decision')
    """A brief explanation of the decision."""


class UsageMetrics(pyd.BaseModel):
    """Individual usage metrics for a single LLM call."""

    prompt_tokens: int = 0
    """Number of prompt tokens used."""
    completion_tokens: int = 0
    """Number of completion tokens used."""
    total_tokens: int = 0
    """Total tokens used (prompt + completion)."""
    cost: float = 0.0
    """Cost in USD for the LLM call."""


class ComponentUsage(pyd.BaseModel):
    """Aggregated usage for a specific component."""

    component_name: str
    """Name of the component (e.g., Planner, Observer, Agent)."""
    call_count: int = 0
    """Number of LLM calls made by the component."""
    total_prompt_tokens: int = 0
    """Total prompt tokens used by the component."""
    total_completion_tokens: int = 0
    """Total completion tokens used by the component."""
    total_tokens: int = 0
    """Total tokens used by the component."""
    total_cost: float = 0.0
    """Total cost in USD for the component."""
