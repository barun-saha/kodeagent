"""A minimalistic approach to building AI agents.
Implements ReAct and CodeAct agents, supported by Planner and Observer.
"""

import asyncio
import inspect
import json
import os
import random
import re
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from json import JSONDecodeError
from typing import Any, ClassVar

import json_repair
import litellm
import pydantic as pyd
import rich
from dotenv import load_dotenv
from tenacity import RetryError

from . import kutils as ku
from . import tools as dtools
from . import tracer
from .code_runner import CODE_ENV_NAMES, CodeRunner
from .file_tracker import OutputInterceptor, install_interceptor
from .history_formatter import (
    CodeActHistoryFormatter,
    HistoryFormatter,
    ReActHistoryFormatter,
)
from .models import (
    AGENT_RESPONSE_TYPES,
    AgentResponse,
    ChatMessage,
    CodeActChatMessage,
    ReActChatMessage,
    Task,
)
from .orchestrator import Observer, Planner
from .tracer import TRACING_TYPES, create_tracer_manager
from .usage_tracker import UsageTracker

# Install the global interceptor patch
install_interceptor()

load_dotenv()

warnings.simplefilter('once', UserWarning)
warnings.filterwarnings('ignore', message='.*Pydantic serializer warnings.*')

logger = ku.get_logger()

REACT_SYSTEM_PROMPT = ku.read_prompt('system/react.txt')
CODE_ACT_SYSTEM_PROMPT = ku.read_prompt('system/codeact.txt')
SALVAGE_RESPONSE_PROMPT = ku.read_prompt('salvage_response.txt')
RELEVANT_TOOLS_PROMPT = ku.read_prompt('relevant_tools.txt')


# Regex for message parsing from LLM response text (case-insensitive, multiline)
THOUGHT_MATCH = re.compile(
    r'Thought:\s*(.+?)(?=\n(?:Action|Answer|Code):|$)', re.DOTALL | re.IGNORECASE
)
ACTION_MATCH = re.compile(r'Action:\s*(\w+)', re.IGNORECASE)
ARGS_MATCH = re.compile(r'Args:\s*(\{.+?\})', re.DOTALL | re.IGNORECASE)
ANSWER_MATCH = re.compile(r'Answer:\s*(.+?)(?=\nSuccessful:|$)', re.DOTALL | re.IGNORECASE)
SUCCESS_MATCH = re.compile(r'Successful:\s*(true|false)', re.IGNORECASE)
CODE_MATCH = re.compile(r'Code:\s*```(?:python)?\s*(.+?)\s*```', re.DOTALL | re.IGNORECASE)

MAX_RESPONSE_PARSING_ATTEMPTS = 3
MAX_TASK_FILES = 10


class Agent(ABC):
    """An abstract agent. Base class for all types of agents."""

    response_format_class: ClassVar[type[pyd.BaseModel]] = ChatMessage

    def __init__(
        self,
        name: str,
        model_name: str,
        description: str | None = None,
        tools: list[Callable] | None = None,
        litellm_params: dict | None = None,
        persona: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
        filter_tools_for_task: bool = False,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        work_dir: str | None = None,
        tracing_type: TRACING_TYPES | None = None,
    ):
        """Create an agent.

        Args:
            name: The name of the agent.
            model_name: The (LiteLLM) model name to use.
            description: Optional brief description about the agent.
            tools: An optional list of tools available to the agent.
            litellm_params: LiteLLM parameters.
            persona: Optional persona for the agent. If provided, this gets added to
             the system prompt.
            system_prompt: Optional system prompt for the agent. If not provided, default is used.
             This is mutually exclusive with persona.
            max_iterations: The max iterations an agent can perform to solve a task.
            filter_tools_for_task: Whether the tools should be filtered for a task. Unused.
            max_retries: Maximum number of retries for LLM calls.
            work_dir: Optional local workspace directory.
            tracing_type: Type of tracing to use.
        """
        self.id = uuid.uuid4()
        self.name: str = name
        self.description = description
        self.model_name: str = model_name
        self.work_dir = work_dir

        self.tools = tools or []
        self.filter_tools_for_task = filter_tools_for_task
        self.litellm_params: dict = litellm_params or {}
        self.persona = persona
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retries = max_retries

        self.tool_names = {t.name for t in tools} if tools else set()
        self.tool_name_to_func = {t.name: t for t in tools} if tools else {}

        self.task: Task | None = None
        self.messages: list[ChatMessage] = []

        # State variables for incremental history formatting
        self._formatted_history_cache: list[dict] = []
        self._history_processed_idx: int = -1
        self._last_tool_call_id: str | None = None
        self._pending_tool_call: bool = False

        # Cache for observer history (accumulated string)
        self._observer_history_str: str = ''
        self._observer_history_idx: int = -1

        self._history_formatter: HistoryFormatter | None = None
        self._tool_descriptions_cache: dict[frozenset[str], str] = {}

        self.msg_idx_of_new_task: int = 0
        self.final_answer_found = False

        self.usage_tracker = UsageTracker()

        # Tracing
        self.tracer_manager: tracer.AbstractTracerManager = create_tracer_manager(tracing_type)
        self.current_trace: tracer.AbstractObservation | None = None

        self.planner: Planner | None = Planner(
            model_name=model_name,
            litellm_params=litellm_params,
            max_retries=self.max_retries,
            usage_tracker=self.usage_tracker,
            tracer_manager=self.tracer_manager,
        )
        self.observer = Observer(
            model_name=model_name,
            litellm_params=litellm_params,
            tool_names=self.tool_names,
            max_retries=self.max_retries,
            usage_tracker=self.usage_tracker,
            tracer_manager=self.tracer_manager,
        )

        self.is_visual_model: bool = llm_vision_support([model_name])[0] or False
        self.task_output_files: list[str] = []

    def __str__(self):
        """String representation of the agent."""
        return f'Agent: {self.name} ({self.id}); LLM: {self.model_name}; Tools: {self.tools}'

    @property
    def current_plan(self) -> str | None:
        """Returns the current plan for the task."""
        if not self.planner or not self.planner.plan:
            return None
        return self.planner.get_formatted_plan()

    @property
    def artifacts(self) -> list[str]:
        """Returns the list of output files generated during task execution."""
        return self.task_output_files

    async def get_relevant_tools(
        self,
        task_description: str,
        task_files: list[str] | None = None,
    ) -> list[Any]:
        """Calls an LLM to determine which tools are relevant for the given task."""
        tool_descriptions = self.get_tools_description()
        prompt = RELEVANT_TOOLS_PROMPT.format(
            task_description=task_description,
            task_files=task_files,
            tool_descriptions=tool_descriptions,
        )

        try:
            tools_response = await ku.call_llm(
                model_name=self.model_name,
                litellm_params=self.litellm_params,
                messages=ku.make_user_message(prompt),
                trace_id=self.task.id if self.task else None,
                usage_tracker=self.usage_tracker,
                component_name='Agent.tool_filter',
            )
            relevant_tool_names = tools_response.split(',') if tools_response.strip() else []
            relevant_tool_names = {t.strip() for t in relevant_tool_names if t.strip()}
            logger.debug('Relevant tool names: %s', relevant_tool_names)
            relevant_tools = [t for t in self.tools if t.name in relevant_tool_names]
            return relevant_tools
        except Exception as e:
            logger.error('Error determining relevant tools: %s', str(e))
            return list(self.tools)

    async def _augment_task_with_previous(self, current_task: str) -> str:
        """Augment current task with previous task context.

        Args:
            current_task: The current task description.

        Returns:
            Augmented task description with previous context.
        """
        if not self.task:
            return current_task

        context_parts = [
            f'\n## Previous Task Context\n\n**Previous Task**: {self.task.description}\n'
        ]

        # Add result if available, or salvage if not finished
        if not self.task.is_finished:
            # Task was interrupted or failed before finishing
            summary = await self.salvage_response()
            context_parts.append(f'**Summary of Progress**: {summary}\n')
        elif self.task.result:
            result_str = str(self.task.result)
            # Truncate if too long (> 2000 chars)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + '... [TRUNCATED]'
            context_parts.append(f'**Result**: {result_str}\n')

        # Add status indicators
        if self.task.is_error:
            context_parts.append('**Status**: ❌ Failed\n')
        elif self.task.is_finished:
            context_parts.append('**Status**: ✅ Completed\n')

        # Add output files if any
        if self.task.output_files:
            files_str = ', '.join(self.task.output_files)
            context_parts.append(f'**Generated Files**: {files_str}\n')

        context_parts.append('\n---\n\n## Current Task\n\n')
        context_parts.append(current_task)

        return ''.join(context_parts)

    def _run_init(
        self, description: str, files: list[str] | None = None, task_id: str | None = None
    ):
        """Initialize the running of a task by an agent.

        Args:
            description: Task description.
            files: Optional files for the task.
            task_id: Optional task ID.
        """
        self.task = Task(description=description, files=files)
        self.task_output_files = []
        if task_id:
            self.task.id = task_id

        if self.planner:
            self.planner.reset()

        if self.observer:
            self.observer.reset()

        # Reset usage tracker for new task
        self.usage_tracker.reset()

        # Initialize root trace for the task
        self.current_trace = self.tracer_manager.start_trace(
            name=f'{self.__class__.__name__}',
            input_data={
                'task': description,
                'files': files,
                'task_id': str(self.task.id),
            },
        )

        self._observer_history_str = ''
        self._observer_history_idx = -1
        self._last_tool_call_id = None
        self._pending_tool_call = False
        self.final_answer_found = False

    def add_output_file(self, path: str):
        """Record a file generated during task execution.

        Args:
            path: Absolute path to the generated file.
        """
        if path not in self.task_output_files:
            self.task_output_files.append(path)
            if self.task:
                self.task.output_files = self.task.output_files + [path]
            logger.info('Recorded output file: %s', path)

    @abstractmethod
    def parse_text_response(self, text: str) -> ChatMessage:
        """Parse a text response from the LLM into a ChatMessage."""

    async def salvage_response(self) -> str:
        """When an agent fails to find an answer, salvage what information could be gathered."""
        prompt = SALVAGE_RESPONSE_PROMPT.format(
            task=self.task.description,
            task_files='\n'.join(self.task.files) if self.task.files else '[None]',
            history=self.get_history(start_idx=self.msg_idx_of_new_task),
        )
        salvaged_response = await ku.call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=ku.make_user_message(prompt),
            trace_id=self.task.id,
            max_retries=self.max_retries,
            usage_tracker=self.usage_tracker,
            component_name='Agent.salvage',
        )
        return salvaged_response

    async def pre_run(self) -> AsyncIterator[AgentResponse]:
        """Hook intended to run before the main agent loop.
        This method acts as an asynchronous generator. Subclasses should override this to:
        - Initialize task-specific state or resources.
        - Validate inputs or preconditions.
        - Emit initial log messages or status updates (by yielding AgentResponse objects).

        If no setup is needed, the default implementation yields nothing.

        Returns:
            AsyncIterator[AgentResponse]: An iterator yielding agent responses (logs, steps, etc.).
        """
        # This makes the method an async generator that yields nothing.
        # Required because the caller iterates over it with `async for`.
        if False:  # pylint: disable=using-constant-test
            yield

    async def post_run(self) -> AsyncIterator[AgentResponse]:
        """Hook intended to run after the main agent loop.
        This method acts as an asynchronous generator and is guaranteed to run (in a finally block).
        Subclasses should override this to:
        - Clean up resources (e.g. temporary files, connections).
        - Log final execution metrics or summaries.
        - Persist agent state.

        Returns:
            AsyncIterator[AgentResponse]: An iterator yielding agent responses.
        """
        # This makes the method an async generator that yields nothing.
        # Required because the caller iterates over it with `async for`.
        if False:  # pylint: disable=using-constant-test
            yield

    @abstractmethod
    async def run(
        self,
        task: str,
        files: list[str] | None = None,
        task_id: str | None = None,
        max_iterations: int | None = None,
        recurrent_mode: bool = False,
        summarize_progress_on_failure: bool = True,
    ) -> AsyncIterator[AgentResponse]:
        """Execute a task using the agent.

        Args:
            task: The task description.
            files: List of files associated with the task.
            task_id: Optional task ID.
            max_iterations: Optional maximum number of iterations.
            recurrent_mode: Whether to run in recurrent mode.
            summarize_progress_on_failure: Whether to summarize progress on failure.

        Returns:
            AsyncIterator[AgentResponse]: An iterator yielding agent responses.
        """

    def response(
        self,
        rtype: AGENT_RESPONSE_TYPES,
        value: Any,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Prepare a response to be sent by the agent.
        Also, store the response in the task result if it is a final response.

        Args:
            rtype: Response type emitted by the agent.
            value: The current update from the agent.
            channel: The response channel.
            metadata: Any metadata associated with the update.

        Returns:
            A response from the agent.
        """
        if rtype == 'final' and self.task:
            self.task.result = value

        return {'type': rtype, 'channel': channel, 'value': value, 'metadata': metadata}

    @abstractmethod
    def formatted_history_for_llm(self) -> list[dict]:
        """Generate a formatted list of chat history messages to send to an LLM."""

    async def _chat(self, response_format: type[pyd.BaseModel] | None = None) -> str | None:
        """Interact with the LLM using the agent's message history.
        Enhanced with retry logic for structured output failures.

        Args:
            response_format: Optional structured response format for the LLM.

        Returns:
            A chat response or an empty string.

        Raises:
            RetryError: If LLM call fails after max retries.
            Exception in case of error.
        """
        formatted_messages = self.formatted_history_for_llm()

        for attempt in range(MAX_RESPONSE_PARSING_ATTEMPTS):
            try:
                chat_response: str = await ku.call_llm(
                    model_name=self.model_name,
                    litellm_params=self.litellm_params,
                    messages=formatted_messages,
                    response_format=response_format,
                    trace_id=self.task.id if self.task else None,
                    max_retries=self.max_retries,
                    usage_tracker=self.usage_tracker,
                    component_name='Agent',
                )
                return chat_response or ''

            except RetryError:
                # LLM call failed after max retries; do not retry at this level
                raise
            except Exception as e:
                logger.warning(
                    'LLM call failed (attempt %d/%d): %s',
                    attempt + 1,
                    MAX_RESPONSE_PARSING_ATTEMPTS,
                    str(e),
                )

                if attempt < MAX_RESPONSE_PARSING_ATTEMPTS - 1:
                    # Add feedback to help LLM correct itself
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    feedback = (
                        f'Error: Previous response had issues: {str(e)}.'
                        ' Please ensure your response follows the exact JSON schema provided.'
                        f' [Timestamp={datetime.now()}]'
                    )
                    formatted_messages.append({'role': 'user', 'content': feedback})
                else:
                    raise

    def add_to_history(self, message: ChatMessage):
        """Add a chat message to the agent's message history."""
        self.messages.append(message)

    def get_tools_description(self, tools: list[Any] | None = None) -> str:
        """Generate a description of all the tools available to the agent. Required args are marked.

        Args:
            tools: Optional list of tools to describe. If None, describes all available tools.

        Returns:
            A formatted string describing each tool, its parameters, and example usage.
        """
        # Create a cache key based on the tools being described
        tools_to_describe = tools if tools is not None else self.tools
        cache_key = frozenset(t.name for t in tools_to_describe)

        if cache_key in self._tool_descriptions_cache:
            return self._tool_descriptions_cache[cache_key]

        description = ''
        filtered_tool_names = {t.name for t in tools_to_describe}

        for t in self.tools:
            if t.name in filtered_tool_names:
                description += f'\n### Tool: {t.name}\n'
                description += f'**Description:** {t.description}\n'

                # Extract and highlight required parameters
                if hasattr(t, 'args_schema') and t.args_schema:
                    schema = t.args_schema.model_json_schema()
                    properties = schema.get('properties', {})
                    required = schema.get('required', [])

                    if properties:
                        description += '\n**Parameters:**\n'
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'any')
                            is_required = param_name in required
                            req_marker = '**REQUIRED**' if is_required else 'Optional'
                            description += f'  - `{param_name}` ({param_type}): {req_marker}\n'

                description += '\n---\n\n'

        self._tool_descriptions_cache[cache_key] = description
        return description

    @property
    def purpose(self) -> str:
        """Describe the name, purpose of, and tools available to an agent."""
        description = f'Name: {self.name}\nDescription: {self.description or "N/A"}'
        description += f'\nTools available to this agent (`{self.name}`):'
        description += f'\n{self.get_tools_description()}'

        return description

    def _get_observer_history(self) -> str:
        """Get the history string specifically optimized for the Observer.

        Optimization strategy:
        1. Maintain an incremental string buffer to avoid O(N) list join operations.
        2. exclude system prompt (Observer knows the task context separately).
        3. Truncate long message content to 1000 chars to save tokens.
        """
        # Start from the next index
        start_idx = self._observer_history_idx + 1

        # If there are new messages, process them
        if start_idx < len(self.messages):
            new_msgs = self.messages[start_idx:]
            new_segments = []

            for msg in new_msgs:
                # Skip system messages
                if msg.role == 'system':
                    continue

                content = str(msg)

                # Truncate any message content exceeding 1000 chars
                if len(content) > 1000:
                    content = content[:1000] + '... [TRUNCATED]'

                new_segments.append(f'[{msg.role}]: {content}')

            if new_segments:
                new_block = '\n'.join(new_segments)
                if self._observer_history_str:
                    self._observer_history_str += '\n' + new_block
                else:
                    self._observer_history_str = new_block

            self._observer_history_idx = len(self.messages) - 1

        return self._observer_history_str

    def get_history(self, start_idx: int = 0, truncate_text: bool = False) -> str:
        """Get a formatted string representation of all the messages.

        Args:
            start_idx: Index to start getting history from.
            truncate_text: If True, truncate message content to 100 chars.
        """
        messages = []
        for msg in self.messages[start_idx:]:
            content = str(msg)
            if truncate_text and len(content) > 100:
                content = content[:100] + '...'
            messages.append(f'[{msg.role}]: {content}')
        return '\n'.join(messages)

    def clear_history(self):
        """Clear the agent's message history."""
        self.messages = []
        self._formatted_history_cache = []
        self._history_processed_idx = -1
        self._last_tool_call_id = None
        self._pending_tool_call = False

    def init_history(self):
        """Initialize the agent's message history, e.g., with a system prompt."""
        self.clear_history()

    def get_usage_report(self, include_breakdown: bool = True) -> str:
        """Get a formatted report of LLM usage for the current/last task.

        Args:
            include_breakdown: Whether to include per-component breakdown.

        Returns:
            Formatted string with usage statistics.
        """
        return self.usage_tracker.format_report(include_breakdown=include_breakdown)

    def get_usage_metrics(self) -> dict:
        """Get raw usage metrics as a dictionary.

        Returns:
            Dictionary with total and per-component usage data.
        """
        total = self.usage_tracker.get_total_usage()
        by_component = self.usage_tracker.get_usage_by_component()

        return {
            'total': total.model_dump(),
            'by_component': {k: v.model_dump() for k, v in by_component.items()},
        }


class ReActAgent(Agent):
    """Reasoning and Acting agent with thought-action-observation loop."""

    response_format_class: ClassVar[type[pyd.BaseModel]] = ReActChatMessage

    def __init__(
        self,
        name: str,
        model_name: str,
        tools: list,
        description: str | None = None,
        litellm_params: dict | None = None,
        persona: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
        filter_tools_for_task: bool = False,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        work_dir: str | None = None,
        tracing_type: TRACING_TYPES | None = None,
    ):
        """Create a ReAct agent."""
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            litellm_params=litellm_params,
            description=description,
            persona=persona,
            system_prompt=system_prompt or REACT_SYSTEM_PROMPT,
            max_iterations=max_iterations,
            filter_tools_for_task=filter_tools_for_task,
            max_retries=max_retries,
            work_dir=work_dir,
            tracing_type=tracing_type,
        )

        if tools:
            logger.info('Created agent: %s; tools: %s', name, [t.name for t in tools])

        self._history_formatter = ReActHistoryFormatter()

    async def _update_plan(self):
        """Update the plan based on the last thought and observation."""
        last_thought = None
        last_observation = None

        # Traverse backwards to find the most recent thought and observation
        for msg in reversed(self.messages):
            # Look for observation (Tool response)
            if last_observation is None and msg.role == 'tool':
                last_observation = msg.content

            # Look for thought (Assistant response with a 'thought' field)
            # We use distinct checks for role and attribute to be robust to custom message types
            if last_thought is None and msg.role == 'assistant':
                # Use getattr for loose coupling - accepts any message object with a 'thought'
                thought = getattr(msg, 'thought', None)
                if thought:
                    last_thought = thought

                # If the message contains a final answer, treat it as the observation
                # This ensures the Planner sees the final result, allowing it to mark "Output..."
                # steps as done
                final_answer = getattr(msg, 'final_answer', None)
                if final_answer:
                    last_observation = f'Final Answer: {final_answer}'

            if last_thought and last_observation:
                break

        # Only update plan if both thought and observation are found
        if not (last_thought and last_observation):
            logger.debug(
                'Skipping plan update: missing thought=%s or observation=%s',
                bool(last_thought),
                bool(last_observation),
            )
            return

        await self.planner.update_plan(
            thought=last_thought,
            observation=last_observation,
            task_id=self.task.id,
            parent_trace=self.current_trace,
        )

    def init_history(self):
        """Initialize the agent's message history with the system prompt."""
        self.messages = [
            ChatMessage(
                role='system',
                content=self.system_prompt.format(
                    persona=self.persona or '', tools=self.get_tools_description()
                ),
            )
        ]
        self._formatted_history_cache = []
        self._history_processed_idx = -1
        self._last_tool_call_id = None
        self._pending_tool_call = False

    async def _create_initial_plan(self):
        """Helper method to create the initial plan.

        Raises:
            RuntimeError: If plan creation fails.
        """
        try:
            await self.planner.create_plan(
                self.task, self.__class__.__name__, parent_trace=self.current_trace
            )
        except RetryError as err:
            logger.warning('Max retries reached during plan creation.')
            error_msg = (
                'Unable to start solving the task: Rate limit exceeded while '
                'creating the initial plan. Please try again later.'
            )
            self.add_to_history(ChatMessage(role='assistant', content=error_msg))
            raise RuntimeError(error_msg) from err

    async def pre_run(self) -> AsyncIterator[AgentResponse]:
        """Perform setup before the main run loop.
        - Initialize task and history.
        - Create initial plan.

        Returns:
            AsyncIterator[AgentResponse]: Iterator of agent responses.
        """
        self.init_history()

        yield self.response(
            rtype='log', value=f'Solving task: `{self.task.description}`', channel='run'
        )

        try:
            await self._create_initial_plan()
        except RuntimeError as e:
            yield self.response(
                rtype='final',
                value=str(e),
                channel='run',
                metadata={'final_answer_found': False, 'is_error': True},
            )
            return

        yield self.response(
            rtype='log', value=f'Plan:\n{self.planner.get_formatted_plan()}', channel='run'
        )

        self.add_to_history(
            ChatMessage(
                role='user', content=f'New Task:\n{self.task.description}', files=self.task.files
            )
        )
        self.add_to_history(
            ChatMessage(role='user', content=f'Plan:\n{self.planner.get_formatted_plan()}')
        )

    async def post_run(self) -> AsyncIterator[AgentResponse]:
        """Perform cleanup after the main run loop.
        - Calculate and log usage metrics.
        - End current trace/span.
        - Flush tracer.

        Returns:
            AsyncIterator[AgentResponse]: Iterator of agent responses.
        """
        if self.task:
            usage_data = self.get_usage_metrics()
            self.task.total_llm_calls = usage_data['total']['call_count']
            self.task.total_prompt_tokens = usage_data['total']['total_prompt_tokens']
            self.task.total_completion_tokens = usage_data['total']['total_completion_tokens']
            self.task.total_tokens = usage_data['total']['total_tokens']
            self.task.total_cost = usage_data['total']['total_cost']
            self.task.usage_by_component = usage_data['by_component']

            # Log usage summary
            logger.info(
                'Task %s usage: %s', self.task.id, self.get_usage_report(include_breakdown=False)
            )

            # End the root trace
            if self.current_trace:
                is_error = getattr(self.task, 'is_error', False)
                self.current_trace.end(
                    result=self.task.result,
                    metadata={
                        'is_error': is_error,
                        'total_tokens': self.task.total_tokens,
                        'total_cost': float(self.task.total_cost),
                        'steps_taken': getattr(self.task, 'steps_taken', 0),
                    },
                )

        # Flush tracer manager
        if self.tracer_manager:
            self.tracer_manager.flush()

        yield self.response(rtype='log', value='Task execution finished', channel='run')

    async def run(
        self,
        task: str,
        files: list[str] | None = None,
        task_id: str | None = None,
        max_iterations: int | None = None,
        recurrent_mode: bool = False,
        summarize_progress_on_failure: bool = True,
    ) -> AsyncIterator[AgentResponse]:
        """Solve a task using ReAct's TAO loop (or CodeAct's TCO loop).

        Args:
            task: A task to be solved by the agent.
            files: An optional list of files related to the task.
            task_id: Optional task ID.
            max_iterations: Optional max iterations for the task.
            recurrent_mode: If True, augment task with previous task context.
            summarize_progress_on_failure: Whether to summarize progress if
             the agent fails to solve the task in max iterations.

        Returns:
            Step updates on the task and the final response.

        Raises:
            ValueError: If task is empty or too many files provided.
            RetryError: If LLM calls fail after max retries.
        """
        if not task or not task.strip():
            raise ValueError('Task description cannot be empty!')
        if files and not isinstance(files, list):
            raise ValueError('Task files must be a list of file paths!')
        if files and len(files) > MAX_TASK_FILES:
            raise ValueError(f'Too many files provided for the task (max {MAX_TASK_FILES})!')

        # Augment task with previous context if recurrent mode enabled
        if recurrent_mode and self.task is not None:
            task = await self._augment_task_with_previous(task)
            logger.debug('Recurrent mode enabled: augmented task with previous context')

        # 1. run() calls _run_init()
        # 2. pre_run() calls init_history() and does the logging/planning
        self._run_init(task, files, task_id)

        # Execute pre-run hook
        async for response in self.pre_run():
            yield response
            # Check if pre_run encountered specific error signal
            if response['type'] == 'final' and response['metadata'].get('is_error'):
                return

        max_iterations = max_iterations or self.max_iterations
        steps_taken = 0
        try:
            # Main Loop
            for idx in range(max_iterations):
                steps_taken = idx + 1
                logger.debug('ITERATION %d/%d', steps_taken, max_iterations)
                yield self.response(
                    rtype='log', channel='run', value=f'* Executing step {steps_taken}'
                )

                try:
                    async for update in self._think():
                        yield update
                    async for update in self._act():
                        yield update
                except asyncio.CancelledError:
                    logger.info('Task cancelled by consumer')
                    raise
                except RetryError as e:
                    logger.warning('Max retries reached for LLM call: %s', e)
                    error_msg = 'Rate limit exceeded. Unable to proceed.'
                    self.add_to_history(ChatMessage(role='assistant', content=error_msg))
                    yield self.response(
                        rtype='final',
                        value=error_msg,
                        channel='run',
                        metadata={'final_answer_found': False, 'is_error': True},
                    )
                    return

                if self.final_answer_found:
                    break

                plan_before_update = None
                if self.planner.plan:
                    plan_before_update = self.current_plan
                    try:
                        await self._update_plan()
                    except RetryError:
                        logger.warning('Max retries reached during plan update.')
                        error_msg = 'Rate limit exceeded during plan update. Unable to proceed.'
                        self.add_to_history(ChatMessage(role='assistant', content=error_msg))
                        yield self.response(
                            rtype='final',
                            value=error_msg,
                            channel='run',
                            metadata={'final_answer_found': False, 'is_error': True},
                        )
                        return

                    self.add_to_history(
                        ChatMessage(
                            role='user',
                            content=(f'Plan progress:\n{self.planner.get_formatted_plan()}'),
                        )
                    )

                try:
                    correction_msg = await self.observer.observe(
                        task=self.task,
                        history=self._get_observer_history(),
                        plan_before=plan_before_update,
                        plan_after=self.current_plan,
                        iteration=idx + 1,
                        parent_trace=self.current_trace,
                    )
                except RetryError:
                    logger.warning('Observer failed due to rate limit. Skipping observation.')
                    correction_msg = None

                if correction_msg:
                    self.add_to_history(
                        ChatMessage(role='user', content=f'Observation: {correction_msg}')
                    )
                    yield self.response(rtype='log', value=correction_msg, channel='observer')

            # Loop iteration over
            if not self.final_answer_found:
                failure_msg = (
                    f'Sorry, I failed to get a complete answer even after {steps_taken} steps!'
                )

                if summarize_progress_on_failure:
                    try:
                        progress_summary = await self.salvage_response()
                        failure_msg += (
                            f"\n\nHere's a summary of progress for the task:\n{progress_summary}"
                        )
                    except RetryError:
                        logger.warning(
                            'Failed to salvage response due to rate limit.'
                            ' Skipping progress summary.'
                        )

                yield self.response(
                    rtype='final',
                    value=failure_msg,
                    channel='run',
                    metadata={'final_answer_found': False},
                )

                self.add_to_history(ChatMessage(role='assistant', content=failure_msg))
            else:
                if self.planner.plan:
                    try:
                        await self._update_plan()
                    except RetryError:
                        logger.warning(
                            'Failed to update plan on final iteration due to '
                            'rate limit. Skipping final plan update.'
                        )

        except asyncio.CancelledError:
            logger.warning('Iteration cancelled')
        finally:
            if self.task:
                self.task.steps_taken = steps_taken

            # Execute post-run hook
            async for response in self.post_run():
                yield response

    async def _think(self) -> AsyncIterator[AgentResponse]:
        """Think about the next step using the new structured response format.

        Returns:
            AsyncIterator[AgentResponse]: An async iterator of AgentResponse
                objects.
        """
        # Create a generation span for the LLM call
        gen_span = self.tracer_manager.start_generation(
            parent=self.current_trace,
            name='think',
            input_data={
                'model': self.model_name,
                'messages_count': len(self.messages),
            },
        )

        thought = await self._record_thought(ReActChatMessage)

        if thought:
            gen_span.update(
                status='success',
                has_thought=bool(thought.thought),
                has_action=bool(getattr(thought, 'action', None)),
                has_final_answer=bool(getattr(thought, 'final_answer', None)),
            )
            gen_span.end(
                output={
                    'thought': thought.thought,
                    'action': getattr(thought, 'action', None),
                },
            )
        else:
            gen_span.update(status='error', error='Failed to parse response')
            gen_span.end(output='parse_failure', is_error=True)

        yield self.response(rtype='step', value=thought, channel='_think')

    async def _record_thought(
        self, response_format_class: type[pyd.BaseModel]
    ) -> ReActChatMessage | CodeActChatMessage | None:
        """Record the agent's thought with improved error handling and fallback parsing.
        Now returns ChatMessage directly instead of separate Response class.

        Args:
            response_format_class (type[pyd.BaseModel]): The response format class.

        Returns:
            Optional[Union[ReActChatMessage, CodeActChatMessage]]: The agent's thought.
        """
        for attempt in range(3):
            try:
                thought_response: str = await self._chat(response_format=response_format_class)

                try:
                    thought_response_cleaned = ku.clean_json_string(thought_response)

                    try:
                        parsed_json = json.loads(thought_response_cleaned)
                    except JSONDecodeError as e:
                        logger.warning(
                            'Initial JSON parse failed: %s. Attempting repair...', str(e)
                        )
                        thought_response_cleaned = json_repair.repair_json(thought_response_cleaned)
                        parsed_json = json.loads(thought_response_cleaned)

                    if 'args' in parsed_json:
                        if isinstance(parsed_json['args'], str):
                            parsed_json['args'] = ku.clean_json_string(parsed_json['args'])
                        elif isinstance(parsed_json['args'], dict):
                            # Ensure deeply nested args are converted to string for Pydantic
                            parsed_json['args'] = json.dumps(parsed_json['args'])

                    # Handle mutual exclusivity violations
                    if response_format_class == ReActChatMessage:
                        has_action = parsed_json.get('action') and parsed_json['action'] != 'FINISH'
                        has_final_answer = parsed_json.get('final_answer')

                        if has_action and has_final_answer:
                            logger.warning(
                                "LLM provided both action ('%s') and final_answer."
                                ' Keeping action, removing final_answer.',
                                parsed_json['action'],
                            )
                            parsed_json['final_answer'] = None
                            parsed_json['task_successful'] = False

                    elif response_format_class == CodeActChatMessage:
                        has_code = parsed_json.get('code')
                        has_final_answer = parsed_json.get('final_answer')

                        if has_code and has_final_answer:
                            logger.warning(
                                'LLM provided both code and final_answer. '
                                'Keeping code, removing final_answer.'
                            )
                            parsed_json['final_answer'] = None
                            parsed_json['task_successful'] = False

                    # Validate and create message directly
                    # CRITICAL: Always force role to 'assistant' for model responses.
                    # Some models (like Gemini 2.5) may hallucinate "role": "user" in
                    # structured JSON, which breaks subsequent tool response pairings.
                    parsed_json['role'] = 'assistant'
                    msg = response_format_class.model_validate(parsed_json)

                    logger.debug('Successfully parsed structured JSON response')

                except (JSONDecodeError, pyd.ValidationError) as parse_error:
                    logger.warning(
                        'Structured parsing failed: %s: %s. Falling back to text parsing...',
                        type(parse_error).__name__,
                        parse_error,
                    )

                    try:
                        msg = self.parse_text_response(thought_response)
                        msg.role = 'assistant'
                        logger.info('Successfully parsed response using text fallback')

                    except Exception as text_error:
                        logger.error('Text parsing also failed: %s', str(text_error))
                        raise ValueError(
                            f'Both structured and text parsing failed. '
                            f'Structured error: {parse_error}. '
                            f'Text parsing error: {text_error}'
                        ) from text_error

                self.add_to_history(msg)
                return msg

            except RetryError:
                raise

            except ValueError as ex:
                logger.error(
                    'Parsing error in _record_thought (attempt %d/3): %s', attempt + 1, str(ex)
                )

                if attempt < 2:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    feedback_message = (
                        f'!Parsing Error: {str(ex)}. '
                        'Please ensure your response follows the required format. '
                        f'[Timestamp={datetime.now()}]'
                    )
                    self.add_to_history(ChatMessage(role='user', content=feedback_message))
                else:
                    logger.error('Failed to parse response after 3 attempts')
                    return None

            except Exception as ex:
                logger.exception(
                    'Unexpected error in _record_thought (attempt %d/3): %s', attempt + 1, str(ex)
                )

                if attempt < 2:
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    feedback_message = (
                        f'!Error: {type(ex).__name__}: {str(ex)}. [Timestamp={datetime.now()}]'
                    )
                    self.add_to_history(ChatMessage(role='user', content=feedback_message))
                else:
                    logger.error('Failed after 3 attempts due to unexpected errors')
                    return None

        return None

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """Take action based on the agent's previous thought.

        Now handles explicit FINISH action with proper error handling and
        hierarchical tracing of the act operation, tool execution, and errors.
        """
        prev_msg: ReActChatMessage = self.messages[-1]  # type: ignore

        # Start root span for the entire act operation
        act_span = self.tracer_manager.start_span(
            parent=self.current_trace,
            name='act',
            input_data={'thought': getattr(prev_msg, 'thought', None)},
        )

        # Check for malformed response
        if not hasattr(prev_msg, 'thought') or not prev_msg.thought:
            act_span.update(status='error', error='Missing or empty thought field')
            self.add_to_history(
                ChatMessage(
                    role='user',
                    content=(
                        '* Error: Response must have a valid `thought` field. '
                        'Please respond strictly following the schema.'
                    ),
                )
            )
            act_span.end(output='malformed_response')
            return

        # Check if this is a final answer
        if hasattr(prev_msg, 'final_answer') and prev_msg.final_answer:
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = not prev_msg.task_successful

            act_span.update(
                status='success',
                operation='final_answer',
                task_successful=prev_msg.task_successful,
            )
            act_span.end(
                output=prev_msg.final_answer,
                metadata={'task_successful': prev_msg.task_successful},
            )

            yield self.response(
                rtype='final',
                value=prev_msg.final_answer,
                channel='_act',
                metadata={'final_answer_found': prev_msg.task_successful},
            )
        else:
            # Tool execution
            tool_name = prev_msg.action
            tool_args = prev_msg.args
            tool_args_dict = {}

            # Validate tool call has required fields
            if not tool_name or not tool_args:
                error_msg = 'Error: Both action and args must be provided for tool calls.'
                act_span.update(
                    status='error',
                    operation='tool_validation_failed',
                    error=error_msg,
                )
                # CRITICAL: Use role='tool' instead of role='user' to maintain
                # conversation format
                self.add_to_history(ChatMessage(role='tool', content=error_msg))
                act_span.end(output='validation_error', is_error=True)
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True},
                )
                return

            try:
                # CRITICAL: Parse the JSON string into a dictionary
                # The args field is a JSON string, not a dict
                tool_args = tool_args.strip().strip('`').strip()
                if tool_args.startswith('json'):
                    tool_args = tool_args[4:].strip()

                try:
                    tool_args_dict = json.loads(tool_args)
                except JSONDecodeError:
                    logger.warning('JSON decode failed, attempting repair...')
                    tool_args_dict = json_repair.loads(tool_args)

                # Validate it's actually a dictionary
                if not isinstance(tool_args_dict, dict):
                    error_msg = f'Tool args must be a dict, got {type(tool_args_dict).__name__}'
                    act_span.update(
                        status='error',
                        operation='args_validation_failed',
                        error=error_msg,
                    )
                    self.add_to_history(ChatMessage(role='tool', content=error_msg))
                    act_span.end(output='args_error', is_error=True)
                    yield self.response(
                        rtype='step',
                        value=error_msg,
                        channel='_act',
                        metadata={'is_error': True},
                    )
                    return

                # Execute tool
                if tool_name in self.tool_names:
                    logger.debug(
                        '🛠 Running tool: %s with args: %s',
                        tool_name,
                        tool_args_dict,
                    )

                    # Create nested span for tool execution
                    tool_span = self.tracer_manager.start_span(
                        parent=act_span,
                        name=tool_name,
                        input_data=tool_args_dict,
                    )

                    # Intercept file creation during tool execution
                    with OutputInterceptor() as interceptor:
                        result = self.tool_name_to_func[tool_name](**tool_args_dict)
                        # Record any files captured by the interceptor
                        generated_files = interceptor.get_manifest()
                        for f in generated_files:
                            self.add_output_file(f)

                    tool_span.update(
                        status='success',
                        file_count=len(generated_files),
                    )
                    tool_span.end(
                        output=str(result),
                        generated_files=generated_files,
                    )

                    # Always use role='tool' for tool results
                    self.add_to_history(ChatMessage(role='tool', content=str(result)))

                    act_span.update(
                        status='success',
                        operation='tool_execution',
                        tool=tool_name,
                    )
                    act_span.end(
                        output=str(result),
                        metadata={
                            'tool': tool_name,
                            'args': tool_args_dict,
                            'generated_files': generated_files,
                        },
                    )

                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={
                            'tool': tool_name,
                            'args': tool_args_dict,
                            'generated_files': generated_files,
                        },
                    )
                else:
                    result = (
                        f'Error: Tool "{tool_name}" not found! '
                        f'Available tools: {", ".join(sorted(self.tool_names))}. '
                        'Please use an exact tool name from the list.'
                    )
                    act_span.update(
                        status='error',
                        operation='tool_not_found',
                        tool=tool_name,
                        error=result,
                    )
                    # Use role='tool' for tool errors too
                    self.add_to_history(ChatMessage(role='tool', content=result))
                    act_span.end(output='tool_not_found', is_error=True)
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'is_error': True},
                    )

            except Exception as ex:
                error_msg = (
                    f'*** Error: Tool execution failed: {type(ex).__name__}: '
                    f'{str(ex)}\n'
                    f'Tool: {tool_name}\n'
                    f'Args provided: {tool_args_dict}\n'
                    f'Please check the tool signature and try again with '
                    f'correct arguments.'
                )
                logger.error(error_msg)
                act_span.update(
                    status='error',
                    operation='tool_execution_exception',
                    tool=tool_name,
                    error_type=type(ex).__name__,
                    error_message=str(ex),
                )
                # Use role='tool' to maintain proper conversation structure
                self.add_to_history(ChatMessage(role='tool', content=error_msg))
                act_span.end(
                    output='exception',
                    is_error=True,
                    error=error_msg,
                )
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True},
                )

    def parse_text_response(self, text: str) -> ReActChatMessage:
        """Parse text-based response when structured output fails.
        Uses regex to extract components from free-form text.

        This is a FALLBACK for when JSON parsing completely fails.
        It only handles text format like:
            Thought: ...
            Action: ...
            Args: {...}
        OR
            Thought: ...
            Answer: ...
            Successful: true/false

        Args:
            text: The raw text response from the LLM.

        Returns:
            Parsed structured response.

        Raises:
            ValueError: If unable to parse action or final_answer fields.
        """
        logger.info('Falling back to text-based regex parsing')

        # Extract thought - REQUIRED
        thought_match = THOUGHT_MATCH.search(text)
        if not thought_match:
            raise ValueError(
                f"Could not extract 'Thought:' field from response. "
                f'Response must start with reasoning. Text: {text[:200]}...'
            )
        thought = thought_match.group(1).strip()

        # Parse ReAct response
        action_match = ACTION_MATCH.search(text)
        action = action_match.group(1).strip() if action_match else None
        args = None

        # Try to extract and validate args
        args_match = ARGS_MATCH.search(text)
        if args_match:
            args_str = args_match.group(1).strip()
            args = ku.clean_json_string(args_str)
            # Validate it's actually valid JSON
            try:
                json.loads(args)
            except (JSONDecodeError, Exception) as e:
                logger.warning('Args extraction failed, invalid JSON: %s', str(e))
                args = None

        # Extract final answer and success status
        answer_match = ANSWER_MATCH.search(text)
        success_match = SUCCESS_MATCH.search(text)
        final_answer = answer_match.group(1).strip() if answer_match else None
        task_successful = success_match.group(1).lower() == 'true' if success_match else False

        # Validation: Must have either (action + args) OR final_answer
        if final_answer:
            # This is a final answer
            return ReActChatMessage(
                role='assistant',
                thought=thought,
                action='FINISH',
                args=None,
                final_answer=final_answer,
                task_successful=task_successful,
            )

        if action:
            # This is a tool call
            if action == 'FINISH':
                raise ValueError(
                    f"Action is 'FINISH' but no Answer field found. Text: {text[:200]}..."
                )
            if not args:
                raise ValueError(
                    f"Action '{action}' specified but no valid Args found. Text: {text[:200]}..."
                )
            return ReActChatMessage(
                role='assistant',
                thought=thought,
                action=action,
                args=args,
                final_answer=None,
                task_successful=False,
            )

        raise ValueError(
            f'Could not extract valid Action or Answer from response. Text: {text[:300]}...'
        )

    def formatted_history_for_llm(self) -> list[dict]:
        """Format message history for LLM with proper tool call structure.
        Uses incremental caching to avoid O(n^2) behavior in the course of solving a task.

        Returns:
            list[dict]: Formatted message history for LLM.
        """
        # Start from the next index
        start_idx = self._history_processed_idx + 1

        # If there are no new messages and we have a cache, just return it (with safety check)
        if start_idx >= len(self.messages) and self._formatted_history_cache:
            # handle pending tool call placeholder logic at end
            pass
        else:
            # Process new messages
            new_msgs = self.messages[start_idx:]

            # Temporary list for new formatted segments
            # We don't append directly to cache to handle user message combination logic safely
            new_formatted_segments = []

            # State for history formatter
            formatter_state = {
                'last_tool_call_id': self._last_tool_call_id,
                'pending_tool_call': self._pending_tool_call,
            }

            for msg in new_msgs:
                d = msg.model_dump()
                role = d.get('role')

                # Handle assistant tool call using strategy pattern if formatter is available
                if self._history_formatter and self._history_formatter.should_format_as_tool_call(
                    msg
                ):
                    formatted_msg = self._history_formatter.format_tool_call(msg, formatter_state)
                    new_formatted_segments.append(formatted_msg)

                elif role == 'tool':
                    tool_msg = {'role': 'tool', 'content': str(d.get('content', ''))}
                    if formatter_state['last_tool_call_id']:
                        tool_msg['tool_call_id'] = formatter_state['last_tool_call_id']
                        # NOTE: We keep last_tool_call_id until we are sure?
                        # In original code (direct from list): last_tool_call_id = None after usage
                        formatter_state['last_tool_call_id'] = None
                        formatter_state['pending_tool_call'] = False
                    new_formatted_segments.append(tool_msg)

                elif role == 'assistant':
                    if hasattr(msg, 'final_answer') and getattr(msg, 'final_answer', None):
                        new_formatted_segments.append(
                            {'role': 'assistant', 'content': getattr(msg, 'final_answer')}
                        )
                    else:
                        new_formatted_segments.append(
                            {'role': 'assistant', 'content': d.get('content', '')}
                        )

                elif role == 'user':
                    usr_msgs = ku.make_user_message(
                        text_content=d.get('content', ''), files=d.get('files', None)
                    )
                    new_formatted_segments.extend(usr_msgs)

                else:
                    # system
                    new_formatted_segments.append({'role': role, 'content': d.get('content', '')})

            # Sync state back to agent
            self._last_tool_call_id = formatter_state['last_tool_call_id']
            self._pending_tool_call = formatter_state['pending_tool_call']

            # Incremental merge with cache
            for seg in new_formatted_segments:
                if (
                    self._formatted_history_cache
                    and self._formatted_history_cache[-1].get('role') == 'user'
                    and seg.get('role') == 'user'
                ):
                    # Merge content
                    prev = self._formatted_history_cache[-1]
                    prev_content = prev['content']
                    curr_content = seg['content']

                    if not isinstance(prev_content, list):
                        prev_content = [prev_content]
                    if not isinstance(curr_content, list):
                        curr_content = [curr_content]

                    # Update in place
                    prev['content'] = prev_content + curr_content
                else:
                    self._formatted_history_cache.append(seg)

            # Update index
            self._history_processed_idx = len(self.messages) - 1

        # Prepare final result (copy to avoid mutation issues if caller modifies)
        result = list(self._formatted_history_cache)

        # Safety check: if we have a pending tool call without response, add a placeholder
        # This is NOT cached, as it's a transient state
        # Delegate check to formatter
        formatter_state = {
            'last_tool_call_id': self._last_tool_call_id,
            'pending_tool_call': self._pending_tool_call,
        }
        if self._history_formatter and self._history_formatter.should_add_pending_placeholder(
            formatter_state
        ):
            logger.warning(
                'Found tool_call without corresponding tool response, adding placeholder'
            )
            result.append(
                {
                    'role': 'tool',
                    'tool_call_id': self._last_tool_call_id,
                    'content': 'Error: Tool execution was interrupted',
                }
            )

        return result


class CodeActAgent(ReActAgent):
    """CodeAct agent using Thought-Code-Observation loop."""

    response_format_class: ClassVar[type[pyd.BaseModel]] = CodeActChatMessage

    def __init__(
        self,
        name: str,
        model_name: str,
        run_env: CODE_ENV_NAMES,
        tools: list[Callable] | None = None,
        description: str | None = None,
        litellm_params: dict | None = None,
        persona: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
        allowed_imports: list[str] | None = None,
        pip_packages: str | None = None,
        timeout: int = 30,
        env_vars_to_set: dict[str, str] | None = None,
        filter_tools_for_task: bool = False,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        work_dir: str | None = None,
        tracing_type: TRACING_TYPES | None = None,
    ):
        """Initialize CodeAct agent."""
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            litellm_params=litellm_params,
            max_iterations=max_iterations,
            persona=persona,
            system_prompt=system_prompt or CODE_ACT_SYSTEM_PROMPT,
            description=description,
            filter_tools_for_task=filter_tools_for_task,
            max_retries=max_retries,
            work_dir=work_dir,
            tracing_type=tracing_type,
        )
        self._history_formatter = CodeActHistoryFormatter()
        self.tools_source_code: str = 'from typing import *\n\n'

        if tools:
            for t in self.tools:
                self.tools_source_code += inspect.getsource(t).replace('@tool\n', '', 1) + '\n'

        self.pip_packages = pip_packages

        if not allowed_imports:
            allowed_imports = []

        self.allowed_imports = allowed_imports + ['datetime', 'typing', 'mimetypes']
        self.code_runner = CodeRunner(
            env=run_env,
            allowed_imports=self.allowed_imports,
            model_name=model_name,
            pip_packages=pip_packages,
            timeout=timeout,
            env_vars_to_set=env_vars_to_set,
            litellm_params=litellm_params,
            work_dir=self.work_dir,
            usage_tracker=self.usage_tracker,
            tool_names=self.tool_names,
        )

    def parse_text_response(self, text: str) -> CodeActChatMessage:
        """Uses regex to extract components from free-form text and parse into messages.

        Returns:
            Parsed structured response.

        Raises:
            ValueError: If unable to parse either code or final_answer fields.
        """
        logger.info('Falling back to text-based regex parsing for CodeAct')

        # NOTE: We intentionally do NOT call super().parse_text_response(text) here.
        # The parent ReActAgent.parse_text_response() enforces the presence of either
        # "Action" (for tools) or "Answer" (final response).
        #
        # A valid CodeAct response might contain "Code" but NO "Action" or "Answer".
        # If we called super(), it would recognize the "Thought" but fail to find
        # "Action"/"Answer" and raise a ValueError before we could check for "Code".
        # Therefore, we independently parse Thought + Code + Answer here.

        # Extract thought - REQUIRED
        thought_match = THOUGHT_MATCH.search(text)
        if not thought_match:
            raise ValueError(
                f"Could not extract 'Thought:' field from response. "
                f'Response must start with reasoning. Text: {text[:200]}...'
            )
        thought = thought_match.group(1).strip()

        # Parse CodeAct response
        code_match = CODE_MATCH.search(text)
        code = code_match.group(1).strip() if code_match else None

        # Also try to find code without markdown blocks
        if not code:
            code_alt_match = re.search(
                r'Code:\s*(.+?)(?=\n(?:Thought|Answer):|$)', text, re.DOTALL | re.IGNORECASE
            )
            if code_alt_match:
                code_raw = code_alt_match.group(1).strip()
                code = code_raw.strip('`').strip()

        # Extract final answer and success status
        answer_match = ANSWER_MATCH.search(text)
        success_match = SUCCESS_MATCH.search(text)
        final_answer = answer_match.group(1).strip() if answer_match else None
        task_successful = success_match.group(1).lower() == 'true' if success_match else False

        # Validation: Must have either code OR final_answer
        if final_answer:
            return CodeActChatMessage(
                role='assistant',
                thought=thought,
                code=None,
                final_answer=final_answer,
                task_successful=task_successful,
            )

        if code:
            return CodeActChatMessage(
                role='assistant',
                thought=thought,
                code=code,
                final_answer=None,
                task_successful=False,
            )

        raise ValueError(
            f'Could not extract valid Code or Answer from response. Text: {text[:300]}...'
        )

    def init_history(self):
        """Initialize message history with system prompt for CodeAct agent."""
        self.messages = [
            ChatMessage(
                role='system',
                content=self.system_prompt.format(
                    persona=self.persona or '',
                    tools=self.get_tools_description(),
                    authorized_imports='\n'.join([f'- {imp}' for imp in self.allowed_imports]),
                ),
            )
        ]
        self._formatted_history_cache = []
        self._history_processed_idx = -1
        self._last_tool_call_id = None
        self._pending_tool_call = False

    async def _think(self) -> AsyncIterator[AgentResponse]:
        """Think step for CodeAct agent.

        Creates a generation span for LLM code generation, tracking model and
        code output with full hierarchical tracing.
        """
        # Create generation span for code generation
        gen_span = self.tracer_manager.start_generation(
            parent=self.current_trace,
            name='think_code',
            input_data={
                'model': self.model_name,
                'messages_count': len(self.messages),
            },
        )

        msg = await self._record_thought(CodeActChatMessage)

        if msg:
            gen_span.update(
                status='success',
                has_thought=bool(msg.thought),
                has_code=bool(getattr(msg, 'code', None)),
                has_final_answer=bool(getattr(msg, 'final_answer', None)),
            )
            gen_span.end(
                output={
                    'thought': msg.thought,
                    'code': getattr(msg, 'code', None)[:100]
                    if getattr(msg, 'code', None)
                    else None,
                },
            )
        else:
            gen_span.update(status='error', error='Failed to parse response')
            gen_span.end(output='parse_failure', is_error=True)

        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """Execute code based on CodeActAgent's previous thought.

        Creates hierarchical spans for code execution with full tracing of
        stdout, stderr, exit status, and generated files.
        """
        prev_msg: CodeActChatMessage = self.messages[-1]  # type: ignore

        # Start root span for the entire act operation
        act_span = self.tracer_manager.start_span(
            parent=self.current_trace,
            name='act',
            input_data={'thought': getattr(prev_msg, 'thought', None)},
        )

        if not hasattr(prev_msg, 'thought') or not prev_msg.thought:
            act_span.update(status='error', error='Missing or empty thought field')
            self.add_to_history(
                ChatMessage(
                    role='user',
                    content=(
                        '* Error: Response must have a valid `thought` field. '
                        'Please respond strictly following the schema.'
                    ),
                )
            )
            act_span.end(output='malformed_response')
            return

        if hasattr(prev_msg, 'final_answer') and prev_msg.final_answer:
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = not prev_msg.task_successful

            act_span.update(
                status='success',
                operation='final_answer',
                task_successful=prev_msg.task_successful,
            )
            act_span.end(
                output=prev_msg.final_answer,
                metadata={'task_successful': prev_msg.task_successful},
            )

            yield self.response(
                rtype='final',
                value=prev_msg.final_answer,
                channel='_act',
                metadata={'final_answer_found': prev_msg.task_successful},
            )
        else:
            try:
                code = prev_msg.code.strip()
                code = code.replace('```py', '')
                code = code.replace('```python', '')
                code = code.replace('```', '').strip()

                logger.debug('🛠 Running code [truncated]: ... %s', code[-100:])

                # Create nested span for code execution
                code_span = self.tracer_manager.start_span(
                    parent=act_span,
                    name='code_execution',
                    input_data={'code_length': len(code)},
                )

                stdout, stderr, exit_status, generated_files = await self.code_runner.run(
                    self.tools_source_code, code, self.task.id
                )

                # Download files from remote environment if necessary
                if generated_files:
                    files = await self.code_runner.download_files_from_remote(generated_files)
                    for f in files:
                        self.add_output_file(f)

                code_span.update(
                    status='success' if exit_status == 0 else 'error',
                    exit_status=exit_status,
                    has_stdout=bool(stdout),
                    has_stderr=bool(stderr),
                    file_count=len(generated_files),
                )
                code_span.end(
                    output={
                        'exit_status': exit_status,
                        'stdout_lines': len(stdout.split('\n')) if stdout else 0,
                        'stderr_lines': len(stderr.split('\n')) if stderr else 0,
                    },
                    generated_files=generated_files,
                    is_error=exit_status != 0,
                )

                observation = f'{stdout}\n{stderr}'.strip()
                msg = ChatMessage(role='tool', content=observation)
                self.add_to_history(msg)

                act_span.update(
                    status='success' if exit_status == 0 else 'warning',
                    operation='code_execution',
                    exit_status=exit_status,
                )
                act_span.end(
                    output=observation[:500],
                    metadata={
                        'is_error': exit_status != 0,
                        'generated_files': generated_files,
                        'exit_status': exit_status,
                    },
                )

                yield self.response(
                    rtype='step',
                    value=observation,
                    channel='_act',
                    metadata={
                        'is_error': exit_status != 0,
                        'generated_files': generated_files,
                    },
                )

            except Exception as ex:
                error_msg = f'*** Error running code: {type(ex).__name__}: {str(ex)}'
                logger.error(error_msg)

                act_span.update(
                    status='error',
                    operation='code_execution_exception',
                    error_type=type(ex).__name__,
                    error_message=str(ex),
                )
                # Respond as the pseudo "tool"
                tool_msg = ChatMessage(role='tool', content=error_msg)
                self.add_to_history(tool_msg)

                act_span.end(
                    output='exception',
                    is_error=True,
                    error=error_msg,
                )

                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True},
                )


def llm_vision_support(model_names: list[str]) -> list[bool]:
    """Check whether images can be used with given LLMs.

    Args:
        model_names (list[str]): List of LLM names.

    Returns:
        list[bool]: List of booleans indicating whether each LLM supports vision.
    """
    status = [litellm.supports_vision(model=model) for model in model_names]
    for model, value in zip(model_names, status):
        print(f'- Vision supported by {model}: {value}')

    return status


def print_response(response: AgentResponse, only_final: bool = True):
    """Print agent's response in terminal with colors.

    Args:
        response (AgentResponse): Agent's response.
        only_final (bool, optional): Whether to print only final response. Defaults to True.
    """
    if response['type'] == 'final':
        msg = response['value'] if isinstance(response['value'], str) else response['value']
        rich.print(f'[green][bold]Agent[/bold]: {msg}[/green]\n')

    if not only_final:
        if response['type'] == 'log':
            rich.print(f'[white]{response}[/white]')
        else:
            rich.print(f'{response}')


async def main():
    """Demonstrate the use of ReActAgent and CodeActAgent."""
    litellm_params = {'temperature': 0, 'timeout': 30}
    model_name = 'gemini/gemini-2.5-flash-lite'
    # model_name = 'openai/gpt-4.1-mini'

    agent = ReActAgent(
        name='Simple agent',
        model_name=model_name,
        tools=[
            dtools.calculator,
            dtools.search_web,
            dtools.read_webpage,
            dtools.extract_as_markdown,
        ],
        max_iterations=5,
        litellm_params=litellm_params,
    )
    # agent = CodeActAgent(
    #     name='Simple agent',
    #     model_name=model_name,
    #     tools=[
    #         dtools.calculator,
    #         dtools.search_web,
    #         dtools.read_webpage,
    #         dtools.extract_as_markdown,
    #     ],
    #     max_iterations=7,
    #     litellm_params=litellm_params,
    #     run_env='host',
    #     allowed_imports=[
    #         'math',
    #         'datetime',
    #         'time',
    #         're',
    #         'typing',
    #         'mimetypes',
    #         'random',
    #         'ddgs',
    #         'bs4',
    #         'urllib.parse',
    #         'requests',
    #         'markitdown',
    #         'pathlib',
    #     ],
    #     pip_packages='ddgs~=9.5.2;beautifulsoup4~=4.14.2;',
    #     work_dir='./agent_workspace',
    # )

    the_tasks = [
        ('What is ten plus 15, raised to 2, expressed in words?', None),
        ('What is the date today? Express it in words like <Month> <Day>, <Year>.', None),
        (
            'Which image has a purple background?',
            [
                'https://www.slideteam.net/media/catalog/product/cache/1280x720/p/r/process_of_natural_language_processing_training_ppt_slide01.jpg',
                'https://cdn.prod.website-files.com/61a05ff14c09ecacc06eec05/66e8522cbe3d357b8434826a_ai-agents.jpg',
            ],
        ),
        (
            'What is four plus seven? Also, what are the festivals in Paris?'
            ' How they differ from Kolkata?',
            None,
        ),
        ('Write an elegant haiku in Basho style. Save it as poem.txt', None),
        # ('generate an image of AI. Use model gemini/imagen-4.0-fast-generate-001', None),
    ]

    print(f'{agent.__class__.__name__} demo\n')

    for task, img_urls in the_tasks:
        rich.print(f'[yellow][bold]User[/bold]: {task}[/yellow]')
        async for response in agent.run(task, files=img_urls):
            print_response(response, only_final=True)

        if agent.artifacts:
            print('Artifacts generated:')
            for art in agent.artifacts:
                print(f'- {art} (size: {os.path.getsize(art)} bytes)')

        if agent.current_plan:
            print(f'Plan:\n{agent.current_plan}')

        await asyncio.sleep(random.uniform(0.15, 0.55))
        print('\n\n')

    print('Demonstrating recurrent mode:\n')
    # Task 1: Perform a calculation or data retrieval
    async for response in agent.run('Find the population of France in 2023'):
        print_response(response, only_final=True)

    # Task 2: Use the result of Task 1 with recurrent_mode=True
    async for response in agent.run(
        'What would it be with a 0.5% growth?',
        recurrent_mode=True,
    ):
        print_response(response, only_final=True)


if __name__ == '__main__':
    os.environ['PYTHONUTF8'] = '1'
    asyncio.run(main())
