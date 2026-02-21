"""Function Calling Agent (FCA) module.
Provides a lightweight agent that uses native LLM function calling.
"""

import inspect
import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from . import kutils as ku
from .models import AgentResponse, Task

load_dotenv()

logger = ku.get_logger('FunctionCallingAgent')


class LocalLoopDetector:
    """Detects repeated tool calls by inspecting chat history.
    This is a simple, history-based detector that doesn't involve LLM calls.
    """

    def __init__(self, window: int = 6, threshold: int = 3):
        """Initialize the loop detector.

        Args:
            window: Number of recent tool calls to consider.
            threshold: Number of occurrences of a (tool, args) pair to trigger a loop detection.
        """
        self.window = window
        self.threshold = threshold

    def is_looping(self, chat_history: list[dict[str, Any]]) -> tuple[bool, str]:
        """Check if the agent is stuck in a loop of calling the same tools with same args.

        Args:
            chat_history: The current chat history.

        Returns:
            A tuple of (is_looping, message).
        """
        # Collect recent tool calls from assistant messages
        tool_calls_history = []
        for message in reversed(chat_history):
            if message.get('role') == 'assistant' and message.get('tool_calls'):
                for tc in message['tool_calls']:
                    func = tc.get('function', {})
                    name = func.get('name')
                    args = func.get('arguments')
                    if name and args:
                        # Canonicalize args for comparison
                        try:
                            args_dict = json.loads(args)
                            canonical_args = json.dumps(args_dict, sort_keys=True)
                        except (json.JSONDecodeError, TypeError):
                            canonical_args = args

                        tool_calls_history.append((name, canonical_args))

            if len(tool_calls_history) >= self.window:
                break

        if not tool_calls_history:
            return False, ''

        # Count occurrences of each (name, args) pair
        counts = {}
        for call in tool_calls_history:
            counts[call] = counts.get(call, 0) + 1
            if counts[call] >= self.threshold:
                name, args = call
                return True, f"Detected repeated call to tool '{name}' with arguments {args}."

        return False, ''


class FunctionCallingChatMessage(BaseModel):
    """Messages for FunctionCallingAgent using native function calling."""

    role: str = Field(description='Role of the message sender', default='assistant')
    content: str | None = Field(
        description='Reasoning/thought or final answer. Required for all responses.',
        default=None,
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        description='Native tool calls from LLM. Populated automatically by LiteLLM.',
        default=None,
    )

    @property
    def is_final(self) -> bool:
        """Check if this is a final answer (no tool calls)."""
        return not self.tool_calls

    def __str__(self) -> str:
        """Return a string representation of the message."""
        if self.is_final:
            return self.content or ''
        parts = []
        if self.content:
            parts.append(f'Reasoning: {self.content}')
        if self.tool_calls:
            for tc in self.tool_calls:
                func = tc.get('function', {})
                name = func.get('name', 'unknown')
                args = func.get('arguments', '{}')
                parts.append(f'Tool: {name}')
                parts.append(f'Args: {args}')
        return '\n'.join(parts)


class FunctionCallingAgent:
    """A minimal, robust agent that uses LLM-native function calling via LiteLLM.
    It maintains OpenAI-compliant chat history and can auto-generate JSON schemas
    for Python functions passed as tools.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        description: str | None = None,
        tools: list[Callable] | None = None,
        litellm_params: dict | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 20,
        max_retries: int = ku.DEFAULT_MAX_LLM_RETRIES,
        work_dir: str | None = None,
    ):
        """Create a function calling agent.

        Args:
            name: The name of the agent.
            model_name: The (LiteLLM) model name to use.
            description: Optional brief description about the agent.
            tools: An optional list of tools available to the agent.
            litellm_params: LiteLLM parameters.
            system_prompt: Optional system prompt for the agent.
            max_iterations: The max iterations an agent can perform to solve a task.
            max_retries: Maximum number of retries for LLM calls.
            work_dir: Optional local workspace directory.
        """
        self.name = name
        self.model_name = model_name
        self.description = description
        self.litellm_params = litellm_params or {}
        if not system_prompt:
            try:
                system_prompt = ku.read_prompt('system/function_calling.txt')
            except Exception:
                system_prompt = 'You are a helpful assistant.'
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.work_dir = work_dir

        # Extract underlying functions if tools are @tool decorated
        normalized_tools = []
        for t in tools or []:
            if hasattr(t, '__wrapped__'):
                normalized_tools.append(t.__wrapped__)
            else:
                normalized_tools.append(t)

        self.tools = normalized_tools
        self.tool_schemas = [self._build_tool_schema(fn) for fn in self.tools]
        self.tool_map = {fn.__name__: fn for fn in self.tools}

        self.chat_history: list[dict[str, Any]] = []
        self.task: Task | None = None
        self._loop_detector = LocalLoopDetector()

    def _run_init(
        self, task: str, files: list[str] | None = None, task_id: str | None = None
    ) -> None:
        """Initialize the running of a task.

        Args:
            task: Task description.
            files: Optional files for the task.
            task_id: Optional task ID.
        """
        self.task = Task(description=task, files=files)
        if task_id:
            self.task.id = task_id

    def _init_history(self) -> None:
        """Initialize message history with system prompt."""
        self.chat_history = [{'role': 'system', 'content': self.system_prompt}]

    def response(
        self,
        rtype: str,
        value: Any,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Prepare a response to be sent by the agent."""
        return {'type': rtype, 'channel': channel, 'value': value, 'metadata': metadata}

    async def run(
        self,
        task: str,
        files: list[str] | None = None,
        task_id: str | None = None,
        max_iterations: int | None = None,
        recurrent_mode: bool = False,
        summarize_progress_on_failure: bool = True,
    ) -> AsyncIterator[AgentResponse]:
        """Run a task through the agent, handling tool calls if needed.

        Args:
            task: Task description.
            files: Optional files for the task.
            task_id: Optional task ID.
            max_iterations: Maximum iterations for this run.
            recurrent_mode: Ignored for now.
            summarize_progress_on_failure: Ignored for now.

        Yields:
            AgentResponse objects.
        """
        self._run_init(task, files, task_id)
        self._init_history()

        yield self.response(rtype='log', value=f'Solving task: `{task}`', channel='run')

        # Add the task to history
        self._append_message({'role': 'user', 'content': task})

        iterations = max_iterations or self.max_iterations

        for idx in range(iterations):
            step_num = idx + 1
            yield self.response(rtype='log', value=f'* Executing step {step_num}', channel='run')

            try:
                response = await litellm.acompletion(
                    model=self.model_name,
                    messages=self.chat_history,
                    tools=self.tool_schemas if self.tool_schemas else None,
                    **self.litellm_params,
                )
            except Exception as e:
                logger.error(f'LiteLLM call failed: {e}')
                yield self.response(
                    rtype='final',
                    value=f'Error calling LLM: {e}',
                    channel='run',
                    metadata={'is_error': True},
                )
                return

            raw_msg = response.choices[0].message
            raw_tool_calls = getattr(raw_msg, 'tool_calls', None)
            tool_calls = [tc.model_dump() for tc in raw_tool_calls] if raw_tool_calls else None

            agent_msg = FunctionCallingChatMessage(
                role=raw_msg.role,
                content=raw_msg.content,
                tool_calls=tool_calls,
            )

            # If the model wants to call a tool
            if not agent_msg.is_final:
                # Add reasoning/tool call to history
                self._append_message(
                    {
                        'role': 'assistant',
                        'content': agent_msg.content,
                        'tool_calls': agent_msg.tool_calls,
                    }
                )

                for tool_call in agent_msg.tool_calls:
                    tool_name = tool_call['function']['name']
                    args_str = tool_call['function'].get('arguments', '{}')
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}

                    tool_id = tool_call.get('id', f'tool_{uuid.uuid4().hex[:8]}')

                    yield self.response(
                        rtype='log', value=f'Tool call: {tool_name}({args_str})', channel='run'
                    )
                    tool_result = self._execute_tool(tool_name, args)

                    self._append_message(
                        {
                            'role': 'tool',
                            'tool_call_id': tool_id,
                            'name': tool_name,
                            'content': str(tool_result),
                        }
                    )

                # Loop detection
                loop_detected, loop_msg = self._loop_detector.is_looping(self.chat_history)
                if loop_detected:
                    logger.warning(loop_msg)
                    warning_msg = f'!!! WARNING: {loop_msg} Please try a different approach.'
                    self._append_message({'role': 'user', 'content': warning_msg})
                    yield self.response(rtype='log', value=loop_msg, channel='loop_detector')
                    # Terminate on loop
                    break

                continue

            # Otherwise, it's a normal assistant message
            self._append_message(agent_msg.model_dump(exclude_none=True))
            yield self.response(
                rtype='final',
                value=agent_msg.content or '',
                channel='run',
                metadata={'final_answer_found': True},
            )
            return

        # exhaustion or loop termination
        failure_msg = f'Sorry, I failed to get a complete answer even after {idx + 1} steps!'
        yield self.response(
            rtype='final', value=failure_msg, channel='run', metadata={'final_answer_found': False}
        )

    def _append_message(self, msg: dict[str, Any]):
        """Append a message to the chat history."""
        self.chat_history.append(msg)

    def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a registered Python function by name."""
        fn = self.tool_map.get(name)
        if not fn:
            return f"Error: tool '{name}' not found."
        try:
            return fn(**args)
        except Exception as e:
            return f"Error executing '{name}': {e}"

    def _build_tool_schema(self, fn: Callable) -> dict[str, Any]:
        """Auto-generate an OpenAI-style tool schema from a Python function."""
        sig = inspect.signature(fn)
        params = {}
        required = []
        for name, param in sig.parameters.items():
            param_type = self._map_type(param.annotation)
            params[name] = {
                'type': param_type,
                'description': f"Parameter '{name}' of type {param_type}",
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            'type': 'function',
            'function': {
                'name': fn.__name__,
                'description': inspect.getdoc(fn) or f'Function {fn.__name__}',
                'parameters': {
                    'type': 'object',
                    'properties': params,
                    'required': required,
                },
            },
        }

    def _map_type(self, t: Any) -> str:
        """Map Python types to JSON schema types."""
        mapping = {
            int: 'integer',
            float: 'number',
            str: 'string',
            bool: 'boolean',
            list: 'array',
            dict: 'object',
            Any: 'string',
        }
        return mapping.get(t, 'string')
