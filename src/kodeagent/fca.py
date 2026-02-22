"""Function Calling Agent.
This module is optimized for Small Language Models (SLMs).
"""

import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict, Literal

import litellm

from . import tools as dtools


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task for the agent to solve. A simplified version for SLMs."""

    id: str | None
    """Optional unique identifier for the task."""
    description: str
    """Text description of the task to be solved."""
    result: str | None
    """Final result or answer after solving the task."""
    steps_taken: int | None
    """Number of steps taken by the agent to solve the task."""


class AgentResponse(TypedDict):
    """Streaming response sent by an agent in the course of solving a task.
    This class is reproduced here to keep this moudle self-contained and optimized.
    """

    type: Literal['step', 'final', 'log']
    """Type of the response: 'step', 'final', or 'log'."""
    channel: str | None
    """Optional channel name for the response."""
    value: Any
    """Value of the response, varies by type."""
    metadata: dict[str, Any] | None
    """Optional metadata associated with the response."""


DATA_TYPES = {
    int: 'integer',
    float: 'number',
    str: 'string',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
    Any: 'string',
}


class FunctionCallingAgent:
    """An agent that uses native function calling to solve tasks,
     optimized for Small Language Models (SLMs). If your're using Ollama, make sure to select
     a model that supports function calling, such as 'ollama/qwen3:8b-q8_0'."""

    def __init__(
        self,
        model: str,
        tools: list[Callable] | None = None,
        system_prompt: str = 'You are a helpful assistant that uses tools to solve tasks.',
        loop_detection_threshold: int = 3,
    ):
        """Initialize the FunctionCallingAgent.

        Args:
            model: Model identifier for LiteLLM.
            tools: Optional list of callable tools.
            system_prompt: System prompt for the agent.
            loop_detection_threshold: Number of consecutive same tool calls
                before triggering loop detection. Default is 3.
        """
        self.model = model
        self.tools = tools or []
        self.tool_schemas = [FunctionCallingAgent._build_tool_schema(fn) for fn in self.tools]
        self.tool_map = {fn.__name__: fn for fn in self.tools}

        self.system_prompt = system_prompt
        self.chat_history: list[dict[str, Any]] = []
        self.loop_detection_threshold = loop_detection_threshold
        self.task: Task | None = None

    @staticmethod
    def _build_tool_schema(fn: Callable) -> dict[str, Any]:
        """Auto-generate an OpenAI-style tool schema from a Python function."""
        sig = inspect.signature(fn)
        params = {}
        required = []

        for name, param in sig.parameters.items():
            param_type = DATA_TYPES.get(param.annotation, 'string')
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

    def _execute_tool(self, tool_call) -> dict[str, str]:
        """Safely executes a specific tool call and returns the message object."""
        name = tool_call.function.name
        args_str = tool_call.function.arguments

        try:
            args = json.loads(args_str)
            logger.info('Agent executing tool: %s with args: %s', name, args)

            if name not in self.tool_map:
                result = f"Error: Tool '{name}' is not defined."
            else:
                result = str(self.tool_map[name](**args))

        except json.JSONDecodeError:
            result = 'Error: Model provided malformed JSON arguments.'
        except Exception as e:
            result = f'Error executing {name}: {str(e)}'

        return {
            'tool_call_id': tool_call.id,
            'role': 'tool',
            'name': name,
            'content': result,
        }

    def _detect_tool_loop(self) -> bool:
        """Detect if the agent is stuck in a tool calling loop.

        Analyzes chat history to identify when the same tool is being called
        consecutively without progress. If a loop is detected, adds an
        intelligent nudge message to guide the agent away from the loop.

        Returns:
            True if a loop was detected and handled, False otherwise.
        """
        # Get recent tool calls from assistant messages
        recent_tool_calls = []
        for i in range(len(self.chat_history) - 1, -1, -1):
            msg = self.chat_history[i]
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                # Extract tool names from all tool calls in this message
                for tool_call in msg.get('tool_calls', []):
                    tool_name = (
                        tool_call.get('function', {}).get('name')
                        if isinstance(tool_call, dict)
                        else getattr(getattr(tool_call, 'function', None), 'name', None)
                    )
                    if tool_name:
                        recent_tool_calls.append(tool_name)

        # Check for consecutive identical tool calls
        if (
            len(recent_tool_calls) >= self.loop_detection_threshold
            and len(set(recent_tool_calls[: self.loop_detection_threshold])) == 1
        ):
            loop_tool = recent_tool_calls[0]
            logger.warning(
                'Loop detected: Tool %s called %d times consecutively.',
                loop_tool,
                self.loop_detection_threshold,
            )

            # Get all available tools except the looping one
            available_tools = [tool for tool in self.tool_map.keys() if tool != loop_tool]

            nudge_message = (
                f' Loop detected: The tool "{loop_tool}" has been called'
                f' {self.loop_detection_threshold} consecutive times without'
                ' making progress. This approach is not working.'
            )

            if available_tools:
                nudge_message += (
                    f'Try a different approach. Consider using one of these'
                    f' tools instead: {", ".join(available_tools)}.'
                )

            nudge_message += (
                'If you have gathered enough information, provide a final'
                ' answer instead of calling a tool.'
            )

            # Add the nudge message as a tool result
            # This maintains OpenAI compliance: assistant message with
            # tool_calls is followed by tool message(s)
            self.chat_history.append(
                {
                    'role': 'tool',
                    'name': loop_tool,
                    'content': nudge_message,
                    'tool_call_id': 'loop-detection',
                }
            )

            return True

        return False

    def _run_init(self, task: str, task_id: str | None = None) -> None:
        """Initialize the running of a task.

        Args:
            task: Task description.
            task_id: Optional task ID.
        """
        if not task or not task.strip():
            raise ValueError('Task description cannot be empty!')

        self.task = Task(description=task, id=task_id, result=None, steps_taken=None)
        self.chat_history = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': task},
        ]

    def _format_history_as_text(self) -> str:
        """Format chat history as readable text.

        Converts the chat history into a human-readable format, excluding
        tool call IDs and other non-essential metadata. Handles assistant
        messages with both content and tool_calls.

        Returns:
            Formatted chat history as a string.
        """
        formatted = []
        for msg in self.chat_history[1:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content')

            if role == 'user':
                if content:
                    formatted.append(f'User: {content}')
            elif role == 'assistant':
                # Assistant can have content and/or tool_calls
                if content:
                    formatted.append(f'Assistant: {content}')
                tool_calls = msg.get('tool_calls')
                if tool_calls:
                    tool_names = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('function', {}).get('name')
                        else:
                            tool_name = getattr(
                                getattr(tool_call, 'function', None),
                                'name',
                                None,
                            )
                        if tool_name:
                            tool_names.append(tool_name)
                    if tool_names:
                        formatted.append(f'Assistant: [Called tools: {", ".join(tool_names)}]')
            elif role == 'tool':
                tool_name = msg.get('name', 'unknown')
                if content:
                    formatted.append(f'Tool ({tool_name}): {content}')

        return '\n'.join(formatted)

    async def _prepare_final_answer(self) -> str:
        """Assess task completion and prepare a user-readable final response.

        Formats the conversation history as readable text and calls the SLM
        with a structured prompt that asks it to determine whether the task
        was completed and to provide a concise final answer. This is a
        standalone SLM call separate from the agent's main loop.

        Returns:
            A user-readable string summarising the result.
        """
        formatted_history = self._format_history_as_text()

        final_system_prompt = (
            'You are a helpful assistant. Based on the conversation history'
            ' below, provide a clear and concise final answer to the user\'s question/task.'
        )
        user_prompt = f'Conversation history:\n{formatted_history}\n\nNow output the JSON object.'

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {'role': 'system', 'content': final_system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )

        final_answer = (response.choices[0].message.content or '').strip()
        logger.debug('Raw final-answer SLM response: %s', final_answer)

        return final_answer

    async def run(self, task: str, max_iterations: int = 10) -> str:
        """Main loop for the agent to process input and execute tools until finished.

        Args:
            task: Task description to process.
            max_iterations: Maximum number of iterations to run.

        Returns:
            A user-readable final response string.
        """
        self._run_init(task)

        n_turns = 0

        # TODO Return streaming response per step of type AgentResponse

        for turn in range(max_iterations):
            logger.info('Turn %d/%d for model %s', turn + 1, max_iterations, self.model)
            n_turns += 1

            response = await litellm.acompletion(
                model=self.model,
                messages=self.chat_history,
                tools=self.tool_schemas,
                tool_choice='auto',
            )

            message = response.choices[0].message
            # Store assistant response in history
            self.chat_history.append(message.model_dump())

            # Check if model wants to call a tool
            if not message.tool_calls:
                break

            # Process all tool calls in the message
            for tool_call in message.tool_calls:
                tool_result_message = self._execute_tool(tool_call)
                self.chat_history.append(tool_result_message)

            # Detect and handle tool loops
            if self._detect_tool_loop():
                logger.info('Loop detection triggered, continuing with nudge.')

        # When max iterations exceeded (or task is finished), prepare final answer
        # TODO Use a flag to skip final answer preparation and present the last content
        prepare_final_answer = True
        self.task.steps_taken = n_turns

        if prepare_final_answer:
            result = await self._prepare_final_answer()
            self.task.result = result
            return result

        # TODO Include a non-LLm version of answer cleaning based on the history

        # Fallback: return the content of the last assistant message
        for msg in reversed(self.chat_history):
            if msg.get('role') == 'assistant' and msg.get('content'):
                return msg['content']
        return 'No response generated.'


async def main():
    """Example usage of the FunctionCallingAgent."""
    # litellm._turn_on_debug()
    # Initialize Agent
    agent = FunctionCallingAgent(
        model='gemini/gemini-2.0-flash-lite',
        # model='ollama/qwen3:8b-q8_0',
        tools=[dtools.search_web, dtools.calculator, dtools.read_webpage],
    )

    # Run Agent
    tasks = [
        'What is 5 times 7?',
        'What is this page about? https://en.wikipedia.org/wiki/Artificial_intelligence',
    ]

    for task in tasks:
        print(f'\nUser Input: {task}')
        final_answer = await agent.run(task, max_iterations=10)
        print(f'\nFinal Result: {final_answer}')
        print(f'>>> {agent.task.result=}')
        print(f'>>> {agent.task.steps_taken=}')
        # print()
        # for msg in agent.chat_history:
        #     print(msg)
        print('-' * 50)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
