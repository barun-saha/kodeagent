"""Function Calling Agent. This module is optimized for Small Language Models (SLMs).
The FC agent uses native function calling to solve tasks. Some of the data structures are
reproduced here to keep this module self-contained and optimized.
"""

import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import litellm

from . import kutils as ku
from . import tools as dtools
from .orchestrator import Planner

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
    files: None = None
    """Unused placeholder only for compatibility with Planner."""


class AgentResponse(TypedDict):
    """Streaming response sent by an agent in the course of solving a task.
    This class is reproduced here to keep this module self-contained and optimized.
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

FCA_SYSTEM_PROMPT = ku.read_prompt('system/function_calling.txt')


def final_answer(result: str) -> str:
    """Provide the final answer to the user's task and end the conversation.

    Args:
        result: The final answer or result of the task.

    Returns:
        The final answer text in user-readable format.
    """
    return result


class FunctionCallingAgent:
    """An agent that uses native function calling to solve tasks,
    optimized for Small Language Models (SLMs). If you're using Ollama, make sure to select
    a model that supports function calling, such as 'ollama/qwen3:8b-q8_0'.
    """

    def __init__(
        self,
        model_name: str,
        tools: list[Callable] | None = None,
        system_prompt: str = FCA_SYSTEM_PROMPT,
        loop_detection_threshold: int = 3,
        litellm_params: dict | None = None,
        max_tool_result_chars: int = 3000,
    ):
        """Initialize the FunctionCallingAgent.

        Args:
            model_name: Model identifier for LiteLLM.
            tools: Optional list of callable tools.
            system_prompt: System prompt for the agent.
            loop_detection_threshold: Number of consecutive same tool calls
             before triggering loop detection. Default is 3.
            litellm_params: Optional dictionary of parameters to pass to LiteLLM calls.
            max_tool_result_chars: Maximum number of characters to store in chat history
             for each tool result. Longer results are truncated with a note. Full results
             are preserved separately for final answer preparation.
        """
        self.model_name = model_name
        self.tools = tools or []

        # Ensure final_answer is always available as a tool
        tool_names = [fn.__name__ for fn in self.tools]
        if 'final_answer' not in tool_names:
            self.tools.append(final_answer)

        self.tool_schemas = [FunctionCallingAgent._build_tool_schema(fn) for fn in self.tools]
        # Exclude final_answer from tool_map — it is executed directly via _execute_tool
        # but its result is extracted separately at the end of the run loop.
        self.tool_map = {fn.__name__: fn for fn in self.tools}

        self.litellm_params = litellm_params
        self.system_prompt = system_prompt
        self.chat_history: list[dict[str, Any]] = []

        self.task: Task | None = None
        self.final_answer_found = False
        self.max_tool_result_chars = max_tool_result_chars
        self.full_tool_results: dict[tuple[str, str], str] = {}

        self.loop_detection_threshold = loop_detection_threshold
        self.nudge_count = 0

    def response(
        self,
        rtype: Literal['step', 'final', 'log'],
        value: Any,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Prepare a response to be sent by the agent.

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

    @staticmethod
    def _parse_param_descriptions(doc: str) -> dict[str, str]:
        """Extract per-parameter descriptions from a docstring.

        Supports Google-style (Args:) and Sphinx-style (:param name:) formats.

        Args:
            doc: The docstring to parse.

        Returns:
            A dictionary mapping parameter names to their descriptions.
        """
        param_docs: dict[str, str] = {}
        if not doc:
            return param_docs

        # Google-style: Args: / Parameters: section
        args_section = re.search(r'(?:Args|Parameters):\s*(.*)', doc, re.DOTALL | re.IGNORECASE)
        if args_section:
            args_text = args_section.group(1)
            for line in args_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'(\w+)\s*(?:\(.*?\))?\s*:\s*(.*)', line)
                if match:
                    param_docs[match.group(1)] = match.group(2).strip()

        # Sphinx-style: :param name: description
        if not param_docs:
            for m in re.finditer(r':param\s+(\w+):\s*(.*)', doc):
                param_docs[m.group(1)] = m.group(2).strip()

        return param_docs

    @staticmethod
    def _build_tool_schema(fn: Callable) -> dict[str, Any]:
        """Auto-generate an OpenAI-style tool schema from a Python function.

        Args:
            fn: The function to build a tool schema for.

        Returns:
            A dictionary representing the tool schema.
        """
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ''
        param_docs = FunctionCallingAgent._parse_param_descriptions(doc)

        params = {}
        required = []

        for name, param in sig.parameters.items():
            param_type = DATA_TYPES.get(param.annotation, 'string')
            description = param_docs.get(name, f'The "{name}" argument ({param_type}).')
            params[name] = {'type': param_type, 'description': description}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        # Use first line only — SLMs lose signal in long descriptions
        description = doc.splitlines()[0].strip() if doc else f'Function {fn.__name__}'

        return {
            'type': 'function',
            'function': {
                'name': fn.__name__,
                'description': description,
                'parameters': {
                    'type': 'object',
                    'properties': params,
                    'required': required,
                },
            },
        }

    def _validate_tool_args(self, name: str, args: dict[str, Any]) -> str | None:
        """Validate tool arguments against the schema.

        Args:
            name: Name of the tool being called.
            args: Arguments provided for the tool call.

        Returns:
            An error message if validation fails, or None if validation succeeds.
        """
        schema = next((s for s in self.tool_schemas if s['function']['name'] == name), None)
        if schema is None:
            return None  # Can't validate, let execution handle it

        parameters = schema['function']['parameters']
        required = parameters.get('required', [])
        properties = parameters.get('properties', {})

        missing = [r for r in required if r not in args]
        if missing:
            return (
                f'Error: Missing required arguments for `{name}`: {missing}. '
                f'Required arguments are: {required}.'
            )

        unexpected = [k for k in args if k not in properties]
        if unexpected:
            return (
                f'Error: Unexpected arguments for `{name}`: {unexpected}. '
                f'Valid arguments are: {list(properties.keys())}.'
            )

        return None

    def _execute_tool(self, tool_call: Any) -> dict[str, str]:
        """Safely executes a specific tool call and returns the message object.

        Args:
            tool_call: The tool call to execute.

        Returns:
            A dictionary representing the tool result message.
        """
        name = tool_call.function.name
        args_str = tool_call.function.arguments

        try:
            args = json.loads(args_str)
            logger.info('Agent executing tool: %s with args: %s', name, args)

            if name not in self.tool_map:
                result = f'Error: Tool `{name}` is not defined.'
            else:
                validation_error = self._validate_tool_args(name, args)
                if validation_error:
                    result = validation_error
                else:
                    tool_result = self.tool_map[name](**args)
                    if tool_result is None:
                        result = (
                            f'Error: Tool `{name}` returned no result. '
                            'Verify your input arguments and try a different approach.'
                        )
                    elif isinstance(tool_result, str) and not tool_result.strip():
                        result = (
                            f'Error: Tool `{name}` returned an empty result. '
                            'The query may have returned no data.'
                        )
                    else:
                        result = str(tool_result)

        except json.JSONDecodeError:
            result = 'Error: Model provided malformed JSON arguments.'
        except TypeError as e:
            result = f'Error: Wrong arguments passed to `{name}`: {str(e)}'
        except Exception as e:
            result = f'Error executing `{name}`: {str(e)}'

        return {
            'tool_call_id': tool_call.id,
            'role': 'tool',
            'name': name,
            'content': result,
        }

    def _detect_tool_loop(self) -> bool:
        """Detect if the agent is stuck in a tool calling loop.

        Analyzes chat history to identify when the same tool is being called
        consecutively without progress. Supports two-stage nudge escalation
        before signalling hard termination.

        Returns:
            True if a loop was detected (nudge added or hard termination signalled),
            False otherwise.
        """
        recent_tool_calls: list[str] = []
        for msg in reversed(self.chat_history):
            if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                for tool_call in msg.get('tool_calls', []):
                    tool_name = (
                        tool_call.get('function', {}).get('name')
                        if isinstance(tool_call, dict)
                        else getattr(getattr(tool_call, 'function', None), 'name', None)
                    )
                    if tool_name:
                        recent_tool_calls.append(tool_name)

        if (
            len(recent_tool_calls) >= self.loop_detection_threshold
            and len(set(recent_tool_calls[: self.loop_detection_threshold])) == 1
        ):
            loop_tool = recent_tool_calls[0]

            # Hard termination after two nudges
            if self.nudge_count >= 2:
                return True

            available_tools = [t for t in self.tool_map if t != loop_tool]

            # Stage 1: gentle nudge
            if self.nudge_count == 0:
                nudge_message = (
                    f'Loop detected: The tool "{loop_tool}" has been called'
                    f' {self.loop_detection_threshold} consecutive times without progress.'
                    ' This approach is not working.'
                )
            # Stage 2: strong nudge
            else:
                nudge_message = (
                    f'[CRITICAL: STOP REPEATING] You are still calling "{loop_tool}"'
                    ' despite the previous warning. You MUST change your strategy or call'
                    ' final_answer with your best answer now.'
                )

            if available_tools:
                nudge_message += (
                    f' Consider using one of these tools instead: {", ".join(available_tools)}.'
                )
            nudge_message += ' If you have gathered enough information, call final_answer now.'

            self.nudge_count += 1
            self.chat_history.append({'role': 'user', 'content': nudge_message})
            return True

        return False

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        """
        Extract all URLs from a text string.

        Args:
            text: The input text to search for URLs.

        Returns:
            A list of URLs found in the text.
        """
        return re.findall(r'https?://\S+', text)

    async def _run_init(
        self,
        task_desc: str,
        task_id: str | None = None,
        use_planning: bool = True,
        recurrent_mode: bool = False,
    ) -> None:
        """Initialize the running of a task.

        Args:
            task_desc: Task description.
            task_id: Optional task ID.
            use_planning: If True, generate a simple plan at the beginning of the task.
            recurrent_mode: If True, the agent will continue to run on the same task
             until it decides to stop, allowing for more dynamic interactions.
        """
        if not task_desc or not task_desc.strip():
            raise ValueError('Task description cannot be empty!')

        if recurrent_mode and self.task is not None:
            task_description = (
                f'## Previous Task\n{self.task.description}\n\n'
                f'## Previous Task Result\n{self.task.result}\n\n'
                f'## New Task:\n{task_desc}'
            )
        else:
            task_description = f'## New Task:\n{task_desc}'

        self.nudge_count = 0
        self.task = Task(description=task_desc, id=task_id, result=None, steps_taken=None)
        self.chat_history = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': task_description},
        ]

        urls = FunctionCallingAgent._extract_urls(task_desc)
        if urls:
            url_list = '\n'.join(f'- {url}' for url in urls)
            self.chat_history.append(
                {
                    'role': 'user',
                    'content': (
                        'The task contains the following URL(s):\n'
                        f'{url_list}\n\n'
                        'If the task requires knowing the content of these pages,'
                        ' use `read_webpage` to fetch them before answering.'
                        ' Do not answer from memory about the content of these pages.'
                    ),
                }
            )
            logger.info('URL pre-injection: %d URL(s) detected, read_webpage required.', len(urls))

        if use_planning:
            planner = Planner(
                model_name=self.model_name,
                litellm_params=self.litellm_params,
            )
            await planner.create_plan(task=self.task, agent_type='fca')
            formatted_plan = planner.get_formatted_plan()
            self.chat_history.append(
                {
                    'role': 'user',
                    'content': f'Here is a plan for this task:\n{formatted_plan}',
                }
            )
            logger.info('Task plan:\n%s\n', formatted_plan)

    def _format_history_as_text(self) -> str:
        """Format chat history as readable text.

        Converts the chat history into a human-readable format, excluding
        tool call IDs and other non-essential metadata.

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
                if content:
                    formatted.append(f'Assistant: {content}')
                tool_calls = msg.get('tool_calls')
                if tool_calls:
                    tool_names = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_name = tool_call.get('function', {}).get('name')
                        else:
                            tool_name = getattr(getattr(tool_call, 'function', None), 'name', None)
                        if tool_name:
                            tool_names.append(tool_name)
                    if tool_names:
                        formatted.append(f'Assistant: [Called tools: {", ".join(tool_names)}]')
            elif role == 'tool':
                tool_name = msg.get('name', 'unknown')
                call_key_match = next(
                    (k for k in self.full_tool_results if k[0] == tool_name),
                    None,
                )
                # Prefer full result for final answer quality; fall back to history version
                full_content = self.full_tool_results[call_key_match] if call_key_match else content
                if full_content:
                    formatted.append(f'Tool ({tool_name}): {full_content}')

        return '\n'.join(formatted)

    def _maybe_truncate(self, content: str) -> str:
        """Truncate or summarise a tool result for storage in chat history.

        Full results are always preserved in self.full_tool_results for use
        by _prepare_final_answer. Only the history-facing version is reduced.

        Note: summarise_long_results=True requires an await — callers must
        use _maybe_truncate_async instead when that flag is set.

        Args:
            content: Full tool result content.

        Returns:
            Possibly truncated content with a note if cut.
        """
        if len(content) <= self.max_tool_result_chars:
            return content
        truncated = content[: self.max_tool_result_chars]
        return (
            f'{truncated}\n\n'
            f'[Truncated — {len(content)} total chars. '
            f'Showing first {self.max_tool_result_chars}.]'
        )

    async def run(
        self,
        task: str,
        max_iterations: int = 10,
        refine_final_answer: bool = True,
        use_planning: bool = True,
        recurrent_mode: bool = False,
        files: None = None,
        loop_threshold: int = 3,
    ) -> AsyncIterator[AgentResponse]:
        """Main loop for the agent to process input and execute tools until finished.

        Args:
            task: Task description to process.
            max_iterations: Maximum number of iterations to run.
            refine_final_answer: If True, calls an additional SLM step to produce a
             clean final answer when the model exits without calling final_answer.
             Recommended for models <=4B that may not use final_answer reliably.
            use_planning: If True, generate a simple plan at the beginning of the task.
            recurrent_mode: If True, the agent continues from the previous task result,
             allowing for multi-turn workflows.
            files: *Unused* — for API compatibility only. Pass URLs or file content
             directly in the task description instead.
            loop_threshold: Number of consecutive same tool calls before triggering loop detection.

        Yields:
            AgentResponse: Streaming log, step, and final responses.
        """
        await self._run_init(task, use_planning=use_planning, recurrent_mode=recurrent_mode)

        n_turns = 0
        self.final_answer_found = False
        consecutive_errors = 0
        executed_tool_calls: dict[tuple[str, str], str] = {}
        self.full_tool_results: dict[tuple[str, str], str] = {}  # full content for final answer

        yield self.response(
            rtype='log', value=f'Solving task: `{self.task.description}`', channel='run'
        )

        for turn in range(max_iterations):
            logger.info('Turn %d/%d for model %s', turn + 1, max_iterations, self.model_name)
            n_turns += 1
            yield self.response(rtype='log', channel='run', value=f'* Executing step {n_turns}')

            response = await litellm.acompletion(
                model=self.model_name,
                messages=self.chat_history,
                tools=self.tool_schemas,
                tool_choice='auto',
                **(self.litellm_params or {}),
            )

            message = response.choices[0].message
            self.chat_history.append(message.model_dump())

            # Model chose to respond without a tool call — treat as done
            if not message.tool_calls:
                self.final_answer_found = True
                break

            # Process all tool calls in this turn
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args_str = tool_call.function.arguments
                call_key = (name, args_str)

                # Deduplicate: return cached result instead of re-executing
                if call_key in executed_tool_calls:
                    prev_result = executed_tool_calls[call_key]
                    tool_result_message = {
                        'tool_call_id': tool_call.id,
                        'role': 'tool',
                        'name': name,
                        'content': (
                            f'You already called `{name}` with these arguments '
                            f'and got: {prev_result}. Use that result.'
                        ),
                    }
                else:
                    tool_result_message = self._execute_tool(tool_call)
                    raw_content = tool_result_message['content']

                    if not raw_content.startswith('Error:'):
                        # Store full result separately for _prepare_final_answer
                        self.full_tool_results[call_key] = raw_content
                        # Truncate/summarise for history
                        tool_result_message = {
                            **tool_result_message,
                            'content': self._maybe_truncate(raw_content),
                        }

                # Track consecutive errors for early exit
                if tool_result_message['content'].startswith('Error:'):
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                self.chat_history.append(tool_result_message)
                yield self.response(
                    rtype='log',
                    value=(
                        f'Executed tool: {name}. Result: {tool_result_message["content"][:100]}...'
                    ),
                    channel='tool',
                )

                if name == 'final_answer':
                    self.final_answer_found = True

            if self.final_answer_found:
                logger.info('Final answer tool called, ending loop.')
                break

            if consecutive_errors >= 3:
                logger.error('Too many consecutive tool errors. Terminating loop early.')
                yield self.response(
                    rtype='log',
                    value='Too many consecutive tool errors. Terminating loop early.',
                    channel='run',
                )
                break

            if self._detect_tool_loop():
                if self.nudge_count >= loop_threshold:
                    logger.error('Loop persisted after nudges. Terminating for safety.')
                    yield self.response(
                        rtype='log',
                        value='Loop persisted after nudges. Terminating for safety.',
                        channel='run',
                    )
                    break
                logger.info('Loop detection triggered, nudging agent.')
                yield self.response(
                    rtype='log', value='Loop detected, nudging agent...', channel='run'
                )

        self.task.steps_taken = n_turns

        # Prefer the final_answer tool result if present
        # for...else: the else block runs only if the loop completed without hitting break
        # i.e. no final_answer tool message was found in history
        result = 'No response generated.'
        for msg in reversed(self.chat_history):
            if msg.get('role') == 'tool' and msg.get('name') == 'final_answer':
                result = msg['content']
                break
        else:
            # Fall back to last assistant text content
            if refine_final_answer:
                result = await self._prepare_final_answer()
            else:
                for msg in reversed(self.chat_history):
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        result = msg['content']
                        break

        yield self.response(rtype='final', value=result, channel='run')

    async def _prepare_final_answer(self) -> str:
        """Summarise the conversation into a clean final answer via a separate SLM call.

        Used as a fallback when the model exits the loop without calling final_answer,
        which is common for models <=4B. Formats the full conversation history as
        plain text and asks the model to produce a concise response.

        Returns:
            A user-readable string summarising the result.
        """
        formatted_history = self._format_history_as_text()
        response = await litellm.acompletion(
            model=self.model_name,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful assistant. Based on the conversation history '
                        "below, provide a clear and concise final answer to the user's task."
                    ),
                },
                {
                    'role': 'user',
                    'content': f'Conversation history:\n{formatted_history}',
                },
            ],
            **(self.litellm_params or {}),
        )
        final_answer_text = (response.choices[0].message.content or '').strip()
        logger.debug('Raw _prepare_final_answer response: %s', final_answer_text)
        return final_answer_text


async def main():
    """Example usage of the FunctionCallingAgent."""
    model_name = 'gemini/gemini-2.0-flash-lite'
    # Some smaller models with 8-bit quantization or higher can perform well with function calling
    # model_name='ollama/qwen3:8b-q8_0'
    # model_name = 'ollama/qwen3:4b-instruct-2507-fp16'
    # model_name = 'ollama/functiongemma:270m-it-fp16'
    # model_name = 'ollama/granite4:7b-a1b-h'
    # model_name = 'ollama/phi4-mini:3.8b-q8_0'

    agent = FunctionCallingAgent(
        model_name=model_name,
        tools=[
            dtools.search_web,
            dtools.calculator,
            dtools.read_webpage,
            dtools.transcribe_youtube,
            dtools.search_wikipedia,
        ],
        litellm_params={'temperature': 0, 'timeout': 90},
    )
    print(f'Using model: {model_name}')

    tasks = [
        'What is 5 times 7?',
        'What is this page about? https://en.wikipedia.org/wiki/Artificial_intelligence',
        'Find the current stock price of NVIDIA & calculate how many shares I can buy with $5000.',
        (
            'Get the transcript of this YouTube video: https://www.youtube.com/watch?v=aircAruvnKk'
            '\nIdentify the main topic, then search Wikipedia for that topic and give me'
            ' a brief summary of what Wikipedia says about it (give Wikipedia page link).'
        ),
    ]

    for idx, task in enumerate(tasks, start=1):
        print(f'\nTask #{idx}: {task}')
        async for response in agent.run(task, max_iterations=10, use_planning=False):
            if response['type'] == 'log':
                print(f'Log: {response["value"]}')
            elif response['type'] == 'final':
                print(f'\nFinal Result: {response["value"]}')

        print(f'>>> {agent.task.result=}')
        print(f'>>> {agent.task.steps_taken=}')
        print('-' * 80)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
