"""
A minimalistic approach to building AI agents.
Implements ReAct and CodeActAgent. Supports multi-agent via SupervisorAgent.
"""
import ast
import asyncio
import inspect
import json
import logging
import os
import random
import re
import shutil
import subprocess as sp
import sys
import tempfile
import textwrap
import uuid
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from json import JSONDecodeError
from typing import (
    AsyncIterator,
    Literal,
    Optional,
    Callable,
    Any,
    Type,
    TypedDict,
    Union,
)

import json_repair
import litellm
import pydantic as pyd
import rich
from dotenv import load_dotenv
from tenacity import stop_after_attempt, wait_random_exponential, AsyncRetrying

from . import kutils as ku


load_dotenv()

warnings.simplefilter('once', UserWarning)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('langfuse').disabled = True
logger = logging.getLogger('KodeAgent')

# litellm.success_callback = ['langfuse']
# litellm.failure_callback = ['langfuse']

# litellm._turn_on_debug()


def _read_prompt(filename: str) -> str:
    """Reads a prompt from the prompts directory."""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', filename)

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f'Prompt file `{filename}` not found in the prompts directory: {prompt_path}'
        ) from fnfe
    except Exception as e:
        raise RuntimeError(
            f'Error reading prompt file `{filename}`: {e}'
        ) from e


REACT_SYSTEM_PROMPT = _read_prompt('system/react.txt')
CODE_ACT_AGENT_PROMPT = _read_prompt('codeact.txt')
RELEVANT_TOOLS_PROMPT = _read_prompt('relevant_tools.txt')
AGENT_PLAN_PROMPT = _read_prompt('agent_plan.txt')
UPDATE_PLAN_PROMPT = _read_prompt('update_plan.txt')
OBSERVATION_PROMPT = _read_prompt('observation.txt')
SALVAGE_RESPONSE_PROMPT = _read_prompt('salvage_response.txt')
# Unused currently
CONTEXTUAL_SUMMARY_PROMPT = _read_prompt('contextual_summary.txt')


def tool(func: Callable) -> Callable:
    """
    A decorator to convert any Python function into a tool with additional metadata.
    Tooling based on async functions is not supported.

    Args:
        func (Callable): The function to be converted into a tool.

    Returns:
        Callable: The decorated function with additional metadata.
    """
    if asyncio.iscoroutinefunction(func):
        raise ValueError(
            'Tooling based on async functions is not supported. Please remove `async` from'
            f' the signature of the `{func.__name__}` function or remove the `@tool` decorator.'
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Create a schema for the function arguments using Pydantic
    signature = inspect.signature(func)
    fields = {name: (param.annotation, ...) for name, param in signature.parameters.items()}

    # Add metadata to the function
    wrapper.name = func.__name__
    wrapper.description = textwrap.dedent(func.__doc__).strip() if func.__doc__ else ''
    wrapper.args_schema = pyd.create_model(func.__name__, **fields)

    return wrapper


@tool
def calculator(expression: str) -> Union[float, None]:
    """
    A simple calculator tool that can evaluate basic arithmetic expressions.
    The expression must contain only the following allowed mathematical symbols:
    digits, +, -, *, /, ., (, )

    The ^ symbol, for example, is not allowed. For exponent, use **.
    In case the expression has any invalid symbol, the function returns `None`.

    Args:
        expression (str): The arithmetic expression as a string.

    Returns:
        The numerical result or `None` in case an incorrect arithmetic expression is provided
         or any other error occurs.
    """
    import re

    # Tools should be self-contained, including the imports
    # This allows their usage in an isolated environment
    # That is why, we import `re` and compute the regex inside this tool definition, not outside
    expression = expression.replace("'", "").replace('^', '**')

    # Define a regex pattern for valid mathematical expressions
    # It's important to define it inside the tool so that the function is complete by itself
    calculator_regex = re.compile(r'^[\d+\-*/().\s]+$')

    if calculator_regex.match(expression) is not None:
        try:
            # Evaluate the expression safely
            result = eval(expression)
            return result
        except Exception as e:
            print(f'calculator:: Error evaluating expression: {e}')
            return None
    else:
        print(f'calculator:: Invalid expression: {expression}')
        return None


@tool
def search_web(query: str, max_results: int = 10, show_description: bool = False) -> str:
    """
    Search the Web using DuckDuckGo. The input should be a search query.
    Use this tool when you need to answer questions about current events and general web search.
    It returns (as Markdown text) the top search results with titles, links, and optional
    descriptions.
    NOTE: The returned URLs can be visited using the `extract_as_markdown` tool to retrieve
    the contents the respective pages.

    Args:
        query: The query string.
        max_results: Maximum no. of search results (links) to return.
        show_description: If `True`, includes the description of each search result.
         Default is `False`.

    Returns:
         The search results.
    """
    import time
    import random

    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError(
            '`ddgs` was not found! Please run `pip install ddgs`.'
        ) from e

    # Note: In general, `verify` should be `True`
    # In some cases, DDGS may fail because of proxy or something else;
    # can set it to `False` but generally not recommended
    results = DDGS(verify=False).text(query, max_results=max_results)
    # DDGS throws a rate limit error
    time.sleep(random.uniform(1.5, 3.5))
    if len(results) == 0:
        return 'No results found! Try a less restrictive/shorter query.'

    if not show_description:
        # If descriptions are not needed, only return titles and links
        results = [f"[{result['title']}]({result['href']})" for result in results]
    else:
        results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
    return '## Search Results\n\n' + '\n\n'.join(results)


@tool
def download_file(url: str) -> str:
    """
    Use this tool only when `extract_as_markdown` cannot be used.
    Download a file from the Web and save it locally on the disk. This tool is to be used to
    when the goal is to download a file (binary) rather than retrieve its contents as text.

    Args:
        url: The URL pointing to the file (must be a correct URL).

    Return:
        Path to the locally saved file or error messages in case of any exception.
    """
    import os
    import requests
    import tempfile

    response = requests.get(url, timeout=20, stream=True, headers={'user-agent': 'kodeagent/0.0.1'})
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
        if os.name == 'nt':
            tmp_file_path = tmp_file_path.replace('\\', '/')
    return tmp_file_path


@tool
def extract_file_contents_as_markdown(
        url_or_file_path: str,
        scrub_links: bool = True,
        max_length: int = None
) -> str:
    """
    Always use this tool to extract the contents of HTML files (.html), PDF files (.pdf),
    Word documents (.docx), and Excel spreadsheets (.xlsx) as Markdown text. No other file type
    is supported. The input can point to a URL or a local file path.
    The extracted text can be used for analysis with LLMs.
    This tool can directly work with URLs, so no need to download the files using
    `file_download` separately.
    NOTE: The output returned by this function can be long and may involve lots of quote marks.

    Args:
        url_or_file_path: URL or Path to a .html, .pdf, .docx, or .xlsx file.
        scrub_links: Defaults to `True`, which removes all links from the extracted Markdown text.
         Set it to `False` if you want to retain the links in the text.
        max_length: If set, limit the output to the first `max_length` characters. This can be used
         to truncate the output but doing so can also lead to loss of information.

    Returns:
        The content of the file in Markdown format.
    """
    import re
    import mimetypes

    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise ImportError(
            '`markitdown` was not found! Please run `pip install markitdown[pdf,docx,xlsx]`.'
        ) from e

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(url_or_file_path.strip()).text_content

        if mimetypes.guess_type(url_or_file_path)[0] == 'application/pdf':
            # Handling (cid:NNN) occurrences in PDFs
            cid_pattern = re.compile(r'\(cid:(\d+)\)')
            matches = set(cid_pattern.findall(result))
            for cid_num in matches:
                cid_str = f'(cid:{cid_num})'
                result = result.replace(cid_str, chr(int(cid_num) + 29))

        if scrub_links:
            # Remove Markdown links [text](url)
            result = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', result)

        if max_length is not None:
            result = result[:max_length]

        return result
    except Exception as e:
        return str(e)


@tool
def search_wikipedia(query: str, max_results: Optional[int] = 3) -> str:
    """
    Search Wikipedia (only) and return the top search results as Markdown text.
    The input should be a search query. The output will contain the title, summary, and link
    to the Wikipedia page.

    Args:
        query: The search query string.
        max_results: The max. no. of search results to consider (default 3).

    Returns:
        The search results in Markdown format.
    """
    try:
        import wikipedia
    except ImportError as e:
        raise ImportError('`wikipedia` was not found! Please run `pip install wikipedia`.') from e

    try:
        results = wikipedia.search(query, results=max_results)
        if not results:
            return 'No results found! Try a less restrictive/shorter query.'

        markdown_results = []
        for title in results:
            page = wikipedia.page(title)
            markdown_results.append(f"### [{page.title}]({page.url})\n{page.summary}")

        return '\n\n'.join(markdown_results)
    except wikipedia.exceptions.DisambiguationError as de:
        return f'DisambiguationError: Please select an option from {", ".join(de.options)}'


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search for academic papers on arXiv.org. The input is a search query.
    This tool is highly specialized and should be used exclusively for
    finding scientific and academic papers. It returns the top search results
    with the title, authors, summary, and a link to the PDF.

    Args:
        query: The search query string for the paper.
        max_results: The maximum number of search results to return (default is 5).

    Returns:
        The search results in Markdown format or a message indicating no results were found.
    """
    try:
        import arxiv

        # Construct the default API client
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))

        if not results:
            return f'No results found for the query: {query}'

        output = f'## ArXiv Search Results for: {query}\n\n'
        for result in results:
            authors = ', '.join([author.name for author in result.authors])
            output += f'### [{result.title}]({result.pdf_url})\n'
            output += f'**Authors:** {authors}\n'
            output += f'**Abstract:** {result.summary}\n'
            output += f'**Published:** {result.published.strftime("%Y-%m-%d")}\n\n'

        return output

    except Exception as e:
        return f'An error occurred during the arXiv search: {str(e)}'


@tool
def get_youtube_transcript(video_id: str) -> str:
    """
    Retrieve the transcript/subtitles for YouTube videos (only). It also works for automatically
    generated subtitles, supports translating subtitles. The input should be a valid YouTube
    video ID. E.g., the URL https://www.youtube.com/watch?v=aBc4E has the video ID `aBc4E`.

    Args:
        video_id: YouTube video ID from the URL.

    Returns:
        The transcript/subtitle of the video, if available.
    """
    from youtube_transcript_api import YouTubeTranscriptApi, _errors as yt_errors

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        transcript_text = ' '.join([item.text for item in transcript.snippets])
    except yt_errors.TranscriptsDisabled:
        transcript_text = (
            '*** ERROR: Could not retrieve a transcript for the video -- subtitles appear to be'
            ' disabled for this video, so this tool cannot help, unfortunately.'
        )
    except yt_errors.NoTranscriptFound:
        return '*** ERROR: No transcript found for this video.'
    except Exception as e:
        return f'*** ERROR: YouTube transcript retrieval failed: {e}'

    return transcript_text


@tool
def get_audio_transcript(file_path: str) -> Any:
    """
    Convert audio files to text using OpenAI's Whisper model via Fireworks API.
    The input should be a path to an audio file (e.g., .mp3, .wav, .flac).
    The audio file should be in a format that Whisper supports.

    Args:
        file_path: Local file system path to the audio file.

    Returns:
        The transcript of the audio file as text.
    """
    import os

    import requests

    with open(file_path, 'rb') as f:
        response = requests.post(
            'https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {os.getenv("FIREWORKS_API_KEY")}'},
            files={'file': f},
            data={
                'model': 'whisper-v3-turbo',
                'temperature': '0',
                'vad_model': 'silero'
            },
            timeout=15,
        )

    if response.status_code == 200:
        return response.json()

    return f'Audio transcription error: {response.status_code}: {response.text}'


# The different types of senders of messages
MESSAGE_ROLES = Literal['user', 'assistant', 'system', 'tool']
# The different types of updates emitted by an agent
AGENT_RESPONSE_TYPES = Literal['step', 'final', 'log']


class Task(pyd.BaseModel):
    """
    Task to be solved by an agent.
    """
    id: str = pyd.Field(description='Auto-generated task ID', default_factory=uuid.uuid4)
    description: str = pyd.Field(description='Task description')
    files: Optional[list[str]] = pyd.Field(description='A list of file paths or URLs')
    result: Optional[Any] = pyd.Field(description='Task result', default=None)
    is_finished: bool = pyd.Field(
        description='Whether the task has finished running', default=False
    )
    is_error: bool = pyd.Field(
        description='Whether the task execution resulted in any error', default=False
    )


class PlanStep(pyd.BaseModel):
    """A single step in an agent's plan."""
    description: str = pyd.Field(description='A brief description of the step')
    is_done: bool = pyd.Field(description='Whether the step has been completed', default=False)


class AgentPlan(pyd.BaseModel):
    """A structured plan for an agent to follow."""
    steps: list[PlanStep] = pyd.Field(description='List of steps to accomplish the task')


class ObserverResponse(pyd.BaseModel):
    """
    The response from the observer after analyzing the agent's behavior.
    """
    is_progressing: bool = pyd.Field(
        description='True if the agent is making meaningful progress on the plan'
    )
    is_in_loop: bool = pyd.Field(
        description='True if the agent is stuck in a repetitive loop'
    )
    reasoning: str = pyd.Field(description='A short reason for the assessment (max 15--20 words)')
    correction_message: Optional[str] = pyd.Field(
        description='A specific, actionable feedback to help the agent self-correct'
    )


class ChatMessage(pyd.BaseModel):
    """
    Generic chat message. This is primarily intended to internal and tool usage.
    Agents shouldn't ask an LLM to respond in this format. In particular, Gemini would fail
    because of `Any`.
    """
    role: MESSAGE_ROLES = pyd.Field(description='Role of the message sender')
    content: Any = pyd.Field(description='Content of the message')


class ReActChatMessage(ChatMessage):
    """
    Messages for the ReAct agent. Do not set the `content` field.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='ALWAYS `None`', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the tool use')
    action: Optional[str] = pyd.Field(
        description='Name of the tool to use; `None` when `final_answer` is available'
    )
    # Gemini complains about empty objects if `args` is defined as dict,
    # hence string type for compatibility
    args: Optional[str] = pyd.Field(
        description='Tool arguments as JSON string; `None` when `final_answer` is available'
    )
    final_answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    task_successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class CodeChatMessage(ChatMessage):
    """
    Messages for the CodeActAgent.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='Unused', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the code')
    code: Optional[str] = pyd.Field(
        description='Python code with tool use; `None` when `final_answer` is available'
    )
    final_answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    task_successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class SupervisorTaskMessage(pyd.BaseModel):
    """
    Messages for the supervisor-agent task delegation.
    """
    agent_id: int = pyd.Field(
        description='Integer agent ID based on the instructions (starting from 0)'
    )
    task: str = pyd.Field(
        description='Task or sub-task description to be delegated to an agent'
    )
    image_files: Optional[list[str]] = pyd.Field(
        description='Optional list of image file paths/URLs associated with the task'
    )
    facts_available: str = pyd.Field(
        description='A list of objective facts collected/observed so far'
    )
    task_complete: bool = pyd.Field(
        description=(
            'Initially False; set to True only when the agent(s) have successfully competed'
            ' all the sub-tasks'
        )
    )
    final_answer: str = pyd.Field(
        description=(
            'The final answer for the user when the main task is done, i.e., `task_complete`'
            ' is True. Set to empty string otherwise.'
        )
    )


class DelegatedTaskStatus(pyd.BaseModel):
    """
    The status of a task delegated by the supervisor.
    """
    status: bool = pyd.Field(description='Either `True` or `False`')
    reason: str = pyd.Field(
        description='Brief explanation for the status, e.g., why the task is incomplete'
    )
    how_to_fix: str = pyd.Field(
        description='Briefly describe how to/what would fix the task result'
    )
    final_answer: str = pyd.Field(
        description='Final solution of the task, if found. Otherwise, empty string.'
    )


class AgentResponse(TypedDict):
    """
    Streaming response sent by an agent in the course of solving a task. The receiver can decide
    what to do with the response based on its type.
    """
    type: AGENT_RESPONSE_TYPES
    channel: Optional[str]
    value: Any
    metadata: Optional[dict[str, Any]]


async def call_llm(
        model_name: str,
        litellm_params: dict,
        messages: list[dict],
        response_format: Optional[Type[pyd.BaseModel]] = None,
        trace_id: Optional[str] = None,
) -> str | None:
    """
    Invoke the LLM to generate a response based on a given list of messages.

    Args:
        model_name: The name of the LLM to be used.
        litellm_params: Optional parameters for LiteLLM.
        messages: A list of messages (and optional images) to be sent to the LLM.
        response_format: Optional type of message the LLM should respond with.
        trace_id: (Optional) Langfuse trace ID.

    Returns:
        The LLM response as string.

    Raises:
        ValueError: If the LLM returns an empty or invalid response body.
    """
    params = {'model': model_name, 'messages': messages}

    if response_format:
        params['response_format'] = response_format

    # Add a timeout to prevent indefinite hangs
    if 'timeout' not in litellm_params:
        params['timeout'] = 30  # seconds

    params.update(litellm_params)

    try:
        # Use AsyncRetrying to handle retries in a non-blocking way
        async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_random_exponential(multiplier=1, max=10)
        ):
            with attempt:
                # Use the asynchronous litellm call
                response = await litellm.acompletion(
                    **params, metadata={'trace_id': str(trace_id)}
                )

                # Check for empty content
                response_content = response.choices[0].message.content
                if not response_content or not response_content.strip():
                    raise ValueError('LLM returned an empty or invalid response body.')

                token_usage = {
                    'cost': response._hidden_params.get('response_cost'),
                    'prompt_tokens': response.usage.get('prompt_tokens'),
                    'completion_tokens': response.usage.get('completion_tokens'),
                    'total_tokens': response.usage.get('total_tokens'),
                }
                logger.info(token_usage)
                return response_content

    except Exception as e:
        logger.exception(
            'LLM call failed after repeated attempts: %s',
            str(e), exc_info=True
        )
        print('\n\ncall_llm MESSAGES:\n', '\n'.join([str(msg) for msg in messages]), '\n\n')
        raise ValueError(
            'Failed to get a valid response from LLM after multiple retries.'
        ) from e


class Planner:
    """
    Given a task, generate and maintain a step-by-step plan to solve it.
    This class is stateful and holds the current plan.
    """
    def __init__(self, model_name: str, litellm_params: Optional[dict] = None):
        self.model_name = model_name
        self.litellm_params = litellm_params or {}
        self.plan: Optional[AgentPlan] = None

    async def create_plan(self, task: Task, agent_type: str) -> AgentPlan:
        """
        Create a plan to solve the given task and store it.
        """
        messages = ku.make_user_message(
            text_content=AGENT_PLAN_PROMPT.format(
                agent_type=agent_type,
                task=task.description,
                task_files='\n'.join(task.files) if task.files else '[None]',
            ),
            files=task.files,
        )
        plan_response = await call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=messages,
            response_format=AgentPlan,
            trace_id=task.id
        )
        self.plan = AgentPlan.model_validate_json(plan_response)
        return self.plan

    async def update_plan(self, thought: str, observation: str, task_id: str):
        """
        Update the plan based on the last thought and observation.
        """
        if not self.plan:
            return

        prompt = UPDATE_PLAN_PROMPT.format(
            plan=self.plan.model_dump_json(indent=2),
            thought=thought,
            observation=observation
        )
        plan_response = await call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=ku.make_user_message(prompt),
            response_format=AgentPlan,
            trace_id=task_id
        )
        self.plan = AgentPlan.model_validate_json(plan_response)

    def get_steps_done(self) -> list[PlanStep]:
        """Returns the completed steps from the current plan."""
        if not self.plan:
            return []
        return [step for step in self.plan.steps if step.is_done]

    def get_steps_pending(self) -> list[PlanStep]:
        """Returns the pending steps from the current plan."""
        if not self.plan:
            return []
        return [step for step in self.plan.steps if not step.is_done]

    def get_formatted_plan(self, scope: Literal['all', 'done', 'pending'] = 'all') -> str:
        """
        Convert the agent's plan into a markdown checklist.
        """
        if not self.plan or not self.plan.steps:
            return ''

        if scope == 'all':
            steps_to_format = self.plan.steps
        elif scope == 'done':
            steps_to_format = self.get_steps_done()
        else:  # pending
            steps_to_format = self.get_steps_pending()

        todo_list = []
        for step in steps_to_format:
            status = 'x' if step.is_done else ' '
            todo_list.append(f'- [{status}] {step.description}')
        return '\n'.join(todo_list)


class Observer:
    """
    Monitors an agent's behavior to detect issues like loops or stalled plans.
    If a problem is detected, it can generate a corrective message to be injected
    into the agent's history to prompt a change in behavior.
    """
    def __init__(
            self,
            model_name: str,
            tool_names: set[str],
            litellm_params: Optional[dict] = None,
            threshold: int = 3
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.tool_names = tool_names
        self.litellm_params = litellm_params or {}
        self.last_correction_iteration: int = 0

    async def observe(
            self,
            iteration: int,
            task: Task,
            history: str,
            plan_before: Optional[str | AgentPlan],
            plan_after: Optional[str | AgentPlan],
    ) -> Optional[str]:
        """
        Observe the agent's state and return a corrective message if a problem is detected.
        """
        if (iteration <= 1) or (self.threshold < 0) or (
                iteration - self.last_correction_iteration < self.threshold
        ):
            return None

        try:
            # Use the LLM to analyze the state and provide feedback
            tool_names = '\n'.join(sorted(list(self.tool_names)))
            prompt = OBSERVATION_PROMPT.format(
                task=task.description,
                plan_before=plan_before,
                plan_after=plan_after,
                history=history,
                tools=tool_names,
            )
            observation_response = await call_llm(
                model_name=self.model_name,
                litellm_params=self.litellm_params,
                messages=[{'role': 'user', 'content': prompt}],
                response_format=ObserverResponse,
            )
            # Parse the structured response
            observation = ObserverResponse.model_validate_json(observation_response)
            print(f'Observer (iteration {iteration}): {observation.model_dump_json()}')

            # Return the correction message if a problem is detected
            if not observation.is_progressing or observation.is_in_loop:
                self.last_correction_iteration = iteration
                msg = (
                        observation.correction_message or observation.reasoning
                        or 'Adjust your approach based on the plan and history.'
                )
                correction = (
                    f'!!!CRITICAL FOR COURSE CORRECTION: {msg}\n'
                )

                if self.tool_names:
                    correction += (
                        f'Here are the exact TOOL names once again for reference:\n{tool_names}'
                    )
                return correction

        except Exception as e:
            # Fallback for LLM or parsing errors
            print(f'LLM Observer failed: {e}')
            return None

        return None

    def reset(self):
        """Reset the observer state."""
        self.last_correction_iteration = 0


class Agent(ABC):
    """
    An abstract agent. This should serve as the base class for all types of agents.
    All subclasses must override at least the `run()` method.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            description: Optional[str] = None,
            tools: Optional[list[Callable]] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            filter_tools_for_task: bool = False,
    ):
        """
        Initialize an agent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used.
            tools: A list of tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: Maximum number of iterations for task solving.
            filter_tools_for_task: Whether to filter tools based on task relevance.
        """
        self.id = uuid.uuid4()
        self.name: str = name
        self.description = description
        self.model_name: str = model_name

        self.tools = tools or []
        self.filter_tools_for_task = filter_tools_for_task
        self.litellm_params: dict = litellm_params or {}
        self.max_iterations = max_iterations

        self.tool_names = {t.name for t in tools} if tools else set()
        self.tool_name_to_func = {t.name: t for t in tools} if tools else {}

        self.task: Optional[Task] = None
        self.messages: list[ChatMessage] = []
        self.msg_idx_of_new_task: int = 0
        self.final_answer_found = False
        self.planner: Optional[Planner] = None
        self.observer = Observer(
            model_name=model_name,
            litellm_params=litellm_params,
            tool_names=self.tool_names
        )

        self.is_visual_model: bool = llm_vision_support([model_name])[0] or False

    def __str__(self):
        return (
            f'Agent: {self.name} ({self.id}); LLM: {self.model_name}; Tools: {self.tools}'
        )


    @property
    def current_plan(self) -> Optional[str]:
        """Returns the current plan for the task."""
        if not self.planner or not self.planner.plan:
            return None
        return self.planner.get_formatted_plan()

    # async def get_history_summary(self) -> str:
    #     """
    #     Generate a summary of the conversation history.
    #     """
    #     history = self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task)
    #     if not history.strip():
    #         return "No activities yet."
    #
    #     prompt = CONTEXTUAL_SUMMARY_PROMPT.format(
    #         task=self.task.description,
    #         history=history
    #     )
    #     summary = await call_llm(
    #         model_name=self.model_name,
    #         litellm_params=self.litellm_params,
    #         messages=ku.make_user_message(prompt),
    #         trace_id=self.task.id if self.task else None
    #     )
    #     return summary

    async def get_relevant_tools(
            self,
            task_description: str,
            task_files: Optional[list[str]] = None,
    ) -> list[Any]:
        """
        Calls an LLM to determine which tools are relevant for the given task.

        Args:
            task_description: The task description.
            task_files: Optional list of files associated with the task.

        Returns:
            A list of relevant tools or all tools, in case of error.
        """
        tool_descriptions = self.get_tools_description()
        prompt = RELEVANT_TOOLS_PROMPT.format(
            task_description=task_description,
            task_files=task_files,
            tool_descriptions=tool_descriptions,
        )

        try:
            tools_response = await call_llm(
                model_name=self.model_name,
                litellm_params=self.litellm_params,
                messages=ku.make_user_message(prompt),
                trace_id=self.task.id if self.task else None
            )
            relevant_tool_names = tools_response.split(',') if tools_response.strip() else []
            relevant_tool_names = {t.strip() for t in relevant_tool_names if t.strip()}
            logger.debug('Relevant tool names: %s', relevant_tool_names)
            relevant_tools = [
                t for t in self.tools if t.name in relevant_tool_names
            ]
            return relevant_tools
        except Exception as e:
            logger.error('Error determining relevant tools: %s', str(e))
            return list(self.tools)

    def _run_init(
            self,
            description: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ):
        """
        Initialize the running of a task by an agent.
        """
        # self.add_to_history(ChatMessage(role='user', content=description))
        self.task = Task(description=description, files=files)
        if task_id:
            self.task.id = task_id
        # self.msg_idx_of_new_task = len(self.messages)
        self.final_answer_found = False
        if self.planner:
            self.planner.plan = None
        self.observer.reset()

    async def salvage_response(self) -> str:
        """
        When an agent fails to find an answer in the stipulated number of steps, this method
        can be called to salvage what little information could be gathered.

        Returns:
            A response from the LLM based on the task and the history.
        """
        prompt = SALVAGE_RESPONSE_PROMPT.format(
            task=self.task.description,
            task_files='\n'.join(self.task.files) if self.task.files else '[None]',
            history=self.get_history(start_idx=self.msg_idx_of_new_task)
        )
        salvaged_response = await call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=ku.make_user_message(prompt),
            trace_id=self.task.id
        )
        return salvaged_response

    def trace(self) -> str:
        """
        Provide a trace of the agent's activities for the current task.
        The trace can be used for debugging.

        Returns:
            A string trace of the agent's thoughts, actions, and observations.
        """
        trace_log = []
        for msg in self.messages[self.msg_idx_of_new_task:]:
            if isinstance(msg, ReActChatMessage):
                trace_log.append(f"Thought: {msg.thought}")
                if msg.action:
                    trace_log.append(f"Action: {msg.action}({msg.args})")
            elif isinstance(msg, CodeChatMessage):
                trace_log.append(f"Thought: {msg.thought}")
                if msg.code:
                    trace_log.append(f"Code:\n{msg.code}")
            elif msg.role == 'tool':
                trace_log.append(f"Observation: {msg.content}")
        return "\n".join(trace_log)

    @abstractmethod
    async def run(
            self,
            task: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Execute a task using the agent.

        Args:
            task: A description of the task.
            files: An optional list of file paths or URLs.
            task_id: (Optional) An ID for the task, if provided by the caller.

        Yields:
            An update from the agent.
        """


    def response(
            self,
            rtype: AGENT_RESPONSE_TYPES,
            value: Any,
            channel: Optional[str] = None,
            metadata: Optional[dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Prepare a response to be sent by the agent. The calling method must yield this response.

        Note: `response` is not made a static method so that only the agents can invoke it.

        Args:
            rtype: The type of the response.
            value: The response value (content).
            channel: Optional channel (e.g., the method name that generated this response).
            metadata: Optional metadata.

        Returns:
            The agent's response.
        """
        return {'type': rtype, 'channel': channel, 'value': value, 'metadata': metadata}

    async def _chat(self, response_format: Optional[Type[pyd.BaseModel]] = None) -> str:
        """
        Interact with the LLM using the agent's message history, which consists of messages with
        different roles: user, assistant, system, and tool. Use the `call_llm` function.
        """
        formatted_messages = []
        last_tool_call_id = None
        for msg in self.messages:
            d = msg.model_dump()
            role = d.get('role')
            # Handle assistant tool call (ReActChatMessage with action/tool use)
            if role == 'assistant' and (
                hasattr(msg, 'action') and getattr(msg, 'action', None)
            ) and not getattr(msg, 'final_answer', None):
                # This is a tool call, not a final answer
                tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                last_tool_call_id = tool_call_id
                formatted_messages.append({
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': getattr(msg, 'action'),
                                'arguments': getattr(msg, 'args')
                            }
                        }
                    ]
                })
            elif role == 'tool':
                # Attach tool_call_id if available
                tool_msg = {'role': 'tool', 'content': str(d.get('content', ''))}
                if last_tool_call_id:
                    tool_msg['tool_call_id'] = last_tool_call_id
                    last_tool_call_id = None
                formatted_messages.append(tool_msg)
            elif role == 'assistant':
                # Final answer or normal assistant message
                # If final_answer is present, use it as content
                if hasattr(msg, 'final_answer') and getattr(msg, 'final_answer', None):
                    formatted_messages.append({'role': 'assistant', 'content': getattr(msg, 'final_answer')})
                else:
                    formatted_messages.append({'role': 'assistant', 'content': d.get('content', '')})
            else:
                # user/system
                formatted_messages.append({'role': role, 'content': d.get('content', '')})

        print('\n\nCHAT MESSAGES:\n', '\n'.join([str(msg) for msg in formatted_messages]), '\n\n')
        chat_response: str = await call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=formatted_messages,
            response_format=response_format,
            trace_id=self.task.id if self.task else None
        )

        return chat_response or ''


    def add_to_history(self, message: ChatMessage):
        """
        Add a chat message, generated by user, AI, or tool, to the agent's message history.

        Args:
            message: The message. Must be a valid `ChatMessage` instance.
        """
        assert isinstance(message, ChatMessage), (
            f'add_to_history() expects a `ChatMessage`; got `{type(message)}`'
        )
        self.messages.append(message)

    def format_messages_for_prompt(self, start_idx: int = 0) -> str:
        """
        Generate a formatted string based on the historical messages that can be injected
        into a prompt. The formatting may differ based on the prompts used by different types
        of agents. Subclasses should override this method accordingly.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
            A formatted string containing the messages.
        """
        return self.get_history(start_idx)

    def get_tools_description(self, tools: Optional[list[Any]] = None) -> str:
        """
        Generate a description of all the tools available to the agent.

        Args:
            tools: Optional list of tools to describe. If not provided, uses the agent's tools.

        Returns:
            A description of the requested or all available tools.
        """
        description = ''
        filtered_tool_names = {t.name for t in (tools or self.tools)}
        for t in self.tools:
            if t.name in filtered_tool_names:
                description += f'- Tool name: {t.name}'
                # description += f'\n  -
                # Schema: {t.args_schema.model_json_schema()}'
                description += f'\n- Tool description: {t.description}'
                description += '\n---\n'

        return description

    @property
    def purpose(self) -> str:
        """
        Describe the name, purpose of, and tools available to an agent.

        Returns:
             A text description of the agent.
        """
        description = f'Name: {self.name}\nDescription: {self.description or "N/A"}'
        description += f'\nTools available to this agent (`{self.name}`):'
        description += f'\n{self.get_tools_description()}'

        return description

    def get_history(self, start_idx: int = 0) -> str:
        """
        Get a formatted string representation of all the messages.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
            A sequence of the messages showing their role and content.
        """
        return '\n'.join([f'[{msg.role}]: {msg.content}' for msg in self.messages[start_idx:]])

    def clear_history(self):
        """
        Clear the agent's message history.
        """
        self.messages = []


class ReActAgent(Agent):
    """
    Reasoning and Acting agent with thought-action-observation loop.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            tools: list,
            description: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            filter_tools_for_task: bool = False,
    ):
        """
        Instantiate a ReAct agent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
        """
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            litellm_params=litellm_params,
            description=description,
            max_iterations=max_iterations,
            filter_tools_for_task=filter_tools_for_task,
        )

        self.planner = Planner(model_name, litellm_params)
        if tools:
            logger.info('Created agent: %s; tools: %s', name, [t.name for t in tools])

    async def _update_plan(self):
        """
        Update the plan based on the last thought and observation.
        """
        last_thought = ''
        last_observation = ''
        if len(self.messages) > 1:
            if isinstance(self.messages[-2], (ReActChatMessage, CodeChatMessage)):
                last_thought = self.messages[-2].thought
            if self.messages[-1].role == 'tool':
                last_observation = self.messages[-1].content
        await self.planner.update_plan(
            thought=last_thought,
            observation=last_observation,
            task_id=self.task.id,
        )

    async def run(
            self,
            task: str,
            files: Optional[list[str]] = None,
            task_id: Optional[str] = None,
            summarize_progress_on_failure: bool = True,
    ) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using ReAct's TAO loop.

        Args:
            task: A description of the task.
            files: An optional list of file paths or URLs.
            task_id: (Optional) An ID for the task, if provided by the caller.
            summarize_progress_on_failure: Whether to summarize the progress made on task failure.

        Yields:
            An update from the agent.
        """
        self._run_init(task, files, task_id)
        self.clear_history()
        self.add_to_history(
            ChatMessage(
                role='system',
                content=REACT_SYSTEM_PROMPT.format(tools=self.get_tools_description())
            )
        )

        yield self.response(
            rtype='log',
            value=f'Solving task: `{self.task.description}`',
            channel='run'
        )

        await self.planner.create_plan(self.task, self.__class__.__name__)
        yield self.response(rtype='log', value=f'Plan:\n{self.planner.plan}', channel='run')

        user_message = f'Task: {self.task.description}\n\nPlan:\n{self.current_plan}'
        self.add_to_history(ChatMessage(role='user', content=user_message))

        for idx in range(self.max_iterations):
            yield self.response(rtype='log', channel='run', value=f'* Executing step {idx + 1}')
            # The thought & observation will get appended to the list of messages
            async for update in self._think():
                yield update
            async for update in self._act():
                yield update

            if self.final_answer_found:
                break

            plan_before_update = None
            if self.planner.plan:
                plan_before_update = self.current_plan
                await self._update_plan()

            # The observer checks for issues and suggests corrections
            correction_msg = await self.observer.observe(
                task=self.task,
                history='\n'.join([str(msg) for msg in self.messages]),
                plan_before=plan_before_update,
                plan_after=self.current_plan,
                iteration=idx + 1
            )
            if correction_msg:
                self.add_to_history(
                    ChatMessage(role='user', content=f'Observation: {correction_msg}')
                )
                yield self.response(rtype='log', value=correction_msg, channel='observer')
            print('-' * 30)

        if not self.final_answer_found:
            failure_msg = f'Sorry, I failed to get a complete answer even after {idx + 1} steps!'

            if summarize_progress_on_failure:
                progress_summary = await self.salvage_response()
                failure_msg += (
                    f'\n\nHere\'s a summary of my progress for this task:\n{progress_summary}'
                )

            yield self.response(
                rtype='final',
                value=failure_msg,
                channel='run',
                metadata={'final_answer_found': False}
            )
            # trace_info = self.trace()
            # if trace_info:
            #     print(trace_info)

            self.add_to_history(ChatMessage(role='assistant', content=failure_msg))
        else:
            # Update the plan one last time after the final answer is found
            if self.planner.plan:
                await self._update_plan()

    async def _think(self) -> AsyncIterator[AgentResponse]:
        """
        Think about the next step to be taken to solve the given task.

        The LLM is prompted with the available tools and the TAO sequence so far. Based on them,
        the LLM will suggest the next action. "Think" of ReAct is also "Observe."

        Yields:
            Update from the think step.
        """
        if self.filter_tools_for_task:
            relevant_tools = await self.get_relevant_tools(
                task_description=self.task.description, task_files=self.task.files
            )
        else:
            relevant_tools = self.tools

        # Preparing the messages for the LLM call
        # task_msg = (
        #     '## Current Task\n'
        #     f'{self.task.description}\n\n'
        #     '## Plan\n'
        #     'Optionally, here is a general plan to follow (ignore if unavailable).'
        #     ' Steps already completed are marked with "x".'
        #     f'{self.current_plan or "<No plan provided; please plan yourself>"}\n\n'
        #     '## History of Thoughts, Actions, and Observations\n'
        #     f'{self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task)}'
        # )
        thought = await self._record_thought(ReActChatMessage)
        yield self.response(rtype='step', value=thought, channel='_think')

    async def _record_thought(
            self,
            response_format_class: Type[ChatMessage]
    ) -> Optional[ChatMessage]:
        """
        Utility method covering the common aspects of the "think" step of the T*O loop.

        Args:
            response_format_class: The type of message used by this agent.

        Returns:
            A message of the `response_format_class` type.
        """
        # Sometimes the LLM does not generate a valid JSON response based on the given format
        # class. It returns a plain text response instead, which leads to a validation error.
        # To handle this, we will retry the LLM call up to 3 times,
        # attempting to parse the response as JSON each time.
        for attempt in range(3):
            try:
                # Call the LLM with the prompt and response format class
                # thought_response = await call_llm(
                #     model_name=self.model_name,
                #     litellm_params=self.litellm_params,
                #     messages=messages,
                #     response_format=response_format_class,
                #     trace_id=self.task.id,
                # )
                print('\n\nBEFORE CALLING thought:')
                print('\n'.join([str(msg) for msg in self.messages]))
                thought_response: str = await self._chat(response_format=response_format_class)
                print(f'{thought_response=}')
                try:
                    json.loads(thought_response)
                except JSONDecodeError:
                    thought_response = json_repair.repair_json(thought_response)

                msg: response_format_class = response_format_class.model_validate_json(
                    thought_response
                )
                msg.role = 'assistant'
                msg.content = msg.final_answer
                self.add_to_history(msg)
                print('\n\n\n\n\nHISTORY SO FAR:\n')
                print('\n'.join([str(msg) for msg in self.messages]))
                return msg

            except Exception as ex:
                print(f'>>> Error in _record_thought(): {ex}')
                # This can happen if the LLM response is not valid JSON
                logger.error('LLM response validation error in _record_thought(). Retrying...')
                await asyncio.sleep(random.uniform(0.5, 1.0))

                if attempt == 0:
                    # Add an explicit observation to the prompt's message history
                    # Add timestamp to avoid potential cached responses
                    feedback_message = (
                        '!Error: Parsing failed because you did not generate the response'
                        ' following the given JSON schema!!! Please ensure your response is'
                        ' a valid JSON object that follows the specified schema.'
                        f' [Timestamp={datetime.now()}]'
                    )
                    self.messages.append(ChatMessage(role='user', content=feedback_message))
                continue

        return None

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Take action based on the agent's previous thought.

        The LLM has suggested an action. This method will identify the tool suggested and
        execute it.

        Yields:
            Updates from the acting step.
        """
        prev_msg: ReActChatMessage = self.messages[-1]  # type: ignore
        if (
                hasattr(prev_msg, 'final_answer') and prev_msg.final_answer == ''
                and (prev_msg.action == '' or prev_msg.args == '' or prev_msg.thought == '')
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields. Please respond strictly'
                        ' following the ReActChatMessage schema.'
                    )
                )
            )
            return

        if hasattr(prev_msg, 'final_answer') and prev_msg.final_answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.task_successful
            # Do NOT add another ChatMessage to history here; it's already added in _record_thought
            yield self.response(
                rtype='final',
                value=prev_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.task_successful}
            )
        else:
            # No answer yet, keep tool calling
            try:
                tool_name, tool_args = prev_msg.action, prev_msg.args
                tool_args = tool_args.strip().strip('`').strip()
                if tool_args.startswith('json'):
                    tool_args = tool_args[4:].strip()

                try:
                    tool_args = json.loads(tool_args)
                except JSONDecodeError:
                    tool_args = json_repair.loads(tool_args)

                if tool_name in self.tool_names:
                    result = self.tool_name_to_func[tool_name](**tool_args)
                    self.add_to_history(ChatMessage(role='tool', content=result))
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'tool': tool_name, 'args': tool_args}
                    )
                else:
                    result = (
                        f'Incorrect tool name generated: {tool_name}!'
                        ' Please suggest a correct tool name from the provided list.'
                    )
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'is_error': True}
                    )

            except Exception as ex:
                error_msg = f'*** An error occurred while taking the suggested action: {ex}'
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True}
                )

    # def format_messages_for_prompt(self, start_idx: int = 0) -> str:
    #     """
    #     Generate a formatted string based on the historical messages for the ReAct agent.
    #
    #     Args:
    #         start_idx: The start index of messages to consider (default 0).
    #
    #     Returns:
    #         A formatted string containing the messages.
    #     """
    #     history = ''
    #
    #     for msg in self.messages[start_idx:]:
    #         if msg.role == 'assistant' and isinstance(msg, ReActChatMessage):
    #             history += f'Thought: {msg.thought}\n'
    #             history += f'Action: {msg.action}\n'
    #             history += f'Args: {msg.args}\n'
    #         elif msg.role == 'tool':
    #             history += f'Observation: {msg.content}\n\n'
    #
    #     return history


# The environments where LLM-generated code can be executed
CODE_ENV_NAMES = Literal['host', 'docker', 'e2b']


class CodeRunner:
    """
    Run Python code generated by an LLM in a given environment.
    """
    def __init__(
            self,
            env: CODE_ENV_NAMES,
            allowed_imports: list[str],
            pip_packages: Optional[str] = None,
            timeout: int = 30,
            env_vars_to_set: Optional[dict[str, str]] = None
    ):
        """
        Create an environment to run Python code.

        Args:
            env: The code execution environment. Must be a string from `CODE_ENV_NAMES`.
            allowed_imports: A list of Python modules that are allowed to be imported.
            pip_packages: Optional Python libs to be installed by `pip` [E2B].
            timeout: Code execution timeout (default 30s).
            env_vars_to_set: Optional environment variables to set in the code execution
             environment (E2B only).
        """
        self.allowed_imports: set[str] = set(allowed_imports)
        self.env: CODE_ENV_NAMES = env
        self.pip_packages: list[str] = re.split('[,;]', pip_packages) if pip_packages else []
        self.default_timeout = timeout
        self.local_modules_to_copy = ['kutils.py']
        self.pip_packages_str = ' '.join(self.pip_packages)
        self.env_vars_to_set = env_vars_to_set

    def check_imports(self, code) -> set[Union[str]]:
        """
        Check whether there is any module imported in a given source code outside the allowed
        Python modules.

        Args:
            code: The source code to scan.

        Returns:
            A (possibly empty) set of module names that are disallowed.
        """
        tree = ast.parse(code)
        imported_modules = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module)

        # Find any disallowed imports
        disallowed = imported_modules - self.allowed_imports

        return disallowed

    def run(self, source_code: str) -> tuple[str, str, int]:
        """
        Run Python code in a pre-specified environment.
        Do not return the stdout or stderr as `None`, since that would get converted to the string
        'None'. Instead, set them to empty strings when required.

        Args:
            source_code: The Python code to run.

        Returns:
            The stdout, stderr, and the process return code (0 if no error).
        """
        try:
            ast.parse(source_code)
        except SyntaxError as se:
            return (
                '',
                f'Code parsing failed due to: {type(se).__name__}\n{se.text}\nError: {str(se)}',
                -1
            )

        disallowed_imports: set = self.check_imports(source_code)
        if len(disallowed_imports) > 0:
            return (
                '',
                f'The following imports are disallowed: {disallowed_imports}'
                f'\nPlease only use the allowed modules for importing.',
                -1
            )

        if self.env == 'host':
            warnings.warn(
                'You are running LLM-generated code on your host. This could be potentially'
                ' dangerous! Please consider using a different code runner environment.',
                UserWarning
            )
            with tempfile.NamedTemporaryFile(
                    mode='w+t', suffix='.py', delete=False, encoding='utf-8'
            ) as code_file:
                code_file.write(source_code)
                code_file.close()  # Close the file before execution

                # Copy the local dependency modules
                for a_file in self.local_modules_to_copy:
                    shutil.copy2(
                        os.path.join(os.path.dirname(__file__), a_file),
                        tempfile.gettempdir()
                    )

                result = sp.run(
                    [sys.executable, code_file.name],
                    shell=False, capture_output=True, text=True,
                    timeout=self.default_timeout,
                    check=False,
                    encoding='utf-8'
                )
                os.remove(code_file.name)
                return result.stdout, result.stderr, result.returncode

        elif self.env =='e2b':
            # Run the code on an E2B sandbox
            try:
                import e2b_code_interpreter as e2b
            except ModuleNotFoundError:
                logger.critical(
                    'The module `e2b_code_interpreter` was not found. Please install E2B as:'
                    ' `pip install e2b-code-interpreter`\nExecution will halt now.'
                )
                sys.exit(-1)

            running_sandboxes = e2b.Sandbox.list()
            logger.info('%d E2B sandboxes are running', len(running_sandboxes))
            if running_sandboxes:
                sbx = e2b.Sandbox.connect(running_sandboxes[0].sandbox_id)
            else:
                sbx = e2b.Sandbox(
                    timeout=self.default_timeout + 15,
                    envs=self.env_vars_to_set or {},
                )
                if self.pip_packages_str:
                    sbx.commands.run(f'pip install {self.pip_packages_str}')

            # Copy the local dependency modules
            for a_file in self.local_modules_to_copy:
                with open(
                        os.path.join(os.path.dirname(__file__), a_file),
                        'r',
                        encoding='utf-8'
                ) as py_file:
                    sbx.files.write(f'/home/user/{a_file}', py_file.read())
                    logger.info('Copied file %s...', a_file)

            logger.info('E2B sandbox info: %s', sbx.get_info())
            execution = sbx.run_code(code=source_code, timeout=self.default_timeout)
            std_out: str = '\n'.join(execution.logs.stdout)
            std_err: str = '\n'.join(execution.logs.stderr)
            ret_code: int = -1 if execution.error else 0
            return std_out, std_err, ret_code

        else:
            raise ValueError(f'Unsupported code execution env: {self.env}')


class CodeActAgent(ReActAgent):
    """
    CodeAct is somewhat like ReAct but uses the Thought-Code-Observation loop rather than
    the Thought-Action-Observation loop. In the TCO loop, Python code is written to invoke
    tools, print & capture the results, and observe the results.

    CodeActAgent will retain most of the functionality from ReActAgent. Only the prompt formatting,
    `_think(), and the `_act()` steps will change.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            run_env: CODE_ENV_NAMES,
            tools: Optional[list[Callable]] = None,
            description: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            allowed_imports: Optional[list[str]] = None,
            pip_packages: Optional[str] = None,
            timeout: int = 30,
            env_vars_to_set: Optional[dict[str, str]] = None,
            filter_tools_for_task: bool = False,
    ):
        """
        Instantiate a CodeActAgent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            run_env: The code execution environment. `host` means code will be run on the system
             where you create this agent. `e2b` means code will be run on an E2B sandbox. You will
             need an E2B API key.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
            allowed_imports: A list of Python modules that the agent is allowed to import.
            pip_packages: Optional Python libs to be installed with `pip` [for E2B].
            timeout: Code execution timeout (default 30s).
            env_vars_to_set: Optional environment variables to set in the code execution.
        """
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            litellm_params=litellm_params,
            max_iterations=max_iterations,
            description=description,
            filter_tools_for_task=filter_tools_for_task,
        )
        # Combine the source code of all tools into one place
        # TODO Somehow dynamically identify and include the modules used by the tools
        self.tools_source_code: str = 'from typing import *\n\nimport kutils as ku\n\n'

        if tools:
            for t in self.tools:
                self.tools_source_code += inspect.getsource(t).replace('@tool\n', '', 1) + '\n'

        self.pip_packages = pip_packages

        if not allowed_imports:
            allowed_imports = []

        # The following imports are allowed by default
        self.allowed_imports = allowed_imports + ['datetime', 'typing', 'mimetypes']
        self.code_runner = CodeRunner(
            env=run_env,
            allowed_imports=self.allowed_imports + ['kutils'],
            pip_packages=pip_packages,
            timeout=timeout,
            env_vars_to_set=env_vars_to_set,
        )

    def format_messages_for_prompt(self, start_idx: int = 0) -> str:
        """
        Generate a formatted string based on the historical messages for the ReAct agent.

        Args:
            start_idx: The start index of messages to consider (default 0).

        Returns:
             A formatted string containing the messages.
        """
        history = ''

        for msg in self.messages[start_idx:]:
            if msg.role == 'assistant' and isinstance(msg, CodeChatMessage):
                history += f'Thought: {msg.thought}\n'
                code = msg.code.strip()
                if not code.startswith('```py'):
                    code = f'```py\n{code}'
                if not code.endswith('```'):
                    code = f'{code}\n```'
                history += f'Code:{code}\n'
            elif msg.role == 'tool':
                history += f'Observation: {msg.content}\n\n'

        return history

    async def _think(self) -> AsyncIterator[AgentResponse]:
        """
        Think about the next step to be taken to solve the given task.

        The LLM is prompted with the available tools and the TCO sequence so far. Based on them,
        the LLM will suggest the next action/code.

        Yields:
            Update from the thing step.
        """
        if self.filter_tools_for_task:
            relevant_tools = await self.get_relevant_tools(
                task_description=self.task.description, task_files=self.task.files
            )
        else:
            relevant_tools = self.tools

        think_prompt = CODE_ACT_AGENT_PROMPT.format(
            task=self.task.description,
            task_files='\n'.join(self.task.files) if self.task.files else '[None]',
            tool_names=self.get_tools_description(relevant_tools),
            authorized_imports=','.join(self.allowed_imports),
            plan=self.current_plan or '[No plan provided; please plan yourself]',
            history=self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task),
        )
        msg = await self._record_thought(CodeChatMessage)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Code action based on CodeActAgent's previous thought.

        The LLM has suggested code. This method will run the code.

        Yields:
            Updates from the acting step.
        """
        prev_msg: CodeChatMessage = self.messages[-1]  # type: ignore

        if (
                not hasattr(prev_msg, 'final_answer') or (
                    not prev_msg.final_answer and (not prev_msg.code or not prev_msg.thought)
                )
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields. Please respond strictly'
                        ' following the CodeChatMessage schema.'
                    )
                )
            )
            return

        if hasattr(prev_msg, 'final_answer') and prev_msg.final_answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.task_successful
            # Do NOT add another ChatMessage to history here; it's already added in _record_thought
            yield self.response(
                rtype='final',
                value=prev_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.task_successful}
            )
        else:
            # No answer yet, keep tool calling
            try:
                code = prev_msg.code.strip()
                code = code.replace('```py', '')
                code = code.replace('```', '').strip()
                code = f'{self.tools_source_code}\n\n{code}'

                stdout, stderr, exit_status = self.code_runner.run(code)
                observation = f'{stdout}\n{stderr}'.strip()
                msg = ChatMessage(role='tool', content=observation)
                self.add_to_history(msg)
                yield self.response(
                    rtype='step',
                    value=observation,
                    channel='_act',
                    metadata={'is_error': exit_status != 0}
                )

            except Exception as ex:
                error_msg = f'*** An error occurred while running the code: {ex}'
                yield self.response(
                    rtype='step',
                    value=error_msg,
                    channel='_act',
                    metadata={'is_error': True}
                )


def llm_vision_support(model_names: list[str]) -> list[bool]:
    """
    Utility function to check whether images can be used with given LLMs.

    Args:
        model_names: A list of LLM names.

    Returns:
        A list of booleans, containing `True` or `False` for each model.
    """
    status = [litellm.supports_vision(model=model) for model in model_names]
    for model, value in zip(model_names, status):
        print(f'- Vision supported by {model}: {value}')

    return status


def print_response(response: AgentResponse, only_final: bool = True):
    """
    A utility function to print agent's response in a terminal, optionally with colors.

    Args:
        response: A response obtained from an agent.
        only_final: If `True`, only print the final response from the agent. Otherwise, print
         all responses, including intermediate steps and logs.
         
    """

    if response['type'] == 'final':
        msg = (
            response['value'].content
            if isinstance(response['value'], ChatMessage) else response['value']
        )
        rich.print(f'[blue][bold]Agent[/bold]: {msg}[/blue]\n')

    if not only_final:
        if response['type'] == 'log':
            rich.print(f'[white]{response}[/white]')
        else:
            rich.print(f'{response}')


async def main():
    """
    Demonstrate the use of ReActAgent and CodeActAgent.
    """
    litellm_params = {'temperature': 0, 'timeout': 30}
    model_name = 'gemini/gemini-2.0-flash-lite'

    # response = await call_llm(
    #     model_name=model_name,
    #     litellm_params=litellm_params,
    #     messages=[
    #         {'role': 'system',
    #          'content': 'You are an expert, helpful agent designed to solve complex tasks using a set of specialized tools.\n\n\n## Core Directives\n\n1.  **Process:** Your process is a strict `Thought` -> `Action` -> `Observation` cycle. Repeat this cycle until the task is complete.\n2.  **Progression:** Each step must meaningfully advance the task. Never repeat a failed attempt without a clear modification. Do not call the same tool with the same arguments twice.\n3.  **Final Answer:** When the task is complete, you must provide the final answer using the structured format below.\n\n\nStrictly adhere to the following Thought-Action-Observation (TAO) cycle:\nThought: Based on the current task status, what is the next logical step to take?\nAction: tool name (one of aforementioned tool names)\nArgs: the input arguments to the tool, in a JSON format representing the kwargs (e.g. {"input": "hello world", "num_beams": 5})\n\n\n## Final Answer Format\n\n* **For a successful task:**\n\n`Thought: I have enough information to answer. I will use the user\'s language.`\n`Answer: <Your final answer in the user\'s language>`\n`Successful: True`\n\n* **For a failed task:**\n\n`Thought: I cannot answer the question with the provided tools/information.`\n`Answer: <Your explanation in the user\'s language>`\n`Successful: False`\n\n\n## Examples\n\nHere are some examples of good and bad practices to help you understand the expected format and behavior using **notional** tasks and tools.\n\n---\n[Task: What color is the sky in this picture? (Image: camera.jpg)]\nThought: The user has provided an image file and wants to know the color of the sky. I have the image directly available to me. Since this is a basic visual analysis task, I will use my innate abilities to answer instead of using a tool.\nAnswer: The sky in the picture is blue.\nSuccessful: True\n\n---\n[Task: Which city has the highest population: Guangzhou or Shanghai?]\n\nThought: The current language of the user is: English. I need to get the populations for both cities and compare them. I will start with Guangzhou and use the tool `web_search`.\nAction: web_search\nArgs: {"query": "Guangzhou population"}\nObservation: [\'Guangzhou has a population of 15 million inhabitants as of 2021.\']\n\nThought: I have the population for Guangzhou. Now I need to find the population of Shanghai using `web_search`.\nAction: web_search\nArgs: {"query": "Shanghai population"}\nObservation: [\'Shanghai population 26 million (2019)\']\n\nThought: Based on the search results from the previous steps, I know that Shanghai has a population of 26 million and Guangzhou has 15 million. So Shanghai has the highest population.\nAnswer: Based on the search results, Shanghai has the highest population.\nSuccessful: True\n\n---\n[Task: Generate a video of the moon.]\n\nThought: The user has asked to generate a video of the moon. Unfortunately, I neither have the innate ability nor any tool that can generate a video. So, I can\'t solve this task.\nAnswer: Unfortunately, I lack the ability to solve this task. May I help you with something else?\nSuccessful: False\n\n---\n[Task: Generate an image of the oldest person in this document.]\n\nThought: The current language of the user is: English. I will begin by identifying the oldest person mentioned in the document using the `document_qa tool`. I only need to print the answer, not the entire document.\nAction: document_qa\nArgs: {"question": "Who is the oldest person mentioned?"}\nObservation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.\n\nThought: Based on the given document, John Doe (55) is the oldest person. He lives in Newfoundland, Canada. As my next logical step, I\'ll use the `image_generator` tool to generate his portrait.\nAction: image_generator\nArgs: {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}\nObservation: image.png\n\nThought: Based on the given document, John Doe (55) is the oldest person. I have also generated his portrait and saved it in the image.png file.\nAnswer: An image of the oldest person has been generated and saved as `image.png`.\nSuccessful: True\n\n\n## Tools\n\nYou have access to the following specialized tools:\n- Tool name: calculator\n- Tool description: A simple calculator tool that can evaluate basic arithmetic expressions.\nThe expression must contain only the following allowed mathematical symbols:\ndigits, +, -, *, /, ., (, )\n\nThe ^ symbol, for example, is not allowed. For exponent, use **.\nIn case the expression has any invalid symbol, the function returns `None`.\n\nArgs:\n    expression (str): The arithmetic expression as a string.\n\nReturns:\n    The numerical result or `None` in case an incorrect arithmetic expression is provided\n     or any other error occurs.\n---\n- Tool name: search_web\n- Tool description: Search the Web using DuckDuckGo. The input should be a search query.\nUse this tool when you need to answer questions about current events and general web search.\nIt returns (as Markdown text) the top search results with titles, links, and optional\ndescriptions.\nNOTE: The returned URLs can be visited using the `extract_as_markdown` tool to retrieve\nthe contents the respective pages.\n\nArgs:\n    query: The query string.\n    max_results: Maximum no. of search results (links) to return.\n    show_description: If `True`, includes the description of each search result.\n     Default is `False`.\n\nReturns:\n     The search results.\n---\n- Tool name: extract_file_contents_as_markdown\n- Tool description: Always use this tool to extract the contents of HTML files (.html), PDF files (.pdf),\nWord documents (.docx), and Excel spreadsheets (.xlsx) as Markdown text. No other file type\nis supported. The input can point to a URL or a local file path.\nThe extracted text can be used for analysis with LLMs.\nThis tool can directly work with URLs, so no need to download the files using\n`file_download` separately.\nNOTE: The output returned by this function can be long and may involve lots of quote marks.\n\nArgs:\n    url_or_file_path: URL or Path to a .html, .pdf, .docx, or .xlsx file.\n    scrub_links: Defaults to `True`, which removes all links from the extracted Markdown text.\n     Set it to `False` if you want to retain the links in the text.\n    max_length: If set, limit the output to the first `max_length` characters. This can be used\n     to truncate the output but doing so can also lead to loss of information.\n\nReturns:\n    The content of the file in Markdown format.\n---\n\n\n\n## Guidelines & Constraints\n\n- Always generate a Thought-Action sequence.\n- Use tools only when needed and only those listed.\n- Always use the correct arguments for tools.\n'},
    #         {'role': 'user',
    #          'content': 'Task: What is four plus seven? Also, what are the festivals in Paris? How they differ from Kolkata?\n\nPlan:\n- [ ] Calculate the sum of four and seven.\n- [ ] Identify the festivals celebrated in Paris.\n- [ ] Identify the festivals celebrated in Kolkata.\n- [ ] Compare the festivals in Paris and Kolkata, highlighting their differences.'},
    #         {'role': 'assistant', 'content': None, 'tool_calls': [
    #             {'id': 'call_6b5e8ec0', 'type': 'function',
    #              'function': {'name': 'calculator', 'arguments': '{"expression": "4 + 7"}'}}]},
    #         {'role': 'tool', 'content': '11', 'tool_call_id': 'call_6b5e8ec0'}
    #     ]
    # )
    # print(response)
    # return

    agent = ReActAgent(
        name='Simple agent',
        model_name=model_name,
        tools=[calculator, search_web, extract_file_contents_as_markdown,],
        max_iterations=5,
        litellm_params=litellm_params,
        filter_tools_for_task=False
    )
    # agent = CodeActAgent(
    #     name='Web agent',
    #     model_name=model_name,
    #     tools=[
    #         search_web, search_arxiv, extract_file_contents_as_markdown, download_file,
    #         get_youtube_transcript,
    #     ],
    #     run_env='host',
    #     max_iterations=6,
    #     litellm_params=litellm_params,
    #     allowed_imports=[
    #         'os', 're', 'time', 'random', 'requests', 'tempfile',
    #         'ddgs', 'markitdown', 'youtube_transcript_api', 'arxiv',
    #     ],
    #     pip_packages='ddgs~=9.5.2;"markitdown[all]";',
    #     filter_tools_for_task=False
    # )

    the_tasks = [
        ('What is ten plus 15, raised to 2, expressed in words?', None),
        ('What is the date today? Express it in words.', None),
        (
            'Which image has a purple background?',
            [
                'https://www.slideteam.net/media/catalog/product/cache/1280x720/p/r/process_of_natural_language_processing_training_ppt_slide01.jpg',
                'https://cdn.prod.website-files.com/61a05ff14c09ecacc06eec05/66e8522cbe3d357b8434826a_ai-agents.jpg',
            ]
        ),
        (
            'What is four plus seven? Also, what are the festivals in Paris?'
            ' How they differ from Kolkata?',
            None
        ),
        (
            'Summarize the notes',
            ['https://web.stanford.edu/class/cs102/lectureslides/ClassificationSlides.pdf',]
        ),
    ]


    print('Agent demo\n')
    for task, img_urls in the_tasks[3:4]:
        rich.print(f'[yellow][bold]User[/bold]: {task}[/yellow]')
        async for response in agent.run(task, files=img_urls):
            print_response(response)

        if agent.current_plan:
            print(f'Plan:\n{agent.current_plan}')

        await asyncio.sleep(random.uniform(0.15, 0.55))
        print('\n\n')


if __name__ == '__main__':
    # For Windows; in case of Unicode error with PDF extraction
    os.environ['PYTHONUTF8'] = '1'

    asyncio.run(main())
