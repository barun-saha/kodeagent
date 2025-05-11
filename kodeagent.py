"""
A minimalistic approach to building AI agents.
Implements ReAct and CodeAgent. Supports multi-agent via SupervisorAgent.
"""
import ast
import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess as sp
import sys
import tempfile
import textwrap
import uuid
import warnings
from abc import ABC, abstractmethod
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

import kutils as ku


load_dotenv()

warnings.simplefilter('once', UserWarning)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get a logger for the current module
logger = logging.getLogger('KodeAgent')

# litellm._turn_on_debug()


REACT_PROMPT = '''
You are an expert assistant, helpful and polite, who can solve any task using tool calls. 
Given a task, you think about how to solve it, suggest a tool to use to solve the current step,
observe the outcome of the current action, and then think again.

## Task

The task description is as follows:
{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Tools

You have access to a set of tools. You can use one or more of these tools in any sequence
you deem appropriate to complete the task at hand. This may require breaking the task into subtasks
and using different tools to complete each subtask.

The following tools are available to you:
{tool_names}


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of aforementioned tool names) if using a tool.
Args: the input arguments to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers.
You may use code markers within your response if you need to.
Please use a valid JSON format for the Args. E.g., do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

Observation: tool response


You should keep repeating the above format (Thought-Action-Observation cycle) till you have enough
information to answer the question without using any more tools.
At that point, you MUST respond in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
Successful: True

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
Successful: False


The `Successful` flag is set to `False` in the second case since the task was failed to be solved.
This flag should be always False until you reach the final step and decide that the task is complete.


## Example Conversations

Below, a few sample conversations using notional tools are provided for your reference.
Please study the patterns carefully.

---
[Sample task: Generate an image of the oldest person in this document.]

Thought: I will begin by identifying the oldest person mentioned in the document. I will use the `document_qa` tool for this purpose.
Action: document_qa
Args: {{"document": "document.pdf", "question": "Who is the oldest person mentioned?"}}
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on document search, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. Now, I'll use the `image_generator` tool to generate his portrait.  
Action: image_generator
Args: {{"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}}
Observation: image.png

Thought: Based on the given document, I have identified John Doe (55) as the oldest person. I have also generated his portrait and saved it in the `image.png` file.
Answer: image.png
Successful: True

---
[Sample task: What is the result of the following operation: 5 + 3 + 1294.678?]

Thought: This is an arithmetic problem. I will use the `calculator` tool to compute the sum.
Action: calculator
Args: {{"expression": "5 + 3 + 1294.678"}}
Observation: 1302.678

Thought: Using the `calculator` tool, the sum of the given numbers is 1302.678.
Answer: 1302.678
Successful: True

---
[Sample task: Generate a video of the moon.]

Thought: The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False


## Additional Instructions:
- Call a tool only when needed, e.g., do not call the search agent if you do not need to search information.
- Do not use non-existent tools. Only use a tool listed earlier. 
- Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
- Never re-do a tool call that you previously did with the exact same parameters.
- Do your best! Don't give up! You're in charge of solving the task, not providing directions to solve it.


## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages (initially empty).

{history}
'''

CODE_AGENT_PROMPT = '''
You are an expert assistant, helpful and polite, who can solve any task using code blobs. 
Given a task, you think about how to solve it, use tools (i.e., user-defined Python functions)
to solve the steps, observe the outcomes, and then think again until a final answer is found.


## Task

The task description is as follows:
{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Tools

You have access to a wide variety of tools. You are responsible for writing Python code and using
the tools in any sequence you deem appropriate to complete the task at hand. This may require
breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_names}


## Allowed Imports

You can use the aforementioned tools to solve a task. In addition, you are allowed to import (only)
the following 
standard Python libraries in the code your write (`*` means you can import any lib):
{authorized_imports}

(You do NOT need to import the tool names -- they are already available to you).


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Code: ```py
Python code that uses one or more aforementioned tool
print(useful information)
```

The `Code` presents simple Python code to solve the step. Your code can use `print()` to save
any important information. These print outputs from code execution will then become available
as `Observation` as input for the next step. The code should be enclosed within triple backticks.

Please ALWAYS start with a Thought.
If this format is used, you will be responded back in the following format:

Observation: tool use response


You should keep repeating the above format (Thought-Code-Observation cycle) till you have enough
information to answer the question without using any more tools.
At that point, you MUST respond in one of the following two formats:

Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
Successful: True

Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
Successful: False


The `Successful` flag is set to `False` in the second case since the task was failed to be solved.
This flag should be always False until you reach the final step and decide that the task is complete.


## Example Conversations

Below, a few sample conversations using notional tools are provided for your reference.
Please study the patterns carefully.

---
[Sample task: Generate an image of the oldest person in this document.]

Thought: I will begin by identifying the oldest person mentioned in the document. I will use the `document_qa` tool for this purpose.
Code: ```py
answer = document_qa(document=document, question='Who is the oldest person mentioned?')
print(answer)
```
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on document search, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. Now, I'll use the `image_generator` tool to generate his portrait.  
Code: ```py
image_path = image_generator(prompt="A portrait of John Doe, a 55-year-old man living in Canada.")
print(f'The output image file is: {{image_path}}')
```
Observation: The output image file is: image.png

Thought: Based on the given document, I have identified John Doe (55) as the oldest person. I have also generated his portrait and saved it in the `image.png` file.
Answer: image.png
Successful: True

---
[Sample task: Which city has the highest population: Guangzhou or Shanghai?]

Thought: I need to get the populations for both cities and compare them: I will use the tool `search` to get the population of both cities.
Code: ```py
for city in ['Guangzhou', 'Shanghai']:
    print(f'Population {{city}}:', search(f'{{city}} population')
```
Observation: Population Guangzhou: ['Guangzhou has a population of 15 million inhabitants as of 2021.']
Population Shanghai: '26 million (2019)'

Thought: Based on the search results, I know that Shanghai has the highest population.
Answer: Shanghai has the highest population.
Successful: True

---
[Sample task: Generate a video of the moon.]

Thought: The user has asked to generate a video of the moon. Unfortunately, I do not have any tool that can generate a video. So, I can't solve this task.
Answer: Unfortunately, I lack the ability to solve this task at this moment. May I help you with something else?
Successful: False


## Additional Instructions:
- ALWAYS generate a Thought-Code sequence.
- In `Code`, only use the variables that you have defined.
- Don't name any new variable with the same name as a tool.
- Call a tool only when needed. Only use a tool listed in the Tools section.
- Always use the right arguments for the tools. Do not pass the arguments as a dict.
  E.g., rather than `answer = wiki({{'query': 'search term'}})`, use `answer = wiki(query='search term')`.
- Avoid having multiple tool calls in the same code when their output format is unpredictable,
  e.g., a search function. In such scenarios, output the results of tool calls with `print()`
  and use those results in the code section.
- Write simple Python code. Avoid writing functions on your own unless it's important.
- Remember to import any required (and allowed) Python module before using them. Also, `Code` is stateless.
  So, the required imports (variables, and your functions, if any) must be mentioned again when needed. 
- Do not print any secrets, e.g., API keys, passwords, and tokens.
- Do your best! Don't give up! You're in charge of solving the task, not providing directions to solve it.


## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages (initially empty).

{history}
'''

SUPERVISOR_PROMPT = '''
You are the supervisor of an AI agency having one or more helpful agents. Given a task, as well as
the capabilities of the agents, you decide which agent(s) and tool(s) of the agent to use to solve
the task. A given task can be complex -- you may need to split it into smaller parts and invoke
different agents to solve each part using their respective tools. In other words, your job is to
efficiently delegate tasks or subtasks to the agents, collect the results, and delegate again until
a final, satisfactory task completion result is found. You should do it carefully without getting
stuck in the same loop.

Important: Carefully read the original given task. When delegating tasks to the agents or when you
need to split into sub-tasks, remember to retain ALL information from the original task.
Even punctuations matter sometimes! Otherwise, you might get stuck in an infinite loop where you
ask an agent something but get a different thing in response. The specifications, expectations,
and responses need to be in sync.

Also, tool usage is efficient, so accept the results obtained by using tools of the agents. Unless,
the results returned indicate some obvious error, in which case you ask the agent again by rephrasing
its task along with your feedback.


## Task

{task}

(Optional) input file paths/URLs associated with this task are as follows:
{task_files}


## Agents

The following agents are available to you, each identified with a unique integer ID starting from 0:
{agents}


## Task Completion

In case you find that the main task and sub-tasks have been successfully completed, generate
a final answer for the user by nicely collating the results of all the sub-tasks. Also, set
`task_complete` to True.
 
 
## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages (initially empty).
Also, agent's response are depicted as user's response.

{history}
'''

SUPERVISOR_TASK_CHECK_PROMPT = '''
Given this task:
{task}

and this sequence of response by agents tool usage:
{response}

determine whether or not the task has been successfully completed.

In case the given attempts appear to effectively capture the final result but fall short only
in some minor way, e.g., not properly formatted, you can give a finishing touch and capture
the result of the task in the `final_answer` field. In this case, also set `status` to `True`.
Otherwise, leave `final_answer` empty when there are significant aspects missing or major deviations
noted from the desired ask result.

Tool usage is generally efficient, so in most cases the results can be accepted unless there is some
indication of obvious error.
'''

SALVATION_PROMPT = '''
You are a helpful AI agency having one or more agents/assistants. You help users by solving their
tasks. Sometimes, due to unpredictable reasons, you might fail to solve the task entirely or
partially. Also, sometimes, you might have completed the task but failed to communicate the final
answer to the user due to some error.

You are here today to address one such scenario.

Given the following task:
{task}

and optional files associated with the task:
{task_files}

Here's a log of what you have done and achieved:
{history}

In the conversation history above, you will find the original task of the user and optionally
delegated sub-tasks.

Your job is to salvage any useful information/output/action related to user's task from the above
sequence of activities. Here's how to respond:
- If you find that the agent/assistant have completed the task satisfactorily (unless there is any
obvious error message or significant deviation from what the task had asked to achieve), simply
generate a final response based on what is available.
- If you notice that one more steps or sub-tasks remain unachieved or failed, begin by apologising.
Identify the useful information available and prepare a final response. After that, display
a bulleted list of what aspects of the task could not be achieved or failed or encountered error. 
- If no portion of the task could be competed, say so and begin by apologising. Then show
a bulleted list of what went wrong. (Skip this part if the task was successful.)

Aside from minor formatting and presenting to the users in a readable way, avoid adding
to the results/facts already found, only report them.
Avoid telling users terms like "logs", "history", and "accepted by you" when responding.
Also, users need to have all information available to them -- avoid telling them to see
agent's previous attempts or tool's previous outputs. 

'''


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
    A simple calculator tool that can evaluate arithmetic expressions.
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
def web_search(query: str) -> str:
    """
    Search the Web for a given query using DuckDuckGo.
    This tool returns the titles of relevant Web page, their URLs, and a short text snippet.
    NOTE: The returned URLs should be visited to retrieve the contents the pages.

    Args:
        query: The query string.

    Returns:
         The search results.
    """
    import time
    import random

    try:
        from duckduckgo_search import DDGS
    except ImportError as e:
        raise ImportError(
            '`duckduckgo_search` was not found! Please run `pip install duckduckgo-search`.'
        ) from e

    # Note: In general, `verify` should be `True`
    # In some cases, DDGS may fail because of proxy or something else;
    # can set it to `False` but generally not recommended
    results = DDGS(verify=True).text(query, max_results=10)
    # DDGS throws a rate limit error
    time.sleep(random.uniform(0.2, 1.2))
    if len(results) == 0:
        return 'No results found! Try a less restrictive/shorter query.'

    results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
    return '## Search Results\n\n' + '\n\n'.join(results)


@tool
def visit_webpage(urls: list[str]) -> str:
    """
    Visit a list of webpages at the given URLs and get their content together as a markdown string.

    Args:
        urls: A list of Web page URLs to visit.

    Returns:
        The combined content of the pages.
    """
    import re
    import requests

    from markdownify import markdownify

    text = ''

    for url in urls:
        try:
            response = requests.get(url, timeout=20, headers={'user-agent': 'my-app/0.0.1'})
            response.raise_for_status()
            if response.status_code == 200:
                md_content = markdownify(response.text).strip()
                md_content = re.sub(r'\n{3,}', '\n\n', md_content)
                if len(md_content) > 8192:
                    text += (
                            md_content[:8192] +
                            '\n..._Content truncated to stay below 8192 characters_...\n'
                    )
        except Exception as e:
            text += f'\nAn error occurred while reading this {url}: {e}\n'

    return text


class Task(pyd.BaseModel):
    """
    Task to be solved by an agent.
    """
    id: str = pyd.Field(description='Auto-generated task ID', default_factory=uuid.uuid4)
    task: str = pyd.Field(description='Task description')
    image_files: Optional[list[str]] = pyd.Field(description='A list of image file paths or URLs')
    result: Optional[Any] = pyd.Field(description='Task result', default=None)
    is_finished: bool = pyd.Field(
        description='Whether the task has finished running', default=False
    )
    is_error: bool = pyd.Field(
        description='Whether the task execution resulted in any error', default=False
    )


# The different types of senders of messages
MESSAGE_ROLES = Literal['user', 'assistant', 'system', 'tool']


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
    Messages for the ReAct agent.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='Unused', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the tool use')
    action: str = pyd.Field(description='Name of the tool to use')
    # Gemini complains about empty objects if `args` is defined as dict,
    # hence string type for compatibility
    args: str = pyd.Field(description='Tool arguments as JSON string')
    answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class CodeChatMessage(ChatMessage):
    """
    Messages for the CodeAgent.
    """
    # The content field will not be used by this message (but the LLM can still assign a value)
    # Higher versions of Pydantic allows to exclude the field altogether
    content: Optional[str] = pyd.Field(description='Unused', exclude=True)
    thought: str = pyd.Field(description='Thoughts behind the code')
    code: str = pyd.Field(description='Python code with tool use')
    answer: Optional[str] = pyd.Field(
        description='Final answer for the task; set only in the final step', default=None
    )
    successful: bool = pyd.Field(description='Task completed or failed? (initially False)')


class SupervisorMessage(pyd.BaseModel):
    """
    Messages for the supervisor-agent interaction.
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
    how_to_fix: str = pyd.Field(description='Briefly describe how to/what would fix the task result')
    final_answer: str = pyd.Field(
        description='Final solution of the task, if found. Otherwise, empty string.'
    )


# The different types of updates emitted by an agent
AGENT_RESPONSE_TYPES = Literal['step', 'final', 'log']


class AgentResponse(TypedDict):
    """
    Streaming response sent by an agent in the course of solving a task. The receiver can decide
    what to do with the response based on its type.
    """
    type: AGENT_RESPONSE_TYPES
    channel: Optional[str]
    value: Any
    metadata: Optional[dict[str, Any]]


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
            vision_model_name: Optional[str] = None,
            tools: Optional[list[Callable]] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
    ):
        """
        Initialize an agent.

        Args:
            name: The name of the agent.
            description: Description of the agent's capabilities or scope. Recommended to have.
            model_name: The name of the LLM to be used.
            vision_model_name: (Optional) vision model to use; None by default.
            tools: A list of tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
        """
        self.id = uuid.uuid4()
        self.name: str = name
        self.description = description
        self.model_name: str = model_name
        self.vision_model_name = vision_model_name or model_name

        self.tools = tools
        self.litellm_params: dict = litellm_params or {}
        self.max_iterations = max_iterations

        self.tool_names = set([t.name for t in tools]) if tools else set([])
        self.tool_name_to_func = {
            t.name: t for t in tools
        } if tools else {}

        self.task: Optional[Task] = None
        self.image_files: Optional[list[str]] = None
        self.messages: list[ChatMessage] = []
        self.msg_idx_of_new_task: int = 0

    def __str__(self):
        """
        A string representation of the agent.
        """
        return (
            f'Agent: {self.name} ({self.id}); LLM: {self.model_name}; Tools: {self.tools}'
        )

    @abstractmethod
    async def run(
            self,
            task: str,
            image_files: Optional[list[str]] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Execute a task using the agent. All subclasses must override this method to solve tasks
        in their own way. The method should yield an `AgentResponse`.

        Args:
            task: A description of the task.
            image_files: An optional list of image file paths or URLs.

        Yields:
            An update from the agent.
        """

    async def _call_llm(
            self,
            messages: list[dict],
            response_format: Optional[Type[ChatMessage | SupervisorMessage | DelegatedTaskStatus]] = None
    ) -> str:
        """
        Invoke the LLM to generate a response based on a given list of messages.

        Args:
            messages: A list of messages (and optional images) to be sent to the LLM.
            response_format: The type of message to respond with.

        Returns:
            The LLM response as string.
        """
        params = {
            'model': self.model_name,
            'messages': messages,
        }
        if response_format:
            params['response_format'] = response_format
        params.update(self.litellm_params)
        response = litellm.completion(**params)

        return response.choices[0].message['content']

    def _create_task(self, description: str, image_files: Optional[list[str]] = None):
        """
        Create a new task to be solved by the agent.

        Args:
            description: The task description.
            image_files: An optional list of image file paths or URLs.
        """
        self.task = Task(task=description, image_files=image_files)
        self.image_files = image_files
        # Since `messages` stores every message generated while interacting with the agent,
        # we need to know which all messages correspond to the current task
        # (so that the ones pertaining to the previous tasks can be ignored)
        self.msg_idx_of_new_task = len(self.messages)

    def _run_init(self, task: str, image_files: Optional[list[str]] = None):
        """
        Initialize the running of a task by an agent.
        """
        self.add_to_history(ChatMessage(role='user', content=task))
        self._create_task(description=task, image_files=image_files)
        self.final_answer_found = False  # Reset from any previous task

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

    def get_tools_description(self) -> str:
        """
        Generate a description of all the tools available to the agent.

        Returns:
            A description of the available tools.
        """
        description = ''
        for t in self.tools:
            description += f'Tool name: {t.name}'
            # description += f'\n  * Schema: {t.args_schema.model_json_schema()}'
            description += f'\nTool description: {t.description}'
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
            vision_model_name: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
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
            vision_model_name=vision_model_name,
            litellm_params=litellm_params,
            description=description,
            max_iterations=max_iterations,
        )

        self.final_answer_found: bool = False
        logger.info('Created agent: %s; tools: %s', name, [t.name for t in tools])

    async def run(
            self,
            task: str,
            image_files: Optional[list[str]] = None
    ) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using ReAct's TAO loop.

        Args:
            task: A description of the task.
            image_files: An optional list of image file paths or URLs.

        Yields:
            An update from the agent.
        """
        self._run_init(task, image_files)

        yield self.response(rtype='log', value=f'Solving task: `{self.task.task}`', channel='run')
        for idx in range(self.max_iterations):
            if self.final_answer_found:
                break

            yield self.response(rtype='log', channel='run', value=f'* Executing step {idx + 1}')
            # The thought & observation will get appended to the list of messages
            async for update in self._think():
                yield update
            async for update in self._act():
                yield update
            print('-' * 30)

        if not self.final_answer_found:
            yield self.response(
                rtype='final',
                value=(
                    f'Sorry, I failed to get a complete answer'
                    f' even after {self.max_iterations} steps!'
                ),
                channel='run'
            )

    async def _think(self) -> AsyncIterator[AgentResponse]:
        """
        Think about the next step to be taken to solve the given task.

        The LLM is prompted with the available tools and the TAO sequence so far. Based on them,
        the LLM will suggest the next action. "Think" of ReAct is also "Observe."

        Yields:
            Update from the thing step.
        """
        # Note: we're not going to chat with the LLM by sending a sequence of messages
        # Instead, every think step will send a single message containing all historical info
        message = REACT_PROMPT.format(
            task=self.task.task,
            task_files='\n'.join(self.task.image_files) if self.task.image_files else '',
            history=self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task),
            tool_names=self.get_tools_description(),
        )
        msg = await self._record_thought(message, ReActChatMessage)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _record_thought(
            self,
            message: str,
            response_format_class: Type[ChatMessage]
    ):
        """
        Utility method covering the common aspects of the "think" step of the T*O loop.

        Args:
            message: A single, formatted message with history to be sent to the LLM.
            response_format_class: The type of message used by this agent.

        Returns:
            A message of the `response_format_class` type.
        """
        response = await self._call_llm(
            messages=ku.make_user_message(text_content=message, image_files=self.task.image_files),
            response_format=response_format_class
        )

        msg: response_format_class = response_format_class.model_validate_json(response)
        msg.role = 'assistant'
        self.add_to_history(msg)
        return msg

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
                prev_msg.answer == ''
                and (prev_msg.action == '' or prev_msg.args == '' or prev_msg.thought == '')
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields.'
                    )
                )
            )
            return

        if prev_msg.answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.successful
            response_msg = ChatMessage(role='assistant', content=prev_msg.answer)
            self.add_to_history(response_msg)

            yield self.response(
                rtype='final',
                value=response_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.successful}
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
                    tool_args = json_repair.json_repair.loads(tool_args)

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
            if msg.role == 'assistant' and isinstance(msg, ReActChatMessage):
                    history += f'Thought: {msg.thought}\n'
                    history += f'Action: {msg.action}\n'
                    history += f'Args: {msg.args}\n'
            elif msg.role == 'tool':
                history += f'Observation: {msg.content}\n\n'

        return history


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
            pip_packages: Optional[str] = None
    ):
        """
        Create an environment to run Python code.

        Args:
            env: The code execution environment. Must be a string from `CODE_ENV_NAMES`.
            allowed_imports: A list of Python modules that are allowed to be imported.
            pip_packages: Optional Python libs to be installed by `pip` [E2B].
        """
        self.allowed_imports: set[str] = set(allowed_imports)
        self.env: CODE_ENV_NAMES = env
        self.pip_packages: list[str] = re.split('[,;]', pip_packages) if pip_packages else []
        self.default_timeout = 15
        self.local_modules_to_copy = ['kutils.py']
        self.pip_packages_str = ' '.join(self.pip_packages)

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
            with tempfile.NamedTemporaryFile(mode='w+t', suffix='.py', delete=False) as code_file:
                code_file.write(source_code)
                code_file.close()  # Close the file before execution

                # Copy the local dependency modules
                for a_file in self.local_modules_to_copy:
                    shutil.copy2(a_file, tempfile.gettempdir())

                result = sp.run(
                    [sys.executable, code_file.name],
                    shell=False, capture_output=True, text=True,
                    timeout=self.default_timeout
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
                sbx = e2b.Sandbox(timeout=30)
                sbx.commands.run(f'pip install {self.pip_packages_str}')

            # Copy the local dependency modules
            for a_file in self.local_modules_to_copy:
                with open(a_file, 'r', encoding='utf-8') as py_file:
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


class CodeAgent(ReActAgent):
    """
    CodeAgent is somewhat like ReAct but uses the Thought-Code-Observation loop rather than
    the Thought-Action-Observation loop. In the TCO loop, Python code is written to invoke
    tools, print & capture the results, and observe the results.

    CodeAgent will retain most of the functionality from ReActAgent. Only the prompt formatting,
    `_think(), and the `_act()` steps will change.
    """
    def __init__(
            self,
            name: str,
            model_name: str,
            run_env: CODE_ENV_NAMES,
            tools: Optional[list[Callable]] = None,
            description: Optional[str] = None,
            vision_model_name: Optional[str] = None,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            allowed_imports: Optional[list[str]] = None,
            pip_packages: Optional[str] = None,
            copy_from_env: bool = False,
    ):
        """
        Instantiate a CodeAgent.

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
            copy_from_env: Whether to copy the keys from the local .env file.
        """
        super().__init__(
            name=name,
            model_name=model_name,
            tools=tools,
            vision_model_name=vision_model_name,
            litellm_params=litellm_params,
            max_iterations=max_iterations,
            description=description,
        )

        # Combine the source code of all tools into one place
        # TODO Somehow dynamically identify and include the modules used by the tools
        self.tools_source_code: str = 'from typing import *\n\nimport kutils as ku\n\n'

        for t in self.tools:
            self.tools_source_code += inspect.getsource(t).strip('@tool') + '\n'

        self.pip_packages = pip_packages

        if not allowed_imports:
            allowed_imports = []

        # The following imports are allowed by default
        self.allowed_imports = allowed_imports + ['datetime', 'typing']
        self.code_runner = CodeRunner(
            env=run_env,
            allowed_imports=self.allowed_imports + ['kutils'],
            pip_packages=pip_packages
        )
        self.copy_from_env = copy_from_env

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
        message = CODE_AGENT_PROMPT.format(
            task=self.task.task,
            task_files='\n'.join(self.task.image_files) if self.task.image_files else '',
            history=self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task),
            tool_names=self.get_tools_description(),
            authorized_imports=','.join(self.allowed_imports),
        )
        msg = await self._record_thought(message, CodeChatMessage)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Code action based on CodeAgent's previous thought.

        The LLM has suggested code. This method will run the code.

        Yields:
            Updates from the acting step.
        """
        prev_msg: CodeChatMessage = self.messages[-1]  # type: ignore

        if (
                not prev_msg.answer and (not prev_msg.code or not prev_msg.thought)
        ):
            self.add_to_history(
                ChatMessage(
                    role='tool',
                    content=(
                        '* Error: incorrect response generated. Must have values for the `answer`'
                        ' or the `action`, `args`, and `thought` fields.'
                    )
                )
            )
            return

        if prev_msg.answer:
            # The final answer has been found!
            self.final_answer_found = True
            self.task.is_finished = True
            self.task.is_error = prev_msg.successful
            response_msg = ChatMessage(role='assistant', content=prev_msg.answer)
            self.add_to_history(response_msg)

            yield self.response(
                rtype='final',
                value=response_msg,
                channel='_act',
                metadata={'final_answer_found': prev_msg.successful}
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


class SupervisorAgent(Agent):
    """
    A supervising agency, consisting of multiple agents, which can solve tasks via delegations.
    """
    def __init__(self, model_name: str, agents: list[Agent], name: str, max_iterations: int = 20):
        """
        Create a supervisor who delegates tasks to a list of agents.

        Args:
            model_name: The name of the LLM to be used.
            agents: A list of agents available to the supervisor.
            name: The name of the supervisor agent.
            max_iterations: The max no. of iterations/attempted delegations made by the supervisor.
        """
        super().__init__(name=name, model_name=model_name, max_iterations=max_iterations)
        self.agents = agents

    async def run(
            self,
            task: str,
            image_files: Optional[list[str]] = None,
    ) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using the supervisor agent/agency.

        Args:
            task: A description of the task.
            image_files: An optional list of image file paths or URLs.

        Yields:
            An update from the agency.
        """
        self._run_init(task, image_files)

        agents_desc = ''
        for idx, agent in enumerate(self.agents):
            agents_desc += f'Agent# {idx}\n{agent.purpose}\n'

        for _ in range(self.max_iterations):
            prompt = SUPERVISOR_PROMPT.format(
                task=task,
                task_files=image_files,
                agents=agents_desc,
                history=self.get_history(),
            ).strip()
            update = await self._call_llm(
                ku.make_user_message(text_content=prompt), SupervisorMessage
            )
            task_delegation_msg: SupervisorMessage = SupervisorMessage.model_validate_json(update)

            if task_delegation_msg.task_complete:
                yield AgentResponse(
                    type='final',
                    channel='supervisor',
                    value=task_delegation_msg.final_answer,
                    metadata=None
                )
                self.add_to_history(
                    message=ChatMessage(
                        role='assistant',
                        content=task_delegation_msg.final_answer
                    )
                )
                return
            else:
                self.add_to_history(
                    message=ChatMessage(
                        role='assistant',
                        content=(
                            f'Delegating to Agent# {task_delegation_msg.agent_id} //'
                            f' Sub-task: {task_delegation_msg.task} //'
                            f' Files: {task_delegation_msg.image_files}'
                        )
                    )
                )

            updates: list[str] = []
            tools_evidence: list[tuple[str, dict]] = []

            async for update in self.agents[task_delegation_msg.agent_id].run(  # type: AgentResponse
                    task=task_delegation_msg.task,
                    image_files=task_delegation_msg.image_files
            ):
                yield update

                if update['type'] == 'final':
                    yield AgentResponse(
                        type='step',
                        channel='supervisor',
                        value=update['value'],
                        metadata=None
                    )
                    updates = [update['value']]
                elif update['type'] == 'step':
                    updates.append(update['value'])
                    metadata = update['metadata']
                    if metadata:
                        tools_evidence.append(
                            (metadata.get('tool', None), metadata.get('args', None))
                        )

            updates = [
                f'Agent\'s attempt {idx}: {v.content if isinstance(v, ChatMessage) else v}'
                for idx, v in enumerate(updates, start=1)
            ]
            evidence = '\n'.join(updates)
            tools_evidence: list[tuple[str, dict]] = [(t, a) for (t, a) in tools_evidence if t]
            evidence += '\n\nTool usage evidence by the agent:\n'
            evidence += '\n'.join([f'Tool: {t} // args: {a}' for (t, a) in tools_evidence])
            status = await self._check_if_task_done(task_delegation_msg.task, evidence)
            if status.status:
                self.add_to_history(
                    message=ChatMessage(
                        role='user',
                        content=(
                            f'I can accept the result: `{status.final_answer}`'
                            '\nYou can proceed to the next subtask, if any.'
                        )
                    )
                )
            else:
                feedback = (
                    f'\nThe result does not look good: {status.final_answer}'
                    f'## Why the task result is incomplete:\n{status.reason}'
                    f'\n## Here\'s what can be done to fix it:\n{status.how_to_fix}'
                )
                self.add_to_history(
                    message=ChatMessage(role='user', content='\n'.join(updates) + feedback)
                )
        # END of supervisor loop
        # The supervisor has exhausted all attempts, but a final answer was not found/returned
        async for update in self._salvage_response():
            yield update

    async def _check_if_task_done(self, task: str, evidence: str) -> DelegatedTaskStatus:
        """
        Check if a task delegated to an agent has reached completion.

        Args:
            task: Delegated task description.
            evidence: Evidence (the steps performed so far).

        Return:
             The delegated task status.
        """
        prompt = SUPERVISOR_TASK_CHECK_PROMPT.format(task=task, response=evidence)
        response = await self._call_llm(
            ku.make_user_message(prompt, None), DelegatedTaskStatus
        )

        return DelegatedTaskStatus.model_validate_json(response)

    async def _salvage_response(self):
        """
        The supervisor has failed to return an answer in stipulated number of steps. This is
        a final result to save face and try salvage what little information could be!
        """
        prompt = SALVATION_PROMPT.format(
            task=self.task,
            task_files=self.image_files,
            history=self.get_history()
        )
        response = await self._call_llm(ku.make_user_message(prompt, None))
        yield AgentResponse(
            type='final',
            channel='supervisor',
            value=response,
            metadata={'salvage': True}
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


def print_response(response: AgentResponse):
    """
    A utility function to print agent's response in a terminal with colors.

    Args:
        response: A response obtained from an agent.
    """
    if response['type'] == 'final':
        msg = (
            response['value'].content
            if isinstance(response['value'], ChatMessage) else response['value']
        )
        rich.print(f'[blue][bold]Agent[/bold]: {msg}[/blue]\n')
    elif response['type'] == 'log':
        rich.print(f'[white]{response}[/white]')
    else:
        rich.print(f'{response}')


async def main():
    """
    Demonstrate the use of ReActAgent and CodeAgent.
    """
    litellm_params = {'temperature': 0}
    model_name = 'gemini/gemini-2.0-flash-lite'
    # model_name = 'azure/gpt-4o'

    agent1 = ReActAgent(
        name='Maths agent',
        model_name=model_name,
        tools=[calculator],
        max_iterations=3,
        litellm_params=litellm_params
    )
    agent2 = CodeAgent(
        name='Web agent',
        model_name=model_name,
        tools=[web_search, visit_webpage],
        run_env='e2b',
        max_iterations=5,
        litellm_params=litellm_params,
        allowed_imports=['os', 're', 'time', 'random', 'requests', 'duckduckgo_search', 'markdownify'],
        pip_packages='duckduckgo_search~=8.0.1;markdownify~=1.1.0',
    )
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
        )
    ]

    print('CodeAgent demo\n')
    for task, img_urls in the_tasks[:-1]:
        rich.print(f'[yellow][bold]User[/bold]: {task}[/yellow]')
        async for response in agent2.run(task, image_files=img_urls):
            print_response(response)

    print('\n\nMulti-agent with supervisor demo\n')

    agency = SupervisorAgent(
        name='Supervisor',
        model_name=model_name,
        agents=[agent1, agent2],
        max_iterations=3
    )
    task = the_tasks[-1]
    rich.print(f'[yellow][bold]User[/bold]: {task[0]}[/yellow]')

    async for response in agency.run(*task):
        print_response(response)


if __name__ == '__main__':
    asyncio.run(main())
