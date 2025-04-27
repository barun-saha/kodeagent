"""
A minimalist agentic framework. Implements ReAct and CodeAgent.
"""
import ast
import asyncio
import inspect
import json
import os
import subprocess as sp
import tempfile
import textwrap
import uuid
import warnings
from abc import ABC, abstractmethod
from functools import wraps
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

import litellm
import pydantic as pyd
import rich
from dotenv import load_dotenv


load_dotenv()
warnings.simplefilter('once', UserWarning)
# litellm._turn_on_debug()


REACT_PROMPT = '''
You are an expert assistant, helpful and polite, who can solve any task using tool calls. 
Given a task, you think about how to solve it, suggest a tool to use to solve the current step,
observe the outcome of the current action, and then think again.

## Task
{task}


## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence
you deem appropriate to complete the task at hand. This may require breaking the task into subtasks
and using different tools to complete each subtask.

You have access to the following tools:
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
- Only use a tool listed earlier. Do not use non-existent tools.
- Always use the right arguments for the tools. Never use variable names as the action arguments, use the value instead.
- Call a tool only when needed, e.g., do not call the search agent if you do not need to search information.
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
{task}


## Allowed Imports

You are allowed to import only the following Python modules in the code your write (`*` means you can import any module):
{authorized_imports}


## Tools

You have access to a wide variety of tools. You are responsible for writing Python code and using
the tools in any sequence you deem appropriate to complete the task at hand. This may require
breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_names}


## Output Format

Please answer in the same language as the question and use the following format:

Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Code: ```py
Python code that uses one or more aforementioned tool
print(useful information)
```<end_code>

The `Code` presents simple Python code to solve the step. Your code can use `print()` to save
any important information. These print outputs from code execution will then become available
as `Observation` as input for the next step. The code must end with `<end_code>` sequence.

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
```<end_code>
Observation: The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland.

Thought: Based on document search, I have identified John Doe, aged 55, as the oldest person. He lives in Newfoundland, Canada. Now, I'll use the `image_generator` tool to generate his portrait.  
Code: ```py
image_path = image_generator(prompt="A portrait of John Doe, a 55-year-old man living in Canada.")
print(f'The output image file is: {{image_path}}')
```<end_code>
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
```<end_code>
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
def get_weather(location: str) -> str:
    """Call to get the weather from a specific location."""
    text = location.lower()
    if any([city in text for city in ['sf', 'san francisco']]):
        return "It's sunny in San Francisco."
    elif 'paris' in text:
        return f'The weather in {location} is beautiful!'
    elif 'london' in text:
        return 'Expect rain today in London.'
    else:
        return f'I am not sure what the weather is in {location}.'


@tool
def say_hello():
    return 'Hello!!!'


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
            requirements_file: Optional[str] = None
    ):
        """
        Create an environment to run Python code.

        Args:
            env: The code execution environment. Must be a string from `CODE_ENV_NAMES`.
            allowed_imports: A list of Python modules that are allowed to be imported.
            requirements_file: Optional `requirements.txt` file to be used by `pip`.
        """
        self.allowed_imports: set[str] = set(allowed_imports)
        self.env: CODE_ENV_NAMES = env
        self.requirements_file = requirements_file

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
            print(f'{disallowed_imports=}')
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
                result = sp.run(
                    ['python', code_file.name], shell=False, capture_output=True, text=True
                )
                os.remove(code_file.name)
                return result.stdout, result.stderr, result.returncode

        elif self.env =='e2b':
            # Run the code on an E2B sandbox
            import e2b_code_interpreter as e2b

            running_sandboxes = e2b.Sandbox.list()
            print(f'{len(running_sandboxes)} E2B sandboxes are running')
            if running_sandboxes:
                sbx = e2b.Sandbox.connect(running_sandboxes[0].sandbox_id)
            else:
                sbx = e2b.Sandbox(timeout=20)

            print('E2B sandbox info:', sbx.get_info())
            execution = sbx.run_code(code=source_code, timeout=15)
            std_out: str = '\n'.join(execution.logs.stdout)
            std_err: str = '\n'.join(execution.logs.stderr)
            ret_code: int = -1 if execution.error else 0
            return std_out, std_err, ret_code

        else:
            raise ValueError(
                f'Unsupported code execution env: {self.env}'
            )


class Task(pyd.BaseModel):
    """
    Task to be solved by an agent.
    """
    id: str = pyd.Field(description='Auto-generated task ID', default_factory=uuid.uuid4)
    task: str = pyd.Field(description='Task description')
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
    successful: bool = pyd.Field(description='Task completed or failed?', default=False)


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
    successful: bool = pyd.Field(description='Task completed or failed?', default=False)


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
            tools: Optional[list[Callable]] = None,
            litellm_params: Optional[dict] = None,
    ):
        """
        Initialize an agent.

        Args:
            name: The name of the agent.
            model_name: The name of the LLM to be used.
            tools: A list of tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
        """
        self.id = uuid.uuid4()
        self.name: str = name
        self.model_name: str = model_name
        self.tools = tools
        self.litellm_params: dict = litellm_params or {}

        self.tool_names = set([t.name for t in tools]) if tools else set([])
        self.tool_name_to_func = {
            t.name: t for t in tools
        } if tools else {}

        self.task: Optional[Task] = None
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
    async def run(self, task: str) -> AsyncIterator[AgentResponse]:
        """
        Execute a task using the agent. All subclasses must override this method to solve tasks
        in their own way. The method should yield an `AgentResponse`.

        Args:
            task: A description of the task.

        Yields:
            An update from the agent.
        """

    async def _call_llm(self, messages: list[dict], response_format: Type[ChatMessage]) -> str:
        """
        Invoke the LLM to generate a response based on a given list of messages.

        Args:
            messages: A list of messages to be sent to the LLM.
            response_format: The type of message to respond with: `ChatMessage` or its subclass.

        Returns:
            The LLM response as string.
        """
        params = {
            'model': self.model_name,
            'messages': messages,
            'response_format': response_format
        }
        params.update(self.litellm_params)
        response = await litellm.acompletion(**params)

        return response.choices[0].message['content']

    def _create_task(self, description: str):
        """
        Create a new task to be solved by the agent.

        Args:
            description: The task description.
        """
        self.task = Task(task=description)
        # Since `messages` stores every message generated while interacting with the agent,
        # we need to know which all messages correspond to the current task
        # (so that the ones pertaining to the previous tasks can be ignored)
        self.msg_idx_of_new_task = len(self.messages)

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

    @staticmethod
    def make_a_single_msg(role: MESSAGE_ROLES, content: str) -> dict:
        """
        Create a single message that can be sent to LiteLLM.
        This can be used to create a `ChatMessage`.

        Args:
            role: The message sender's role.
            content: The content of the message.

        Returns:
            A dict item representing the message.
        """
        return {'role': role, 'content': content}

    def add_to_history(self, message: ChatMessage):
        """
        Add a chat message, generated by user, AI, or tool, to the agent's message history.

        Args:
            message: The message. Must be a valid `ChatMessage` instance.
        """
        assert isinstance(message, ChatMessage), (
            f'add_to_history() expects a ChatMessage; got {type(message)}'
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
            description += f'- Tool: {t.name}'
            description += f'\n  * Schema: {t.args_schema.model_json_schema()}'
            description += f'\n  * Description: {t.description}'
            description += '\n\n'

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
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
    ):
        """
        Instantiate a ReAct agent.

        Args:
            name: The name of the agent.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
        """
        super().__init__(name, model_name, tools, litellm_params)

        self.max_iterations: int = max_iterations
        self.final_answer_found: bool = False

    def _run_init(self, task: str):
        self.add_to_history(ChatMessage(role='user', content=task))
        self._create_task(description=task)  # This step is redundant in this class
        self.final_answer_found = False  # Reset from any previous task

    async def run(self, task: str) -> AsyncIterator[AgentResponse]:
        """
        Solve a task using ReAct's TAO loop.

        Args:
            task: A description of the task.

        Yields:
            An update from the agent.
        """
        self._run_init(task)

        yield self.response(
            rtype='log',
            value=f'---= Agent {self.name}: Solving the task `{self.task.task}` =---',
            channel='run'
        )
        for idx in range(self.max_iterations):
            if self.final_answer_found:
                break

            yield self.response(
                rtype='log',
                channel='run',
                value=f'  * Executing step {idx + 1}'
            )
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
            history=self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task),
            tool_names=self.get_tools_description(),
        )
        response = await self._call_llm(
            messages=[Agent.make_a_single_msg(role='user', content=message)],
            response_format=ReActChatMessage
        )

        msg = ReActChatMessage.model_validate_json(response)
        msg.role = 'assistant'
        self.add_to_history(msg)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Take action based on the agent's previous thought.

        The LLM has suggested an action. This method will identify the tool suggested and
        execute it.

        Yields:
            Updates from the acting step.
        """
        prev_msg: ReActChatMessage = self.messages[-1]

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

                # prev_thought.args is a JSON string; convert it into a dict
                tool_args = tool_args.strip().strip('`').strip()
                if tool_args.startswith('json'):
                    tool_args = tool_args[4:].strip()
                tool_args = json.loads(tool_args)

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
                        ' Please suggest a correct tool name from <TOOLS>.'
                    )
                    yield self.response(
                        rtype='step',
                        value=result,
                        channel='_act',
                        metadata={'tool': tool_name, 'args': tool_args, 'is_error': True}
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
            tools: Optional[list[Callable]],
            run_env: CODE_ENV_NAMES,
            litellm_params: Optional[dict] = None,
            max_iterations: int = 20,
            allowed_imports: Optional[list[str]] = None,
            requirements_file: Optional[str] = None,
    ):
        """
        Instantiate a CodeAgent.

        Args:
            name: The name of the agent.
            model_name: The name of the LLM to be used (use names from LiteLLM).
            tools: The tools available to the agent.
            run_env: The code execution environment. `host` means code will be run on the system
             where you create this agent. `e2b` means code will be run on an E2B sandbox. You will
             need an E2B API key.
            litellm_params: Optional parameters for LiteLLM.
            max_iterations: The maximum number of steps that the agent should try to solve a task.
            allowed_imports: A list of Python modules that the agent is allowed to import.
            requirements_file: Optional `requirements.txt` file to be used with `pip` [Unused].
        """
        super().__init__(name, model_name, tools, litellm_params, max_iterations)

        # Combine the source code of all tools into one place
        # TODO Somehow dynamically identify and include the modules used by the tools
        self.tools_source_code: str = 'from typing import *\n\n'

        for t in self.tools:
            self.tools_source_code += inspect.getsource(t).strip('@tool') + '\n'

        self.requirements_file = requirements_file

        if not allowed_imports:
            allowed_imports = []

        # The following imports are allowed by default
        self.allowed_imports = allowed_imports + ['datetime', 'typing']
        self.code_runner = CodeRunner(env=run_env, allowed_imports=self.allowed_imports)
        self.end_code_marker = '<end_code>'

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
                history += f'Code:{msg.code.strip()}\n'
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
            history=self.format_messages_for_prompt(start_idx=self.msg_idx_of_new_task),
            tool_names=self.get_tools_description(),
            authorized_imports='datetime,'
        )
        response = await self._call_llm(
            messages=[Agent.make_a_single_msg(role='user', content=message)],
            response_format=CodeChatMessage
        )

        msg = CodeChatMessage.model_validate_json(response)
        msg.role = 'assistant'
        self.add_to_history(msg)
        yield self.response(rtype='step', value=msg, channel='_think')

    async def _act(self) -> AsyncIterator[AgentResponse]:
        """
        Code action based on CodeAgent's previous thought.

        The LLM has suggested code. This method will run the code.

        Yields:
            Updates from the acting step.
        """
        prev_msg: CodeChatMessage = self.messages[-1]

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
                if code.endswith(self.end_code_marker):
                    code = code[:len(self.end_code_marker)]
                code = code.strip('`')
                if code.startswith('py'):
                    code = code[2:].strip()
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

            except FileExistsError as ex:  #Exception as ex:
                error_msg = f'*** An error occurred while running the code: {ex}'
                yield self.response(rtype='step',
                                     value=error_msg,
                                     channel='_act',
                                     metadata={'is_error': True}
                                     )


async def main():
    """
    Demonstrate the use of ReActAgent and CodeAgent.
    """
    litellm_params = {'temperature': 0}
    model_name = 'gemini/gemini-2.0-flash-lite'

    # agent = ReActAgent(
    #     name='Agent ReAct',
    #     model_name=model_name,
    #     tools=[get_weather, say_hello, calculator],
    #     max_iterations=3,
    #     litellm_params=litellm_params
    # )
    agent = CodeAgent(
        name='Agent Code',
        model_name=model_name,
        tools=[get_weather, calculator],
        run_env='e2b',
        max_iterations=3,
        litellm_params=litellm_params,
        allowed_imports=['re'],
    )

    for task in [
        'How is the weather of the capital of France?',
        'What is ten plus 15, raised to 2, expressed in words?',
        'What is the date today? Express it in words.'
    ]:
        rich.print(f'[blue][bold]User[/bold]: {task}[/blue]')

        async for response in agent.run(task):
            if response['type'] == 'final':
                msg = (
                    response['value'].content
                    if isinstance(response['value'], ChatMessage) else response['value']
                )
                rich.print(f'[green][bold]Agent[/bold]: {msg}[/green]\n')
            else:
                print(response)


if __name__ == '__main__':
    asyncio.run(main())
