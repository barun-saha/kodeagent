# Usage


## Run Tasks

Using KodeAgent, you can create a ReAct agent and run a task like this:

```python
from kodeagent import ReActAgent, print_response, extract_as_markdown, search_web

agent = ReActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[search_web, extract_as_markdown],
    max_iterations=7,
)

for task in [
    'What are the festivals in Paris? How they differ from Kolkata?',
]:
    print(f'User: {task}')

    async for response in agent.run(task):
        print_response(response, only_final=True)
```

The `print_response` function displays the agent's responses using a rich format. Setting `only_final=True` ensures that only the final response is printed. Otherwise, the intermediate streaming responses from the agent are also shown.

Tasks can also be run with input files. The files can be local files, remote files, or URLs. Run a task with files in this way:
```python
async for response in agent.run(
    task='Caption these images',
    files=[
        '/home/user/image1.jpg',
        'http://example.com/image2.jpg',
    ],
):
    print_response(response, only_final=True)
```

You can also create a CodeAct agent:

```python
from kodeagent import CodeActAgent, search_web, extract_as_markdown

agent = CodeActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[search_web, extract_as_markdown],
    run_env='host',
    max_iterations=7,
    allowed_imports=['re', 'requests', 'duckduckgo_search', 'markitdown'],
    pip_packages='ddgs~=9.5.2;"markitdown[all]";',
)
```

The `run_env` parameter specifies the environment where the agent's code will execute. Setting it to `'host'` allows the agent to run code directly on the host machine. You can also use `'e2b'` to run the code on [E2B sandbox](https://e2b.dev/).


## Task Result and State

The only way to execute any task is by invoking the `run()` method with the task description. The `run()` method provides streaming responses from the task execution, which can be iterated over asynchronously. However, often you may want to access the final response from the agent. This can be accessed via `agent.task.result`:

```python
for _ in agent.run('Some task description'):
    pass
final_response = agent.task.result
print(final_response)
```

> ⓘ NOTE:
> 
> The `result` is available only when the agent has found the final answer. Otherwise, it will be `None`. An agent can fail to find a final answer due to several reasons, e.g., a timeout, an error in the code, and max iterations reached. See the next section for more details on how to check if the final answer has been found.

`agent.task` provides access to the current task object, which contains useful information such as the task inputs, the final result, and LLM usage statistics. Refer to the `Task` model's API documentation for more details.

The agent also tracks the LLM usage statistics (reported by LiteLLM) for each task, detailing the component-wise token usage and the cost in USD. This can be accessed in two ways: raw data (`dict`) and formatted report (`str`), for example:

```python
# Access raw LLM usage data
llm_usage_data = agent.get_usage_metrics()
# Print formatted LLM usage report
llm_usage_report = agent.get_usage_report()
print(llm_usage_report)
```

You can also view the plan followed by the agent to complete the task:
```python
print(agent.current_plan)
```


## Streaming Responses

The `agent.run()` method is an asynchronous generator that yields `AgentResponse` objects. This allows you to monitor the agent's progress in real-time.

### Response Structure

Each `AgentResponse` is a dictionary with the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `type` | `str` | The type of update: `step`, `final`, or `log`. |
| `value` | `Any` | The content of the update. For `final`, it's the final answer. For `step`, it's a `ChatMessage`. |
| `channel` | `str` | Optional identifier for the source of the response (e.g., 'run', 'think', 'act'). |
| `metadata`| `dict`| Optional dictionary containing additional information like `is_error` or `usage`. |

### Handling Responses

You can use the response type to filter or format the output:

```python
async for response in agent.run(task):
    if response['type'] == 'final':
        print(f"Final Answer: {response['value']}")
    elif response['type'] == 'step':
        # value is a ChatMessage object (ReActChatMessage or CodeActChatMessage)
        print(f"Step: {response['value'].thought}")
    elif response['type'] == 'log':
        print(f"Log: {response['value']}")
```


## Setting API Keys

Set your LLM and other API keys as environment variables. KodeAgent relies on several environment variables for model access, code execution, and observability.

| Category | Variable | Description |
| :--- | :--- | :--- |
| **LLM (LiteLLM)** | `GOOGLE_API_KEY` | API key for Gemini models. |
| | `OPENAI_API_KEY` | API key for OpenAI models. |
| | `ANTHROPIC_API_KEY` | API key for Claude models. |
| **Code Execution** | `E2B_API_KEY` | Required for `run_env='e2b'`. |
| **Observability** | `LANGFUSE_PUBLIC_KEY` | Public key for Langfuse tracing. |
| | `LANGFUSE_SECRET_KEY` | Secret key for Langfuse tracing. |
| | `LANGFUSE_HOST` | Host URL for Langfuse (default: https://cloud.langfuse.com). |
| **Tools** | `FIREWORKS_API_KEY` | Required for `transcribe_audio` tool. |


### Setting Variables

For example, if your are using Linux, you can add the following lines to your `~/.bashrc` or `~/.bash_profile`:

```bash
# OpenAI API key (if using OpenAI models)
export OPENAI_API_KEY=your_openai_api_key
# Gemini API key (if using Gemini models); for Vertex AI, see LiteLLM documentation
export GOOGLE_API_KEY=your_google_api_key
# For remote code execution on E2B
export E2B_API_KEY=your_e2b_key
# For logging with Langfuse
export LANGFUSE_PUBLIC_KEY=pk_something
export LANGFUSE_SECRET_KEY=sk_something
export LANGFUSE_HOST=your_url
```

On Windows, navigate to Settings > Environment Variables and add the variables there.

If you are using KodeAgent in development mode, you can also create a `.env` file in the root directory of the project and add your keys there:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Observation and Feedback

The `Observer` is an internal component that monitors the agent's work to detect if it's stuck in a loop or has stalled. It can provide corrective feedback to help the agent get back on track.

By default, the `Observer` is enabled and triggered based on a predefined threshold in the agent's logic. It analyzes the chat history and the current plan to ensure the agent is making meaningful progress.
