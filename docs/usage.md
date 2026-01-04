# Usage

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


## Additional Information

The only way to make the agent execute any task is by invoking the `run()` method with the task description. The `run()` method provides streaming responses from the task execution, which can be iterated over asynchronously. However, often time you may want to the access the final response from the agent. This can be accessed via `agent.task.result`, for example:

```python
for _ in agent.run('Some task description'):
    pass
final_response = agent.task.result
print(final_response)
```

`agent.task` provides access to the current task object, which contains useful information such as the task inputs, the final result, and LLM usage statistics. Refer to the API documentation for more details.

The agent also tracks the LLM usage statistics for each task. This can be accessed in two ways: raw data (`dict`) and formatted report (`str`), for example:

```python
# Access raw LLM usage data
llm_usage_data = agent.get_usage_metrics()
# Print formatted LLM usage report
llm_usage_report = agent.get_usage_report()
print(llm_usage_report)
```


## Setting API Keys

Set your LLM and other API keys as environment variables. For example, if your are using Linux, you can add the following lines to your `~/.bashrc` or `~/.bash_profile`:

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