# Usage

Using KodeAgent, you can create a ReAct agent and run a task like this:

```python
from kodeagent import ReActAgent, print_response, extract_file_contents_as_markdown, search_web


agent = ReActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[search_web, extract_file_contents_as_markdown],
    max_iterations=7,
)

for task in [
    'What are the festivals in Paris? How they differ from Kolkata?',
]:
    print(f'User: {task}')

    async for response in agent.run(task):
        print_response(response, only_final=True)
```

You can also create a CodeAct agent:

```python
from kodeagent import CodeActAgent, search_web, extract_file_contents_as_markdown


agent = CodeActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[search_web, extract_file_contents_as_markdown],
    run_env='host',
    max_iterations=7,
    allowed_imports=['re', 'requests', 'duckduckgo_search', 'markitdown'],
    pip_packages='ddgs~=9.5.2;"markitdown[all]";',
)
```