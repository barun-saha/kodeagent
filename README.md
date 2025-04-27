# KodeAgent

KodeAgent is a minimalistic, framework-less agentic solution built from scratch.

Written in about 1200 lines (including prompts and documentation), KodeAgent comes with built-in ReAct and CodeAgent. Or you can create your own agent by subclassing `Agent`.

A key motivation beyond KodeAgent is also to teach building agentic frameworks from scratch. KodeAgent introduces a few primitives and code flows that should help you to get an idea about how such frameworks typically work. 


## Usage

KodeAgent has just about five direct dependencies. Create a virtual environment and install them as follows:
```bash
pip install -r requirements.txt
```

Copy the `kodeagent.py` file to your source code location. Yes, it's just a single file!

In your application code, create a ReAct agent like this:
```python
from kodeagent import ReActAgent


agent = ReActAgent(
    name='Agent ReAct',
    model_name='gemini/gemini-2.0-flash-lite',
    tools=[get_weather, calculator],
    max_iterations=3,
)
```

Or if you want to use CodeAgent:
```python
from kodeagent import CodeAgent


agent = CodeAgent(
    name='Agent Code',
    model_name='gemini/gemini-2.0-flash-lite',
    tools=[get_weather, calculator],
    run_env='e2b',
    max_iterations=3,
    allowed_imports=['re'],
)
```

Now let your agent solve the tasks like this:
```python
from kodeagent import ChatMessage


for task in [
    'What is 10 + 15, raised to 2, expressed in words?',
]:
    print(f'User: {task}')

    async for response in agent.run(task):
        if response['type'] == 'final':
            msg = (
                response['value'].content
                if isinstance(response['value'], ChatMessage) else response['value']
            )
            print(f'Agent: {msg}')
        else:
            print(response)
```

That's it! Your agent should start solving the task and keep streaming the updates.

The `get_weather`, `calculator` tools are inbuilt. The former returns dummy weather status. The latter performs real arithmetic calculations.

KodeAgent uses [LiteLLM](https://github.com/BerriAI/litellm), enabling it to work with any capable LLM. Currently, KodeAgent has been tested with Gemini 2.0 Flash Lite.

LLM model names, parameters, and keys should be set as per [LiteLLM documentation](https://docs.litellm.ai/docs/set_keys). For example, add `GEMINI_API_KEY` to the `.env` to use Gemini API.

### Code Execution

CodeAgent executes LLM-generated code to leverage the tools. KodeAgent currently supports two different code run environments:
- `host`: The Python code will be run on the system where you created this agent. In other words, where the application is running.
- `e2b`:  The Python code will be run on an [E2B sandbox](https://e2b.dev/). You will need an E2B API key and add to your `.env` file.

KodeAgent is very much experimental. Capabilities are limited. Use with caution.


## Acknowledgement

KodeAgent heavily borrows code and ideas from different places, such as:
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)
- [Smolagents](https://github.com/huggingface/smolagents/tree/main)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Building ReAct Agents from Scratch: A Hands-On Guide using Gemini](https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae)
- [LangGraph Tutorial: Build Your Own AI Coding Agent](https://medium.com/@mariumaslam499/build-your-own-ai-coding-agent-with-langgraph-040644343e73)

