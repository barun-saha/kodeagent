# 🤖 KodeAgent: The Minimal Agent Engine

[![pypi](https://img.shields.io/pypi/v/kodeagent.svg)](https://pypi.org/project/kodeagent/)
[![codecov](https://codecov.io/gh/barun-saha/kodeagent/branch/main/graph/badge.svg)](https://codecov.io/gh/barun-saha/kodeagent)
[![Documentation Status](https://readthedocs.org/projects/kodeagent/badge/?version=latest)](https://kodeagent.readthedocs.io/en/latest/?badge=latest)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Known Vulnerabilities](https://snyk.io/test/github/barun-saha/kodeagent/badge.svg)](https://snyk.io/test/github/barun-saha/kodeagent)


KodeAgent is a frameworkless, minimalistic approach to building AI agents. KodeAgent is a small (~2,000 lines of pure Python) engine designed to be the robust reasoning core inside your larger system, not the entire platform.

![KodeAgent Demo](examples/kodeagent_demo.gif)


## ✅ Why KodeAgent?

KodeAgent adheres to the **Unix Philosophy**: do one thing well and integrate seamlessly.

Use KodeAgent because it offers:
- **Memoryless and Scalable:** Designed to be memoryless with zero overhead, making it perfect for serverless and high-throughput microservices (ephemeral-task agents).
- **ReAct & CodeAct:** KodeAgent supports both ReAct and CodeAct agent paradigms out-of-the-box, enabling agents to reason and act using tools or by generating and executing code.
- **LLM Agnostic:** Built on LiteLLM, KodeAgent easily swaps between models, such as Gemini and OpenAI, without requiring any changes to the core agent logic.
- **Lightweight Foundation:** At only ~2,000 lines of pure Python and just a few dependencies, KodeAgent provides the agent primitives without any of the architectural bloat found in some frameworks.
- **The Glass Box:** The architecture is frameworkless and minimal, allowing you to read the entire Pure Python source and debug without fighting opaque abstraction layers.


## ✋ Why Not?

Also, here are a few reasons why you shouldn't use KodeAgent:

- KodeAgent is actively evolving, meaning some aspects may change.
- You want to use some of the well-known frameworks.
- You need a full-fledged platform with built-in memory management.


## 🚀 Quick Start

<a target="_blank" href="https://colab.research.google.com/drive/1D9ly3qi9sPZn_sUFdfC1XCFehldCVPXM?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


Install [KodeAgent](https://pypi.org/project/kodeagent/) via pip:
```bash
pip install kodeagent
```

Or if you want to clone the KodeAgent GitHub repository locally and run from there, use:
```bash
git clone https://github.com/barun-saha/kodeagent.git

python -m venv venv
source venv/bin/activate
# venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
```

Now, in your application code, create a ReAct agent and run a task like this (see `examples/kodeagent_quick_start.py`):

```python
from kodeagent import ReActAgent, print_response
from kodeagent.tools import read_webpage, search_web

agent = ReActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[search_web, read_webpage],
    max_iterations=5,
)

for task in [
    'What are the festivals in Paris? How they differ from Kolkata?',
]:
    print(f'User: {task}')

    async for response in agent.run(task):
        print_response(response, only_final=True)
```

You can also create a CodeActAgent, which leverages the core CodeAct pattern to generate and execute Python code on the fly for complex tasks. For example:

```python
from kodeagent import CodeActAgent
from kodeagent.tools import read_webpage, search_web, extract_as_markdown

agent = CodeActAgent(
    name='Web agent',
    model_name='gemini/gemini-2.0-flash-lite',
    tools=[search_web, read_webpage, extract_as_markdown],
    run_env='host',
    max_iterations=7,
    allowed_imports=[
        're', 'requests', 'ddgs', 'urllib', 'requests', 'bs4',
        'pathlib', 'urllib.parse', 'markitdown'
    ],
    pip_packages='ddgs~=9.5.2;beautifulsoup4~=4.14.2;"markitdown[all]";',
)
```

That's it! Your agent should start solving the task and keep streaming the updates. For more examples, including how to provide files as inputs, see the [kodeagent.py](src/kodeagent/kodeagent.py) module.

KodeAgent uses [LiteLLM](https://github.com/BerriAI/litellm), enabling it to work with any capable LLM. Currently, KodeAgent has been tested with Gemini 2.5 Flash Lite. For advanced tasks, you can try Gemini 2.5 Pro.

LLM model names, parameters, and keys should be set as per [LiteLLM API Keys documentation](https://docs.litellm.ai/docs/set_keys). For example, set the `GEMINI_API_KEY` environment variable (add in the `.env` file you are running from source code) to use [Gemini API](https://aistudio.google.com/api-keys). Additionally, you can set `OPENAI_API_KEY` for OpenAI models; set `ANTHROPIC_API_KEY` for Claude models; and so on. For [Azure OpenAI](https://docs.litellm.ai/docs/providers/azure/) models, set `AZURE_API_KEY`, `AZURE_API_BASE`, and `AZURE_API_VERSION` environment variables.


### Code Execution

`CodeActAgent` executes LLM-generated code to leverage the tools. KodeAgent currently supports two different code run environments:
- `host`: The Python code will be run on the system where you created this agent. In other words, where the application is running.
- `e2b`:  The Python code will be run on an [E2B sandbox](https://e2b.dev/). You will need to set the `E2B_API_KEY` environment variable.

With `host` as the code running environment, no special steps are required, since it uses the current Python installation. However, with `e2b`, code (and tools) are copied to a different environment and are executed there. Therefore, some additional setup may be required.

You can also specify a `work_dir` to serve as a local workspace. For the `e2b` environment, any files generated by the agent in the sandbox will be automatically downloaded to this local `work_dir`. If specified, `work_dir` could be relative or absolute path, but it **must exist**; otherwise, a temporary directory will be created and used for each run.

```python
from kodeagent import CodeActAgent

agent = CodeActAgent(
    name='Data Agent',
    model_name='gemini/gemini-2.0-flash-lite',
    run_env='e2b',
    work_dir='./agent_workspace',  # Local workspace directory to copy files to/from E2B
    # ... other parameters
)
```


For example, the Python modules that are allowed to be used in code should be explicitly specified using `allowed_imports`. In addition, any additional Python package that may need to be installed should be specified as a comma-separated list via `pip_packages`.  

KodeAgent is under active development. Capabilities are limited. Use with caution.


## 🛠️ Tools

KodeAgent comes with the following built-in [tools](src/kodeagent/tools.py):
- **`calculator`**: A simple calculator tool to perform basic arithmetic operations. It imports the `ast`, `operator`, and `re` Python libraries.
- **`download_file`**: A tool to download a file from a given URL. It imports the `requests`, `re`, `tempfile`, `pathlib`, and `urllib.parse` Python libraries.
- **`extract_as_markdown`**: A tool to read file contents and return as Markdown using MarkItDown. It imports the `re`, `pathlib`, `urllib.parse`, and `markitdown` Python libraries.
- **`generate_image`**: A tool to generate an image based on a text prompt using the specified model. The (LiteLLM) model name to be used must be mentioned in the task, system prompt, or somehow. It imports the `os`, `base64`, and `litellm` Python libraries.
- **`read_webpage`**: A tool to read a webpage using BeautifulSoup. It imports the `re`, `requests`, `urllib.parse`, and `bs4` Python libraries.
- **`search_arxiv`**: A tool to search arXiv for research papers and return summaries and links. It imports the `arxiv` library.
- **`search_web`**: A web search tool using DuckDuckGo to fetch top search results. It imports the `datetime`, `random`, and `time` Python libraries.
- **`search_wikipedia`**: A tool to search Wikipedia and return summaries and links. It imports the `wikipedia` library.
- **`transcribe_audio`**: A tool to transcribe audio files using OpenAI's Whisper via [Fireworks API](https://fireworks.ai/). Need to set the `FIREWORKS_API_KEY` environment variable. It imports the `os` and `requests` Python libraries.
- **`transcribe_youtube`**: A tool to fetch YouTube video transcripts. It imports the `youtube_transcript_api` library.

Check out the docstrings of these tools in the [tools.py](src/kodeagent/tools.py) module for more details.

To add a new tool, use the `@tool` decorator from `kodeagent.tools` module. For example:
```python
from kodeagent import tool

@tool
def my_tool(param1: str) -> str:
    """Description of the tool.
    Args:
        param1 (str): Description of param1.
    Returns:
        str: Description of the return value.
    """
    # Tool implementation here
    return 'result'
```

Module imports and all variables should be inside the tool function. If you're using `CodeActAgent`, KodeAgent will execute the tool function in isolation.
For further details, refer to the [API documentation](https://kodeagent.readthedocs.io/en/latest/).

## 🔭 Observability

KodeAgent logs the LLM calls and usage using [Langfuse](https://langfuse.com/). The LiteLLM calls set the trace ID to the task ID. The model name as well as the response format and retry attempts for the generations are also logged (see the screenshot below). To enable tracing, create your Langfuse account and set the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` environment variables. Read more about [Langfuse integration with LiteLLM](https://docs.litellm.ai/docs/observability/langfuse_integration).

<img width="80%" height="80%" alt="KodeAgent trace on Langfuse dashboard" src="https://github.com/user-attachments/assets/78390575-9d12-4ca3-aac3-5eff084c88f0" />


## ⊷ Sequence Diagram for CodeAct Agent (via CodeRabbit)
```mermaid
sequenceDiagram
  autonumber
  actor User
  participant Agent
  participant Planner
  participant LLM as LLM/Prompts
  participant Tools

  User->>Agent: run(task)
  Agent->>Planner: create_plan(task)
  Planner->>LLM: request AgentPlan JSON (agent_plan.txt)
  LLM-->>Planner: AgentPlan JSON
  Planner-->>Agent: planner.plan set

  loop For each step
    Agent->>Planner: get_formatted_plan()
    Agent->>LLM: codeact prompt + {plan, history}
    LLM-->>Agent: Thought + Code
    Agent->>Tools: execute tool call(s)
    Tools-->>Agent: Observation
    Agent->>Planner: update_plan(thought, observation, task_id)
  end

  Agent-->>User: Final Answer / Failure (per codeact spec)
```


# 🧪 Run Tests

To run unit tests, use:
```bash
python -m pytest .\tests\unit -v --cov --cov-report=html
```

For integration tests involving calls to APIs, use:
```bash
python -m pytest .\tests\integration -v --cov --cov-report=html
```

Gemini and E2B API keys should be set in the `.env` file for integration tests to work.

A [Kaggle notebook](https://www.kaggle.com/code/barunsaha/kodeagent-benchmark/) for benchmarking KodeAgent is also available.


## 🗺️ Roadmap & Contributions

To be updated.


## 🙏 Acknowledgement

KodeAgent heavily borrows code and ideas from different places, such as:
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/)
- [Smolagents](https://github.com/huggingface/smolagents/tree/main)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Building ReAct Agents from Scratch: A Hands-On Guide using Gemini](https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae)
- [LangGraph Tutorial: Build Your Own AI Coding Agent](https://medium.com/@mariumaslam499/build-your-own-ai-coding-agent-with-langgraph-040644343e73)
- Aider, Antigravity, CodeRabbit, GitHub Copilot, Jules, ...


## ⚠️ DISCLAIMER & LIABILITY

AI agents can occasionally cause unintended or unpredictable side effects. We urge users to **use KodeAgent with caution**. Always review generated code and test agents rigorously in a constrained, non-production environment before deployment.

**LIMITATION OF LIABILITY:**
By using this software, you agree that KodeAgent, its developers, contributors, supporters, and any other associated entities shall not be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software.
