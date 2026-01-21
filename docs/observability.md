# Observability

KodeAgent includes built-in support for observability through platforms such as [Langfuse](https://langfuse.com/) and [LangSmith](https://www.langchain.com/langsmith). Hierarchical tracing is integrated into both `ReActAgent` and `CodeActAgent`, providing complete observability into their decision-making and execution loops.

**Key Features**

✅ **Hierarchical Tracing:** Traces, spans, and generations properly nested.  
✅ **Backend Integrations:** Direct support for Langfuse and LangSmith.  
✅ **Graceful Degradation:** No-op mode when tracing is disabled (default).  
✅ **Zero Breaking Changes:** Fully backward compatible. 


## Enabling Observability

By default, both `ReActAgent` and `CodeActAgent` have observability disabled (using a no-op tracer). To enable observability, set the `tracing_type` parameter when initializing the agent. 

> ⓘ NOTE:
>
> While `langfuse` is included with KodeAgent by default, `langsmith` is not and must be installed separately using `pip install langsmith`.

```python
from kodeagent import ReActAgent, CodeActAgent
from kodeagent.tools import calculator

# Enable Langfuse tracing
agent = ReActAgent(
    name='Simple agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[calculator],
    tracing_type='langfuse',
)

# Enable LangSmith tracing
agent = CodeActAgent(
    name='Simple agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[calculator],
    tracing_type='langsmith',
)
```

### Configuration

Ensure the relevant environment variables are set for your chosen platform:

**Langfuse:**
```bash
export LANGFUSE_PUBLIC_KEY='your_public_key'
export LANGFUSE_SECRET_KEY='your_secret_key'
export LANGFUSE_HOST='https://api.langfuse.com'
```

**LangSmith:**
```bash
export LANGCHAIN_TRACING_V2='true'
export LANGCHAIN_API_KEY='your_api_key'
export LANGCHAIN_PROJECT='your_project_name'  # Optional
```


## Viewing Traces

Once observability is enabled, all agent interactions will be traced and sent to the configured backend (Langfuse or LangSmith). You can log in to your [Langfuse dashboard](https://cloud.langfuse.com) (or [LangSmith dashboard](https://smith.langchain.com)) to view detailed traces of each agent run, including:
- **Agent Decisions**: See each thought, action, and observation made by the agent.
- **Tool Usage**: Monitor which tools were invoked and their outputs (or what code was written).
- **Plan and Observations**: Review the agent's plan progress and observations made during the task.
- **Performance Metrics**: Analyze response times and resource usage.

To view the traces:

1. Go to https://cloud.langfuse.com (or https://smith.langchain.com)
2. Find trace by agent class name
3. Expand to see hierarchical trace tree

A screenshot of a sample trace in Langfuse:

<img width="90%" height="90%" alt="KodeAgent trace on Langfuse dashboard" src="https://github.com/user-attachments/assets/52530ccd-57dd-4be0-afe9-70cdab279a2e" />

A screenshot of a sample trace in LangSmith:

<img width="90%" height="90%" alt="KodeAgent trace on LangSmith dashboard" src="https://github.com/user-attachments/assets/8cd947d2-d575-4037-8b0a-1d36492ce21c" />


## Trace Hierarchy

The resulting trace hierarchy looks like this:

```text
Agent.run() [root trace]
├── plan_creation [span]
│   └── (LLM call via ku.call_llm with component_name='Planner.create')
├── [iterations...]
│   ├── think [span] (for ReAct) or think_code [span] (for CodeAct)
│   ├── act [span]
│   ├── plan_update [span]
│   │   └── (LLM call via ku.call_llm with component_name='Planner.update')
│   └── observe [span]
│       └── (LLM call via ku.call_llm with component_name='Observer')
└── post_run [span] (if implemented)
```


## Tracer Module

**Key Classes:**

- `AbstractObservation` - Universal observation interface
- `AbstractTracerManager` - Universal manager interface
- `NoOpObservation` - No-op observation implementation
- `NoOpTracerManager` - No-op manager implementation
- `LangfuseTracerManager` - Langfuse integration
- `LangSmithTracerManager` - LangSmith integration
- `LangSmithObservation` - LangSmith-specific observation wrapper


### Architecture

```
AbstractObservation (interface)
├── NoOpObservation (no-op impl)
├── Langfuse objects (referenced directly)
└── LangSmithObservation (wraps RunTree)

AbstractTracerManager (interface)
├── NoOpTracerManager (no-op impl)
├── LangfuseTracerManager (Langfuse impl)
└── LangSmithTracerManager (LangSmith impl)
```


## Tracing Concepts Mapping

The following table summarizes how OpenTelemetry (OTel) concepts map to the underlying SDK methods and KodeAgent's internal tracer.

| OTel Concept | Langfuse SDK | LangSmith SDK | KodeAgent Tracer (`tracer.py`) |
| :--- | :--- | :--- | :--- |
| **Trace** (Root) | `langfuse.trace()` | `RunTree(run_type='chain')` | `start_trace()` |
| **Span** (Sub-node) | `parent.span()` | `parent.create_child(run_type='tool')` | `start_span()` |
| **Generation** (LLM) | `parent.generation()` | `parent.create_child(run_type='llm')` | `start_generation()` |
| **Event / Update** | `observation.update()` | `run_tree.error = ...` | `update()` |
| **End / Finalize** | `observation.end()` | `run_tree.end()` + `run_tree.patch()` | `end()` |
| **Flush / Sync** | `langfuse.flush()` | `client.flush()` | `flush()` |

### LangSmith Implementation Details

The `LangSmithTracerManager` requires a few extra steps to ensure traces are recorded reliably. Unlike the Langfuse SDK which often handles background syncing transparently, the LangSmith implementation explicitly calls `RunTree.post()` upon creation to initialize the run in the backend. During finalization in the `end()` method, it calls both `RunTree.end()` to record the outputs and `RunTree.patch()` to send the final state. Finally, the `flush()` method is used to ensure all buffered runs are transmitted before the process exits.
