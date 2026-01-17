# Observability

KodeAgent includes built-in support for observability through platforms, such as [Langfuse](https://langfuse.com/).
Hierarchical tracing has been integrated into both `ReActAgen`t and `CodeActAgent`, providing complete observability into their decision-making and execution loops. Traces are captured automatically when agents run, with support for Langfuse and other observability platforms.

**Key Features**

- ✅ **Hierarchical Tracing:** Traces, spans, and generations properly nested.
- ✅ **Langfuse Integration:** Direct support for Langfuse observability platform.
- ✅ **Multi-Backend Support:** Extensible design supports LangSmith and others.
- ✅ **Graceful Degradation:** No-op mode when tracing disabled (default).
- ✅ **Zero Breaking Changes:** Fully backward compatible.


## Enabling Observability

By default, both `ReActAgent` and `CodeActAcgent` have observability disabled. More specifically, they use a no-op tracer that does not record any data.
To enable Langfuse observability, you need to set the `tracing_type` parameter when initializing the agent. For example:

```python
from kode_agent import ReActAgent, CodeActAgent
from kodeagent.tools import calculator

agent = ReActAgent(
    name='Simple agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[calculator],
    tracing_type='langfuse',  # Enable Langfuse tracing
)

# For CodeActAgent:
agent = CodeActAgent(
    name='Simple agent',
    model_name='gemini/gemini-2.5-flash-lite',
    tools=[calculator],
    tracing_type='langfuse',  # Enable Langfuse tracing
)
```

Now, run you agent as usual:

```python
async for response in agent.run('
                                '):
    print(response)
    # Traces automatically logged to Langfuse
```

Of course, Langfuse must be installed. In addition, the `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` keys should be set via environment variables, e.g.:

```bash
export LANGFUSE_PUBLIC_KEY='your_public_key'
export LANGFUSE_SECRET_KEY='your_secret_key'
export LANGFUSE_HOST='https://api.langfuse.com'
```

Currently, only Langfuse and no-op tracing is supported. However, the design is extensible, and support for other platforms like LangSmith can be added in the future.


## Viewing Traces

Once observability is enabled, all agent interactions will be traced and sent to Langfuse. You can log in to your [Langfuse dashboard](https://cloud.langfuse.com) to view detailed traces of each agent run, including:
- **Agent Decisions**: See each thought, action, and observation made by the agent.
- **Tool Usage**: Monitor which tools were invoked and their outputs (or what code was written).
- **Plan and Observations**: Review the agent's plan progress and observations made during the task.
- **Performance Metrics**: Analyze response times and resource usage.

To view the traces:

1. Go to https://cloud.langfuse.com
2. Find trace by agent class name
3. Expand to see hierarchical trace tree

A screenshot of a sample trace in Langfuse:

<img width="90%" height="90%" alt="KodeAgent trace on Langfuse dashboard" src="https://github.com/user-attachments/assets/52530ccd-57dd-4be0-afe9-70cdab279a2e" />


## Trace Hierarchy

The resulting trace hierarchy looks like this:

```
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


### Architecture

```
AbstractObservation (interface)
├── NoOpObservation (no-op impl)
└── Langfuse objects (directly used)

AbstractTracerManager (interface)
├── NoOpTracerManager (no-op impl)
└── LangfuseTracerManager (Langfuse impl)
```

### Methods

```python
# Create spans/traces
observation = manager.start_trace(name='task', input_data={...})
observation = manager.start_span(parent=trace, name='op', input_data={...})
observation = manager.start_generation(parent=trace, name='llm', input_data={...})

# Update during execution
observation.update(status='processing', progress=50)

# End observation
observation.end(output='result', metadata={...})

# Context manager support
with observation:
    # Do work
    pass  # observation.end() called automatically
```
