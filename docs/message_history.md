## Message History

KodeAgent manages conversation history by maintaining two complementary views of your dialogue with the LLM. This "dual storage" approach ensures that while the LLM gets exactly what it needs to function, we keep a rich, human-readable record of the agent's internal state.

### Dual Storage: Structured vs. API Format

- **The Internal Log (`self.messages`):** This is the master record. Messages are stored as structured Pydantic objects (like `ChatMessage` or `CodeActChatMessage`). They don't just hold text; they capture the agent's "thoughts," specific tool actions, and code blocks in an organized, searchable way.
- **The LLM View:** LLM APIs require a specific format (usually a list of dictionaries with `role` and `content`). Before every call, KodeAgent transforms the internal log into this API-ready format. This includes mapping agent actions into official `tool_calls` so the LLM stays in sync with its own reasoning loop.

> **ⓘ NOTE**
>
> Ideally, it should be possible to maintain only the latter view. However, this has proved to be a bit challenging so far, especially with tool call IDs. Hence, we maintain both views.

### Smart Formatting & Optimization

Formatting isn't a simple 1:1 conversion. KodeAgent uses specialized **History Formatters** tailored to each agent type:
- **ReAct Agents:** Convert "Action" and "Args" fields into standard API tool calls.
- **CodeAct Agents:** Treat Python code blocks as "pseudo" tool calls (named `code_execution`).
- **Incremental Caching:** To keep things fast, the agent caches formatted messages. It only processes new messages since the last turn, avoiding the overhead of re-formatting the entire history on every step.

### The Message Lifecycle

1.  **Input:** Your request is wrapped in a `ChatMessage` and added to `self.messages`.
2.  **Transformation:** When it's time to "think," `formatted_history_for_llm()` scans the history, applies the appropriate transformation (ReAct vs. CodeAct), and prepares the payload for the LLM.
3.  **Inference:** The LLM responds. KodeAgent parses this response—extracting thoughts, actions, or final answers—and creates a new structured message object.
4.  **Observation:** Any tool output or code result is recorded as a `tool` role message, and the cycle repeats until the task is complete.

This system ensures that the conversation remains robust, provider-agnostic, and easy to debug.

### Extending with New Formatters

If you're building a new agent type with unique message requirements, you can implement a custom formatter:

1.  **Define a Strategy:** Create a new class inheriting from `HistoryFormatter` in `src/kodeagent/history_formatter.py`.
2.  **Implement Core Methods:** You'll need to define `should_format_as_tool_call` (to detect agent actions), `format_tool_call` (to transform them), and `should_add_pending_placeholder` (for handling interrupted turns).
3.  **Reference:** Use `ReActHistoryFormatter` as your primary template. It demonstrates how to handle standard tool call IDs and intermediate state.
4.  **Register with Agent:** Assign your new formatter to `self._history_formatter` in your agent's `__init__`.
