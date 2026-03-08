# Message History

KodeAgent manages conversation history by maintaining a single, unified view of the dialogue that is directly compatible with LLM APIs while still preserving rich, structured information for the agent's internal reasoning.

## Unified History (`self.chat_history`)

Starting from v0.10.0, KodeAgent uses a single `chat_history` member (a list of dictionaries) as the master record. This approach offers several advantages:

- **API Compatibility**: The history is already in the format expected by most LLM APIs (e.g., OpenAI, Gemini, Anthropic), containing `role`, `content`, and optional `tool_calls`.
- **Consistency**: There's no longer a "dual storage" paradigm to keep in sync, reducing complexity and potential for bugs.
- **Provider Agnostic**: The history handles multimodal content and tool calls in a standard way across different LLM and SLM providers through LiteLLM.

## Structured Messaging & Formatters

While messages are stored as dictionaries, the agent still reasons using structured Pydantic objects (like `ChatMessage`, `ReActChatMessage`, or `CodeActChatMessage`). 

To bridge this gap, KodeAgent uses specialized **History Formatters**:
- **Bidirectional Mapping**: Formatters convert structured agent objects into API-compliant dictionaries and vice versa.
- **Agent Specificity**: Each agent type (ReAct, CodeAct, FCA) has a dedicated formatter that knows how to map its specific thoughts, actions, and observations into the standard `tool`/`assistant` roles.
- **Rich Content**: Thoughts and intermediate reasoning are often stored as text in the assistant's response or as metadata, ensuring they are preserved for context without breaking API schemas.

## The Message Lifecycle

1.  **Input**: Your request is wrapped in a `ChatMessage`, formatted into a dictionary, and added to `self.chat_history`.
2.  **Inference**: When it's time to "think," the agent sends `self.chat_history` (excluding the system prompt) to the LLM.
3.  **Parsing**: The LLM responds. KodeAgent's `parse_text_response` (or native parsing) extracts thoughts, actions, or final answers, which are then added back to the history.
4.  **Observation**: Any tool output or code result is recorded as a `tool` role message in the unified history, and the cycle repeats.

## Truncation and Cleanup

To handle long conversations and prevent token overflow:
- **Automatic Truncation**: Large tool results or long message contents are truncated when being added to history or when being formatted for display/observation.
- **History Management**: Methods like `get_history()` allow for easy inspection of the dialogue, while `clear_history()` resets the agent's state for a new task.

This unified system ensures that KodeAgent remains lightweight, scalable, and easy to debug while maintaining full compatibility with modern LLM capabilities.
