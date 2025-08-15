import pytest

from kodeagent import (
    ReActAgent,
    tool,
    ChatMessage,
    ReActChatMessage,
    calculator,
    web_search,
    file_download
)


@tool
def dummy_tool_one(param1: str) -> str:
    """Description for dummy tool one."""
    return f"tool one executed with {param1}"


@pytest.fixture
def react_agent():
    """Fixture to create a ReActAgent instance for testing."""
    agent = ReActAgent(
        name='test_react_agent',
        model_name='gemini/gemini-2.0-flash-lite',
        tools=[dummy_tool_one, calculator, web_search, file_download],
        description='Test ReAct agent for unit tests',
        max_iterations=3
    )
    return agent

def test_react_agent_initialization(react_agent):
    """Test the initialization of ReActAgent."""
    assert react_agent.name == "test_react_agent"
    assert react_agent.model_name == "gemini/gemini-2.0-flash-lite"
    assert len(react_agent.tools) == 4  # dummy_tool_one, calculator, web_search, file_download
    assert react_agent.max_iterations == 3
    assert "dummy_tool_one" in react_agent.tool_names
    assert "calculator" in react_agent.tool_names

def test_add_to_history(react_agent):
    """Test adding messages to agent's history."""
    msg = ChatMessage(role="user", content="test message")
    react_agent.add_to_history(msg)
    assert len(react_agent.messages) == 1
    assert react_agent.messages[0].role == "user"
    assert react_agent.messages[0].content == "test message"

    # Test adding invalid message type
    with pytest.raises(AssertionError):
        react_agent.add_to_history("invalid message")

def test_format_messages_for_prompt(react_agent):
    """Test formatting of message history for prompt."""
    msg1 = ReActChatMessage(
        role="assistant",
        thought="test thought",
        action="dummy_tool_one",
        args='{"param1": "test"}',
        content="",  # Added missing content field
        successful=False,
        answer=None
    )
    msg2 = ChatMessage(role="tool", content="tool response")

    react_agent.add_to_history(msg1)
    react_agent.add_to_history(msg2)

    formatted = react_agent.format_messages_for_prompt()
    assert "Thought: test thought" in formatted
    assert "Action: dummy_tool_one" in formatted
    assert "Observation: tool response" in formatted

@pytest.mark.asyncio
async def test_react_agent_run_success(react_agent):
    """Test successful task execution by ReActAgent."""
    responses = []
    async for response in react_agent.run("Add 2 and 2"):
        responses.append(response)

    # Check that we got the expected responses
    assert any(r["type"] == "final" for r in responses)
    assert react_agent.final_answer_found
    assert react_agent.task.is_finished
    # Verify we got a numerical answer since we used a calculator task
    final_response = next(r for r in responses if r["type"] == "final")
    assert "4" in final_response["value"].content

@pytest.mark.asyncio
async def test_react_agent_run_with_tool_error(react_agent):
    """Test ReActAgent handling tool execution errors."""
    # Create a broken tool that always raises an exception
    @tool
    def broken_tool(param1: str) -> str:
        """A tool that always fails."""
        raise Exception("Tool error")

    # Add the broken tool to the agent
    react_agent.tools.append(broken_tool)

    responses = []
    async for response in react_agent.run("Use the broken tool"):
        responses.append(response)

    # Check that error was captured in the response
    error_responses = [r for r in responses if r["metadata"] and r["metadata"].get("is_error")]
    assert len(error_responses) > 0
    assert "Incorrect tool name generated" in str(error_responses[0]["value"])

@pytest.mark.asyncio
async def test_think_step(react_agent):
    """Test the think step of ReActAgent."""
    # Initialize the task first
    react_agent._run_init("Calculate 5 plus 3")

    responses = []
    async for response in react_agent._think():
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]["type"] == "step"
    assert isinstance(responses[0]["value"], ReActChatMessage)
    assert responses[0]["value"].thought is not None
    assert len(responses[0]["value"].thought) > 0

@pytest.mark.asyncio
async def test_act_step_with_invalid_tool(react_agent):
    """Test the act step with an invalid tool name."""
    invalid_response = ReActChatMessage(
        thought="Test thought",
        action="nonexistent_tool",
        args='{"param1": "test"}',
        answer=None,
        successful=False,
        role="assistant",
        content=""
    )

    react_agent.add_to_history(invalid_response)

    responses = []
    async for response in react_agent._act():
        responses.append(response)

    assert len(responses) == 1
    assert "Incorrect tool name" in responses[0]["value"]
    assert responses[0]["metadata"]["is_error"]

def test_get_tools_description(react_agent):
    """Test getting tool descriptions."""
    desc = react_agent.get_tools_description()
    assert "dummy_tool_one" in desc
    assert "calculator" in desc
    assert "web_search" in desc
    assert "file_download" in desc
    assert "Description for dummy tool one" in desc


@pytest.mark.asyncio
async def test_get_relevant_tools(react_agent):
    """Test filtering relevant tools for a task."""
    task_description = 'What is 2 plus 3?'  # Simple calculator task

    print("\nStarting test_get_relevant_tools...")
    print(f"Task description: {task_description}")
    print(f"Available tools: {[t.name for t in react_agent.tools]}")
    print(f"Tools description: {react_agent.get_tools_description()}")

    # Initialize the task with a proper task description
    react_agent._run_init(task_description)

    try:
        # This will make an actual API call to determine relevant tools
        print("\nCalling get_relevant_tools...")
        tools = await react_agent.get_relevant_tools(task_description)

        print(f"\nReturned tools: {[t.name for t in tools]}")
        print(f"Returned tool count: {len(tools)}")

        # The task requires calculation, so calculator should be relevant
        assert len(tools) > 0, "No tools were returned from get_relevant_tools"
        tool_names = {t.name for t in tools}
        print(f"Tool names set: {tool_names}")
        assert "calculator" in tool_names, "calculator should be relevant for arithmetic"
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        print(f"Agent state: Task initialized={hasattr(react_agent, 'task')}")
        print(f"Agent history: {react_agent.messages}")
        raise

def test_clear_history(react_agent):
    """Test clearing agent's message history."""
    msg = ChatMessage(role="user", content="test message")
    react_agent.add_to_history(msg)
    assert len(react_agent.messages) == 1

    react_agent.clear_history()
    assert len(react_agent.messages) == 0
