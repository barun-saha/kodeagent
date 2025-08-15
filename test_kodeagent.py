"""
Unit tests for the KodeAgent ReActAgent class.
"""
import datetime
import pytest

from kodeagent import (
    ReActAgent,
    tool,
    ChatMessage,
    ReActChatMessage,
    calculator,
    web_search,
    file_download, CodeActAgent
)


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@tool
def dummy_tool_one(param1: str) -> str:
    """Description for dummy tool one."""
    return f'tool one executed with {param1}'


@pytest.fixture
def react_agent():
    """Fixture to create a ReActAgent instance for testing."""
    agent = ReActAgent(
        name='test_react_agent',
        model_name=MODEL_NAME,
        tools=[dummy_tool_one, calculator, web_search, file_download],
        description='Test ReAct agent for unit tests',
        max_iterations=3
    )
    return agent


def test_react_agent_initialization(react_agent):
    """Test the initialization of ReActAgent."""
    assert react_agent.name == 'test_react_agent'
    assert react_agent.model_name == MODEL_NAME
    assert len(react_agent.tools) == 4  # dummy_tool_one, calculator, web_search, file_download
    assert react_agent.max_iterations == 3
    assert 'dummy_tool_one' in react_agent.tool_names
    assert 'calculator' in react_agent.tool_names


def test_add_to_history(react_agent):
    """Test adding messages to agent's history."""
    msg = ChatMessage(role='user', content='test message')
    react_agent.add_to_history(msg)
    assert len(react_agent.messages) == 1
    assert react_agent.messages[0].role == 'user'
    assert react_agent.messages[0].content == 'test message'

    # Test adding invalid message type
    with pytest.raises(AssertionError):
        react_agent.add_to_history('invalid message')


def test_format_messages_for_prompt(react_agent):
    """Test formatting of message history for prompt."""
    msg1 = ReActChatMessage(
        role='assistant',
        thought='test thought',
        action='dummy_tool_one',
        args='{"param1": "test"}',
        content='',  # Added missing content field
        successful=False,
        answer=None
    )
    msg2 = ChatMessage(role='tool', content='tool response')

    react_agent.add_to_history(msg1)
    react_agent.add_to_history(msg2)

    formatted = react_agent.format_messages_for_prompt()
    assert 'Thought: test thought' in formatted
    assert 'Action: dummy_tool_one' in formatted
    assert 'Observation: tool response' in formatted


@pytest.mark.asyncio
async def test_react_agent_run_success(react_agent):
    """Test successful task execution by ReActAgent."""
    responses = []
    async for response in react_agent.run('Add 2 and 2'):
        responses.append(response)

    # Check that we got the expected responses
    assert any(r['type'] == 'final' for r in responses)
    assert react_agent.final_answer_found
    assert react_agent.task.is_finished
    # Verify we got a numerical answer since we used a calculator task
    final_response = next(r for r in responses if r['type'] == 'final')
    assert '4' in str(final_response['value'])


@pytest.mark.asyncio
async def test_react_agent_run_with_tool_error(react_agent):
    """Test ReActAgent handling tool execution errors."""
    # Create a broken tool that always raises an exception
    @tool
    def broken_tool(param1: str) -> str:
        """A tool that always fails."""
        raise Exception('Tool error')

    # Add the broken tool to the agent
    react_agent.tools.append(broken_tool)

    responses = []
    async for response in react_agent.run('Use the broken tool'):
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

    # Initialize the task with a proper task description
    react_agent._run_init(task_description)

    try:
        # This will make an actual API call to determine relevant tools
        tools = await react_agent.get_relevant_tools(task_description)

        # The task requires calculation, so calculator should be relevant
        assert len(tools) > 0, "No tools were returned from get_relevant_tools"
        tool_names = {t.name for t in tools}
        assert "calculator" in tool_names, "calculator should be relevant for arithmetic"
    except Exception as e:
        raise


def test_clear_history(react_agent):
    """Test clearing agent's message history."""
    msg = ChatMessage(role='user', content='test message')
    react_agent.add_to_history(msg)
    assert len(react_agent.messages) == 1

    react_agent.clear_history()
    assert len(react_agent.messages) == 0


@pytest.mark.asyncio
async def test_unsupported_task(react_agent):
    """Test that agent fails appropriately when given an unsupported task."""
    task_description = 'Generate a 30-second video animation of a flying bird'

    responses = []
    async for response in react_agent.run(task_description):
        responses.append(response)

    response = ' | '.join([str(r) for r in responses])
    assert (
        'cannot' in response or
        'no relevant tool' in response or
        'unable' in response or
        'failed' in response or
        'unfortunately' in response
    ), 'Agent should have failed for unsupported video generation task'


async def _codeact_agent_date_(code_agent) -> tuple[bool, str]:
    """Helper function to run a code block and return the response."""
    task = "What is today's date? Express it in words without time."
    responses = []
    async for response in code_agent.run(task):
        responses.append(response['value'])

    # Get today's date for verification
    today = datetime.datetime.now().strftime('%B %d, %Y')
    response = ' | '.join([str(r) for r in responses])

    # The agent's response should contain today's date
    return today.lower() in response.lower(), f'Expected {today} in response but got: {response}'


@pytest.mark.asyncio
async def test_codeact_agent_host():
    """Test the CodeActAgent functionality on a local system."""
    code_agent1 = CodeActAgent(
        name='Code agent',
        model_name=MODEL_NAME,
        run_env='host',
        max_iterations=3,
        allowed_imports=['datetime'],
        description='Agent that can write and execute Python code'
    )

    status, assert_error = await _codeact_agent_date_(code_agent1)
    assert status, assert_error


@pytest.mark.asyncio
async def test_codeact_agent_e2b():
    """Test the CodeActAgent functionality ona remote E2B sandbox."""
    code_agent2 = CodeActAgent(
        name='Code agent',
        model_name=MODEL_NAME,
        run_env='e2b',
        max_iterations=3,
        allowed_imports=['datetime'],
        description='Agent that can write and execute Python code',
        pip_packages=None,
    )

    status, assert_error = await _codeact_agent_date_(code_agent2)
    assert status, assert_error


@pytest.mark.asyncio
async def test_codeact_agent_unsupported():
    """Test the CodeActAgent functionality on an unsupported env."""
    code_agent = CodeActAgent(
        name='Code agent',
        model_name=MODEL_NAME,
        run_env='docker',
        max_iterations=3,
        allowed_imports=['datetime'],
        description='Agent that can write and execute Python code',
        pip_packages=None,
    )

    responses = []
    async for response in code_agent.run('What is the date today?'):
        responses.append(response)

    response = ' | '.join([str(r) for r in responses])
    print(f'{response=}')
    assert (
        'Unsupported code execution' in response
    ), 'Expected code execution to fail on unsupported env'
