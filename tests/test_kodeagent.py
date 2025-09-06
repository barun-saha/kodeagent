"""
Unit tests for the KodeAgent ReActAgent class.
"""
import datetime
import pytest
from unittest.mock import AsyncMock, patch

from kodeagent import (
    ReActAgent,
    tool,
    ChatMessage,
    ReActChatMessage,
    calculator,
    search_web,
    download_file,
    search_arxiv,
    CodeActAgent,
    CodeChatMessage,
    AgentPlan,
    PlanStep,
    Planner,
    call_llm,
    Task,
    Observer,
    ObserverResponse
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
        tools=[dummy_tool_one, calculator, search_web, download_file],
        description='Test ReAct agent for unit tests',
        max_iterations=3
    )
    return agent


@pytest.fixture
def planning_react_agent():
    """Fixture to create a ReActAgent instance with planning enabled."""
    agent = ReActAgent(
        name='planning_react_agent',
        model_name=MODEL_NAME,
        tools=[dummy_tool_one, calculator, search_web, download_file],
        description='Test ReAct agent with planning for unit tests',
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
    assert "search_web" in desc
    assert "download_file" in desc
    assert "Description for dummy tool one" in desc


def test_search_arxiv():
    """Test the arxiv search tool for research papers."""
    query = "attention is all you need vaswani"
    results = search_arxiv(query=query, max_results=2)
    assert "attention is all you need" in results.lower()
    assert "vaswani" in results.lower()
    assert "## ArXiv Search Results for:" in results


@pytest.mark.asyncio
async def test_get_relevant_tools(react_agent):
    """Test filtering relevant tools for a task."""
    task_description = 'What is 2 plus 3?'  # Simple calculator task

    # Initialize the task with a proper task description
    react_agent._run_init(task_description)

    # This will make an actual API call to determine relevant tools
    tools = await react_agent.get_relevant_tools(task_description)

    # The task requires calculation, so calculator should be relevant
    assert len(tools) > 0, "No tools were returned from get_relevant_tools"
    tool_names = {t.name for t in tools}
    assert "calculator" in tool_names, "calculator should be relevant for arithmetic"


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




def test_planner_helpers(planner):
    """Test helper methods of the Planner class."""
    planner.plan = AgentPlan(steps=[
        PlanStep(description='First step.', is_done=False),
        PlanStep(description='Second step.', is_done=True)
    ])

    # test get_steps_done
    done_steps = planner.get_steps_done()
    assert len(done_steps) == 1
    assert done_steps[0].description == 'Second step.'

    # test get_steps_pending
    pending_steps = planner.get_steps_pending()
    assert len(pending_steps) == 1
    assert pending_steps[0].description == 'First step.'

    # test get_formatted_plan
    formatted_plan = planner.get_formatted_plan()
    expected = '- [ ] First step.\n- [x] Second step.'
    assert formatted_plan == expected

    formatted_done = planner.get_formatted_plan(scope='done')
    assert formatted_done == '- [x] Second step.'

    formatted_pending = planner.get_formatted_plan(scope='pending')
    assert formatted_pending == '- [ ] First step.'


@pytest.mark.asyncio
async def test_run_with_planning(planning_react_agent):
    """Test a successful run with planning enabled."""
    responses = []
    async for response in planning_react_agent.run('Add 2 and 2'):
        responses.append(response)

    plan_logs = [r for r in responses if r['type'] == 'log' and 'Plan:' in r['value']]
    assert len(plan_logs) == 1
    assert any(r['type'] == 'final' for r in responses)


@pytest.mark.asyncio
async def test_call_llm():
    """Test the public `call_llm` function."""
    response = await call_llm(
        model_name=MODEL_NAME,
        litellm_params={},
        messages=[{'role': 'user', 'content': 'Hello!'}],
        trace_id='test-trace-id'
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.fixture
def planner():
    """Fixture to create a Planner instance for testing."""
    return Planner(model_name=MODEL_NAME)


@pytest.mark.asyncio
async def test_planner_create_plan(planner):
    """Test plan creation by the Planner."""
    task = Task(description='Solve 2+2', files=[])
    plan = await planner.create_plan(task, 'ReActAgent')
    assert plan is not None
    assert isinstance(plan, AgentPlan)
    assert len(plan.steps) > 0
    assert planner.plan == plan


@pytest.mark.asyncio
async def test_planner_update_plan(planner):
    """Test plan progress update by the Planner."""
    planner.plan = AgentPlan(steps=[
        PlanStep(description='Use calculator to find 2+2.', is_done=False),
        PlanStep(description='Provide the final answer.', is_done=False)
    ])
    thought = 'I used the calculator and got the result.'
    observation = '4'
    task_id = 'test-task-123'
    await planner.update_plan(thought, observation, task_id)

    assert planner.plan is not None
    assert isinstance(planner.plan, AgentPlan)
    assert planner.plan.steps[0].is_done is True


@pytest.fixture
def observer():
    """Fixture to create an Observer instance for testing."""
    return Observer(
        model_name=MODEL_NAME,
        tool_names={'dummy_tool_one', 'calculator'},
        threshold=3
    )


@pytest.mark.asyncio
async def test_observer_threshold(observer):
    """Test that observer does not trigger before threshold."""
    task = Task(description='test task', files=None)
    history = 'some history'
    plan = AgentPlan(steps=[PlanStep(description='Step 1', is_done=False)])

    # Iteration 1, should not trigger
    correction = await observer.observe(1, task, history, plan, plan)
    assert correction is None
    assert observer.last_correction_iteration == 0

    # Iteration 2 (threshold=3), should not trigger
    correction = await observer.observe(2, task, history, plan, plan)
    assert correction is None
    assert observer.last_correction_iteration == 0


@pytest.mark.asyncio
@patch('kodeagent.call_llm', new_callable=AsyncMock)
async def test_observer_no_issues(mock_call_llm, observer):
    """Test that observer returns None when there are no issues."""
    mock_call_llm.return_value = '{"is_progressing": true, "is_in_loop": false, "reasoning": "all good"}'
    task = Task(description='test task', files=None)
    history = 'some history'
    plan = AgentPlan(steps=[PlanStep(description='Step 1', is_done=False)])

    correction = await observer.observe(4, task, history, plan, plan)
    assert correction is None
    mock_call_llm.assert_called_once()
    _args, kwargs = mock_call_llm.call_args
    assert kwargs['model_name'] == MODEL_NAME
    assert kwargs['response_format'] == ObserverResponse


@pytest.mark.asyncio
@patch('kodeagent.call_llm', new_callable=AsyncMock)
async def test_observer_detects_loop(mock_call_llm, observer):
    """Test observer returns a correction when a loop is detected."""
    mock_call_llm.return_value = (
        '{"is_progressing": false, "is_in_loop": true, "reasoning": "stuck in a loop",'
        ' "correction_message": "try something else"}'
    )
    task = Task(description='test task', files=None)
    history = 'some history'
    plan = AgentPlan(steps=[PlanStep(description='Step 1', is_done=False)])

    correction = await observer.observe(4, task, history, plan, plan)
    assert correction is not None
    assert 'try something else' in correction
    assert observer.last_correction_iteration == 4
    mock_call_llm.assert_called_once()
    _args, kwargs = mock_call_llm.call_args
    assert kwargs['model_name'] == MODEL_NAME
    assert kwargs['response_format'] == ObserverResponse


def test_observer_reset(observer):
    """Test that observer state is reset."""
    observer.last_correction_iteration = 5
    observer.reset()
    assert observer.last_correction_iteration == 0


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
    """Test the CodeActAgent functionality on a remote E2B sandbox."""
    import os
    if not os.getenv('E2B_API_KEY'):
        pytest.skip('E2B_API_KEY environment variable is not set.')

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


@pytest.mark.asyncio
async def test_codeact_agent_format_messages():
    """Test CodeActAgent message formatting."""
    code_agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    # Add some messages
    code_agent.add_to_history(ChatMessage(role='user', content='Test message'))
    code_agent.add_to_history(CodeChatMessage(
        role='assistant',
        thought='Testing code',
        code='print("hello")',
        content='',
        successful=False,
        answer=None
    ))
    code_agent.add_to_history(ChatMessage(role='tool', content='hello'))

    formatted = code_agent.format_messages_for_prompt()
    assert 'Thought: Testing code' in formatted
    assert 'Code:```py\nprint("hello")' in formatted
    assert 'Observation: hello' in formatted


@pytest.mark.asyncio
@patch('kodeagent.call_llm')
async def test_codeact_think_step(mock_call_llm):
    """Test CodeActAgent's think step."""
    mock_call_llm.return_value = (
        '{"role": "assistant", "thought": "test thought", "code": "print(2+2)", '
        '"content": "", "successful": false, "answer": null}'
    )

    code_agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    code_agent._run_init('Test task')
    responses = []
    async for response in code_agent._think():
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]['type'] == 'step'
    assert isinstance(responses[0]['value'], CodeChatMessage)
    assert responses[0]['value'].thought == 'test thought'
    assert responses[0]['value'].code == 'print(2+2)'


@pytest.mark.asyncio
async def test_llm_vision_support():
    """Test checking vision support for LLMs."""
    from kodeagent import llm_vision_support

    # Test with known models
    models = ['gemini/gemini-pro-vision', 'gpt-4-vision-preview']
    support_status = llm_vision_support(models)
    assert len(support_status) == len(models)
    assert all(isinstance(status, bool) for status in support_status)


def test_code_chat_message_validation():
    """Test CodeChatMessage validation."""
    # Valid message
    msg = CodeChatMessage(
        role="assistant",
        thought="test thought",
        code="print('test')",
        content="",
        successful=False,
        answer=None
    )
    assert msg.role == "assistant"
    assert msg.thought == "test thought"
    assert msg.code == "print('test')"

    # Missing required fields
    with pytest.raises(ValueError):
        CodeChatMessage(
            role="assistant",
            thought="",  # Empty thought
            code="print('test')",
            content="",
            successful=False,
            answer=None
        )


@pytest.mark.asyncio
async def test_agent_with_no_tools():
    """Test agent initialization with no tools."""
    agent = ReActAgent(
        name='no_tools_agent',
        model_name=MODEL_NAME,
        tools=[]
    )
    assert len(agent.tools) == 0
    assert len(agent.tool_names) == 0

    responses = []
    async for response in agent.run('Simple task'):
        responses.append(response)

    assert any('Simple task' in str(r['value']) for r in responses)


def test_agent_response_helper():
    """Test the response helper method of Agent."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # Test different response types
    step_response = agent.response('step', 'test value', 'test_channel')
    assert step_response['type'] == 'step'
    assert step_response['value'] == 'test value'
    assert step_response['channel'] == 'test_channel'

    # Test with metadata
    meta_response = agent.response('log', 'test', metadata={'key': 'value'})
    assert meta_response['metadata'] == {'key': 'value'}

    # Test final response
    final_response = agent.response('final', 'done')
    assert final_response['type'] == 'final'
    assert final_response['value'] == 'done'

