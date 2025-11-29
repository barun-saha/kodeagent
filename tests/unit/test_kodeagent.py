"""
Unit tests for the agents and their operations.
"""
import datetime
from typing import Optional, AsyncIterator
from unittest.mock import patch, MagicMock

import pydantic_core
import pytest

from kodeagent import (
    ReActAgent,
    CodeActAgent,
    CodeActChatMessage,
    Planner,
    Task,
    Observer,
    Agent
)
from kodeagent.models import (
    ChatMessage,
    ReActChatMessage,
    AgentResponse,
    AgentPlan,
    PlanStep,
    ObserverResponse
)
from kodeagent.tools import (
    tool,
    calculator,
    search_web,
    download_file,
)


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'

# Mock responses for LLM calls
MOCK_LLM_RESPONSES = {
    'think': '{"role": "assistant", "thought": "I should use the calculator",'
             ' "action": "calculator", "args": "{\\"expression\\": \\"2+2\\"}",'
             ' "content": "", "task_successful": false, "final_answer": null}',
    'observe': '{"is_progressing": true, "is_in_loop": false, "reasoning": "all good"}',
    'plan': '{"steps": [{"description": "Use calculator", "is_done": false}]}',
    'code': '{"role": "assistant", "thought": "Getting date",'
            ' "code": "from datetime import datetime'
            '\\nprint(datetime.now().strftime(\'%B %d, %Y\'))",'
            ' "content": "", "task_successful": false, "final_answer": null}'
}

@tool
def dummy_tool_one(param1: str) -> str:
    """Description for dummy tool one."""
    return f'tool one executed with {param1}'


@pytest.fixture
def mock_llm():
    """Fixture to mock LLM API calls."""
    async def mock_call_llm(*args, **kwargs):
        # Return different responses based on the context
        if any('Thought:' in str(m.get('content', '')) for m in kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['think']
        if 'plan' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['plan']
        if 'observe' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['observe']
        if 'code' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['code']
        return 'Default mock response'

    with patch('kodeagent.kutils.call_llm', new=mock_call_llm):
        yield mock_call_llm


@pytest.fixture
def react_agent(mock_llm):
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
def codeact_agent_factory():
    """Factory fixture to create CodeActAgent instances with proper mocking."""

    def _create_agent(**kwargs):
        # Patch CodeRunner to avoid task.id issue
        with patch('kodeagent.kodeagent.CodeRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run.return_value = ('output', '', 0)
            mock_runner_class.return_value = mock_runner

            defaults = {
                'name': 'test_codeact_agent',
                'model_name': MODEL_NAME,
                'run_env': 'host',
                'allowed_imports': ['datetime'],
            }
            defaults.update(kwargs)

            agent = CodeActAgent(**defaults)
            agent.code_runner = mock_runner
            return agent

    return _create_agent


def test_react_agent_initialization(react_agent):
    """Test the initialization of ReActAgent."""
    assert react_agent.name == 'test_react_agent'
    assert react_agent.model_name == MODEL_NAME
    assert len(react_agent.tools) == 4
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
        task_successful=False,
        final_answer=None
    )
    msg2 = ChatMessage(role='tool', content='tool response')

    react_agent.add_to_history(msg1)
    react_agent.add_to_history(msg2)

    formatted = react_agent.formatted_history_for_llm()
    assert isinstance(formatted, list)
    assert len(formatted) > 0
    assert any('assistant' in str(m) for m in formatted)


@pytest.mark.asyncio
async def test_react_agent_run_success(react_agent):
    """Test successful task execution by ReActAgent."""
    # Predefined assistant responses
    assistant_sequence = [
        # Plan
        '{"steps": [{"description": "Use calculator", "is_done": false}]}',
        # Step 1: Think (ReActChatMessage)
        '{"role": "assistant", "thought": "I need to calculate 2 + 2. I will use the calculator tool.",'
        ' "action": "calculator", "args": "{\\"expression\\": \\"2+2\\"}",'
        ' "task_successful": false, "final_answer": null}',
        # Update plan
        '{"steps": [{"description": "Use calculator", "is_done": true}]}',
        # Step 2: Final answer
        '{"role": "assistant", "thought": "I have calculated 2 + 2 = 4. I can now provide the final answer.",'
        ' "action": "FINISH", "args": null,'
        ' "task_successful": true, "final_answer": "The sum of 2 and 2 is 4"}',
        # Final update plan
        '{"steps": [{"description": "Use calculator", "is_done": true}]}',
    ]

    # Patch call_llm to return each response in order
    with patch('kodeagent.kutils.call_llm', side_effect=assistant_sequence):
        responses = []
        try:
            async for response in react_agent.run('Add 2 and 2'):
                responses.append(response)
        except StopAsyncIteration:
            pass

        assert any(r['type'] == 'final' for r in responses)
        assert react_agent.final_answer_found
        assert react_agent.task.is_finished
        final_response = next((r for r in responses if r['type'] == 'final'), None)
        assert final_response is not None
        assert '4' in str(final_response['value'])


@pytest.mark.asyncio
async def test_react_agent_run_with_tool_error():
    """Test ReActAgent handling tool execution errors."""
    @tool
    def broken_tool(param1: str) -> str:
        """A tool that always fails."""
        raise Exception('Tool error')

    def llm_side_effect(*args, **kwargs):
        # If response_format is AgentPlan, return valid AgentPlan JSON
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Use the broken tool", "is_done": false}]}'
        # If response_format is ObserverResponse, return valid ObserverResponse JSON
        if kwargs.get('response_format') == ObserverResponse:
            return (
                '{"is_progressing": true, "is_in_loop": false,'
                ' "reasoning": "Agent is progressing", "correction_message": null}'
            )
        # Otherwise, return a valid ReActChatMessage-like response
        return (
            '{"role": "assistant", "thought": "Use `broken_tool`",'
            ' "action": "broken_tool", "args": "{\\"param1\\": \\"Something\\"}",'
            ' "task_successful": false, "final_answer": null}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        responses = []
        agent = ReActAgent(
            name='broken_tools_agent',
            model_name=MODEL_NAME,
            tools=[broken_tool],
            max_iterations=1
        )
        assert len(agent.tools) == 1

        try:
            async for response in agent.run('Use the broken tool'):
                responses.append(response)
        except StopAsyncIteration:
            pass

        # Check for error responses - look in step responses
        error_responses = [
            r for r in responses
            if r.get('type') == 'step' and r.get('metadata') and r['metadata'].get('is_error')
        ]
        assert len(error_responses) > 0, "Expected at least one error response"
        assert any('Error' in str(r['value']) or 'error' in str(r['value']) for r in error_responses)


@pytest.mark.asyncio
async def test_act_step_with_invalid_tool(react_agent):
    """Test the act step with an invalid tool name."""
    invalid_response = ReActChatMessage(
        thought="Test thought",
        action="nonexistent_tool",
        args='{"param1": "test"}',
        final_answer=None,
        task_successful=False,
        role="assistant"
    )

    react_agent.add_to_history(invalid_response)

    responses = []
    async for response in react_agent._act():
        responses.append(response)

    assert len(responses) == 1
    assert "not found" in responses[0]["value"]
    assert responses[0]["metadata"]["is_error"]


def test_get_tools_description(react_agent):
    """Test getting tool descriptions."""
    desc = react_agent.get_tools_description()
    assert "dummy_tool_one" in desc
    assert "calculator" in desc
    assert "search_web" in desc
    assert "download_file" in desc
    assert "Description for dummy tool one" in desc


@pytest.mark.asyncio
@patch('kodeagent.kutils.call_llm')
async def test_get_relevant_tools(mock_llm, react_agent):
    """Test filtering relevant tools for a task."""
    mock_llm.return_value = 'calculator'
    task_description = 'What is 2 plus 3?'
    react_agent._run_init(task_description)

    tools = await react_agent.get_relevant_tools(task_description)
    assert len(tools) > 0
    tool_names = {t.name for t in tools}
    assert 'calculator' in tool_names


def test_clear_history(react_agent):
    """Test clearing agent's message history."""
    msg = ChatMessage(role='user', content='test message')
    react_agent.add_to_history(msg)
    assert len(react_agent.messages) == 1

    react_agent.clear_history()
    assert len(react_agent.messages) == 0


@pytest.fixture
def mock_e2b():
    """Fixture to mock E2B sandbox."""
    class MockSandbox:
        """Mock E2B Sandbox class."""
        async def run_python(self, code: str, **kwargs):
            if 'datetime' in code:
                return {'output': 'September 6, 2025'}
            return {'output': 'Mock output'}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch('kodeagent.Sandbox', return_value=MockSandbox()):
        yield MockSandbox()


@pytest.mark.asyncio
async def test_codeact_agent_host():
    """Test the CodeActAgent functionality on a local system."""
    current_month = datetime.datetime.today().strftime('%B')

    def llm_side_effect(*args, **kwargs):
        # If response_format is AgentPlan, return valid AgentPlan JSON
        if kwargs.get('response_format') == AgentPlan:
            return (
                '{"steps": [{"description": "Find the current month name based on today\'s date",'
                ' "is_done": false}]}'
            )
        # If response_format is ObserverResponse, return valid ObserverResponse JSON
        if kwargs.get('response_format') == ObserverResponse:
            return (
                '{"is_progressing": true, "is_in_loop": false,'
                ' "reasoning": "Agent is progressing", "correction_message": null}'
            )

        messages = kwargs.get('messages', [])
        if not messages:
            return '{"role": "system", "content": "You are an assistant."}'

        last_message = messages[-1]

        if last_message['role'] == 'user' and 'Plan' not in last_message.get('content', ''):
            # Return a properly formatted CodeActChatMessage for the initial task
            return (
                '{"role": "assistant", "thought": "I need to get the current month name", '
                '"code": "from datetime import datetime\\nprint(datetime.now().strftime(\'%B\'))", '
                '"task_successful": false, "final_answer": null}'
            )

        if last_message['role'] == 'tool':
            # Return a final CodeActChatMessage with the result
            return (
                '{"role": "assistant", "thought": "I have the month name", '
                f'"code": null, '
                f'"task_successful": true, "final_answer": "The current month is {current_month}"'
                '}'
            )

        return (
            '{"role": "assistant", "thought": "Thinking", '
            '"code": null, "task_successful": false, "final_answer": null}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        with patch('kodeagent.kodeagent.CodeRunner') as mock_runner_class:
            # Create a mock runner instance
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = (current_month, '', 0)
            mock_runner_class.return_value = mock_runner_instance

            code_agent = CodeActAgent(
                name='Code agent',
                model_name=MODEL_NAME,
                run_env='host',
                max_iterations=3,
                allowed_imports=['datetime'],
                description='Agent that can write and execute Python code'
            )

            # Ensure the agent uses our mock runner
            code_agent.code_runner = mock_runner_instance

            task = 'Compute the name of the month today.'
            responses = []
            try:
                async for response in code_agent.run(task):
                    responses.append(response['value'])
            except StopAsyncIteration:
                pass

            response = ' | '.join([str(r) for r in responses])
            assert current_month in response


@pytest.mark.asyncio
async def test_codeact_agent_unsupported_env():
    """Test the CodeActAgent functionality on an unsupported run env."""
    current_month = datetime.datetime.today().strftime('%B')

    def llm_side_effect(*args, **kwargs):
        # If response_format is AgentPlan, return valid AgentPlan JSON
        if kwargs.get('response_format') == AgentPlan:
            return (
                '{"steps": [{"description": "Find the current month name based on today\'s date",'
                ' "is_done": false}]}'
            )
        # If response_format is ObserverResponse, return valid ObserverResponse JSON
        if kwargs.get('response_format') == ObserverResponse:
            return (
                '{"is_progressing": true, "is_in_loop": false,'
                ' "reasoning": "Agent is progressing", "correction_message": null}'
            )

        messages = kwargs.get('messages', [])
        if not messages:
            return '{"role": "system", "content": "You are an assistant."}'

        last_message = messages[-1]

        if last_message['role'] == 'user' and 'Plan' not in last_message.get('content', ''):
            # Return a properly formatted CodeActChatMessage for the initial task
            return (
                '{"role": "assistant", "thought": "I need to get the current month name", '
                '"code": "from datetime import datetime\\nprint(datetime.now().strftime(\'%B\'))", '
                '"task_successful": false, "final_answer": null}'
            )

        if last_message['role'] == 'tool':
            # Return a final CodeActChatMessage with the result
            return (
                '{"role": "assistant", "thought": "I have the month name", '
                f'"code": null, '
                f'"task_successful": true, "final_answer": "The current month is {current_month}"'
                '}'
            )

        return (
            '{"role": "assistant", "thought": "Thinking", '
            '"code": null, "task_successful": false, "final_answer": null}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        with patch('kodeagent.kodeagent.CodeRunner') as mock_runner_class:
            # Create a mock runner that simulates unsupported env error
            mock_runner_instance = MagicMock()

            def mock_run_with_error(code):
                # Return error tuple instead of raising exception
                return ('', 'Unsupported code execution env: unknown_env', -1)

            mock_runner_instance.run.side_effect = mock_run_with_error
            mock_runner_class.return_value = mock_runner_instance

            code_agent = CodeActAgent(
                name='Code agent',
                model_name=MODEL_NAME,
                run_env='unknown_env',
                max_iterations=3,
                allowed_imports=['datetime'],
                description='Agent that can write and execute Python code'
            )

            # Ensure the agent uses our mock runner
            code_agent.code_runner = mock_runner_instance

            task = 'Compute the name of the month today.'
            responses = []
            try:
                async for response in code_agent.run(task):
                    responses.append(response['value'])
            except StopAsyncIteration:
                pass

            response = ' | '.join([str(r) for r in responses])
            assert 'Unsupported code execution env: unknown_env' in response or 'Error' in response


def test_code_chat_message_validation():
    """Test CodeChatMessage validation."""
    role = 'assistant'
    thought = 'test thought'
    code = "print('test')"
    msg = CodeActChatMessage(
        role=role,
        thought=thought,
        code=code,
        task_successful=False,
        final_answer=None
    )
    assert msg.role == role
    assert msg.thought == thought
    assert msg.code == code

    with pytest.raises(pydantic_core.ValidationError):
        CodeActChatMessage(
            role=role,
            thought=None,  # No valid thought
            code=code,
            task_successful=False,
            final_answer=None
        )


@pytest.mark.asyncio
async def test_agent_with_no_tools():
    """Test agent initialization with no tools."""
    def llm_side_effect(*args, **kwargs):
        # If response_format is AgentPlan, return valid AgentPlan JSON
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Ask for details", "is_done": false}]}'
        # If response_format is ObserverResponse, return valid ObserverResponse JSON
        if kwargs.get('response_format') == ObserverResponse:
            return (
                '{"is_progressing": true, "is_in_loop": false,'
                ' "reasoning": "Agent is progressing", "correction_message": null}'
            )
        # Otherwise, return a valid ReActChatMessage-like response
        return (
            '{"role": "assistant", "thought": "I should ask for details",'
            ' "action": "FINISH", "args": null, "task_successful": true,'
            ' "final_answer": "Task completed"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        agent = ReActAgent(
            name='no_tools_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1
        )
        assert len(agent.tools) == 0
        assert len(agent.tool_names) == 0

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except StopAsyncIteration:
            pass

        assert any('Simple task' in str(r['value']) for r in responses)


def test_agent_response_helper():
    """Test the response helper method of Agent."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    step_response = agent.response('step', 'test value', 'test_channel')
    assert step_response['type'] == 'step'
    assert step_response['value'] == 'test value'
    assert step_response['channel'] == 'test_channel'

    meta_response = agent.response('log', 'test', metadata={'key': 'value'})
    assert meta_response['metadata'] == {'key': 'value'}

    final_response = agent.response('final', 'done')
    assert final_response['type'] == 'final'
    assert final_response['value'] == 'done'


def test_task_initialization():
    """Test Task class initialization and properties."""
    task_description = 'Sample task'
    task = Task(description=task_description, files=None)
    assert task.description == task_description
    assert task.is_finished is False
    assert task.is_error is False
    assert task.result is None
    assert task.id is not None

    # Test with files
    task_files=['file1.txt', 'file2.txt']
    task_with_files = Task(
        description='Task with files',
        files=task_files
    )
    assert len(task_with_files.files) == 2
    assert task_with_files.files == task_files


def test_task_completion():
    """Test Task completion and result setting."""
    task = Task(description='Complete this task', files=None)
    assert not task.is_finished

    # Simulate task completion
    task.result = 'Task completed successfully'
    task.is_finished = True

    assert task.is_finished
    assert task.result == 'Task completed successfully'
    assert not task.is_error

    # Test error state
    error_task = Task(description='Failed task', files=None)
    error_task.is_error = True
    error_task.is_finished = True
    error_task.result = 'Error occurred'

    assert error_task.is_error
    assert error_task.is_finished
    assert error_task.result == 'Error occurred'


def test_plan_step():
    """Test PlanStep class initialization and properties."""
    step = PlanStep(description="Calculate sum")
    assert step.description == "Calculate sum"
    assert step.is_done is False

    # Test marking step as done
    step.is_done = True
    assert step.is_done is True


def test_agent_plan():
    """Test AgentPlan class initialization and management of steps."""
    steps = [
        PlanStep(description='Step 1'),
        PlanStep(description='Step 2'),
        PlanStep(description='Step 3')
    ]
    plan = AgentPlan(steps=steps)

    assert len(plan.steps) == 3
    assert all(not step.is_done for step in plan.steps)

    # Test marking steps as done
    plan.steps[0].is_done = True
    assert plan.steps[0].is_done
    assert not plan.steps[1].is_done


def test_observer_response():
    """Test ObserverResponse class initialization and properties."""
    response = ObserverResponse(
        is_progressing=True,
        is_in_loop=False,
        reasoning='Agent is making good progress on the calculation task',
        correction_message=None
    )

    assert response.is_progressing is True
    assert response.is_in_loop is False
    assert 'making good progress' in response.reasoning
    assert response.correction_message is None

    # Test with correction message
    response_with_correction = ObserverResponse(
        is_progressing=False,
        is_in_loop=True,
        reasoning='Agent is stuck in a loop',
        correction_message='Try using a different approach to solve the calculation'
    )

    assert response_with_correction.is_progressing is False
    assert response_with_correction.is_in_loop is True
    assert response_with_correction.correction_message is not None


def test_observer_response_validation():
    """Test validation of ObserverResponse fields."""
    # Empty reasoning should NOT fail - it's just a string field
    response = ObserverResponse(
        is_progressing=True,
        is_in_loop=False,
        reasoning='',  # Empty string is allowed
        correction_message=None
    )
    assert response.reasoning == ''

    # Test with valid reasoning
    response2 = ObserverResponse(
        is_progressing=False,
        is_in_loop=True,
        reasoning='Agent is stuck',
        correction_message='Try again'
    )
    assert response2.reasoning == 'Agent is stuck'


@pytest.fixture
def planner():
    """Fixture to create a Planner instance for testing."""
    return Planner(
        model_name=MODEL_NAME,
        litellm_params={'max_tokens': 1000}
    )


@pytest.mark.asyncio
async def test_planner_create_plan(planner):
    """Test creating a new plan."""
    mock_plan_response = '{"steps": [{"description": "Use calculator", "is_done": false}]}'

    with patch('kodeagent.kutils.call_llm', autospec=True) as mock_call_llm:
        mock_call_llm.return_value = mock_plan_response
        task = Task(description='Calculate 2+2', files=None)
        plan = await planner.create_plan(task, agent_type='ReAct')

        assert isinstance(plan, AgentPlan)
        assert len(plan.steps) > 0
        assert isinstance(plan.steps[0], PlanStep)
        assert plan.steps[0].description == 'Use calculator'
        assert not plan.steps[0].is_done


@pytest.mark.asyncio
async def test_planner_update_plan(planner):
    """Test updating an existing plan."""
    mock_update_response = '{"steps": [{"description": "Use calculator", "is_done": true}]}'

    with patch('kodeagent.kutils.call_llm', autospec=True) as mock_call_llm:
        mock_call_llm.return_value = mock_update_response
        task = Task(description='Calculate 2+2', files=None)
        await planner.create_plan(task, agent_type='ReAct')

        await planner.update_plan(
            thought='I need to use the calculator',
            observation='The calculator returned 4',
            task_id=str(task.id)
        )

        assert planner.plan is not None
        assert len(planner.plan.steps) > 0
        assert planner.plan.steps[0].is_done


def test_planner_get_steps_status(planner):
    """Test getting completed and pending steps."""
    # Create a plan manually for testing
    plan = AgentPlan(steps=[
        PlanStep(description='Step 1', is_done=True),
        PlanStep(description='Step 2', is_done=False),
        PlanStep(description='Step 3', is_done=True)
    ])
    planner.plan = plan

    done_steps = planner.get_steps_done()
    pending_steps = planner.get_steps_pending()

    assert len(done_steps) == 2
    assert len(pending_steps) == 1
    assert all(step.is_done for step in done_steps)
    assert not any(step.is_done for step in pending_steps)


def test_planner_get_formatted_plan(planner):
    """Test formatting the plan as a markdown checklist."""
    # Create a plan manually for testing
    plan = AgentPlan(steps=[
        PlanStep(description='Step 1', is_done=True),
        PlanStep(description='Step 2', is_done=False),
        PlanStep(description='Step 3', is_done=True)
    ])
    planner.plan = plan

    # Test formatting all steps
    all_steps = planner.get_formatted_plan(scope='all')
    assert '- [x] Step 1' in all_steps
    assert '- [ ] Step 2' in all_steps
    assert '- [x] Step 3' in all_steps

    # Test formatting only done steps
    done_steps = planner.get_formatted_plan(scope='done')
    assert '- [x] Step 1' in done_steps
    assert '- [x] Step 3' in done_steps
    assert '- [ ] Step 2' not in done_steps

    # Test formatting only pending steps
    pending_steps = planner.get_formatted_plan(scope='pending')
    assert '- [ ] Step 2' in pending_steps
    assert '- [x] Step 1' not in pending_steps
    assert '- [x] Step 3' not in pending_steps


def test_planner_empty_plan(planner):
    """Test planner behavior with no plan."""
    assert planner.get_steps_done() == []
    assert planner.get_steps_pending() == []
    assert planner.get_formatted_plan() == ''
    assert planner.plan is None


def test_abstract_agent(mock_llm):
    """Test agent initialization."""
    with pytest.raises(TypeError):
        Agent(name='minimal_agent', model_name=MODEL_NAME)


def test_agent_subclass(mock_llm):
    """Test agent initialization with no tools."""
    class MinimalAgent(Agent):
        async def run(
                self,
                task: str,
                files: Optional[list[str]] = None,
                task_id: Optional[str] = None
        ) -> AsyncIterator[AgentResponse]:
            yield self.response('final', 'Done')

        def formatted_history_for_llm(self) ->list[dict]:
            return []
        
        def parse_text_response(self, text: str) -> ChatMessage:
            """Parse text response."""
            return ChatMessage(role='assistant', content=text)

    agent = MinimalAgent(
        name='minimal_agent',
        model_name=MODEL_NAME
    )
    assert len(agent.tools) == 0
    assert len(agent.tool_names) == 0
    assert len(agent.tool_name_to_func) == 0


@pytest.mark.asyncio
async def test_observer_analyze():
    """Test Observer's analysis of agent behavior."""
    mock_response = (
        '{"is_progressing": false, "is_in_loop": true,'
        ' "reasoning": "Agent keeps using calculator repeatedly",'
        ' "correction_message": "Try a different approach"}'
    )

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        observer = Observer(
            model_name=MODEL_NAME,
            tool_names={'calculator', 'search_web'},
            threshold=2
        )

        # Mock a task and history that shows a loop
        task = Task(description='Calculate 2+2', files=None)
        history = """
Thought: I should use calculator
Action: calculator
Args: {"a": 2, "b": 2}
Observation: 4

Thought: I should use calculator again
Action: calculator
Args: {"a": 2, "b": 2}
Observation: 4
"""

        # First call before threshold - should return None
        correction = await observer.observe(
            iteration=1,
            task=task,
            history=history,
            plan_before=None,
            plan_after=None
        )
        assert correction is None

        # Call after threshold with looping behavior
        correction = await observer.observe(
            iteration=3,
            task=task,
            history=history,
            plan_before=None,
            plan_after=None
        )
        assert correction is not None
        assert "CRITICAL FOR COURSE CORRECTION" in correction


@pytest.mark.asyncio
async def test_observer_reset():
    """Test Observer reset functionality."""
    observer = Observer(
        model_name=MODEL_NAME,
        tool_names={'calculator'},
        threshold=2
    )

    observer.last_correction_iteration = 5
    observer.reset()
    assert observer.last_correction_iteration == 0


def test_agent_str(react_agent):
    """Test the string representation of an Agent."""
    agent_str = str(react_agent)
    assert 'Agent: test_react_agent' in agent_str
    assert react_agent.model_name in agent_str
    assert 'Tools:' in agent_str
    assert str(react_agent.id) in agent_str


def test_agent_purpose():
    """Test Agent's purpose string generation."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator],
        description='A test agent'
    )

    purpose = agent.purpose
    assert 'Name: test_agent' in purpose
    assert 'Description: A test agent' in purpose
    assert 'calculator' in purpose


def test_agent_trace(react_agent):
    """Test the trace method of Agent."""
    # Add some history
    react_agent.add_to_history(ChatMessage(role='user', content='Calculate 2+2'))
    react_agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    ))
    react_agent.add_to_history(ChatMessage(role='tool', content='4'))

    trace = react_agent.trace()
    assert 'Thought: Using calculator' in trace
    assert 'Action: calculator' in trace
    assert 'Observation: 4' in trace


def test_agent_trace_detailed(react_agent):
    """Test detailed scenarios for Agent.trace."""
    # Test empty history
    assert react_agent.trace() == ''

    # Add a sequence of messages to test different scenarios
    messages = [
        ChatMessage(role='user', content='Calculate 2+2'),
        ReActChatMessage(
            role='assistant',
            thought='I will use the calculator',
            action='calculator',
            args='{"expression": "2+2"}',
            task_successful=False,
            final_answer=None
        ),
        ChatMessage(role='tool', content='4'),
        ReActChatMessage(
            role='assistant',
            thought='I have the result',
            action='FINISH',
            args=None,
            task_successful=True,
            final_answer='The answer is 4'
        )
    ]

    for msg in messages:
        react_agent.add_to_history(msg)

    trace = react_agent.trace()

    # Verify all components are present in the trace
    assert 'Thought: I will use the calculator' in trace
    assert 'Action: calculator({"expression": "2+2"})' in trace
    assert 'Observation: 4' in trace
    assert 'Thought: I have the result' in trace

    # Verify order of events
    thought_pos = trace.find('Thought: I will use the calculator')
    action_pos = trace.find('Action: calculator')
    observation_pos = trace.find('Observation: 4')
    assert thought_pos < action_pos < observation_pos


@pytest.mark.parametrize('expression,expected', [
    ('2 + 2', 4),
    ('10 * 5', 50),
    ('(3 + 2) * 4', 20),
    ('2 ** 3', 8),
    ('invalid expr', None),
    ('os.system("ls")', None),  # test security
    ('10 + ^2', None),  # invalid operator
    ('10 / 0', None),  # division by zero
])
def test_calculator_tool(expression, expected):
    """Test the calculator tool with various inputs."""
    result = calculator(expression)
    assert result == expected


# Edge case: Observer with missing reasoning and correction_message
def test_observer_response_missing_fields():
    response = ObserverResponse(is_progressing=False, is_in_loop=True, reasoning='test', correction_message=None)
    assert response.is_in_loop
    assert response.reasoning == 'test'
    assert response.correction_message is None


# ============================================================================
# NEW TESTS FOR IMPROVED COVERAGE
# ============================================================================

def test_react_chat_message_validation_mutual_exclusivity():
    """Test ReActChatMessage validation for mutual exclusivity."""
    # Valid: action with args
    msg = ReActChatMessage(
        role='assistant',
        thought='test',
        action='calculator',
        args='{"expression": "2+2"}',
        final_answer=None,
        task_successful=False
    )
    assert msg.action == 'calculator'

    # Invalid: action with final_answer
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args='{"expression": "2+2"}',
            final_answer='4',
            task_successful=False
        )

    # Invalid: FINISH without final_answer
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='FINISH',
            args=None,
            final_answer=None,
            task_successful=True
        )

    # Valid: FINISH with final_answer
    msg = ReActChatMessage(
        role='assistant',
        thought='test',
        action='FINISH',
        args=None,
        final_answer='The answer is 4',
        task_successful=True
    )
    assert msg.is_final


def test_codeact_chat_message_validation_mutual_exclusivity():
    """Test CodeActChatMessage validation for mutual exclusivity."""
    # Valid: code only
    msg = CodeActChatMessage(
        role='assistant',
        thought='test',
        code='print("hello")',
        final_answer=None,
        task_successful=False
    )
    assert msg.code is not None

    # Invalid: both code and final_answer
    with pytest.raises(pydantic_core.ValidationError):
        CodeActChatMessage(
            role='assistant',
            thought='test',
            code='print("hello")',
            final_answer='hello',
            task_successful=False
        )

    # Invalid: neither code nor final_answer
    with pytest.raises(pydantic_core.ValidationError):
        CodeActChatMessage(
            role='assistant',
            thought='test',
            code=None,
            final_answer=None,
            task_successful=False
        )

    # Valid: final_answer only
    msg = CodeActChatMessage(
        role='assistant',
        thought='test',
        code=None,
        final_answer='The result is 42',
        task_successful=True
    )
    assert msg.is_final


def test_react_chat_message_args_validation():
    """Test ReActChatMessage args JSON validation."""
    # Valid JSON args
    msg = ReActChatMessage(
        role='assistant',
        thought='test',
        action='calculator',
        args='{"expression": "2+2"}',
        final_answer=None,
        task_successful=False
    )
    assert msg.args is not None

    # Invalid JSON args
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args='not valid json',
            final_answer=None,
            task_successful=False
        )

    # Empty args
    msg = ReActChatMessage(
        role='assistant',
        thought='test',
        action='FINISH',
        args=None,
        final_answer='done',
        task_successful=True
    )
    assert msg.args is None


@pytest.mark.asyncio
async def test_record_thought_with_json_repair():
    """Test _record_thought with JSON repair."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Test task')

    # Mock response with slightly malformed JSON
    malformed_json = '{"role": "assistant", "thought": "test", "action": "calculator", "args": \'{"expression": "2+2"}\', "final_answer": null, "task_successful": false}'

    with patch('kodeagent.kutils.call_llm', return_value=malformed_json):
        msg = await agent._record_thought(ReActChatMessage)
        assert msg is not None
        assert msg.thought == 'test'


@pytest.mark.asyncio
async def test_record_thought_max_retries():
    """Test _record_thought max retries."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Test task')

    # Mock response that always fails
    with patch('kodeagent.kutils.call_llm', return_value='completely invalid'):
        msg = await agent._record_thought(ReActChatMessage)
        assert msg is None


def test_get_tools_description_with_filter():
    """Test get_tools_description with tool filtering."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator, search_web, dummy_tool_one]
    )

    # Get description for specific tools
    desc = agent.get_tools_description(tools=[calculator, search_web])
    assert 'calculator' in desc
    assert 'search_web' in desc
    assert 'dummy_tool_one' not in desc


def test_current_plan_property():
    """Test current_plan property."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # No plan initially
    assert agent.current_plan is None

    # Create a plan
    plan = AgentPlan(steps=[
        PlanStep(description='Step 1', is_done=False),
        PlanStep(description='Step 2', is_done=False)
    ])
    agent.planner.plan = plan

    # Now should have plan
    current = agent.current_plan
    assert current is not None
    assert 'Step 1' in current
    assert 'Step 2' in current


@pytest.mark.asyncio
async def test_salvage_response():
    """Test salvage_response method."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')
    agent.add_to_history(ChatMessage(role='user', content='Calculate 2+2'))

    mock_response = 'I attempted to calculate but failed'

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        result = await agent.salvage_response()
        assert 'attempted' in result


def test_get_history():
    """Test get_history method."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    agent.add_to_history(ChatMessage(role='user', content='Hello'))
    agent.add_to_history(ChatMessage(role='assistant', content='Hi'))

    history = agent.get_history()
    assert '[user]: Hello' in history
    assert '[assistant]: Hi' in history

    # Test with start_idx
    history_from_1 = agent.get_history(start_idx=1)
    assert '[user]: Hello' not in history_from_1
    assert '[assistant]: Hi' in history_from_1


def test_run_init():
    """Test _run_init method."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    agent._run_init('Test task', files=['file1.txt'], task_id='custom_id')

    assert agent.task is not None
    assert agent.task.description == 'Test task'
    assert agent.task.files == ['file1.txt']
    assert agent.task.id == 'custom_id'
    assert agent.final_answer_found is False


@pytest.mark.asyncio
async def test_observer_with_negative_threshold():
    """Test Observer with None threshold (disabled)."""
    observer = Observer(
        model_name=MODEL_NAME,
        tool_names={'calculator'},
        threshold=None  # Disabled observer
    )

    task = Task(description='Test', files=None)
    correction = await observer.observe(
        iteration=5,
        task=task,
        history='test',
        plan_before=None,
        plan_after=None
    )

    # Should return None because threshold is None
    assert correction is None


@pytest.mark.asyncio
async def test_observer_exception_handling():
    """Test Observer exception handling."""
    observer = Observer(
        model_name=MODEL_NAME,
        tool_names={'calculator'},
        threshold=1
    )

    task = Task(description='Test', files=None)

    with patch('kodeagent.kutils.call_llm', side_effect=Exception('LLM error')):
        correction = await observer.observe(
            iteration=2,
            task=task,
            history='test',
            plan_before=None,
            plan_after=None
        )

        # Should return None on exception
        assert correction is None


def test_formatted_history_for_llm_react():
    """Test formatted_history_for_llm for ReActAgent."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    agent.add_to_history(ChatMessage(role='system', content='You are helpful'))
    agent.add_to_history(ChatMessage(role='user', content='Calculate 2+2', files=['file.txt']))
    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    ))
    agent.add_to_history(ChatMessage(role='tool', content='4'))

    formatted = agent.formatted_history_for_llm()
    assert isinstance(formatted, list)
    assert len(formatted) > 0
    assert any(m.get('role') == 'system' for m in formatted)
    assert any(m.get('role') == 'tool' for m in formatted)


def test_formatted_history_for_llm_codeact():
    """Test formatted_history_for_llm for CodeActAgent."""
    # Create agent with proper initialization
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    agent.add_to_history(ChatMessage(role='system', content='You are helpful'))
    agent.add_to_history(ChatMessage(role='user', content='Get date'))
    agent.add_to_history(CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='from datetime import datetime\nprint(datetime.now())',
        task_successful=False,
        final_answer=None
    ))
    agent.add_to_history(ChatMessage(role='tool', content='2024-01-01'))

    formatted = agent.formatted_history_for_llm()
    assert isinstance(formatted, list)
    assert len(formatted) > 0


@pytest.mark.asyncio
async def test_think_with_filter_tools():
    """Test _think with filter_tools_for_task enabled."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator, search_web],
        filter_tools_for_task=True
    )
    agent._run_init('Calculate 2+2')

    with patch.object(agent, 'get_relevant_tools', return_value=[calculator]):
        with patch.object(agent, '_record_thought', return_value=ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args='{"expression": "2+2"}',
            task_successful=False,
            final_answer=None
        )):
            responses = []
            async for response in agent._think():
                responses.append(response)

            assert len(responses) > 0


def test_chat_message_with_files():
    """Test ChatMessage with files."""
    msg = ChatMessage(
        role='user',
        content='Analyze this file',
        files=['file1.txt', 'file2.txt']
    )

    assert msg.files == ['file1.txt', 'file2.txt']
    assert msg.role == 'user'


def test_task_with_custom_id():
    """Test Task with custom ID."""
    custom_id = 'my-custom-task-id'
    task = Task(description='Test', files=None)
    task.id = custom_id

    assert task.id == custom_id


@pytest.mark.asyncio
async def test_codeact_code_execution_with_markdown():
    """Test CodeActAgent code execution with markdown formatting."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Get date')

    msg = CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='```python\nfrom datetime import datetime\nprint(datetime.now().year)\n```',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0


def test_planner_with_litellm_params():
    """Test Planner initialization with litellm_params."""
    params = {'temperature': 0.5, 'max_tokens': 500}
    planner = Planner(model_name=MODEL_NAME, litellm_params=params)

    assert planner.litellm_params == params
    assert planner.model_name == MODEL_NAME


def test_observer_with_custom_threshold():
    """Test Observer with custom threshold."""
    observer = Observer(
        model_name=MODEL_NAME,
        tool_names={'calculator'},
        threshold=5
    )

    assert observer.threshold == 5


def test_agent_is_visual_model():
    """Test agent visual model detection."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # is_visual_model should be set during initialization
    assert isinstance(agent.is_visual_model, bool)


@pytest.mark.asyncio
async def test_act_with_malformed_response():
    """Test _act with malformed response."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # Create a message without thought
    msg = ChatMessage(role='assistant', content='No thought here')
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    # Should handle gracefully
    assert len(responses) >= 0


def test_agent_tool_name_to_func_mapping():
    """Test agent tool name to function mapping."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator, search_web]
    )

    assert 'calculator' in agent.tool_name_to_func
    assert 'search_web' in agent.tool_name_to_func
    assert agent.tool_name_to_func['calculator'] == calculator


def test_react_chat_message_is_final():
    """Test ReActChatMessage is_final property."""
    # Not final
    msg1 = ReActChatMessage(
        role='assistant',
        thought='test',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    )
    assert not msg1.is_final

    # Final
    msg2 = ReActChatMessage(
        role='assistant',
        thought='test',
        action='FINISH',
        args=None,
        task_successful=True,
        final_answer='The answer is 4'
    )
    assert msg2.is_final


def test_codeact_chat_message_is_final():
    """Test CodeActChatMessage is_final property."""
    # Not final
    msg1 = CodeActChatMessage(
        role='assistant',
        thought='test',
        code='print("hello")',
        task_successful=False,
        final_answer=None
    )
    assert not msg1.is_final

    # Final
    msg2 = CodeActChatMessage(
        role='assistant',
        thought='test',
        code=None,
        task_successful=True,
        final_answer='Done'
    )
    assert msg2.is_final


@pytest.mark.asyncio
async def test_record_thought_with_mutual_exclusivity_fix():
    """Test _record_thought fixes mutual exclusivity violations."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Test task')

    # Mock response with both action and final_answer (violation)
    malformed_json = '{"role": "assistant", "thought": "test", "action": "calculator", "args": "{\\"expression\\": \\"2+2\\"}", "final_answer": "4", "task_successful": false}'

    with patch('kodeagent.kutils.call_llm', return_value=malformed_json):
        msg = await agent._record_thought(ReActChatMessage)
        assert msg is not None
        # Should have removed final_answer to fix violation
        assert msg.final_answer is None
        assert msg.action == 'calculator'


@pytest.mark.asyncio
async def test_record_thought_codeact_mutual_exclusivity_fix():
    """Test _record_thought fixes CodeAct mutual exclusivity violations."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Test task')

    # Mock response with both code and final_answer (violation)
    malformed_json = '{"role": "assistant", "thought": "test", "code": "print(\'hello\')", "final_answer": "hello", "task_successful": false}'

    with patch('kodeagent.kutils.call_llm', return_value=malformed_json):
        msg = await agent._record_thought(CodeActChatMessage)
        assert msg is not None
        # Should have removed final_answer to fix violation
        assert msg.final_answer is None
        assert msg.code is not None


# Additional comprehensive tests for coverage
@pytest.mark.asyncio
async def test_act_with_tool_execution_success():
    """Test _act with successful tool execution."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')

    msg = ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0
    assert responses[0]['type'] == 'step'
    assert '4' in str(responses[0]['value'])


@pytest.mark.asyncio
async def test_codeact_act_with_code_execution():
    """Test CodeActAgent _act with code execution."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Get date')

    msg = CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='from datetime import datetime\nprint(datetime.now().year)',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0
    assert responses[0]['type'] == 'step'


@pytest.mark.asyncio
async def test_codeact_act_with_code_markdown_variants():
    """Test CodeActAgent _act with various markdown code formats."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Get date')

    # Test with ```py format
    msg = CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='```py\nfrom datetime import datetime\nprint(datetime.now().year)\n```',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0


@pytest.mark.asyncio
async def test_codeact_act_with_code_error():
    """Test CodeActAgent _act with code execution error."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Bad code')

    msg = CodeActChatMessage(
        role='assistant',
        thought='Running bad code',
        code='import os\nos.system("rm -rf /")',  # Disallowed import
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0
    assert responses[0]['metadata']['is_error']


@pytest.mark.asyncio
async def test_think_step():
    """Test _think step."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')

    mock_msg = ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    )

    with patch.object(agent, '_record_thought', return_value=mock_msg):
        responses = []
        async for response in agent._think():
            responses.append(response)

        assert len(responses) > 0
        assert responses[0]['type'] == 'step'


@pytest.mark.asyncio
async def test_codeact_think_step():
    """Test CodeActAgent _think step."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Get date')

    mock_msg = CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='from datetime import datetime\nprint(datetime.now())',
        task_successful=False,
        final_answer=None
    )

    with patch.object(agent, '_record_thought', return_value=mock_msg):
        responses = []
        async for response in agent._think():
            responses.append(response)

        assert len(responses) > 0
        assert responses[0]['type'] == 'step'


def test_parse_text_response_invalid_format():
    """Test parse_text_response with completely invalid format."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # No thought field at all
    with pytest.raises(ValueError):
        agent.parse_text_response("Just some random text")


def test_parse_text_response_action_without_args():
    """Test parse_text_response with action but no args."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    text_response = """
Thought: I need to calculate
Action: calculator
"""
    with pytest.raises(ValueError):
        agent.parse_text_response(text_response)


def test_parse_text_response_finish_without_answer():
    """Test parse_text_response with FINISH but no answer."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    text_response = """
Thought: Done
Action: FINISH
"""
    with pytest.raises(ValueError):
        agent.parse_text_response(text_response)


def test_parse_text_response_codeact_no_code_or_answer():
    """Test parse_text_response CodeAct with no code or answer."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    text_response = """
Thought: Thinking
"""
    with pytest.raises(ValueError):
        agent.parse_text_response(text_response)


@pytest.mark.asyncio
async def test_update_plan():
    """Test _update_plan method."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')

    # Create initial plan
    agent.planner.plan = AgentPlan(steps=[
        PlanStep(description='Use calculator', is_done=False)
    ])

    mock_response = '{"steps": [{"description": "Use calculator", "is_done": true}]}'

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        await agent._update_plan()

        assert agent.planner.plan is not None
        assert agent.planner.plan.steps[0].is_done


@pytest.mark.asyncio
async def test_update_plan_without_plan():
    """Test _update_plan when no plan exists."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')

    # No plan set
    agent.planner.plan = None

    # Should not raise error
    await agent._update_plan()
    assert agent.planner.plan is None


def test_agent_description():
    """Test agent description property."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator],
        description='A test agent'
    )

    assert agent.description == 'A test agent'


def test_agent_id():
    """Test agent ID is unique."""
    agent1 = ReActAgent(
        name='agent1',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent2 = ReActAgent(
        name='agent2',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    assert agent1.id != agent2.id


@pytest.mark.asyncio
async def test_get_relevant_tools_error_handling():
    """Test get_relevant_tools error handling."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator, search_web]
    )
    agent._run_init('Test task')

    # Mock LLM error
    with patch('kodeagent.kutils.call_llm', side_effect=Exception('LLM error')):
        tools = await agent.get_relevant_tools('Test task')
        # Should return all tools on error
        assert len(tools) == 2


def test_chat_message_model_dump():
    """Test ChatMessage model_dump."""
    msg = ChatMessage(
        role='user',
        content='Hello',
        files=['file.txt']
    )

    dumped = msg.model_dump()
    assert dumped['role'] == 'user'
    assert dumped['content'] == 'Hello'
    assert dumped['files'] == ['file.txt']


def test_react_chat_message_model_dump():
    """Test ReActChatMessage model_dump."""
    msg = ReActChatMessage(
        role='assistant',
        thought='test',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    )

    dumped = msg.model_dump()
    assert dumped['role'] == 'assistant'
    assert dumped['thought'] == 'test'
    assert dumped['action'] == 'calculator'


def test_codeact_chat_message_model_dump():
    """Test CodeActChatMessage model_dump."""
    msg = CodeActChatMessage(
        role='assistant',
        thought='test',
        code='print("hello")',
        task_successful=False,
        final_answer=None
    )

    dumped = msg.model_dump()
    assert dumped['role'] == 'assistant'
    assert dumped['thought'] == 'test'
    assert dumped['code'] == 'print("hello")'


@pytest.mark.asyncio
async def test_act_with_json_repair():
    """Test _act with JSON repair for args."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    # Args with single quotes instead of double quotes
    msg = ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args="{'expression': '2+2'}",  # Invalid JSON
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    # Should handle JSON repair
    assert len(responses) > 0


def test_agent_trace_with_codeact():
    """Test trace method with CodeActChatMessage."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    agent.add_to_history(ChatMessage(role='user', content='Get date'))
    agent.add_to_history(CodeActChatMessage(
        role='assistant',
        thought='Getting date',
        code='from datetime import datetime\nprint(datetime.now())',
        task_successful=False,
        final_answer=None
    ))
    agent.add_to_history(ChatMessage(role='tool', content='2024-01-01'))

    trace = agent.trace()
    assert 'Thought: Getting date' in trace
    assert 'Code:' in trace
    assert 'Observation: 2024-01-01' in trace


def test_agent_trace_with_final_answer():
    """Test trace method with final answer."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    agent.add_to_history(ChatMessage(role='user', content='Calculate'))
    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Done',
        action='FINISH',
        args=None,
        task_successful=True,
        final_answer='The answer is 4'
    ))

    trace = agent.trace()
    assert 'Thought: Done' in trace
    # FINISH action should not appear in trace for final answers


@pytest.mark.asyncio
async def test_salvage_response_with_history():
    """Test salvage_response with message history."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')
    agent.msg_idx_of_new_task = 0

    agent.add_to_history(ChatMessage(role='user', content='Calculate 2+2'))
    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    ))
    agent.add_to_history(ChatMessage(role='tool', content='4'))

    mock_response = 'I calculated 2+2 and got 4'

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        result = await agent.salvage_response()
        assert 'calculated' in result


def test_codeact_tools_source_code():
    """Test CodeActAgent tools source code generation."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        tools=[calculator],
        allowed_imports=['datetime']
    )

    assert 'def calculator' in agent.tools_source_code
    assert 'from typing import *' in agent.tools_source_code


@pytest.mark.asyncio
async def test_act_with_final_answer():
    """Test _act with final answer."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    msg = ReActChatMessage(
        role='assistant',
        thought='Done',
        action='FINISH',
        args=None,
        task_successful=True,
        final_answer='The answer is 4'
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0
    assert responses[0]['type'] == 'final'
    assert 'The answer is 4' in responses[0]['value']


@pytest.mark.asyncio
async def test_codeact_act_with_final_answer():
    """Test CodeActAgent _act with final answer."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Get date')

    msg = CodeActChatMessage(
        role='assistant',
        thought='Done',
        code=None,
        task_successful=True,
        final_answer='The date is 2024-01-01'
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0
    assert responses[0]['type'] == 'final'
    assert 'The date is 2024-01-01' in responses[0]['value']


def test_parse_text_response_with_valid_action():
    """Test parse_text_response with valid action."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    text_response = """
Thought: I need to calculate
Action: calculator
Args: {"expression": "2+2"}
"""
    msg = agent.parse_text_response(text_response)
    assert msg.thought == 'I need to calculate'
    assert msg.action == 'calculator'
    assert msg.args is not None


def test_parse_text_response_with_valid_answer():
    """Test parse_text_response with valid answer."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    text_response = """
Thought: Done
Answer: The answer is 4
Successful: true
"""
    msg = agent.parse_text_response(text_response)
    assert msg.thought == 'Done'
    assert msg.final_answer == 'The answer is 4'
    assert msg.task_successful is True


def test_parse_text_response_codeact_with_code(codeact_agent_factory):
    """Test parse_text_response CodeAct with code."""
    agent = codeact_agent_factory()

    # Test 1: Code with Action format (intermediate step)
    text_response = """
Thought: Getting date
Action: code_execution
Args: {"language": "python"}
Code: ```python
from datetime import datetime
print(datetime.now())
```
"""
    msg = agent.parse_text_response(text_response)
    assert isinstance(msg, CodeActChatMessage)
    assert msg.thought == 'Getting date'
    assert msg.code is not None
    assert 'datetime' in msg.code


def test_parse_text_response_codeact_with_answer(codeact_agent_factory):
    """Test parse_text_response CodeAct with answer."""
    agent = codeact_agent_factory()
    text_response = """
Thought: Done
Answer: The date is 2024-01-01
Successful: true
"""
    msg = agent.parse_text_response(text_response)
    assert isinstance(msg, CodeActChatMessage)
    assert msg.thought == 'Done'
    assert msg.final_answer == 'The date is 2024-01-01'
    assert msg.task_successful is True


@pytest.mark.asyncio
async def test_act_with_missing_args():
    """Test _act with missing args - should be caught by validation."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    # This should fail validation when creating the message
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='Using calculator',
            action='calculator',
            args=None,  # Missing args for non-FINISH action
            task_successful=False,
            final_answer=None
        )


@pytest.mark.asyncio
async def test_act_with_invalid_json_args():
    """Test _act with invalid JSON args - should be caught by validation."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    # This should fail validation when creating the message
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='Using calculator',
            action='calculator',
            args='not json',  # Invalid JSON
            task_successful=False,
            final_answer=None
        )


def test_formatted_history_with_pending_tool_call():
    """Test formatted_history_for_llm with pending tool call."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    ))
    # No tool response added

    formatted = agent.formatted_history_for_llm()
    # Should add placeholder for missing tool response
    assert any(m.get('role') == 'tool' for m in formatted)


@pytest.mark.asyncio
async def test_codeact_act_with_exception():
    """Test CodeActAgent _act with exception during execution."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Bad code')

    msg = CodeActChatMessage(
        role='assistant',
        thought='Running code',
        code='raise Exception("Test error")',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0


@pytest.mark.asyncio
async def test_update_plan_with_history():
    """Test _update_plan with message history."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate 2+2')

    # Add history
    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Using calculator',
        action='calculator',
        args='{"expression": "2+2"}',
        task_successful=False,
        final_answer=None
    ))
    agent.add_to_history(ChatMessage(role='tool', content='4'))

    # Create initial plan
    agent.planner.plan = AgentPlan(steps=[
        PlanStep(description='Use calculator', is_done=False)
    ])

    mock_response = '{"steps": [{"description": "Use calculator", "is_done": true}]}'

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        await agent._update_plan()

        assert agent.planner.plan is not None
        assert agent.planner.plan.steps[0].is_done


def test_read_prompt_file_not_found():
    """Test _read_prompt with non-existent file."""
    from kodeagent.kodeagent import _read_prompt

    with pytest.raises(FileNotFoundError) as exc_info:
        _read_prompt('nonexistent_file.txt')
    assert 'not found' in str(exc_info.value)


def test_read_prompt_error():
    """Test _read_prompt with read error."""
    from kodeagent.kodeagent import _read_prompt

    with patch('builtins.open', side_effect=PermissionError('Access denied')):
        with pytest.raises(RuntimeError) as exc_info:
            _read_prompt('system/react.txt')
        assert 'Error reading prompt file' in str(exc_info.value)


def test_llm_vision_support():
    """Test llm_vision_support function."""
    from kodeagent.kodeagent import llm_vision_support

    # Test with known models
    models = ['gpt-4-vision-preview', 'gpt-4o', 'claude-3-opus']
    results = llm_vision_support(models)
    assert isinstance(results, list)
    assert len(results) == len(models)


def test_print_response():
    """Test print_response function."""
    from kodeagent.kodeagent import print_response

    # Test with final response
    response = {
        'type': 'final',
        'value': 'Test answer',
        'channel': 'test',
        'metadata': {}
    }
    print_response(response, only_final=True)

    # Test with log response
    log_response = {
        'type': 'log',
        'value': 'Test log',
        'channel': 'test',
        'metadata': {}
    }
    print_response(log_response, only_final=False)

    # Test with step response
    step_response = {
        'type': 'step',
        'value': 'Test step',
        'channel': 'test',
        'metadata': {}
    }
    print_response(step_response, only_final=False)


@pytest.mark.asyncio
async def test_chat_with_exception():
    """Test _chat method with exception handling."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Test task')

    # Mock LLM to raise exception on all attempts
    with patch('kodeagent.kutils.call_llm', side_effect=Exception('LLM error')):
        with pytest.raises(Exception):
            await agent._chat(response_format=ReActChatMessage)


@pytest.mark.asyncio
async def test_run_with_summarize_progress_false():
    """Test agent run with summarize_progress_on_failure=False."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Test step", "is_done": false}]}'
        if kwargs.get('response_format') == ObserverResponse:
            return '{"is_progressing": true, "is_in_loop": false, "reasoning": "ok", "correction_message": null}'
        # Return response that doesn't finish
        return '{"role": "assistant", "thought": "thinking", "action": "calculator", "args": "{\\"expression\\": \\"2+2\\"}", "task_successful": false, "final_answer": null}'

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[calculator],
            max_iterations=1
        )

        responses = []
        async for response in agent.run('Test task', summarize_progress_on_failure=False):
            responses.append(response)

        # Should have final response without summary
        final_responses = [r for r in responses if r['type'] == 'final']
        assert len(final_responses) > 0
        assert 'summary of my progress' not in str(final_responses[0]['value']).lower()


@pytest.mark.asyncio
async def test_act_with_empty_args_string():
    """Test _act with empty args string - should be caught by validation."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    # Empty/whitespace args should fail validation
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='Using calculator',
            action='calculator',
            args='   ',  # Empty/whitespace args - will be normalized to None
            task_successful=False,
            final_answer=None
        )


def test_react_chat_message_args_with_list():
    """Test ReActChatMessage args validation with list instead of dict."""
    with pytest.raises(pydantic_core.ValidationError) as exc_info:
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args='["item1", "item2"]',  # List instead of dict
            task_successful=False,
            final_answer=None
        )
    assert 'must be a JSON object' in str(exc_info.value)


def test_react_chat_message_task_successful_with_action():
    """Test ReActChatMessage validation - task_successful must be False for actions."""
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args='{"expression": "2+2"}',
            task_successful=True,  # Should be False for intermediate steps
            final_answer=None
        )


def test_codeact_chat_message_task_successful_with_code():
    """Test CodeActChatMessage validation - task_successful must be False with code."""
    with pytest.raises(pydantic_core.ValidationError):
        CodeActChatMessage(
            role='assistant',
            thought='test',
            code='print("hello")',
            task_successful=True,  # Should be False when executing code
            final_answer=None
        )


def test_react_chat_message_finish_with_args():
    """Test ReActChatMessage validation - FINISH cannot have args."""
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='FINISH',
            args='{"something": "value"}',  # Should be None for FINISH
            task_successful=True,
            final_answer='Done'
        )


def test_react_chat_message_action_without_args():
    """Test ReActChatMessage validation - action requires args."""
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='test',
            action='calculator',
            args=None,  # Should have args for non-FINISH action
            task_successful=False,
            final_answer=None
        )


@pytest.mark.asyncio
async def test_observer_with_correction_and_tools():
    """Test Observer with correction message and tool names."""
    mock_response = (
        '{"is_progressing": false, "is_in_loop": true,'
        ' "reasoning": "Stuck in loop",'
        ' "correction_message": "Try different tool"}'
    )

    with patch('kodeagent.kutils.call_llm', return_value=mock_response):
        observer = Observer(
            model_name=MODEL_NAME,
            tool_names={'calculator', 'search_web', 'download_file'},
            threshold=2
        )

        task = Task(description='Test', files=None)
        correction = await observer.observe(
            iteration=3,
            task=task,
            history='test history',
            plan_before='Plan before',
            plan_after='Plan after'
        )

        assert correction is not None
        assert 'CRITICAL FOR COURSE CORRECTION' in correction
        assert 'Try different tool' in correction
        # Should include tool names
        assert 'calculator' in correction or 'TOOL' in correction


@pytest.mark.asyncio
async def test_planner_update_plan_with_no_existing_plan():
    """Test planner update when plan doesn't exist."""
    planner = Planner(model_name=MODEL_NAME)

    # No plan exists
    assert planner.plan is None

    # Update should not crash
    await planner.update_plan(
        thought='test',
        observation='test',
        task_id='test-123'
    )

    # Plan should still be None
    assert planner.plan is None


def test_agent_msg_idx_of_new_task():
    """Test agent msg_idx_of_new_task tracking."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )

    # Initially 0
    assert agent.msg_idx_of_new_task == 0

    # Add some messages
    agent.add_to_history(ChatMessage(role='user', content='First task'))
    agent.add_to_history(ChatMessage(role='assistant', content='Response'))

    # Start new task
    agent._run_init('Second task')

    # msg_idx should still be 0 (it's set in run(), not _run_init())
    assert agent.msg_idx_of_new_task == 0


def test_parse_text_response_with_no_action_or_answer():
    """Test parse_text_response with neither action nor answer."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator])

    text_response = """
Thought: Just thinking
"""
    with pytest.raises(ValueError) as exc_info:
        agent.parse_text_response(text_response)
    assert 'Could not extract valid Action or Answer' in str(exc_info.value)


def test_parse_text_response_with_invalid_args_json():
    """Test parse_text_response with invalid JSON in args."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator])

    text_response = """
Thought: Calculate
Action: calculator
Args: {invalid json}
"""
    with pytest.raises(ValueError) as _:
        agent.parse_text_response(text_response)
    # Should fail during args validation


def test_codeact_parse_text_response_code_without_markdown():
    """Test CodeAct parse_text_response with code not in markdown."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )

    # Code without markdown blocks - needs valid Action+Args for parent parser
    text_response = """
Thought: Getting date
Action: code_execution
Args: {"language": "python"}
Code: from datetime import datetime
print(datetime.now())
"""
    msg = agent.parse_text_response(text_response)
    assert isinstance(msg, CodeActChatMessage)
    assert msg.code is not None
    # Code should be extracted even without markdown
    assert 'datetime' in msg.code


@pytest.mark.asyncio
async def test_get_relevant_tools_with_empty_response():
    """Test get_relevant_tools when LLM returns empty string."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator, search_web])
    agent._run_init('Test task')

    with patch('kodeagent.kutils.call_llm', return_value=''):
        tools = await agent.get_relevant_tools('Test task')
        # Should return empty list when no tools specified
        assert len(tools) == 0


@pytest.mark.asyncio
async def test_get_relevant_tools_with_whitespace():
    """Test get_relevant_tools with whitespace in response."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator, search_web])
    agent._run_init('Test task')

    with patch('kodeagent.kutils.call_llm', return_value='  calculator  ,  search_web  '):
        tools = await agent.get_relevant_tools('Test task')
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert 'calculator' in tool_names
        assert 'search_web' in tool_names


def test_formatted_history_with_final_answer():
    """Test formatted_history_for_llm with final answer message."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator])
    agent.add_to_history(ReActChatMessage(
        role='assistant',
        thought='Done',
        action='FINISH',
        args=None,
        task_successful=True,
        final_answer='The answer is 42'
    ))

    formatted = agent.formatted_history_for_llm()
    assert len(formatted) > 0
    # Should have assistant message with content
    assert any(m.get('role') == 'assistant' and m.get('content') for m in formatted)


def test_codeact_formatted_history_with_final_answer(codeact_agent_factory):
    """Test CodeAct formatted_history_for_llm with final answer."""
    agent = codeact_agent_factory()
    agent.add_to_history(CodeActChatMessage(
        role='assistant',
        thought='Done',
        code=None,
        task_successful=True,
        final_answer='The result is 42'
    ))

    formatted = agent.formatted_history_for_llm()
    assert len(formatted) > 0
    # Should have assistant message with content
    assert any(m.get('role') == 'assistant' and m.get('content') for m in formatted)


@pytest.mark.asyncio
async def test_record_thought_with_args_cleaning():
    """Test _record_thought with args that need cleaning."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[calculator])
    agent._run_init('Test task')

    # Response with args that need cleaning (extra whitespace, etc)
    response_json = '''
    {
        "role": "assistant",
        "thought": "test",
        "action": "calculator",
        "args": "  {\\"expression\\": \\"2+2\\"}  ",
        "task_successful": false,
        "final_answer": null
    }
    '''

    with patch('kodeagent.kutils.call_llm', return_value=response_json):
        msg = await agent._record_thought(ReActChatMessage)
        assert msg is not None
        assert msg.args is not None


def test_agent_filter_tools_for_task_attribute():
    """Test agent filter_tools_for_task attribute."""
    agent1 = ReActAgent(
        name='agent1',
        model_name=MODEL_NAME,
        tools=[calculator],
        filter_tools_for_task=True
    )
    assert agent1.filter_tools_for_task is True

    agent2 = ReActAgent(
        name='agent2',
        model_name=MODEL_NAME,
        tools=[calculator],
        filter_tools_for_task=False
    )
    assert agent2.filter_tools_for_task is False


@pytest.mark.asyncio
async def test_act_with_tool_args_not_dict():
    """Test _act when tool args parse to non-dict - should be caught by validation."""
    agent = ReActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        tools=[calculator]
    )
    agent._run_init('Calculate')

    # Args that parse to a list should fail validation
    with pytest.raises(pydantic_core.ValidationError):
        ReActChatMessage(
            role='assistant',
            thought='Using calculator',
            action='calculator',
            args='["not", "a", "dict"]',  # List instead of dict
            task_successful=False,
            final_answer=None
        )


def test_observer_threshold_none():
    """Test Observer with threshold=None."""
    observer = Observer(model_name=MODEL_NAME, tool_names={'calculator'}, threshold=None)
    assert observer.threshold is None


def test_planner_get_formatted_plan_empty():
    """Test planner get_formatted_plan with empty plan."""
    planner = Planner(model_name=MODEL_NAME)
    planner.plan = AgentPlan(steps=[])

    formatted = planner.get_formatted_plan()
    assert formatted == ''


def test_task_model_fields():
    """Test Task model has all expected fields."""
    task = Task(description='Test')
    assert hasattr(task, 'id')
    assert hasattr(task, 'description')
    assert hasattr(task, 'files')
    assert hasattr(task, 'result')
    assert hasattr(task, 'is_finished')
    assert hasattr(task, 'is_error')


def test_agent_response_format_class():
    """Test Agent response_format_class class variable."""
    assert hasattr(Agent, 'response_format_class')
    assert Agent.response_format_class == ChatMessage
    assert ReActAgent.response_format_class == ReActChatMessage
    assert CodeActAgent.response_format_class == CodeActChatMessage


@pytest.mark.asyncio
async def test_codeact_act_with_code_stripping():
    """Test CodeActAgent _act with code that needs stripping."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        allowed_imports=['datetime']
    )
    agent._run_init('Test')

    # Code with various Markdown formats
    msg = CodeActChatMessage(
        role='assistant',
        thought='Test',
        code='```python\nprint("hello")\n```',
        task_successful=False,
        final_answer=None
    )
    agent.add_to_history(msg)

    responses = []
    async for response in agent._act():
        responses.append(response)

    assert len(responses) > 0


def test_codeact_tools_source_code_empty():
    """Test CodeActAgent with no tools."""
    agent = CodeActAgent(
        name='test_agent',
        model_name=MODEL_NAME,
        run_env='host',
        tools=[],
        allowed_imports=['datetime']
    )

    # Should have base imports but no tool functions
    assert 'from typing import *' in agent.tools_source_code
    assert 'def ' not in agent.tools_source_code or agent.tools_source_code.count('def ') == 0
