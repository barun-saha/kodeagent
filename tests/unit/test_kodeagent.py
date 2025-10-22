"""
Unit tests for the agents and their operations.
"""
import datetime
from typing import Optional, AsyncIterator
from unittest import mock
from unittest.mock import MagicMock, patch

import pydantic_core
import pytest
import wikipedia

from kodeagent import (
    ReActAgent,
    ChatMessage,
    ReActChatMessage,
    CodeActAgent,
    CodeActChatMessage,
    AgentPlan,
    PlanStep,
    Planner,
    Task,
    Observer,
    ObserverResponse,
    Agent,
    CodeRunner,
    AgentResponse
)
from kodeagent.tools import (
    tool,
    calculator,
    search_web,
    download_file,
    extract_file_contents_as_markdown,
    search_wikipedia,
    get_audio_transcript,
)


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'

# Mock responses for LLM calls
MOCK_LLM_RESPONSES = {
    'think': '{"role": "assistant", "thought": "I should use the calculator",'
             ' "action": "calculator", "args": "{\\"a\\": 2, \\"b\\": 2}",'
             ' "content": "", "successful": false, "answer": null}',
    'observe': '{"is_progressing": true, "is_in_loop": false, "reasoning": "all good"}',
    'plan': '{"steps": [{"description": "Use calculator", "is_done": false}]}',
    'code': '{"role": "assistant", "thought": "Getting date",'
            ' "code": "from datetime import datetime'
            '\\nprint(datetime.now().strftime(\'%B %d, %Y\'))",'
            ' "content": "", "successful": false, "answer": null}'
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
        content='',
        task_successful=False,
        final_answer=None
    )
    msg2 = ChatMessage(role='tool', content='tool response')

    react_agent.add_to_history(msg1)
    react_agent.add_to_history(msg2)

    formatted = '\n'.join([str(m) for m in react_agent.formatted_history_for_llm()])
    assert "'role': 'assistant'" in formatted
    assert "'role': 'tool', 'content': 'tool response'" in formatted
    assert "'name': 'dummy_tool_one'" in formatted


@pytest.mark.asyncio
async def test_react_agent_run_success(react_agent):
    """Test successful task execution by ReActAgent."""
    # Predefined assistant responses
    assistant_sequence = [
        # Plan
        '{"steps": [{"description": "Use calculator", "is_done": false}]}',
        # Step 1: Think (ReActChatMessage)
        '{"role": "assistant", "thought": "I need to calculate 2 + 2.'
        ' I will use the calculator tool.",'
        ' "action": "calculator", "args": "{\\"expression\\": \\"2+2\\"}",'
        ' "task_successful": false, "content": null, "final_answer": null}',
        # Update plan
        '{"steps": [{"description": "Use calculator", "is_done": true}]}',
        # Step 1: Final after tool (ReActChatMessage)
        '{"role": "assistant", "thought": "I have calculated 2 + 2 = 4.'
        ' I can now provide the final answer.",'
        ' "action": null, "args": null, "content": null,'
        ' "task_successful": true, "final_answer": "The sum of 2 and 2 is 4"}',
        # Final update plan
        '{"steps": [{"description": "Use calculator", "is_done": true}]}',
    ]

    # Patch call_llm to return each response in order
    with patch('kodeagent.kutils.call_llm', side_effect=assistant_sequence):
        responses = []
        async for response in react_agent.run('Add 2 and 2'):
            responses.append(response)

        assert any(r['type'] == 'final' for r in responses)
        assert react_agent.final_answer_found
        assert react_agent.task.is_finished
        final_response = next(r for r in responses if r['type'] == 'final')
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
            ' "action": "broken_tool", "args": "{\"param1\": \"Something\"}",'
            ' "content": "", "task_successful": false, "final_answer": false}'
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

        async for response in agent.run('Use the broken tool'):
            responses.append(response)

        error_responses = [r for r in responses if r["metadata"] and r["metadata"].get("is_error")]
        assert len(error_responses) > 0
        assert 'An error occurred while taking' in str(error_responses[0]['value'])


@pytest.mark.asyncio
async def test_act_step_with_invalid_tool(react_agent):
    """Test the act step with an invalid tool name."""
    invalid_response = ReActChatMessage(
        thought="Test thought",
        action="nonexistent_tool",
        args='{"param1": "test"}',
        final_answer=None,
        task_successful=False,
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

        if last_message['role'] == 'user':
            # Return a properly formatted CodeActChatMessage for the initial task
            return (
                '{"role": "assistant", "thought": "I need to get the current month name", '
                '"code": "from datetime import datetime\\nprint(datetime.now().strftime(\'%B\'))", '
                '"content": "", "task_successful": false, "final_answer": null}'
            )

        if last_message['role'] == 'tool':
            # Return a final CodeActChatMessage with the result
            return (
                '{"role": "assistant", "thought": "I have the month name", '
                f'"code": "", "content": "The current month is {current_month}", '
                f'"task_successful": true, "final_answer": "The current month is {current_month}"'
                '}'
            )

        return (
            '{"role": "assistant", "content": "I am not sure what to do next.",'
            ' "task_successful": false, "final_answer": null}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        code_agent = CodeActAgent(
            name='Code agent',
            model_name=MODEL_NAME,
            run_env='host',
            max_iterations=3,
            allowed_imports=['datetime'],
            description='Agent that can write and execute Python code'
        )

        task = 'Compute the name of the month today.'
        responses = []
        async for response in code_agent.run(task):
            responses.append(response['value'])

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

        if last_message['role'] == 'user':
            # Return a properly formatted CodeActChatMessage for the initial task
            return (
                '{"role": "assistant", "thought": "I need to get the current month name", '
                '"code": "from datetime import datetime\\nprint(datetime.now().strftime(\'%B\'))", '
                '"content": "", "task_successful": false, "final_answer": null}'
            )

        if last_message['role'] == 'tool':
            # Return a final CodeActChatMessage with the result
            return (
                '{"role": "assistant", "thought": "I have the month name", '
                f'"code": "", "content": "The current month is {current_month}", '
                f'"task_successful": true, "final_answer": "The current month is {current_month}"'
                '}'
            )

        return (
            '{"role": "assistant", "content": "I am not sure what to do next.",'
            ' "task_successful": false, "final_answer": null}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        code_agent = CodeActAgent(
            name='Code agent',
            model_name=MODEL_NAME,
            run_env='unknown_env',
            max_iterations=3,
            allowed_imports=['datetime'],
            description='Agent that can write and execute Python code'
        )

        task = 'Compute the name of the month today.'
        responses = []
        async for response in code_agent.run(task):
            responses.append(response['value'])

        response = ' | '.join([str(r) for r in responses])
        assert 'Unsupported code execution env: unknown_env' in response


def test_code_chat_message_validation():
    """Test CodeChatMessage validation."""
    role = 'assistant'
    thought = 'test thought'
    code = "print('test')"
    msg = CodeActChatMessage(
        role=role,
        thought=thought,
        code=code,
        content='',
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
            content='',
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
            ' "action": "", "args": "", "content": "", "task_successful": true,'
            ' "final_answer": true}'
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
    with pytest.raises(ValueError):
        # Should fail because reasoning is required
        ObserverResponse(
            is_progressing=True,
            is_in_loop=False,
            reasoning=None,  # Empty reasoning should fail
            correction_message=None
        )


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

    agent = MinimalAgent(
        name='minimal_agent',
        model_name=MODEL_NAME
    )
    assert len(agent.tools) == 0
    assert len(agent.tool_names) == 0
    assert len(agent.tool_name_to_func) == 0


def test_code_runner_initialization():
    """Test CodeRunner initialization and configuration."""
    runner = CodeRunner(
        env='host',
        allowed_imports=['os', 'datetime'],
        pip_packages='requests==2.31.0',
        timeout=45,
        env_vars_to_set={'TEST_VAR': 'test_value'}
    )
    assert runner.env == 'host'
    assert 'os' in runner.allowed_imports
    assert 'datetime' in runner.allowed_imports
    assert runner.pip_packages == ['requests==2.31.0']
    assert runner.default_timeout == 45
    assert runner.env_vars_to_set == {'TEST_VAR': 'test_value'}


def test_code_runner_check_imports():
    """Test import checking functionality of CodeRunner."""
    runner = CodeRunner(env='host', allowed_imports=['os', 'datetime'])

    # Test allowed imports
    code_with_allowed = """
import os
from datetime import datetime
print('test')
"""
    assert len(runner.check_imports(code_with_allowed)) == 0

    # Test disallowed imports
    code_with_disallowed = """
import os
import requests
from flask import Flask
"""
    disallowed = runner.check_imports(code_with_disallowed)
    assert 'requests' in disallowed
    assert 'flask' in disallowed


def test_code_runner_syntax_error():
    """Test CodeRunner handling of syntax errors."""
    runner = CodeRunner(env='host', allowed_imports=['os'])
    code_with_syntax_error = '''
print('Hello'
print('World')  # Missing parenthesis above
'''
    _, stderr, exit_code = runner.run(code_with_syntax_error)
    assert exit_code != 0
    assert 'SyntaxError' in stderr


def test_code_runner():
    """Test the CodeRunner's code execution functionality."""
    # Initialize CodeRunner with host environment and basic allowed imports
    runner = CodeRunner(
        env='host',
        allowed_imports=['math', 'datetime'],
        timeout=5
    )

    # Test successful code execution
    code = '''
import math
result = math.sqrt(16)
print(f"Square root is {result}")
'''
    stdout, stderr, return_code = runner.run(code)
    assert return_code == 0
    assert 'Square root is 4.0' in stdout
    assert stderr == ''

    # Test disallowed imports
    code_with_unauthorized_import = '''
import os
os.getcwd()
'''
    stdout, stderr, return_code = runner.run(code_with_unauthorized_import)
    assert return_code != 0
    assert 'unauthorized' in stderr.lower() or 'disallowed' in stderr.lower()

    # Test syntax error handling
    invalid_code = '''
print("Hello
'''
    stdout, stderr, return_code = runner.run(invalid_code)
    assert return_code != 0
    assert 'SyntaxError' in stderr


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
        content='',
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
            content='',
            task_successful=False,
            final_answer=None
        ),
        ChatMessage(role='tool', content='4'),
        ReActChatMessage(
            role='assistant',
            thought='I have the result',
            action=None,
            args=None,
            content='The result is 4',
            task_successful=True,
            final_answer='The answer is 4'
        )
    ]

    for msg in messages:
        react_agent.add_to_history(msg)

    trace = react_agent.trace()
    print(trace)

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
    response = ObserverResponse(is_progressing=False, is_in_loop=True, reasoning='', correction_message=None)
    assert response.is_in_loop
    assert response.reasoning == ''
    assert response.correction_message is None


