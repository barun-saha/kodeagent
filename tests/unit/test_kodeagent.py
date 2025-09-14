"""
Unit tests for the agents and their operations.
"""
from typing import Optional, AsyncIterator
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic_core
import pytest
import wikipedia

from kodeagent import (
    tool,
    calculator,
    search_web,
    download_file,
    extract_file_contents_as_markdown,
    search_wikipedia,
    search_arxiv,
    get_youtube_transcript,
    get_audio_transcript,
    ReActAgent,
    ChatMessage,
    ReActChatMessage,
    CodeActAgent,
    CodeChatMessage,
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


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'

# Mock responses for LLM calls
MOCK_LLM_RESPONSES = {
    'think': '{"role": "assistant", "thought": "I should use the calculator", "action": "calculator", "args": "{\\"a\\": 2, \\"b\\": 2}", "content": "", "successful": false, "answer": null}',
    'observe': '{"is_progressing": true, "is_in_loop": false, "reasoning": "all good"}',
    'plan': '{"steps": [{"description": "Use calculator", "is_done": false}]}',
    'code': '{"role": "assistant", "thought": "Getting date", "code": "from datetime import datetime\\nprint(datetime.now().strftime(\'%B %d, %Y\'))", "content": "", "successful": false, "answer": null}'
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
        elif 'plan' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['plan']
        elif 'observe' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['observe']
        elif 'code' in str(kwargs.get('messages', [])):
            return MOCK_LLM_RESPONSES['code']
        return 'Default mock response'

    with patch('kodeagent.call_llm', new=mock_call_llm):
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
def planning_react_agent(mock_llm):
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

    assert any(r['type'] == 'final' for r in responses)
    assert react_agent.final_answer_found
    assert react_agent.task.is_finished
    final_response = next(r for r in responses if r['type'] == 'final')
    assert '4' in str(final_response['value'])


@pytest.mark.asyncio
async def test_react_agent_run_with_tool_error(react_agent):
    """Test ReActAgent handling tool execution errors."""
    @tool
    def broken_tool(param1: str) -> str:
        """A tool that always fails."""
        raise Exception('Tool error')

    react_agent.tools.append(broken_tool)

    responses = []
    async for response in react_agent.run('Use the broken tool'):
        responses.append(response)

    error_responses = [r for r in responses if r["metadata"] and r["metadata"].get("is_error")]
    assert len(error_responses) > 0
    assert "Incorrect tool name generated" in str(error_responses[0]["value"])


@pytest.mark.asyncio
async def test_think_step(react_agent):
    """Test the think step of ReActAgent."""
    react_agent._run_init("Calculate 5 plus 3")

    responses = []
    async for response in react_agent._think():
        responses.append(response)

    assert len(responses) == 1
    assert responses[0]["type"] == "step"
    assert isinstance(responses[0]["value"], ReActChatMessage)
    assert responses[0]["value"].thought is not None


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


@pytest.mark.asyncio
@patch('kodeagent.call_llm')
async def test_get_relevant_tools(mock_llm, react_agent):
    """Test filtering relevant tools for a task."""
    mock_llm.return_value = '["calculator", "dummy_tool_one"]'
    task_description = 'What is 2 plus 3?'
    react_agent._run_init(task_description)

    tools = await react_agent.get_relevant_tools(task_description)
    assert len(tools) > 0
    tool_names = {t.name for t in tools}
    assert "calculator" in tool_names


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
    with patch('kodeagent.call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MOCK_LLM_RESPONSES['code']

        code_agent = CodeActAgent(
            name='Code agent',
            model_name=MODEL_NAME,
            run_env='host',
            max_iterations=3,
            allowed_imports=['datetime'],
            description='Agent that can write and execute Python code'
        )

        task = "What is today's date? Express it in words without time."
        responses = []
        async for response in code_agent.run(task):
            responses.append(response['value'])

        response = ' | '.join([str(r) for r in responses])
        assert any('September' in str(r) for r in responses)


def test_code_chat_message_validation():
    """Test CodeChatMessage validation."""
    role = 'assistant'
    thought = 'test thought'
    code = "print('test')"
    msg = CodeChatMessage(
        role=role,
        thought=thought,
        code=code,
        content='',
        successful=False,
        answer=None
    )
    assert msg.role == role
    assert msg.thought == thought
    assert msg.code == code

    with pytest.raises(pydantic_core.ValidationError):
        CodeChatMessage(
            role=role,
            thought=None,  # No valid thought
            code=code,
            content='',
            successful=False,
            answer=None
        )


@pytest.mark.asyncio
async def test_agent_with_no_tools():
    """Test agent initialization with no tools."""
    with patch('kodeagent.call_llm', new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = 'I cannot help with that task as I have no tools.'

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

        print(f'{responses=}')
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
            reasoning=None  # Empty reasoning should fail
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

    with patch('kodeagent.kodeagent.call_llm', autospec=True) as mock_call_llm:
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

    with patch('kodeagent.kodeagent.call_llm', autospec=True) as mock_call_llm:
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
    code_with_syntax_error = """
print('Hello'
print('World')  # Missing parenthesis above
"""
    _, stderr, exit_code = runner.run(code_with_syntax_error)
    assert exit_code != 0
    assert 'SyntaxError' in stderr


@pytest.mark.asyncio
async def test_observer_analyze():
    """Test Observer's analysis of agent behavior."""
    mock_response = '{"is_progressing": false, "is_in_loop": true, "reasoning": "Agent keeps using calculator repeatedly", "correction_message": "Try a different approach"}'

    with patch('kodeagent.call_llm', return_value=mock_response):
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
        successful=False,
        answer=None
    ))
    react_agent.add_to_history(ChatMessage(role='tool', content='4'))

    trace = react_agent.trace()
    assert 'Thought: Using calculator' in trace
    assert 'Action: calculator' in trace
    assert 'Observation: 4' in trace


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


def test_search_web():
    """Test search_web with mock results."""
    mock_results = [
        {'title': 'Result 1', 'href': 'http://test1.com', 'body': 'Description 1'},
        {'title': 'Result 2', 'href': 'http://test2.com', 'body': 'Description 2'}
    ]

    with patch('ddgs.DDGS') as mock_ddgs:
        mock_instance = mock_ddgs.return_value
        mock_instance.text.return_value = mock_results

        # Test without descriptions
        result = search_web('test query', max_results=2)
        assert '[Result 1](http://test1.com)' in result
        assert '[Result 2](http://test2.com)' in result
        assert 'Description' not in result

        # Test with descriptions
        result = search_web('test query', max_results=2, show_description=True)


def test_download_file():
    """Test file download functionality."""
    with patch('requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.iter_content.return_value = [b'test content']

        result = download_file('http://test.com/file.txt')
        mock_get.assert_called_with(
            'http://test.com/file.txt',
            timeout=20,
            stream=True,
            headers={'user-agent': 'kodeagent/0.0.1'}
        )
        assert isinstance(result, str)
        assert len(result) > 0

        # Test error case
        mock_get.return_value.raise_for_status.side_effect = Exception("404 Not Found")
        with pytest.raises(Exception):
            download_file('http://test.com/error')


def test_extract_file_contents():
    """Test file content extraction."""
    with patch('markitdown.MarkItDown') as mock_markitdown:
        mock_instance = mock_markitdown.return_value

        # Test PDF with CID replacement
        mock_instance.convert.return_value.text_content = 'Test content with (cid:123)'
        result = extract_file_contents_as_markdown('test.pdf')
        assert 'Test content with' in result
        assert '(cid:123)' not in result

        # Test link handling
        mock_instance.convert.return_value.text_content = '[Link](http://test.com) text'
        result = extract_file_contents_as_markdown('test.html', scrub_links=False)
        assert '[Link](http://test.com)' in result

        result = extract_file_contents_as_markdown('test.html', scrub_links=True)
        assert '[Link](http://test.com)' not in result
        assert 'Link' in result

        # Test max length
        mock_instance.convert.return_value.text_content = 'A' * 100
        result = extract_file_contents_as_markdown('test.txt', max_length=50)
        assert len(result) <= 50


def test_search_wikipedia():
    """Test Wikipedia search functionality."""
    with patch('wikipedia.search') as mock_search:
        with patch('wikipedia.page') as mock_page:
            mock_search.return_value = ['Page1', 'Page2']

            class MockPage:
                def __init__(self, title):
                    self.title = title
                    self.url = f'https://en.wikipedia.org/wiki/{title}'
                    self.summary = f'Summary of {title}'

            mock_page.side_effect = lambda title: MockPage(title)
            result = search_wikipedia("test query", max_results=2)

            assert '[Page1](https://en.wikipedia.org/wiki/Page1)' in result
            assert 'Summary of Page1' in result

            # Test no results
            mock_search.return_value = []
            result = search_wikipedia('nonexistent')
            assert 'No results found!' in result

            # Test disambiguation
            mock_search.side_effect = wikipedia.exceptions.DisambiguationError(
                'Query', ['Option1', 'Option2']
            )
            result = search_wikipedia('ambiguous')
            assert 'DisambiguationError' in result
            assert 'Option1' in result


def test_get_audio_transcript():
    """Test audio transcript retrieval with mocked responses."""
    expected_response = {'text': 'This is the transcribed text'}
    mock_file = mock.mock_open(read_data=b'fake audio content')

    # Test successful case
    with patch('builtins.open', mock_file), patch('requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_post.return_value = mock_response

        result = get_audio_transcript('test_audio.mp3')
        assert result == expected_response
