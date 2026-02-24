"""Unit tests for the FunctionCallingAgent in fca.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kodeagent.fca import (
    DATA_TYPES,
    AgentResponse,
    FunctionCallingAgent,
    Task,
)


@pytest.fixture
def mock_params():
    """Default LiteLLM params for testing."""
    return {'temperature': 0}


@pytest.fixture
def fca_agent(mock_params):
    """Fixture to create a FunctionCallingAgent instance for testing."""

    def dummy_tool(a: int, b: str = 'default') -> str:
        """A dummy tool for testing.

        Args:
            a: An integer.
            b: A string.
        """
        return f'Result: {a}, {b}'

    return FunctionCallingAgent(
        model_name='test-model',
        tools=[dummy_tool],
        system_prompt='Test prompt',
        litellm_params=mock_params,
    )


def test_data_types():
    """Test DATA_TYPES mapping."""
    assert DATA_TYPES[int] == 'integer'
    assert DATA_TYPES[str] == 'string'
    assert DATA_TYPES[list] == 'array'


def test_task_initialization():
    """Test Task class initialization."""
    task = Task(id='123', description='Test task', result=None, steps_taken=None)
    assert task.id == '123'
    assert task.description == 'Test task'
    assert task.result is None
    assert task.steps_taken is None


def test_agent_response_type():
    """Test AgentResponse type behavior (as TypedDict)."""
    response: AgentResponse = {'type': 'step', 'channel': 'run', 'value': 'test', 'metadata': None}
    assert response['type'] == 'step'
    assert response['value'] == 'test'


def test_fca_initialization(fca_agent):
    """Test FunctionCallingAgent initialization."""
    assert fca_agent.model_name == 'test-model'
    assert len(fca_agent.tools) == 1
    assert fca_agent.system_prompt == 'Test prompt'
    assert 'dummy_tool' in fca_agent.tool_map
    assert len(fca_agent.tool_schemas) == 1
    assert fca_agent.tool_schemas[0]['function']['name'] == 'dummy_tool'


def test_fca_response_method(fca_agent):
    """Test the response method."""
    fca_agent.task = Task(id='1', description='test', result=None, steps_taken=None)

    # Test step response
    resp = fca_agent.response('step', 'working')
    assert resp['type'] == 'step'
    assert resp['value'] == 'working'

    # Test final response updates task result
    resp = fca_agent.response('final', 'done')
    assert resp['type'] == 'final'
    assert resp['value'] == 'done'
    assert fca_agent.task.result == 'done'


def test_build_tool_schema():
    """Test _build_tool_schema with various parameter configurations."""

    def complex_tool(x: int, y: list, z: str = 'optional') -> bool:
        """Complex tool docstring."""
        return True

    schema = FunctionCallingAgent._build_tool_schema(complex_tool)
    assert schema['type'] == 'function'
    assert schema['function']['name'] == 'complex_tool'
    assert schema['function']['description'] == 'Complex tool docstring.'

    params = schema['function']['parameters']
    assert params['type'] == 'object'
    assert params['properties']['x']['type'] == 'integer'
    assert params['properties']['y']['type'] == 'array'
    assert params['properties']['z']['type'] == 'string'
    assert 'x' in params['required']
    assert 'y' in params['required']
    assert 'z' not in params['required']


def test_execute_tool_success(fca_agent):
    """Test successful tool execution."""
    tool_call = MagicMock()
    tool_call.id = 'call_1'
    tool_call.function.name = 'dummy_tool'
    tool_call.function.arguments = json.dumps({'a': 10, 'b': 'hello'})

    result = fca_agent._execute_tool(tool_call)
    assert result['tool_call_id'] == 'call_1'
    assert result['role'] == 'tool'
    assert result['name'] == 'dummy_tool'
    assert 'Result: 10, hello' in result['content']


def test_execute_tool_errors(fca_agent):
    """Test tool execution error cases."""
    # 1. Undefined tool
    tool_call = MagicMock()
    tool_call.id = 'call_2'
    tool_call.function.name = 'non_existent_tool'
    tool_call.function.arguments = '{}'

    result = fca_agent._execute_tool(tool_call)
    assert "Error: Tool 'non_existent_tool' is not defined." in result['content']

    # 2. Malformed JSON arguments
    tool_call.function.name = 'dummy_tool'
    tool_call.function.arguments = '{invalid_json}'
    result = fca_agent._execute_tool(tool_call)
    assert 'Error: Model provided malformed JSON arguments.' in result['content']

    # 3. Exception during execution
    def failing_tool(**kwargs):
        raise Exception('Tool failure')

    fca_agent.tool_map['failing_tool'] = failing_tool
    tool_call.function.name = 'failing_tool'
    tool_call.function.arguments = '{}'
    result = fca_agent._execute_tool(tool_call)
    assert 'Error executing failing_tool: Tool failure' in result['content']


def test_detect_tool_loop(fca_agent):
    """Test tool loop detection."""
    # No loop
    fca_agent.chat_history = [
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'tool1'}}]},
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'tool2'}}]},
    ]
    assert fca_agent._detect_tool_loop() is False

    # Loop detected
    # Need 3 (threshold) same tool calls
    fca_agent.chat_history = [
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'tool1'}}]},
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'tool1'}}]},
        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'tool1'}}]},
    ]
    assert fca_agent._detect_tool_loop() is True
    assert fca_agent.chat_history[-1]['role'] == 'user'
    assert 'Loop detected' in fca_agent.chat_history[-1]['content']


def test_format_history_as_text(fca_agent):
    """Test chat history formatting."""
    # Assistant message with tool_calls as objects (mocking litellm response)
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = 'tool1'

    fca_agent.chat_history = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'thinking', 'tool_calls': [mock_tool_call]},
        {'role': 'tool', 'name': 'tool1', 'content': 'output'},
    ]

    formatted = fca_agent._format_history_as_text()
    assert 'User: hello' in formatted
    assert 'Assistant: thinking' in formatted
    assert 'Assistant: [Called tools: tool1]' in formatted
    assert 'Tool (tool1): output' in formatted


@pytest.mark.asyncio
async def test_prepare_final_answer(fca_agent):
    """Test _prepare_final_answer."""
    fca_agent.chat_history = [{'role': 'user', 'content': 'test'}]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = ' Final answer. '

    with patch('litellm.acompletion', new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = mock_response

        result = await fca_agent._prepare_final_answer()
        assert result == 'Final answer.'
        mock_complete.assert_called_once()


@pytest.mark.asyncio
async def test_run_init(fca_agent):
    """Test _run_init with and without planning."""
    # 1. Successful init without planning
    await fca_agent._run_init('Test task', use_planning=False)
    assert fca_agent.task.description == 'Test task'
    assert len(fca_agent.chat_history) == 2
    assert fca_agent.chat_history[0]['content'] == 'Test prompt'

    # 2. Empty task error
    with pytest.raises(ValueError):
        await fca_agent._run_init('', use_planning=False)

    # 3. Recurrent mode
    fca_agent.task = Task(id='1', description='prev', result='result', steps_taken=1)
    await fca_agent._run_init('new', use_planning=False, recurrent_mode=True)
    assert 'Previous Task' in fca_agent.chat_history[1]['content']

    # 4. With planning (mocking Planner)
    with patch('kodeagent.fca.Planner') as mock_planner_class:
        mock_planner = mock_planner_class.return_value
        mock_planner.create_plan = AsyncMock()
        mock_planner.get_formatted_plan.return_value = 'Step 1'

        await fca_agent._run_init('Task with plan', use_planning=True)
        assert any('Plan:\nStep 1' in msg.get('content', '') for msg in fca_agent.chat_history)


@pytest.mark.asyncio
async def test_run_main_loop(fca_agent):
    """Test the main run loop."""
    # Set up mocks
    mock_response_1 = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.id = 'c1'
    mock_tool_call.function.name = 'dummy_tool'
    mock_tool_call.function.arguments = json.dumps({'a': 1})
    mock_response_1.choices[0].message.tool_calls = [mock_tool_call]
    mock_response_1.choices[0].message.model_dump.return_value = {
        'role': 'assistant',
        'tool_calls': [{'id': 'c1', 'function': {'name': 'dummy_tool', 'arguments': '{"a": 1}'}}],
    }

    mock_response_2 = MagicMock()
    mock_response_2.choices[0].message.tool_calls = None
    mock_response_2.choices[0].message.content = 'Final result'
    mock_response_2.choices[0].message.model_dump.return_value = {
        'role': 'assistant',
        'content': 'Final result',
    }

    with patch('litellm.acompletion', side_effect=[mock_response_1, mock_response_2]):
        with patch.object(fca_agent, '_prepare_final_answer', return_value='Refined Answer'):
            responses = []
            try:
                async for resp in fca_agent.run('Solve this', max_iterations=2, use_planning=False):
                    responses.append(resp)
            except StopAsyncIteration:
                pass

            # Assertions
            assert any(r['type'] == 'log' and 'Solving task' in r['value'] for r in responses)
            assert any(
                r['type'] == 'log' and 'Executed tool: dummy_tool' in r['value'] for r in responses
            )
            assert any(r['type'] == 'final' and r['value'] == 'Refined Answer' for r in responses)
            assert fca_agent.task.steps_taken == 2


@pytest.mark.asyncio
async def test_run_fallback_no_refinement(fca_agent):
    """Test run loop when refine_final_answer is False."""
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = 'Simple Answer'
    mock_response.choices[0].message.model_dump.return_value = {
        'role': 'assistant',
        'content': 'Simple Answer',
    }

    with patch('litellm.acompletion', return_value=mock_response):
        responses = []
        try:
            async for resp in fca_agent.run('Task', refine_final_answer=False, use_planning=False):
                responses.append(resp)
        except StopAsyncIteration:
            pass

        assert any(r['type'] == 'final' and r['value'] == 'Simple Answer' for r in responses)


@pytest.mark.asyncio
async def test_run_max_iterations(fca_agent):
    """Test run loop hitting max iterations."""
    mock_response = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.name = 'dummy_tool'
    mock_tool_call.function.arguments = '{}'
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.choices[0].message.model_dump.return_value = {
        'role': 'assistant',
        'tool_calls': [],
    }

    with patch('litellm.acompletion', return_value=mock_response):
        with patch.object(fca_agent, '_prepare_final_answer', return_value='Time up'):
            responses = []
            try:
                async for resp in fca_agent.run(
                    'Endless task', max_iterations=1, use_planning=False
                ):
                    responses.append(resp)
            except StopAsyncIteration:
                pass

            assert fca_agent.task.steps_taken == 1
            assert any(r['type'] == 'final' and r['value'] == 'Time up' for r in responses)
