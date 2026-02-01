"""Unit tests for FunctionCallingAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kodeagent import (
    ChatMessage,
    FunctionCallingAgent,
    FunctionCallingChatMessage,
    ToolCall,
)
from kodeagent.history_formatter import FunctionCallingHistoryFormatter
from kodeagent.tools import tool


# Properly decorated tools using @tool from tools.py
@tool
def tool_add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@tool
def tool_greet(name: str) -> str:
    """Greet a person.

    Args:
        name: Name to greet

    Returns:
        Greeting string
    """
    return f'Hello, {name}!'


@pytest.fixture
def mock_litellm_support():
    """Mock LiteLLM function calling support."""
    with patch('kodeagent.kutils.supports_function_calling', return_value=True) as mock:
        yield mock


@pytest.fixture
def agent(mock_litellm_support):
    return FunctionCallingAgent(
        name='TestAgent',
        model_name='test-model',
        tools=[tool_add, tool_greet],
        litellm_params={'temperature': 0},
    )


def test_initialization_check():
    """Test that agent raises ValueError if model doesn't support function calling."""
    with patch('kodeagent.kutils.supports_function_calling', return_value=False):
        with pytest.raises(ValueError, match='does not support function calling'):
            FunctionCallingAgent(name='Test', model_name='bad-model')


def test_schema_generation(agent):
    """Test that tool schemas are generated correctly."""
    schemas = agent._generate_tool_schemas()
    assert len(schemas) == 2
    assert schemas[0]['function']['name'] == 'tool_add'
    assert schemas[1]['function']['name'] == 'tool_greet'
    # Verify cache usage
    with patch('kodeagent.kutils.generate_function_schema') as mock_gen:
        schemas2 = agent._generate_tool_schemas()
        assert schemas2 is schemas
        mock_gen.assert_not_called()


def test_formatted_history_for_llm(agent):
    """Test history formatting for LLM."""
    # Setup history
    agent.messages = [
        ChatMessage(role='system', content='System info'),
        ChatMessage(role='user', content='User query'),
        FunctionCallingChatMessage(
            role='assistant',
            content='Thinking...',
            tool_calls=[ToolCall(id='call_1', name='tool_add', arguments='{"a": 1, "b": 2}')],
        ),
        ChatMessage(role='tool', content='3', tool_call_id='call_1'),
    ]

    formatted = agent.formatted_history_for_llm()
    assert len(formatted) == 4
    assert formatted[0]['role'] == 'system'
    assert formatted[1]['role'] == 'user'

    # Assistant message
    assert formatted[2]['role'] == 'assistant'
    assert formatted[2]['content'] == 'Thinking...'
    assert len(formatted[2]['tool_calls']) == 1
    assert formatted[2]['tool_calls'][0]['id'] == 'call_1'

    # Tool response
    assert formatted[3]['role'] == 'tool'
    assert formatted[3]['content'] == '3'
    assert formatted[3]['tool_call_id'] == 'call_1'


@pytest.mark.asyncio
async def test_chat_returns_object(agent):
    """Test _chat method returns FunctionCallingChatMessage."""
    # Mock return value of call_llm
    mock_message = MagicMock()
    mock_message.role = 'assistant'
    mock_message.content = 'Thought process'

    # Configure mock tool call with string attributes
    mock_tool_call = MagicMock()
    mock_tool_call.id = 'call_123'
    # Configure function object within tool call
    mock_function = MagicMock()
    mock_function.name = 'tool_add'  # Explicitly set name attribute
    mock_function.arguments = '{"a": 1, "b": 2}'  # Explicitly set arguments

    mock_tool_call.function = mock_function
    mock_message.tool_calls = [mock_tool_call]

    with patch('kodeagent.kutils.call_llm', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_message

        response = await agent._chat()

        assert isinstance(response, FunctionCallingChatMessage)
        assert response.role == 'assistant'
        assert response.content == 'Thought process'
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == 'tool_add'

        # Verify call_llm was called with return_full_message=True
        mock_call.assert_called_once()
        assert mock_call.call_args.kwargs['return_full_message'] is True
        assert 'tools' in mock_call.call_args.kwargs['litellm_params']
        assert mock_call.call_args.kwargs['litellm_params']['tool_choice'] == 'auto'


@pytest.mark.asyncio
async def test_think_execution(agent):
    """Test _think logic."""
    mock_msg = FunctionCallingChatMessage(
        role='assistant', tool_calls=[ToolCall(id='1', name='tool_add', arguments='{}')]
    )

    with patch.object(agent, '_chat', new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_msg

        responses = []
        async for resp in agent._think():
            responses.append(resp)

        assert len(responses) == 1
        assert responses[0]['type'] == 'step'
        assert responses[0]['value'] == mock_msg

        # Verify history updated
        assert agent.messages[-1] == mock_msg


@pytest.mark.asyncio
async def test_act_execution(agent):
    """Test _act executes tool."""
    agent.messages.append(
        FunctionCallingChatMessage(
            role='assistant',
            tool_calls=[ToolCall(id='call_999', name='tool_add', arguments='{"a": 5, "b": 7}')],
        )
    )

    responses = []
    async for resp in agent._act():
        responses.append(resp)

    assert len(responses) == 1
    assert responses[0]['type'] == 'step'
    assert responses[0]['value'] == 12  # 5+7

    # Verify history updated
    assert len(agent.messages) == 3  # Assistant, Tool response, User continuation
    tool_msg = agent.messages[1]
    assert tool_msg.role == 'tool'
    assert tool_msg.content == '12'
    assert tool_msg.tool_call_id == 'call_999'
    assert tool_msg.name == 'tool_add'

    # Verify Gemini continuation
    cont_msg = agent.messages[2]
    assert cont_msg.role == 'user'
    assert 'continue' in cont_msg.content.lower()


@pytest.mark.asyncio
async def test_act_tools_not_found(agent):
    """Test _act handles missing tools."""
    agent.messages.append(
        FunctionCallingChatMessage(
            role='assistant',
            tool_calls=[ToolCall(id='call_x', name='missing_tool', arguments='{}')],
        )
    )

    responses = []
    async for resp in agent._act():
        responses.append(resp)

    assert responses[0]['metadata']['is_error'] is True
    assert 'Tool "missing_tool" not found' in str(responses[0]['value'])


@pytest.mark.asyncio
async def test_final_answer(agent):
    """Test _act handles final answer."""
    agent.messages.append(
        FunctionCallingChatMessage(
            role='assistant', final_answer='The answer is 42', task_successful=True
        )
    )

    responses = []
    async for resp in agent._act():
        responses.append(resp)

    assert len(responses) == 1
    assert responses[0]['type'] == 'final'
    assert responses[0]['value'] == 'The answer is 42'
    assert responses[0]['metadata']['final_answer_found'] is True


def test_history_formatter_pending_placeholder():
    """Test placeholder logic in formatter."""
    formatter = FunctionCallingHistoryFormatter()
    state = {'pending_tool_call': True, 'last_tool_call_id': 'abc'}
    assert formatter.should_add_pending_placeholder(state) is True

    chat_msg = ChatMessage(role='assistant', content='content')  # No tool calls
    formatted = formatter.format_tool_call(chat_msg, state)
    assert formatted['content'] == 'content'
    # Logic in formatter handles msg.tool_calls iteration, if none, returns empty list?
    # Actually format_tool_call expects msg to have tool_calls if called

    # Let's test checking logic
    msg_with_tools = FunctionCallingChatMessage(
        role='assistant', tool_calls=[ToolCall(id='1', name='foo', arguments='{}')]
    )
    assert formatter.should_format_as_tool_call(msg_with_tools) is True

    formatted = formatter.format_tool_call(msg_with_tools, state)
    assert len(formatted['tool_calls']) == 1
    assert formatted['tool_calls'][0]['id'] == '1'
    assert state['last_tool_call_id'] == '1'


@pytest.mark.asyncio
async def test_update_plan_aggregates_observations(agent):
    """Test that _update_plan aggregates multiple tool observations."""
    agent.task = MagicMock()
    agent.task.id = 'task_123'
    agent.planner.update_plan = AsyncMock()

    # Setup history with assistant message followed by multiple tool messages
    agent.messages = [
        FunctionCallingChatMessage(
            role='assistant',
            thought='I will add 1+2 and greet Joe',
            tool_calls=[
                ToolCall(id='c1', name='tool_add', arguments='{"a":1, "b":2}'),
                ToolCall(id='c2', name='tool_greet', arguments='{"name":"Joe"}'),
            ],
        ),
        ChatMessage(role='tool', content='3', tool_call_id='c1', name='tool_add'),
        ChatMessage(role='tool', content='Hello, Joe!', tool_call_id='c2', name='tool_greet'),
    ]

    await agent._update_plan()

    # Verify planner.update_plan was called with aggregated observations
    agent.planner.update_plan.assert_called_once()
    kwargs = agent.planner.update_plan.call_args.kwargs
    assert kwargs['thought'] == 'I will add 1+2 and greet Joe'
    assert '[tool_add]: 3' in kwargs['observation']
    assert '[tool_greet]: Hello, Joe!' in kwargs['observation']


@pytest.mark.asyncio
async def test_chat_populates_thought_fallback(agent):
    """Test that _chat populates thought from content or tool names."""
    mock_message = MagicMock()
    mock_message.role = 'assistant'
    mock_message.content = None  # No content

    mock_tool_call = MagicMock()
    mock_tool_call.id = 'call_1'
    mock_tool_call.function.name = 'tool_add'
    mock_tool_call.function.arguments = '{"a": 1, "b": 2}'
    mock_message.tool_calls = [mock_tool_call]

    with patch('kodeagent.kutils.call_llm', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_message
        response = await agent._chat()

        assert response.thought == 'I will use tools: tool_add'
        assert response.content is None
