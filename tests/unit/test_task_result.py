"""Test the TaskResult class."""

from unittest.mock import MagicMock, patch

import pytest

from kodeagent import ReActAgent

MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.fixture
def mock_llm():
    """Fixture to mock LLM API calls."""

    async def mock_call_llm(*args, **kwargs):
        return 'Mock response'

    with patch('kodeagent.kutils.call_llm', new=mock_call_llm):
        yield mock_call_llm


@pytest.mark.asyncio
async def test_task_result_stored_on_success(mock_llm):
    """Test that task.result is stored when agent succeeds."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[])

    # Sequence: Create plan -> Think -> Act (Final) -> Update plan
    assistant_sequence = [
        '{"steps": [{"description": "done", "is_done": false}]}',  # Create plan
        '{"role": "assistant", "thought": "done", "action": "FINISH", "final_answer": "42", "task_successful": true}',  # Think
        '{"steps": [{"description": "done", "is_done": true}]}',  # Update plan
    ]

    with patch('kodeagent.kutils.call_llm', side_effect=assistant_sequence):
        async for _ in agent.run('test task'):
            pass

    assert agent.task.result == '42'
    assert agent.task.is_finished is True


@pytest.mark.asyncio
async def test_task_result_stored_on_failure(mock_llm):
    """Test that task.result is stored when agent fails (max iterations)."""
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[], max_iterations=1)

    # We need to distinguish between Planner, Observer and Agent calls
    def side_effect(*args, **kwargs):
        comp = kwargs.get('component_name', '')
        if comp == 'Planner.create' or comp == 'Planner.update':
            return '{"steps": [{"description": "step", "is_done": false}]}'
        if comp == 'Agent':
            return '{"role": "assistant", "thought": "thinking", "action": "calculator", "args": "{\\"expression\\": \\"1+1\\"}"}'
        if comp == 'Observer':
            return '{"is_progressing": true, "is_in_loop": false, "reasoning": "ok"}'
        if comp == 'Agent.salvage':
            return 'Salvagable progress summary'
        return 'default'

    with patch('kodeagent.kutils.call_llm', side_effect=side_effect):
        async for _ in agent.run('test task'):
            pass

    assert 'failed to get a complete answer' in agent.task.result
    assert 'Salvagable progress summary' in agent.task.result
    assert agent.task.is_finished is False


@pytest.mark.asyncio
async def test_task_result_stored_on_init_error(mock_llm):
    """Test that task.result is stored when error occurs during initialization."""
    from tenacity import RetryError

    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[])

    # Force RetryError on plan creation
    with patch('kodeagent.kutils.call_llm', side_effect=RetryError(MagicMock())):
        async for _ in agent.run('test task'):
            pass

    assert 'Unable to start solving the task' in agent.task.result
    assert agent.task.is_finished is False
