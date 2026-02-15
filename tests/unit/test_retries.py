"""Unit tests for retry logic in call_llm and agent behavior on rate limit errors."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import RateLimitError
from tenacity import RetryError

from kodeagent import ReActAgent
from kodeagent.kutils import DEFAULT_MAX_LLM_RETRIES, call_llm

MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.mark.asyncio
async def test_call_llm_rate_limit_retry_success():
    """Test that call_llm retries on RateLimitError and succeeds if it clears within limits."""
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='Success'))]
    mock_response._hidden_params = {}
    mock_response.usage = {}

    # Mock litellm.acompletion to raise RateLimitError 2 times, then return success
    # This should succeed because DEFAULT_MAX_LLM_RETRIES is 3
    side_effects = [
        RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME),
        RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME),
        mock_response,
    ]

    with patch('litellm.acompletion', side_effect=side_effects) as mock_acompletion:
        response = await call_llm(MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}])

        assert response == 'Success'
        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_call_llm_retries_on_generic_error():
    """Test that call_llm retries on generic errors (as per user configuration)."""
    # Mock litellm.acompletion to raise a generic Exception
    error = ValueError('Some other error')

    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError) as _:
            await call_llm(MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}])

        # Verify call count is DEFAULT_MAX_LLM_RETRIES (because we retry on Exception now)
        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_call_llm_custom_max_retries():
    """Test that call_llm respects custom max_retries."""
    # Mock litellm.acompletion to raise a generic Exception
    error = ValueError('Some other error')
    custom_retries = 5

    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError):
            await call_llm(
                MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}], max_retries=custom_retries
            )

        # Verify call count matches custom retries
        assert mock_acompletion.call_count == custom_retries


@pytest.mark.asyncio
async def test_call_llm_rate_limit_retry_failure():
    """Test that call_llm fails after MAX_LLM_RETRIES if RateLimitError persists."""
    # Mock litellm.acompletion to always raise RateLimitError
    # Raise it MAX_LLM_RETRIES + 1 times to ensure we exceed the limit
    error = RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME)

    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError) as _:
            await call_llm(MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}])

        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_graceful_exit_on_rate_limit_during_plan():
    """Test that the agent exits gracefully when rate limit is hit during plan creation."""
    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # Simulate rate limit error during plan creation
        mock_call_llm.side_effect = RetryError(last_attempt=MagicMock())

        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1,
            system_prompt='You are a test agent.',
        )

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(f'Agent raised an exception instead of exiting gracefully: {e}')

        # Verify final response is an error
        final_response = next((r for r in responses if r['type'] == 'final'), None)
        assert final_response is not None, 'No final response found'
        assert final_response['metadata']['is_error'] is True
        assert 'Rate limit exceeded' in final_response['value']
        assert 'initial plan' in final_response['value']

        # Verify history contains the error message
        assert any('Rate limit exceeded' in str(msg) for msg in agent.chat_history), (
            'Error message not found in history'
        )
        assert len(agent.chat_history) > 0, 'History is empty'


@pytest.mark.asyncio
async def test_graceful_exit_on_rate_limit_during_execution():
    """Test that the agent exits gracefully when rate limit is hit during execution."""
    # Create a mock response object for plan
    mock_plan_response = '{"steps": [{"description": "Step 1", "is_done": false}]}'

    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # First call succeeds (Planner)
        # Second call (Think) fails with RetryError
        mock_call_llm.side_effect = [
            mock_plan_response,  # Plan
            RetryError(last_attempt=MagicMock()),  # Think fails
        ]

        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1,
            system_prompt='You are a test agent.',
        )

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(f'Agent raised an exception instead of exiting gracefully: {e}')

        # Verify final response is an error
        final_response = next((r for r in responses if r['type'] == 'final'), None)
        assert final_response is not None, 'No final response found'
        assert final_response['metadata']['is_error'] is True
        assert 'Rate limit exceeded' in final_response['value']

        # Verify history contains the error message
        assert any('Rate limit exceeded' in str(msg) for msg in agent.chat_history), (
            'Error message not found in history'
        )


@pytest.mark.asyncio
async def test_graceful_exit_on_exhaustion():
    """Test that the agent exits gracefully when retries are exhausted."""
    # Create an agent with mocked dependencies
    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # First call succeeds (Planner)
        # Second call (Think) fails with RetryError
        mock_call_llm.side_effect = [
            '{"steps": [{"description": "Step 1", "is_done": false}]}',  # Plan
            RetryError(last_attempt=MagicMock()),
        ]
        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1,
            system_prompt='You are a test agent.',
        )

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(f'Agent raised an exception instead of exiting gracefully: {e}')

        # Verify final response is an error
        final_response = next((r for r in responses if r['type'] == 'final'), None)
        assert final_response is not None
        assert final_response['metadata']['is_error'] is True
        assert 'Rate limit exceeded' in final_response['value']

        # Verify history contains the error message
        assert any('Rate limit exceeded' in str(msg) for msg in agent.chat_history)


@pytest.mark.asyncio
async def test_graceful_exit_on_rate_limit_during_plan_update():
    """Ensure the agent exits gracefully when rate limit is hit during plan update.
    This forces the _update_plan() path to raise RetryError.
    """
    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # Keep call order simple: create_plan -> think -> act
        mock_call_llm.side_effect = [
            # Initial plan creation: make planner.plan truthy
            '{"steps": [{"description": "Step 1", "is_done": false}]}',
            # Think
            'Thinking about Step 1.',
            # Act
            'Acting on Step 1 now.',
        ]

        agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[], max_iterations=1)

        # Force _update_plan to raise RetryError so we hit the plan-update error branch
        with patch.object(agent, '_update_plan', side_effect=RetryError(last_attempt=MagicMock())):
            responses = []
            try:
                async for response in agent.run('Simple task'):
                    responses.append(response)
            except Exception as e:
                pytest.fail(f'Agent raised an exception instead of exiting gracefully: {e}')

    # Verify final response is the plan-update error
    final_response = next((r for r in responses if r['type'] == 'final'), None)
    assert final_response is not None, 'No final response found'
    assert final_response['metadata']['final_answer_found'] is False
    assert final_response['metadata'].get('is_error') is True
    assert final_response['value'] == 'Rate limit exceeded during plan update. Unable to proceed.'


class EmptyAsyncIter:
    """An async iterator that yields nothing."""

    def __aiter__(self):
        """Return the async iterator itself."""
        return self

    async def __anext__(self):
        """Raise StopAsyncIteration to signal the end of iteration."""
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_observer_returns_none_on_retryerror():
    """Ensure that when observer.observe raises RetryError,
    correction_msg is set to None and no observer response is yielded.
    """
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[], max_iterations=3)

    # Make planner.plan truthy so plan update path runs
    agent.planner.plan = True
    agent.planner.get_formatted_plan = lambda: 'Step 1'
    agent._update_plan = AsyncMock(return_value=None)

    # Patch _think and _act to yield nothing
    agent._think = lambda: EmptyAsyncIter()
    agent._act = lambda: EmptyAsyncIter()

    # Force observer.observe to raise RetryError
    agent.observer.observe = AsyncMock(side_effect=RetryError(last_attempt=MagicMock()))

    responses = []
    async for response in agent.run('Task with observer failure'):
        responses.append(response)

    # Verify no observer channel response exists
    assert all(r.get('channel') != 'observer' for r in responses)
