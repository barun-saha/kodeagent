"""
Unit tests for retry logic in call_llm and agent behavior on rate limit errors.
"""
from unittest.mock import patch, MagicMock

import pytest
from tenacity import RetryError
from litellm.exceptions import RateLimitError

from kodeagent import ReActAgent
from kodeagent.kutils import call_llm, DEFAULT_MAX_LLM_RETRIES


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.mark.asyncio
async def test_call_llm_rate_limit_retry_success():
    """
    Test that call_llm retries on RateLimitError and succeeds if it clears within limits.
    """
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
    mock_response._hidden_params = {}
    mock_response.usage = {}

    # Mock litellm.acompletion to raise RateLimitError 2 times, then return success
    # This should succeed because DEFAULT_MAX_LLM_RETRIES is 3
    side_effects = [
        RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME),
        RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME),
        mock_response
    ]

    with patch('litellm.acompletion', side_effect=side_effects) as mock_acompletion:
        response = await call_llm(
            MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}]
        )

        assert response == 'Success'
        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_call_llm_retries_on_generic_error():
    """
    Test that call_llm retries on generic errors (as per user configuration).
    """
    # Mock litellm.acompletion to raise a generic Exception
    error = ValueError('Some other error')

    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError) as excinfo:
            await call_llm(MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}])

        # Verify call count is DEFAULT_MAX_LLM_RETRIES (because we retry on Exception now)
        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_call_llm_custom_max_retries():
    """
    Test that call_llm respects custom max_retries.
    """
    # Mock litellm.acompletion to raise a generic Exception
    error = ValueError('Some other error')
    custom_retries = 5
    
    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError):
            await call_llm(
                MODEL_NAME, {},
                [{'role': 'user', 'content': 'hi'}], max_retries=custom_retries
            )
        
        # Verify call count matches custom retries
        assert mock_acompletion.call_count == custom_retries


@pytest.mark.asyncio
async def test_call_llm_rate_limit_retry_failure():
    """
    Test that call_llm fails after MAX_LLM_RETRIES if RateLimitError persists.
    """
    # Mock litellm.acompletion to always raise RateLimitError
    # Raise it MAX_LLM_RETRIES + 1 times to ensure we exceed the limit
    error = RateLimitError(message='Rate limit exceeded', llm_provider='gemini', model=MODEL_NAME)

    with patch('litellm.acompletion', side_effect=error) as mock_acompletion:
        with pytest.raises(RetryError) as excinfo:
            await call_llm(MODEL_NAME, {}, [{'role': 'user', 'content': 'hi'}])

        assert mock_acompletion.call_count == DEFAULT_MAX_LLM_RETRIES


@pytest.mark.asyncio
async def test_graceful_exit_on_rate_limit_during_plan():
    """
    Test that the agent exits gracefully when rate limit is hit during plan creation.
    """
    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # Simulate rate limit error during plan creation
        mock_call_llm.side_effect = RetryError(last_attempt=MagicMock())

        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1,
            system_prompt='You are a test agent.'
        )

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(
                f'Agent raised an exception instead of exiting gracefully: {e}'
            )

        # Verify final response is an error
        final_response = next(
            (r for r in responses if r['type'] == 'final'), None
        )
        assert final_response is not None, 'No final response found'
        assert final_response['metadata']['is_error'] is True
        assert 'Rate limit exceeded' in final_response['value']
        assert 'initial plan' in final_response['value']

        # Verify history contains the error message
        assert any(
            'Rate limit exceeded' in str(msg) for msg in agent.messages
        ), 'Error message not found in history'
        assert len(agent.messages) > 0, 'History is empty'


@pytest.mark.asyncio
async def test_graceful_exit_on_rate_limit_during_execution():
    """
    Test that the agent exits gracefully when rate limit is hit during execution.
    """
    # Create a mock response object for plan
    mock_plan_response = '{"steps": [{"description": "Step 1", "is_done": false}]}'

    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # First call succeeds (Planner)
        # Second call (Think) fails with RetryError
        mock_call_llm.side_effect = [
            mock_plan_response,  # Plan
            RetryError(last_attempt=MagicMock())  # Think fails
        ]

        agent = ReActAgent(
            name='test_agent',
            model_name=MODEL_NAME,
            tools=[],
            max_iterations=1,
            system_prompt='You are a test agent.'
        )

        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(
                f'Agent raised an exception instead of exiting gracefully: {e}'
            )

        # Verify final response is an error
        final_response = next(
            (r for r in responses if r['type'] == 'final'), None
        )
        assert final_response is not None, 'No final response found'
        assert final_response['metadata']['is_error'] is True
        assert 'Rate limit exceeded' in final_response['value']

        # Verify history contains the error message
        assert any(
            'Rate limit exceeded' in str(msg) for msg in agent.messages
        ), 'Error message not found in history'


@pytest.mark.asyncio
async def test_graceful_exit_on_exhaustion():
    """
    Test that the agent exits gracefully when retries are exhausted.
    """
    # Create an agent with mocked dependencies
    with patch('kodeagent.kutils.call_llm') as mock_call_llm:
        # First call succeeds (Planner)
        # Second call (Think) fails with RetryError
        mock_call_llm.side_effect = [
            '{"steps": [{"description": "Step 1", "is_done": false}]}', # Plan
            RetryError(last_attempt=MagicMock())
        ]
        
        agent = ReActAgent(
            name='test_agent', 
            model_name=MODEL_NAME, 
            tools=[],
            max_iterations=1,
            system_prompt="You are a test agent."
        )
        
        responses = []
        try:
            async for response in agent.run('Simple task'):
                responses.append(response)
        except Exception as e:
            pytest.fail(f"Agent raised an exception instead of exiting gracefully: {e}")
            
        # Verify final response is an error
        final_response = next((r for r in responses if r['type'] == 'final'), None)
        assert final_response is not None
        assert final_response['metadata']['is_error'] is True
        assert "Rate limit exceeded" in final_response['value']
        
        # Verify history contains the error message
        assert any("Rate limit exceeded" in str(msg) for msg in agent.messages)
