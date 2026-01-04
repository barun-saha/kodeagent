"""
Unit tests for the ReActAgent pre-run and post-run hooks.
"""
from unittest.mock import MagicMock, AsyncMock
from typing import AsyncIterator

import pytest
from kodeagent.kodeagent import ReActAgent, AgentResponse


class TestAgent(ReActAgent):
    """Test agent that implements hooks with side effects."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_run_called = False
        self.post_run_called = False
    
    async def pre_run(self) -> AsyncIterator[AgentResponse]:
        self.pre_run_called = True
        # Call parent pre_run to ensure standard initialization happens
        async for r in super().pre_run():
            yield r
            
    async def post_run(self) -> AsyncIterator[AgentResponse]:
        self.post_run_called = True
        # Call parent post_run to ensure standard cleanup happens
        async for r in super().post_run():
            yield r


@pytest.mark.asyncio
async def test_hooks_execution():
    """Verify that pre_run and post_run hooks are executed."""
    agent = TestAgent(name='Test Agent', model_name='test-model', tools=[], max_iterations=1)
    # Mock planner and other components to avoid actual LLM calls
    agent.planner = MagicMock()
    agent.planner.create_plan = AsyncMock()
    agent.planner.get_formatted_plan = MagicMock(return_value='Test Plan')
    agent._think = MagicMock()
    agent._think.return_value = iter([]) # async iterator mock is tricky, let's just mock run dependencies
    
    # We need to mock _think and _act to return async iterators or ensure they are harmless
    async def mock_async_gen():
        if False: yield
        
    agent._think = MagicMock(return_value=mock_async_gen())
    agent._act = MagicMock(return_value=mock_async_gen())
    agent.observer = MagicMock()
    agent.observer.observe = AsyncMock(return_value=None)
    
    # Run the agent
    async for response in agent.run('Test task'):
        pass

    assert agent.pre_run_called is True
    assert agent.post_run_called is True


@pytest.mark.asyncio
async def test_pre_run_failure_stops_execution():
    """Verify that if pre_run yields an error, execution stops."""
    class FailingPreRunAgent(ReActAgent):
        async def pre_run(self) -> AsyncIterator[AgentResponse]:
            # Simulate failure
            yield {
                'type': 'final',
                'value': 'Setup failed',
                'channel': 'run',
                'metadata': {'is_error': True}
            }

    agent = FailingPreRunAgent(name='Fail Agent', model_name='test-model', tools=[])
    agent._think = MagicMock() # Should not be called
    
    responses = []
    async for response in agent.run('Test task'):
        responses.append(response)
        
    # Check that we got the error response
    assert len(responses) > 0
    assert responses[0]['type'] == 'final'
    assert responses[0]['value'] == 'Setup failed'
    
    # Verify _think was NOT called
    agent._think.assert_not_called()
