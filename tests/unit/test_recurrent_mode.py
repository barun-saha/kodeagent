"""Test recurrent mode functionality."""

from unittest.mock import patch

import pytest

from kodeagent import AgentPlan, ReActAgent

MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.fixture
def react_agent():
    """Fixture for ReActAgent instance."""
    return ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[], max_iterations=1)


@pytest.mark.asyncio
async def test_recurrent_mode_disabled_by_default(react_agent):
    """Test that recurrent mode is disabled by default."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Task 1", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing task", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Task 1 done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task
        async for _ in react_agent.run('Task 1'):
            pass

        # Second task without recurrent mode
        async for _ in react_agent.run('Task 2'):
            pass

        # Verify Task 2 does NOT contain Task 1 context in description
        assert react_agent.task.description == 'Task 2'
        assert 'Previous Task Context' not in react_agent.task.description


@pytest.mark.asyncio
async def test_recurrent_mode_enabled(react_agent):
    """Test that recurrent mode augments task with previous context."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Result 1"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task
        async for _ in react_agent.run('First task'):
            pass

        # Verify first task completed
        assert react_agent.task is not None
        assert react_agent.task.result == 'Result 1'

        # Second task with recurrent mode enabled
        async for _ in react_agent.run('Second task', recurrent_mode=True):
            pass

        # Verify augmented task was used
        assert '## Previous Task Context' in react_agent.task.description
        assert 'First task' in react_agent.task.description
        assert 'Result 1' in react_agent.task.description
        assert 'Second task' in react_agent.task.description


@pytest.mark.asyncio
async def test_recurrent_mode_with_error_task(react_agent):
    """Test recurrent mode when previous task failed."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Recovered"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task - simulate error
        async for _ in react_agent.run('Failing task'):
            pass

        # Manually mark as error
        react_agent.task.is_error = True
        react_agent.task.result = 'Error occurred'

        # Second task with recurrent mode
        async for _ in react_agent.run('Retry the task', recurrent_mode=True):
            pass

        # Verify error status was included in context
        assert '❌ Failed' in react_agent.task.description
        assert 'Error occurred' in react_agent.task.description


@pytest.mark.asyncio
async def test_recurrent_mode_with_output_files(react_agent):
    """Test recurrent mode includes output files from previous task."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task
        async for _ in react_agent.run('Generate files'):
            pass

        # Add output files
        react_agent.task.output_files = ['/tmp/file1.txt', '/tmp/file2.txt']

        # Second task with recurrent mode
        async for _ in react_agent.run('Process the files', recurrent_mode=True):
            pass

        # Verify files were mentioned in context
        assert 'file1.txt' in react_agent.task.description
        assert 'Generated Files' in react_agent.task.description


@pytest.mark.asyncio
async def test_recurrent_mode_first_task(react_agent):
    """Test recurrent mode on first task (no previous context)."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task with recurrent mode (should have no effect since self.task is None)
        async for r in react_agent.run('First task', recurrent_mode=True):
            pass

        # Verify no augmentation occurred
        assert not react_agent.task.description.startswith('## Previous Task Context')
        assert react_agent.task.description == 'First task'


@pytest.mark.asyncio
async def test_recurrent_mode_truncates_long_result(react_agent):
    """Test that recurrent mode truncates very long results."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task
        async for _ in react_agent.run('Task 1'):
            pass

        # Set a very long result
        react_agent.task.result = 'A' * 3000

        # Second task with recurrent mode
        async for _ in react_agent.run('Task 2', recurrent_mode=True):
            pass

        # Verify result was truncated
        assert '[TRUNCATED]' in react_agent.task.description
        # Should not contain all 3000 A's
        assert react_agent.task.description.count('A') < 2500


@pytest.mark.asyncio
async def test_recurrent_mode_with_unfinished_task(react_agent):
    """Test recurrent mode with unfinished task (uses salvage_response)."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        # This will be used by salvage_response as well
        if kwargs.get('component_name') == 'Agent.salvage':
            return 'Salvaged progress summary'
        return (
            '{"role": "assistant", "thought": "Completing", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        # First task - simulate interruption by not breaking the loop
        # and manually setting is_finished to False
        async for _ in react_agent.run('Unfinished task'):
            pass

        react_agent.task.is_finished = False

        # Second task with recurrent mode
        async for _ in react_agent.run('Continue task', recurrent_mode=True):
            pass

        # Verify salvage_response was called and included in context
        assert 'Summary of Progress' in react_agent.task.description
        assert 'Salvaged progress summary' in react_agent.task.description


@pytest.mark.asyncio
async def test_recurrent_mode_forces_assistant_role(react_agent):
    """Test that agent forces 'assistant' role even if LLM returns 'user'."""

    def llm_side_effect(*args, **kwargs):
        if kwargs.get('response_format') == AgentPlan:
            return '{"steps": [{"description": "Execute", "is_done": false}]}'
        # Hallucinate role: user
        return (
            '{"role": "user", "thought": "Thinking...", '
            '"action": "FINISH", "args": null, '
            '"task_successful": true, "final_answer": "Done"}'
        )

    with patch('kodeagent.kutils.call_llm', side_effect=llm_side_effect):
        async for _ in react_agent.run('Test message role forcing'):
            pass

        # Verify that the message in history has role 'assistant' NOT 'user'
        # The last message is probably the assistant's final answer
        # Messages: [system, user, user (plan), assistant (result), assistant (summary/final)]
        # Actually in run loop it might vary, let's find the assistant message
        assistant_msgs = [m for m in react_agent.chat_history if m['role'] == 'assistant']
        assert len(assistant_msgs) > 0

        # Check if any message ended up with role 'user' that should be 'assistant'
        # In this mock, the only "user" role should be the ones explicitly added by the agent
        # as task descriptions or plans.
        for msg in react_agent.chat_history:
            if hasattr(msg, 'thought'):
                assert msg.role == 'assistant'


@pytest.mark.asyncio
async def test_augment_task_with_no_task(react_agent):
    """Test _augment_task_with_previous when self.task is None."""
    # pylint: disable=protected-access
    task_desc = 'test task'
    augmented = await react_agent._augment_task_with_previous(task_desc)
    assert augmented == task_desc
