"""Tests for verifying system prompts in Planner and Observer."""

from unittest.mock import AsyncMock, patch

import pytest

from kodeagent.kodeagent import Task
from kodeagent.models import AgentPlan
from kodeagent.orchestrator import (
    OBSERVER_SYSTEM_PROMPT,
    PLAN_UPDATER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    Observer,
    Planner,
)


@pytest.fixture
def planner():
    """Create a Planner instance."""
    return Planner(model_name='test-model')


@pytest.fixture
def observer():
    """Create an Observer instance."""
    return Observer(model_name='test-model', tool_names={'tool1', 'tool2'}, threshold=1)


@pytest.mark.asyncio
async def test_planner_create_plan_system_prompt(planner):
    """Verify that Planner.create_plan uses the PLANNER_SYSTEM_PROMPT."""
    task = Task(description='test task')

    with patch('kodeagent.kutils.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = '{"steps": []}'
        await planner.create_plan(task, agent_type='ReAct')

        call_args = mock_call_llm.call_args
        assert call_args is not None, 'call_llm was not called'
        messages = call_args.kwargs['messages']

        # Verify the first message is the system prompt
        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == PLANNER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_planner_update_plan_system_prompt(planner):
    """Verify that Planner.update_plan uses the PLAN_UPDATER_SYSTEM_PROMPT."""
    planner.plan = AgentPlan(steps=[])

    with patch('kodeagent.kutils.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = '{"steps": []}'
        await planner.update_plan(thought='thinking', observation='observed', task_id='123')

        call_args = mock_call_llm.call_args
        assert call_args is not None, 'call_llm was not called'
        messages = call_args.kwargs['messages']

        # Verify the first message is the system prompt
        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == PLAN_UPDATER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_observer_observe_system_prompt(observer):
    """Verify that Observer.observe uses the OBSERVER_SYSTEM_PROMPT."""
    task = Task(description='test task')

    with patch('kodeagent.kutils.call_llm', new_callable=AsyncMock) as mock_call_llm:
        mock_call_llm.return_value = (
            '{"is_progressing": true, "is_in_loop": false, "reasoning": "ok"}'
        )

        # Ensure iteration allows observation (iteration 2, last 0, threshold 1 -> 2 >= 1)
        await observer.observe(
            iteration=2,
            task=task,
            history='history',
            plan_before='plan_before',
            plan_after='plan_after',
        )

        call_args = mock_call_llm.call_args
        assert call_args is not None, 'call_llm was not called'
        messages = call_args.kwargs['messages']

        # Verify the first message is the system prompt
        assert len(messages) >= 1
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == OBSERVER_SYSTEM_PROMPT
