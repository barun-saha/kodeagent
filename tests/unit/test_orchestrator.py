"""Unit tests for the orchestrator (Planner and Observer)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from kodeagent import (
    Observer,
    Planner,
    ReActAgent,
    Task,
)
from kodeagent.models import (
    AgentPlan,
    ObserverResponse,
    PlanStep,
)

MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.fixture
def planner():
    """Fixture to create a Planner instance for testing."""
    return Planner(model_name=MODEL_NAME, litellm_params={'max_tokens': 1000})


def test_plan_step():
    """Test PlanStep class initialization and properties."""
    step = PlanStep(description='Calculate sum')
    assert step.description == 'Calculate sum'
    assert step.is_done is False

    # Test marking step as done
    step.is_done = True
    assert step.is_done is True


def test_agent_plan():
    """Test AgentPlan class initialization and management of steps."""
    steps = [
        PlanStep(description='Step 1'),
        PlanStep(description='Step 2'),
        PlanStep(description='Step 3'),
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
        correction_message=None,
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
        correction_message='Try using a different approach to solve the calculation',
    )

    assert response_with_correction.is_progressing is False
    assert response_with_correction.is_in_loop is True
    assert response_with_correction.correction_message is not None


def test_observer_response_validation():
    """Test validation of ObserverResponse fields."""
    # Empty reasoning should NOT fail - it's just a string field
    response = ObserverResponse(
        is_progressing=True,
        is_in_loop=False,
        reasoning='',  # Empty string is allowed
        correction_message=None,
    )
    assert response.reasoning == ''

    # Test with valid reasoning
    response2 = ObserverResponse(
        is_progressing=False,
        is_in_loop=True,
        reasoning='Agent is stuck',
        correction_message='Try again',
    )
    assert response2.reasoning == 'Agent is stuck'


def test_observer_response_missing_fields():
    """Test ObserverResponse with missing optional fields."""
    response = ObserverResponse(
        is_progressing=False, is_in_loop=True, reasoning='test', correction_message=None
    )
    assert response.is_in_loop
    assert response.reasoning == 'test'
    assert response.correction_message is None


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
            task_id=str(task.id),
        )

        assert planner.plan is not None
        assert len(planner.plan.steps) > 0
        assert planner.plan.steps[0].is_done


def test_planner_get_steps_status(planner):
    """Test getting completed and pending steps."""
    # Create a plan manually for testing
    plan = AgentPlan(
        steps=[
            PlanStep(description='Step 1', is_done=True),
            PlanStep(description='Step 2', is_done=False),
            PlanStep(description='Step 3', is_done=True),
        ]
    )
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
    plan = AgentPlan(
        steps=[
            PlanStep(description='Step 1', is_done=True),
            PlanStep(description='Step 2', is_done=False),
            PlanStep(description='Step 3', is_done=True),
        ]
    )
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


def test_planner_reset(planner):
    """Test resetting the planner."""
    # Create a plan manually for testing
    plan = AgentPlan(steps=[PlanStep(description='Step 1', is_done=True)])
    planner.plan = plan
    planner.reset()
    assert planner.plan is None


def test_planner_with_litellm_params():
    """Test Planner initialization with litellm_params."""
    params = {'temperature': 0.5, 'max_tokens': 500}
    planner = Planner(model_name=MODEL_NAME, litellm_params=params)

    assert planner.litellm_params == params
    assert planner.model_name == MODEL_NAME


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
            model_name=MODEL_NAME, tool_names={'calculator', 'search_web'}, threshold=2
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
            iteration=1, task=task, history=history, plan_before=None, plan_after=None
        )
        assert correction is None

        # Call after threshold with looping behavior
        correction = await observer.observe(
            iteration=3, task=task, history=history, plan_before=None, plan_after=None
        )
        assert correction is not None
        assert 'CRITICAL FOR COURSE CORRECTION' in correction


@pytest.mark.asyncio
async def test_observer_reset():
    """Test Observer reset functionality."""
    observer = Observer(model_name=MODEL_NAME, tool_names={'calculator'}, threshold=2)

    observer.last_correction_iteration = 5
    observer.reset()
    assert observer.last_correction_iteration == 0


def test_observer_with_custom_threshold():
    """Test Observer with custom threshold."""
    observer = Observer(model_name=MODEL_NAME, tool_names={'calculator'}, threshold=5)
    assert observer.threshold == 5


@pytest.mark.asyncio
async def test_observer_with_negative_threshold():
    """Test Observer with None threshold (disabled)."""
    observer = Observer(
        model_name=MODEL_NAME,
        tool_names={'calculator'},
        threshold=None,  # Disabled observer
    )
    task = Task(description='Test', files=None)
    correction = await observer.observe(
        iteration=5, task=task, history='test', plan_before=None, plan_after=None
    )

    # Should return None because threshold is None
    assert correction is None


@pytest.mark.asyncio
async def test_observer_exception_handling():
    """Test Observer exception handling."""
    observer = Observer(model_name=MODEL_NAME, tool_names={'calculator'}, threshold=1)
    task = Task(description='Test', files=None)

    with patch('kodeagent.kutils.call_llm', side_effect=Exception('LLM error')):
        correction = await observer.observe(
            iteration=2, task=task, history='test', plan_before=None, plan_after=None
        )

        # Should return None on exception
        assert correction is None


@pytest.mark.asyncio
async def test_observer_yields_correction_message():
    """Ensure that when observer.observe returns a correction message,
    the agent adds it to history and yields an observer log response.
    """
    agent = ReActAgent(name='test_agent', model_name=MODEL_NAME, tools=[], max_iterations=3)

    # Patch planner so plan exists and no real LLM call is made
    agent.planner.plan = AgentPlan(steps=[PlanStep(description='Step 1', is_done=False)])
    agent.planner.get_formatted_plan = lambda *args, **kwargs: 'Step 1'
    agent.planner.create_plan = AsyncMock(return_value=None)  # No-op

    # Patch _think and _act to yield nothing
    class EmptyAsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    agent._think = lambda: EmptyAsyncIter()
    agent._act = lambda: EmptyAsyncIter()

    # Patch _update_plan to succeed
    agent._update_plan = AsyncMock(return_value=None)

    # Patch observer.observe to return a correction message
    agent.observer.observe = AsyncMock(return_value='Please adjust step 1')

    responses = []
    async for response in agent.run('Task with observer correction'):
        responses.append(response)

    # Verify an observer channel response exists with the correction message
    observer_response = next((r for r in responses if r.get('channel') == 'observer'), None)
    assert observer_response is not None
    assert 'Please adjust step 1' in observer_response['value']

    # Verify history contains the observation message
    assert any('Observation: Please adjust step 1' in str(msg) for msg in agent.messages)


@pytest.mark.asyncio
async def test_observer_retry_error():
    """Test Observer rate limit handling."""
    agent = ReActAgent(name='CoverageAgent', model_name='test-model', tools=[], max_iterations=1)
    agent.usage_tracker = MagicMock()
    agent.task = Task(description='Task')
    agent.planner = MagicMock()
    agent.planner.plan = None

    # Fix: pre_run mock
    async def mock_pre_run():
        if False:
            yield

    agent.pre_run = MagicMock(return_value=mock_pre_run())

    async def success_gen():
        yield {'type': 'step', 'value': 'stuff'}

    agent._think = MagicMock(return_value=success_gen())
    agent._act = MagicMock(return_value=success_gen())

    # Observer raises RetryError
    retry_err = RetryError(last_attempt=MagicMock())
    agent.observer = MagicMock()
    agent.observer.observe = AsyncMock(side_effect=retry_err)

    responses = []
    async for resp in agent.run('Task'):
        responses.append(resp)

    # If we got here without exception, it passed the try/except block
    assert True
