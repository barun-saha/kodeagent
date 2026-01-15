"""Unit tests for the tracing module and its integration in kodeagent."""

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from kodeagent import tracer
from kodeagent.kodeagent import CodeActAgent, Observer, Planner, ReActAgent
from kodeagent.models import (
    AgentPlan,
    CodeActChatMessage,
    ObserverResponse,
    ReActChatMessage,
    Task,
)


class TestTracerModule:
    """Tests for the tracer module (tracer.py)."""

    def test_noop_observation(self) -> None:
        """Verify NoOpObservation methods do nothing and return None."""
        obs = tracer.NoOpObservation()
        assert obs.update(key='value') is None
        assert obs.end(result='ok') is None

    def test_noop_tracer_manager(self) -> None:
        """Verify NoOpTracerManager returns NoOpObservation."""
        manager = tracer.NoOpTracerManager()
        obs = manager.start_trace('test', {'input': 1})
        assert isinstance(obs, tracer.NoOpObservation)

        span = manager.start_span(obs, 'span', {'input': 2})
        assert isinstance(span, tracer.NoOpObservation)

        gen = manager.start_generation(obs, 'gen', {'input': 3})
        assert isinstance(gen, tracer.NoOpObservation)

    def test_abstract_observation_context_manager(self) -> None:
        """Verify AbstractObservation can be used as a context manager."""

        class MockObservation(tracer.AbstractObservation):
            def __init__(self) -> None:
                self.ended = False

            def update(self, **kwargs: Any) -> None:
                pass

            def end(self, **kwargs: Any) -> None:
                self.ended = True

        obs = MockObservation()
        with obs as entered_obs:
            assert entered_obs is obs
            assert not obs.ended
        assert obs.ended

    @patch('langfuse.client.Langfuse')
    def test_langfuse_tracer_manager(self, mock_langfuse_class: MagicMock) -> None:
        """Verify LangfuseTracerManager interacts correctly with Langfuse client."""
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client

        manager = tracer.LangfuseTracerManager()
        assert manager.client is mock_client

        # Test start_trace
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace
        res_trace = manager.start_trace('root', {'in': 1})
        mock_client.trace.assert_called_once_with(name='root', input={'in': 1})
        assert res_trace is mock_trace

        # Test start_span
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        res_span = manager.start_span(mock_trace, 'child', {'in': 2})
        mock_trace.span.assert_called_once_with(name='child', input={'in': 2})
        assert res_span is mock_span

        # Test start_generation
        mock_gen = MagicMock()
        mock_trace.generation.return_value = mock_gen
        res_gen = manager.start_generation(mock_trace, 'gen', {'in': 3})
        mock_trace.generation.assert_called_once_with(name='gen', input={'in': 3})
        assert res_gen is mock_gen


class TestTracingIntegration:
    """Tests for tracing integration in kodeagent.py."""

    @pytest.fixture
    def mock_tracer_manager(self) -> MagicMock:
        """Fixture for a mocked tracer manager."""
        manager = MagicMock(spec=tracer.AbstractTracerManager)
        # Ensure start_span returns a mock with an end method to avoid Errors
        manager.start_span.return_value = MagicMock(spec=tracer.AbstractObservation)
        manager.start_trace.return_value = MagicMock(spec=tracer.AbstractObservation)
        manager.start_generation.return_value = MagicMock(spec=tracer.AbstractObservation)
        return manager

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_planner_tracing(
        self, mock_call_llm: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify Planner creates spans during plan creation and update."""
        planner = Planner(model_name='test-model', tracer_manager=mock_tracer_manager)
        task = Task(description='test task', files=['f1.txt'])

        # Mock LLM response
        mock_call_llm.return_value = AgentPlan(steps=[]).model_dump_json()

        # Test create_plan
        await planner.create_plan(task, 'ReActAgent')
        mock_tracer_manager.start_span.assert_any_call(
            parent=ANY,
            name='plan_creation',
            input_data={
                'agent_type': 'ReActAgent',
                'task_id': task.id,
                'task_description': 'test task',
                'file_count': 1,
            },
        )

        mock_tracer_manager.start_span.return_value.end.assert_called()

        # Test update_plan
        planner.plan = AgentPlan(steps=[])
        await planner.update_plan('thought', 'obs', task.id)
        mock_tracer_manager.start_span.assert_any_call(
            parent=ANY,
            name='plan_update',
            input_data={
                'task_id': task.id,
                'thought_length': 7,
                'observation_length': 3,
                'current_steps': 0,
            },
        )

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_observer_tracing(
        self, mock_call_llm: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify Observer creates spans during observation."""
        observer = Observer(
            model_name='test-model', tool_names={'tool1'}, tracer_manager=mock_tracer_manager
        )
        task = Task(description='test task')

        # Mock LLM response for ObserverResponse
        obs_res = ObserverResponse(
            is_progressing=True,
            is_in_loop=False,
            reasoning='moving forward',
            correction_message=None,
        )
        mock_call_llm.return_value = obs_res.model_dump_json()

        await observer.observe(
            iteration=4, task=task, history='h', plan_before=None, plan_after=None
        )

        mock_tracer_manager.start_span.assert_called_with(
            parent=ANY,
            name='observe',
            input_data={
                'iteration': 4,
                'task_id': task.id,
                'history_length': 1,
                'tool_count': 1,
            },
        )

        mock_tracer_manager.start_span.return_value.end.assert_called()

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_observer_tracing_correction(
        self, mock_call_llm: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify Observer tracing captures corrections."""
        observer = Observer(
            model_name='test-model', tool_names={'tool1'}, tracer_manager=mock_tracer_manager
        )
        task = Task(description='test task')

        # Mock LLM response that triggers correction
        obs_res = ObserverResponse(
            is_progressing=False,
            is_in_loop=True,
            reasoning='looping',
            correction_message='stop looping',
        )
        mock_call_llm.return_value = obs_res.model_dump_json()

        await observer.observe(
            iteration=4, task=task, history='h', plan_before=None, plan_after=None
        )

        mock_tracer_manager.start_span.return_value.end.assert_called_with(
            output={
                'is_progressing': False,
                'is_in_loop': True,
                'correction_issued': True,
                'observation': 'stop looping',
            }
        )

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_observer_tracing_error(
        self, mock_call_llm: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify Observer tracing handles exceptions."""
        observer = Observer(
            model_name='test-model', tool_names={'tool1'}, tracer_manager=mock_tracer_manager
        )
        task = Task(description='test task')

        # Force an error in ObserverResponse validation
        mock_call_llm.side_effect = Exception('llm error')

        await observer.observe(
            iteration=4, task=task, history='h', plan_before=None, plan_after=None
        )

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(status='error', error='llm error')
        mock_span.end.assert_called_with(is_error=True)

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_agent_langfuse_init(self, mock_call_llm: MagicMock) -> None:
        """Verify Agent initializes LangfuseTracerManager when requested."""
        with patch('langfuse.client.Langfuse'):
            agent = ReActAgent(
                name='test-agent', model_name='test-model', tools=[], tracing_type='langfuse'
            )
            assert isinstance(agent.tracer_manager, tracer.LangfuseTracerManager)

    @pytest.mark.asyncio
    @patch('kodeagent.kutils.call_llm')
    async def test_agent_root_trace(self, mock_call_llm: MagicMock) -> None:
        """Verify Agent starts a root trace in _run_init."""
        # Using NoOpTracerManager by default, but we can verify it calls start_trace if we provide a mock
        mock_manager = MagicMock(spec=tracer.AbstractTracerManager)
        mock_manager.start_trace.return_value = MagicMock(spec=tracer.AbstractObservation)

        agent = ReActAgent(
            name='test-agent',
            model_name='test-model',
            tools=[],
            tracing_type='none',  # Will be overridden by explicitly passing tracer_manager to component
        )
        # Manually set tracer_manager because it's initialized in __init__
        agent.tracer_manager = mock_manager

        agent._run_init('test task', files=['f1.txt'])
        mock_manager.start_trace.assert_called_once_with(
            name='ReActAgent',
            input_data={
                'task': 'test task',
                'files': ['f1.txt'],
                'task_id': ANY,
            },
        )

    @pytest.mark.asyncio
    @patch('kodeagent.kodeagent.ReActAgent._record_thought')
    async def test_react_agent_think_tracing(
        self, mock_record_thought: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify ReActAgent._think creates a generation span."""
        agent = ReActAgent(name='test', model_name='m', tools=[])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()

        mock_record_thought.return_value = ReActChatMessage(
            thought='thinking', action='tool1', args='{}'
        )

        async for _ in agent._think():
            pass

        mock_tracer_manager.start_generation.assert_called_once_with(
            parent=agent.current_trace, name='think', input_data={'model': 'm', 'messages_count': 0}
        )

    @pytest.mark.asyncio
    async def test_react_agent_act_tracing(self, mock_tracer_manager: MagicMock) -> None:
        """Verify ReActAgent._act creates spans for tool execution."""
        mock_tool = MagicMock()
        mock_tool.name = 'tool1'
        mock_tool.return_value = 'result'

        agent = ReActAgent(name='test', model_name='m', tools=[mock_tool])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()

        # Setup message with tool call
        msg = ReActChatMessage(thought='using tool', action='tool1', args='{"arg1": 1}')
        agent.messages.append(msg)
        agent.task = Task(description='t')

        async for _ in agent._act():
            pass

        mock_tracer_manager.start_span.assert_any_call(
            parent=ANY, name='tool1', input_data={'arg1': 1}
        )

    @pytest.mark.asyncio
    async def test_react_agent_act_validation_error_tracing(
        self, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify ReActAgent._act traces validation errors."""
        agent = ReActAgent(name='test', model_name='m', tools=[])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Message with missing action
        msg = ReActChatMessage.model_construct(thought='thinking', action=None, args='{}')
        agent.messages.append(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='tool_validation_failed',
            error=ANY,
        )
        mock_span.end.assert_called_with(output='validation_error', is_error=True)

    @pytest.mark.asyncio
    async def test_react_agent_act_args_error_tracing(self, mock_tracer_manager: MagicMock) -> None:
        """Verify ReActAgent._act traces args parsing errors."""
        agent = ReActAgent(name='test', model_name='m', tools=[])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Message with non-dict args (after parsing)
        msg = ReActChatMessage.model_construct(thought='thinking', action='tool1', args='[1, 2, 3]')
        agent.messages.append(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='args_validation_failed',
            error=ANY,
        )
        mock_span.end.assert_called_with(output='args_error', is_error=True)

    @pytest.mark.asyncio
    @patch('kodeagent.kodeagent.ReActAgent._record_thought')
    async def test_code_act_agent_think_tracing(
        self, mock_record_thought: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._think creates a generation span."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()

        mock_record_thought.return_value = CodeActChatMessage(thought='thinking', code='print(1)')

        async for _ in agent._think():
            pass

        mock_tracer_manager.start_generation.assert_called_once_with(
            parent=agent.current_trace,
            name='think_code',
            input_data={'model': 'm', 'messages_count': 0},
        )

    @pytest.mark.asyncio
    @patch('kodeagent.kodeagent.ReActAgent._record_thought')
    async def test_code_act_agent_think_error_tracing(
        self, mock_record_thought: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._think traces parsing failures."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()

        mock_record_thought.return_value = None

        async for _ in agent._think():
            pass

        mock_gen = mock_tracer_manager.start_generation.return_value
        mock_gen.update.assert_called_with(status='error', error='Failed to parse response')
        mock_gen.end.assert_called_with(output='parse_failure', is_error=True)

    @pytest.mark.asyncio
    @patch('kodeagent.code_runner.CodeRunner.run')
    async def test_code_act_agent_act_tracing(
        self, mock_runner_run: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._act creates spans for code execution."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Setup message with code
        msg = CodeActChatMessage(thought='running code', code='print(1)')
        agent.messages.append(msg)

        mock_runner_run.return_value = ('out', '', 0, [])

        async for _ in agent._act():
            pass

        mock_tracer_manager.start_span.assert_any_call(
            parent=ANY, name='code_execution', input_data={'code_length': 8}
        )

    @pytest.mark.asyncio
    async def test_code_act_agent_act_missing_thought_tracing(
        self, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._act traces missing thought error."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Message with missing thought
        msg = CodeActChatMessage.model_construct(thought=None, code='print(1)')
        agent.messages.append(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(status='error', error='Missing or empty thought field')
        mock_span.end.assert_called_with(output='malformed_response')

    @pytest.mark.asyncio
    @patch('kodeagent.code_runner.CodeRunner.run')
    async def test_code_act_agent_act_exception_tracing(
        self, mock_runner_run: MagicMock, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._act traces execution exceptions."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        msg = CodeActChatMessage(thought='running', code='print(1)')
        agent.messages.append(msg)

        mock_runner_run.side_effect = Exception('runner crashed')

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='code_execution_exception',
            error_type='Exception',
            error_message='runner crashed',
        )
        mock_span.end.assert_called_with(
            output='exception',
            is_error=True,
            error=ANY,
        )

    @pytest.mark.asyncio
    async def test_code_act_agent_act_final_answer_tracing(
        self, mock_tracer_manager: MagicMock
    ) -> None:
        """Verify CodeActAgent._act creates spans for final answer."""
        agent = CodeActAgent(name='test', model_name='m', run_env='host')
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Setup message with final answer
        msg = CodeActChatMessage(thought='done', final_answer='the answer')
        agent.messages.append(msg)

        async for _ in agent._act():
            pass

        mock_tracer_manager.start_span.assert_called_with(
            parent=agent.current_trace, name='act', input_data={'thought': 'done'}
        )
        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='success',
            operation='final_answer',
            task_successful=False,
        )
        mock_span.end.assert_called_with(
            output='the answer',
            metadata={'task_successful': False},
        )
