"""Unit tests for the tracing module and its integration in kodeagent."""

from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from kodeagent import tracer
from kodeagent.kodeagent import CodeActAgent, ReActAgent
from kodeagent.models import (
    AgentPlan,
    CodeActChatMessage,
    ObserverResponse,
    ReActChatMessage,
    Task,
)
from kodeagent.orchestrator import Observer, Planner


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
        res_trace_obs = manager.start_trace('root', {'in': 1})
        mock_client.trace.assert_called_once_with(name='root', input={'in': 1})
        assert isinstance(res_trace_obs, tracer.LangfuseObservation)
        assert res_trace_obs.obj is mock_trace

        # Test start_span
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        res_span_obs = manager.start_span(res_trace_obs, 'child', {'in': 2})
        mock_trace.span.assert_called_once_with(name='child', input={'in': 2})
        assert isinstance(res_span_obs, tracer.LangfuseObservation)
        assert res_span_obs.obj is mock_span

        # Test start_generation
        mock_gen = MagicMock()
        mock_trace.generation.return_value = mock_gen
        res_gen_obs = manager.start_generation(res_trace_obs, 'gen', {'in': 3})
        mock_trace.generation.assert_called_once_with(name='gen', input={'in': 3})
        assert isinstance(res_gen_obs, tracer.LangfuseObservation)
        assert res_gen_obs.obj is mock_gen

    @patch('langsmith.run_trees.RunTree')
    @patch('langsmith.Client')
    def test_langsmith_tracer_manager(
        self, mock_client_class: MagicMock, mock_runtree_class: MagicMock
    ) -> None:
        """Verify LangSmithTracerManager interacts correctly with RunTree."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        manager = tracer.LangSmithTracerManager()
        assert manager.client is mock_client

        # Test start_trace
        mock_run_tree = MagicMock()
        mock_runtree_class.return_value = mock_run_tree

        res_obs = manager.start_trace('root', {'in': 1})

        # Verify RunTree created and posted
        mock_runtree_class.assert_called_once_with(
            name='root', run_type='chain', inputs={'in': 1}, client=mock_client
        )
        mock_run_tree.post.assert_called_once()
        assert isinstance(res_obs, tracer.LangSmithObservation)
        assert res_obs.run_tree is mock_run_tree

        # Test start_span (child)
        mock_child_tree = MagicMock()
        mock_run_tree.create_child.return_value = mock_child_tree

        res_span = manager.start_span(res_obs, 'child', {'in': 2})

        mock_run_tree.create_child.assert_called_once_with(
            name='child', run_type='tool', inputs={'in': 2}
        )
        # Verify wrapped in observation
        assert isinstance(res_span, tracer.LangSmithObservation)
        assert res_span.run_tree is mock_child_tree

        # Test end with output
        res_span.end(output='done', metadata={'m': 1})
        mock_child_tree.add_metadata.assert_called_once_with({'m': 1})
        mock_child_tree.end.assert_called_once_with(outputs={'output': 'done'}, error=None)
        mock_child_tree.patch.assert_called_once()

        # Test end with result (mapping check)
        mock_run_tree.end.reset_mock()
        res_obs.end(result='finished')
        mock_run_tree.end.assert_called_once_with(outputs={'output': 'finished'}, error=None)

    def test_tracer_manager_flush(self) -> None:
        """Verify flush method delegates to client."""
        # NoOp
        noop = tracer.NoOpTracerManager()
        noop.flush()  # Should not raise

        # Langfuse
        with patch('langfuse.client.Langfuse') as mock_lf:
            mock_client = MagicMock()
            mock_lf.return_value = mock_client
            lf_manager = tracer.LangfuseTracerManager()
            lf_manager.flush()
            mock_client.flush.assert_called_once()

        # LangSmith
        with patch('langsmith.Client') as mock_ls:
            mock_client = MagicMock()
            mock_ls.return_value = mock_client
            ls_manager = tracer.LangSmithTracerManager()
            ls_manager.flush()
            mock_client.flush.assert_called_once()


class TestTracingIntegration:
    """Tests for tracing integration in kodeagent.py."""

    @pytest.fixture
    def mock_tracer_manager(self) -> MagicMock:
        """Fixture for a mocked tracer manager."""
        manager = MagicMock(spec=tracer.AbstractTracerManager)

        def make_mock_obs():
            obs = MagicMock(spec=tracer.AbstractObservation)
            obs.__enter__.return_value = obs
            return obs

        # Ensure start calls return a mock with an end method and context manager support
        manager.start_trace.return_value = make_mock_obs()
        manager.start_span.return_value = make_mock_obs()
        manager.start_generation.return_value = make_mock_obs()
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
        agent.add_to_history(msg)
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
        agent.add_to_history(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='tool_validation_failed',
            error=ANY,
            output='validation_error',
            is_error=True,
        )

    @pytest.mark.asyncio
    async def test_react_agent_act_args_error_tracing(self, mock_tracer_manager: MagicMock) -> None:
        """Verify ReActAgent._act traces args parsing errors."""
        agent = ReActAgent(name='test', model_name='m', tools=[])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')

        # Message with non-dict args (after parsing)
        msg = ReActChatMessage.model_construct(thought='thinking', action='tool1', args='[1, 2, 3]')
        agent.add_to_history(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='args_validation_failed',
            error=ANY,
            output='args_error',
            is_error=True,
        )

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
        mock_gen.update.assert_called_with(
            status='error',
            error='Failed to parse response',
            output='parse_failure',
            is_error=True,
        )

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
        agent.add_to_history(msg)

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
        agent.add_to_history(msg)

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            error='Missing or empty thought field',
            output='malformed_response',
        )

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
        agent.add_to_history(msg)

        mock_runner_run.side_effect = Exception('runner crashed')

        async for _ in agent._act():
            pass

        mock_span = mock_tracer_manager.start_span.return_value
        mock_span.update.assert_called_with(
            status='error',
            operation='code_execution_exception',
            error_type='Exception',
            error_message='runner crashed',
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
        agent.add_to_history(msg)

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
            output='the answer',
            metadata={'task_successful': False},
        )

    @pytest.mark.asyncio
    async def test_agent_post_run_tracing(self, mock_tracer_manager: MagicMock) -> None:
        """Verify Agent.post_run ends trace and flushes manager."""
        agent = ReActAgent(name='test', model_name='m', tools=[])
        agent.tracer_manager = mock_tracer_manager
        agent.current_trace = MagicMock()
        agent.task = Task(description='t')
        agent.task.result = 'final'

        # Mock usage metrics so post_run populates task with these values
        with patch.object(agent, 'get_usage_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'total': {
                    'call_count': 1,
                    'total_prompt_tokens': 50,
                    'total_completion_tokens': 50,
                    'total_tokens': 100,
                    'total_cost': 0.01,
                },
                'by_component': {},
            }

            async for _ in agent.post_run():
                pass

        agent.current_trace.end.assert_called_with(
            result='final',
            metadata={
                'is_error': False,
                'total_tokens': 100,
                'total_cost': 0.01,
                'steps_taken': 0,
            },
        )
        mock_tracer_manager.flush.assert_called_once()


class TestTracingCoverage:
    """Extra tests to achieve high coverage for tracer.py."""

    def test_create_tracer_manager_factory(self) -> None:
        """Test all branches of create_tracer_manager."""
        assert isinstance(tracer.create_tracer_manager('langfuse'), tracer.LangfuseTracerManager)
        assert isinstance(tracer.create_tracer_manager('langsmith'), tracer.LangSmithTracerManager)
        assert isinstance(tracer.create_tracer_manager(None), tracer.NoOpTracerManager)
        assert isinstance(tracer.create_tracer_manager('unknown'), tracer.NoOpTracerManager)

    def test_langfuse_init_import_error(self) -> None:
        """Test LangfuseTracerManager when langfuse is not installed."""
        with patch.dict('sys.modules', {'langfuse.client': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                manager = tracer.LangfuseTracerManager()
                assert manager.client is None
                assert isinstance(manager.start_trace('n', {}), tracer.NoOpObservation)
                assert isinstance(
                    manager.start_span(tracer.NoOpObservation(), 'n', {}), tracer.NoOpObservation
                )
                assert isinstance(
                    manager.start_generation(tracer.NoOpObservation(), 'n', {}),
                    tracer.NoOpObservation,
                )

    def test_langsmith_init_import_error(self) -> None:
        """Test LangSmithTracerManager when langsmith is not installed."""
        with patch.dict('sys.modules', {'langsmith': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                manager = tracer.LangSmithTracerManager()
                assert manager.client is None
                assert isinstance(manager.start_trace('n', {}), tracer.NoOpObservation)

    def test_langsmith_start_trace_import_error(self) -> None:
        """Test LangSmithTracerManager.start_trace when RunTree cannot be imported."""
        manager = tracer.LangSmithTracerManager()
        manager.client = MagicMock()
        with patch('builtins.__import__', side_effect=ImportError):
            assert isinstance(manager.start_trace('n', {}), tracer.NoOpObservation)

    def test_langsmith_observation_post_error(self) -> None:
        """Test LangSmithObservation handling post() exception."""
        mock_run_tree = MagicMock()
        mock_run_tree.post.side_effect = Exception('post failed')
        # Should not raise
        obs = tracer.LangSmithObservation(mock_run_tree)
        assert not obs.ended

    def test_langsmith_observation_update_error(self) -> None:
        """Test LangSmithObservation mapping error status."""
        mock_run_tree = MagicMock()
        obs = tracer.LangSmithObservation(mock_run_tree)
        obs.update(error='some error')
        assert mock_run_tree.error == 'some error'

    def test_langsmith_observation_end_exception(self) -> None:
        """Test LangSmithObservation handling exceptions in end()."""
        mock_run_tree = MagicMock()
        mock_run_tree.patch.side_effect = Exception('patch failed')
        obs = tracer.LangSmithObservation(mock_run_tree)
        # Should catch exception and log it
        obs.end(output='test')
        assert obs.ended

    def test_langsmith_manager_client_none(self) -> None:
        """Test LangSmithTracerManager methods when client is None."""
        manager = tracer.LangSmithTracerManager()
        manager.client = None
        assert isinstance(manager.start_trace('n', {}), tracer.NoOpObservation)
        assert isinstance(
            manager.start_span(tracer.NoOpObservation(), 'n', {}), tracer.NoOpObservation
        )
        assert isinstance(
            manager.start_generation(tracer.NoOpObservation(), 'n', {}), tracer.NoOpObservation
        )

    def test_langsmith_manager_wrong_parent(self) -> None:
        """Test LangSmithTracerManager methods with wrong parent type."""
        manager = tracer.LangSmithTracerManager()
        manager.client = MagicMock()
        noop = tracer.NoOpObservation()
        assert isinstance(manager.start_span(noop, 'n', {}), tracer.NoOpObservation)
        assert isinstance(manager.start_generation(noop, 'n', {}), tracer.NoOpObservation)

    def test_langfuse_observation_mapping(self) -> None:
        """Verify LangfuseObservation maps result to output and handles missing end."""
        # Case 1: Trace (no end)
        mock_trace = MagicMock(spec=['update'])
        obs = tracer.LangfuseObservation(mock_trace)
        obs.end(result='final ok')
        mock_trace.update.assert_called_once_with(output='final ok')

        # Case 2: Span/Gen (has end)
        mock_span = MagicMock()
        obs2 = tracer.LangfuseObservation(mock_span)
        obs2.end(result='span ok')
        mock_span.end.assert_called_once_with(output='span ok')

    def test_langfuse_observation_update(self) -> None:
        """Verify LangfuseObservation.update delegates to wrapped object."""
        mock_obj = MagicMock()
        obs = tracer.LangfuseObservation(mock_obj)
        obs.update(status='success')
        mock_obj.update.assert_called_once_with(status='success')
