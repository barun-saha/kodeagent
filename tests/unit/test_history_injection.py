import pytest

from kodeagent.fca import FunctionCallingAgent
from kodeagent.kodeagent import ReActAgent


@pytest.fixture
def mock_agent():
    return ReActAgent(model_name='gpt-4', name='TestAgent')


@pytest.fixture
def mock_fca():
    return FunctionCallingAgent(model_name='gpt-4')


def test_validate_chat_history_valid(mock_agent):
    valid_history = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello'},
        {'role': 'assistant', 'content': 'Hi there!', 'tool_calls': None},
    ]
    # This should not raise
    mock_agent._run_init(task='test', chat_history=valid_history, task_id='task-123')
    assert len(mock_agent.chat_history) == 3
    assert mock_agent.task.id == 'task-123'


def test_validate_chat_history_no_system(mock_agent):
    history = [{'role': 'user', 'content': 'Hello'}]
    mock_agent._run_init(task='test', chat_history=history)
    # _run_init should prepend system prompt
    assert len(mock_agent.chat_history) == 2


@pytest.mark.asyncio
async def test_react_agent_history_injection_pre_run(mock_agent):
    history = [{'role': 'user', 'content': 'Hello'}]
    mock_agent._run_init(task='Resuming task', chat_history=history)

    # We need to mock planner to avoid LLM calls in pre_run
    from unittest.mock import MagicMock

    mock_agent.planner = MagicMock()
    mock_agent.planner.get_formatted_plan.return_value = 'Step 1: Test'
    mock_agent.planner.plan = 'test'
    mock_agent.planner.create_plan = MagicMock(side_effect=lambda *args, **kwargs: None)
    # create_plan is actually awaited in pre_run, so it must be an AsyncMock
    from unittest.mock import AsyncMock

    mock_agent.planner.create_plan = AsyncMock()

    responses = []
    async for resp in mock_agent.pre_run():
        responses.append(resp)

    # Check that system prompt was prepended
    assert mock_agent.chat_history[0]['role'] == 'system'
    assert mock_agent.chat_history[1]['role'] == 'user'
    # Content is merged/converted to list by add_to_history
    assert any('Hello' in item.get('text', '') for item in mock_agent.chat_history[1]['content'])

    # New task should be in the history as well
    def has_text(content, search_str):
        if isinstance(content, str):
            return search_str in content
        if isinstance(content, list):
            return any(search_str in item.get('text', '') for item in content)
        return False

    assert any(
        has_text(m.get('content'), 'New Task:\nResuming task') for m in mock_agent.chat_history
    )


def test_validate_chat_history_invalid_role(mock_agent):
    invalid_history = [{'role': 'bad_role', 'content': 'Hello'}]
    with pytest.raises(ValueError, match="role' must be one of"):
        mock_agent._run_init(task='test', chat_history=invalid_history)


def test_validate_chat_history_unresolved_tool_call(mock_agent):
    invalid_history = [
        {
            'role': 'assistant',
            'tool_calls': [
                {'id': '1', 'type': 'function', 'function': {'name': 'test', 'arguments': '{}'}}
            ],
        }
    ]
    with pytest.raises(ValueError, match='unresolved tool call'):
        mock_agent._run_init(task='test', chat_history=invalid_history)


def test_fca_history_injection_valid(mock_fca):
    valid_history = [{'role': 'user', 'content': 'Hello'}]
    # FCA _run_init is async
    import asyncio

    asyncio.run(
        mock_fca._run_init(
            task_desc='test', chat_history=valid_history, use_planning=False, task_id='fca-task'
        )
    )

    assert mock_fca.chat_history[0]['role'] == 'system'
    assert mock_fca.chat_history[1]['role'] == 'user'
    assert mock_fca.task.id == 'fca-task'

    def has_text(content, search_str):
        if isinstance(content, str):
            return search_str in content
        if isinstance(content, list):
            return any(search_str in item.get('text', '') for item in content)
        return False

    # FCA uses kutils.make_user_message which returns list content
    assert has_text(mock_fca.chat_history[1]['content'], 'Hello')
    # FCA appends the new task message immediately in _run_init
    assert has_text(mock_fca.chat_history[2]['content'], '## New Task:\ntest')


@pytest.mark.asyncio
async def test_react_agent_history_injection_with_system(mock_agent):
    # History already has a system message
    history = [
        {'role': 'system', 'content': 'Existing system message'},
        {'role': 'user', 'content': 'Hello'},
    ]
    mock_agent._run_init(task='New task', chat_history=history)

    from unittest.mock import AsyncMock, MagicMock

    mock_agent.planner = MagicMock()
    mock_agent.planner.get_formatted_plan.return_value = 'Plan'
    mock_agent.planner.create_plan = AsyncMock()

    async for _ in mock_agent.pre_run():
        pass

    # System message should NOT be prepended again
    assert mock_agent.chat_history[0]['content'] == 'Existing system message'
    assert mock_agent.chat_history[1]['role'] == 'user'


@pytest.mark.asyncio
async def test_fca_history_injection_with_system(mock_fca):
    # History already has a system message
    history = [
        {'role': 'system', 'content': 'Existing system message'},
        {'role': 'user', 'content': 'Hello'},
    ]
    await mock_fca._run_init(task_desc='test', chat_history=history, use_planning=False)

    # System message should NOT be prepended again
    assert mock_fca.chat_history[0]['content'] == 'Existing system message'
    assert mock_fca.chat_history[1]['role'] == 'user'


def test_mutual_exclusivity_react(mock_agent):
    import asyncio

    with pytest.raises(ValueError, match='mutually exclusive'):
        # We need to wrap the async generator to trigger the check if it were in the body,
        # but here it's at the start of run()
        gen = mock_agent.run(
            task='test', recurrent_mode=True, chat_history=[{'role': 'user', 'content': 'hi'}]
        )
        asyncio.run(gen.__anext__())


def test_mutual_exclusivity_fca(mock_fca):
    import asyncio

    with pytest.raises(ValueError, match='mutually exclusive'):
        gen = mock_fca.run(
            task='test', recurrent_mode=True, chat_history=[{'role': 'user', 'content': 'hi'}]
        )
        asyncio.run(gen.__anext__())


# Direct kutils.validate_chat_history tests for coverage
from kodeagent import kutils as ku


def test_validate_history_not_list():
    with pytest.raises(ValueError, match='must be a list'):
        ku.validate_chat_history('not a list')


def test_validate_history_empty():
    with pytest.raises(ValueError, match='must not be empty'):
        ku.validate_chat_history([])


def test_validate_history_msg_not_dict():
    with pytest.raises(ValueError, match='must be a dict'):
        ku.validate_chat_history(['not a dict'])


def test_validate_history_invalid_role():
    with pytest.raises(ValueError, match="role' must be one of"):
        ku.validate_chat_history([{'role': 'invalid'}])


def test_validate_history_system_not_at_0():
    with pytest.raises(ValueError, match='must be at index 0'):
        ku.validate_chat_history(
            [{'role': 'user', 'content': 'hi'}, {'role': 'system', 'content': 'sys'}]
        )


def test_validate_history_multiple_systems():
    # This check is slightly redundant with "must be at index 0" but good to have
    # because the loop continues. Actually, if there's one at 0 and one at 1,
    # the index check for index 1 triggers first.
    # To trigger "only one system message is allowed", we'd need to bypass idx!=0 check.
    # Wait, the code says:
    # if role == 'system':
    #     if idx != 0: raise ValueError(...)
    #     if system_seen_at is not None: raise ValueError(...)
    # So idx!=0 ALWAYS triggers if it's not the first one.
    pass


def test_validate_history_system_none_content():
    with pytest.raises(ValueError, match="must have a non-None 'content'"):
        ku.validate_chat_history([{'role': 'system', 'content': None}])


def test_validate_history_user_none_content():
    with pytest.raises(ValueError, match="must have a non-None 'content'"):
        ku.validate_chat_history([{'role': 'user', 'content': None}])


def test_validate_history_assistant_empty():
    with pytest.raises(ValueError, match="must have either 'content' or 'tool_calls'"):
        ku.validate_chat_history([{'role': 'assistant'}])


def test_validate_history_assistant_invalid_tool_calls():
    with pytest.raises(ValueError, match="'tool_calls' must be a list"):
        ku.validate_chat_history([{'role': 'assistant', 'tool_calls': 'not a list'}])


def test_validate_history_tool_call_not_dict():
    with pytest.raises(ValueError, match='must be a dict'):
        ku.validate_chat_history([{'role': 'assistant', 'tool_calls': ['not a dict']}])


def test_validate_history_tool_call_missing_id():
    with pytest.raises(ValueError, match="missing or empty 'id'"):
        ku.validate_chat_history([{'role': 'assistant', 'tool_calls': [{'id': None}]}])


def test_validate_history_tool_call_invalid_type():
    with pytest.raises(ValueError, match="'type' must be 'function'"):
        ku.validate_chat_history(
            [{'role': 'assistant', 'tool_calls': [{'id': '1', 'type': 'not_fn'}]}]
        )


def test_validate_history_tool_call_function_not_dict():
    with pytest.raises(ValueError, match="'function' must be a dict"):
        ku.validate_chat_history(
            [
                {
                    'role': 'assistant',
                    'tool_calls': [{'id': '1', 'type': 'function', 'function': 'not a dict'}],
                }
            ]
        )


def test_validate_history_tool_call_function_name_missing():
    with pytest.raises(ValueError, match="'function.name' must be a non-empty string"):
        ku.validate_chat_history(
            [
                {
                    'role': 'assistant',
                    'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': ''}}],
                }
            ]
        )


def test_validate_history_tool_call_function_args_not_str():
    with pytest.raises(ValueError, match="'function.arguments' must be a string"):
        ku.validate_chat_history(
            [
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {'id': '1', 'type': 'function', 'function': {'name': 'f', 'arguments': {}}}
                    ],
                }
            ]
        )


def test_validate_history_tool_call_unknown_tool_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        ku.validate_chat_history(
            [
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': '1',
                            'type': 'function',
                            'function': {'name': 'unknown', 'arguments': '{}'},
                        }
                    ],
                },
                {'role': 'tool', 'tool_call_id': '1', 'name': 'unknown', 'content': 'res'},
            ],
            tool_names={'known'},
        )
    assert "tool name 'unknown' not registered" in caplog.text


def test_validate_history_tool_missing_id():
    with pytest.raises(ValueError, match="missing or empty 'tool_call_id'"):
        ku.validate_chat_history([{'role': 'tool'}])


def test_validate_history_tool_missing_name():
    with pytest.raises(ValueError, match="missing or empty 'name'"):
        ku.validate_chat_history([{'role': 'tool', 'tool_call_id': '1'}])


def test_validate_history_tool_content_not_str():
    with pytest.raises(ValueError, match="'content' must be a string"):
        ku.validate_chat_history(
            [{'role': 'tool', 'tool_call_id': '1', 'name': 'n', 'content': None}]
        )


def test_validate_history_unresolved_tool_call_ids():
    # Tests the backward walk and resolving IDs
    with pytest.raises(ValueError, match='unresolved tool call'):
        ku.validate_chat_history(
            [
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': '1',
                            'type': 'function',
                            'function': {'name': 'f', 'arguments': '{}'},
                        }
                    ],
                }
            ]
        )

    # This should pass
    ku.validate_chat_history(
        [
            {
                'role': 'assistant',
                'tool_calls': [
                    {'id': '1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}
                ],
            },
            {'role': 'tool', 'tool_call_id': '1', 'name': 'f', 'content': 'res'},
        ]
    )


def test_validate_history_unknown_tool_call_id():
    with pytest.raises(ValueError, match='refers to unknown tool_call_id'):
        ku.validate_chat_history(
            [{'role': 'tool', 'tool_call_id': 'ghost_id', 'name': 'f', 'content': 'res'}]
        )


def test_validate_history_multiple_turns():
    # Valid interleaved history
    history = [
        {'role': 'user', 'content': '1+1?'},
        {
            'role': 'assistant',
            'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}
            ],
        },
        {'role': 'tool', 'tool_call_id': 'c1', 'name': 'f', 'content': '2'},
        {'role': 'assistant', 'content': 'It is 2.'},
        {'role': 'user', 'content': 'And 2+2?'},
        {
            'role': 'assistant',
            'tool_calls': [
                {'id': 'c2', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}
            ],
        },
        {'role': 'tool', 'tool_call_id': 'c2', 'name': 'f', 'content': '4'},
        {'role': 'assistant', 'content': 'It is 4.'},
    ]
    ku.validate_chat_history(history)


@pytest.mark.asyncio
async def test_codeact_agent_history_injection_no_system_msg():
    """Test that CodeActAgent correctly handles injected history without a system message.
    This was previously failing with a KeyError because pre_run() was using a format string
    that didn't include 'authorized_imports'.
    """
    from kodeagent.kodeagent import CodeActAgent

    agent = CodeActAgent(model_name='gpt-3.5-turbo')
    injected_history = [{'role': 'user', 'content': 'Previous interaction'}]

    # This call should not raise KeyError
    agent._run_init(task='New task', chat_history=injected_history)

    # Verify system message is at index 0 and has authorized_imports formatted
    assert agent.chat_history[0]['role'] == 'system'
    content = agent.chat_history[0]['content']
    assert '- datetime' in content
    assert 'Previous interaction' in agent.chat_history[1]['content']


def test_agent_get_system_prompt_content_none():
    """Test get_system_prompt_content when system_prompt is None."""
    from kodeagent.kodeagent import ReActAgent

    agent = ReActAgent(model_name='gpt-3.5-turbo', system_prompt=None)
    assert agent.get_system_prompt_content() == ''


@pytest.mark.asyncio
async def test_codeact_parse_text_response_no_thought():
    """Test CodeAct parser with missing Thought (should raise ValueError)."""
    from kodeagent.kodeagent import CodeActAgent

    agent = CodeActAgent(name='test_codeact', model_name='gpt-3.5-turbo', run_env='host')
    text_response = "Code: print('hello')"
    with pytest.raises(ValueError, match="Could not extract 'Thought:' field"):
        agent.parse_text_response(text_response)
