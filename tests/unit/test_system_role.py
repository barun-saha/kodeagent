"""Tests for verifying system role propagation in ReAct and CodeAct agents."""

import pytest
from unittest.mock import MagicMock

from kodeagent.kodeagent import (
    ReActAgent,
    CodeActAgent,
    REACT_SYSTEM_PROMPT,
    CODE_ACT_SYSTEM_PROMPT,
)


def test_react_agent_system_role_provided():
    """Verify ReActAgent uses system_role in system prompt when provided."""
    system_role = "You're a technical writer."
    agent = ReActAgent(name='TestAgent', model_name='gpt-4', tools=[], system_role=system_role)
    agent.init_history()

    expected_prompt = REACT_SYSTEM_PROMPT.format(
        system_role=system_role, tools=agent.get_tools_description()
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_react_agent_system_role_none():
    """Verify ReActAgent uses empty string for system_role when not provided."""
    agent = ReActAgent(name='TestAgent', model_name='gpt-4', tools=[])
    agent.init_history()

    expected_prompt = REACT_SYSTEM_PROMPT.format(
        system_role='', tools=agent.get_tools_description()
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_codeact_agent_system_role_provided():
    """Verify CodeActAgent uses system_role in system prompt when provided."""
    system_role = "You're a data scientist."
    agent = CodeActAgent(
        name='TestAgent', model_name='gpt-4', run_env='host', tools=[], system_role=system_role
    )
    agent.init_history()

    expected_prompt = CODE_ACT_SYSTEM_PROMPT.format(
        system_role=system_role,
        tools=agent.get_tools_description(),
        authorized_imports='\n'.join([f'- {imp}' for imp in agent.allowed_imports]),
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_codeact_agent_system_role_none():
    """Verify CodeActAgent uses empty string for system_role when not provided."""
    agent = CodeActAgent(name='TestAgent', model_name='gpt-4', run_env='host', tools=[])
    agent.init_history()

    expected_prompt = CODE_ACT_SYSTEM_PROMPT.format(
        system_role='',
        tools=agent.get_tools_description(),
        authorized_imports='\n'.join([f'- {imp}' for imp in agent.allowed_imports]),
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt
