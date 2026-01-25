"""Tests for verifying persona propagation in ReAct and CodeAct agents."""

import pytest
from kodeagent.kodeagent import (
    ReActAgent,
    CodeActAgent,
    REACT_SYSTEM_PROMPT,
    CODE_ACT_SYSTEM_PROMPT,
)


def test_react_agent_persona_provided():
    """Verify ReActAgent uses persona in system prompt when provided."""
    persona = "You're a technical writer."
    agent = ReActAgent(name='TestAgent', model_name='gpt-4', tools=[], persona=persona)
    agent.init_history()

    expected_prompt = REACT_SYSTEM_PROMPT.format(
        persona=persona, tools=agent.get_tools_description()
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_react_agent_persona_none():
    """Verify ReActAgent uses empty string for persona when not provided."""
    agent = ReActAgent(name='TestAgent', model_name='gpt-4', tools=[])
    agent.init_history()

    expected_prompt = REACT_SYSTEM_PROMPT.format(persona='', tools=agent.get_tools_description())
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_codeact_agent_persona_provided():
    """Verify CodeActAgent uses persona in system prompt when provided."""
    persona = "You're a data scientist."
    agent = CodeActAgent(
        name='TestAgent', model_name='gpt-4', run_env='host', tools=[], persona=persona
    )
    agent.init_history()

    expected_prompt = CODE_ACT_SYSTEM_PROMPT.format(
        persona=persona,
        tools=agent.get_tools_description(),
        authorized_imports='\n'.join([f'- {imp}' for imp in agent.allowed_imports]),
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt


def test_codeact_agent_persona_none():
    """Verify CodeActAgent uses empty string for persona when not provided."""
    agent = CodeActAgent(name='TestAgent', model_name='gpt-4', run_env='host', tools=[])
    agent.init_history()

    expected_prompt = CODE_ACT_SYSTEM_PROMPT.format(
        persona='',
        tools=agent.get_tools_description(),
        authorized_imports='\n'.join([f'- {imp}' for imp in agent.allowed_imports]),
    )
    assert agent.messages[0].role == 'system'
    assert agent.messages[0].content == expected_prompt
