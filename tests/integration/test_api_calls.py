"""
Integration tests for the KodeAgent package.
These tests make real API calls and should not be run in unit test suites.
"""
import datetime
import os
import pytest

from kodeagent import (
    ReActAgent,
    CodeActAgent,
    calculator,
    search_arxiv
)
from kodeagent.kutils import call_llm


MODEL_NAME = 'gemini/gemini-2.0-flash-lite'


@pytest.mark.asyncio
async def test_call_llm():
    """Test the public `call_llm` function with real API call."""
    response = await call_llm(
        model_name=MODEL_NAME,
        litellm_params={},
        messages=[{'role': 'user', 'content': 'Hello!'}],
        trace_id='integration-test'
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_search_arxiv():
    """Test the arxiv search tool for research papers."""
    query = 'attention is all you need vaswani'
    results = search_arxiv(query=query, max_results=3)
    assert query in results.lower()
    assert '## ArXiv Search Results for:' in results


@pytest.mark.asyncio
async def test_get_relevant_tools_integration():
    """Integration test for getting relevant tools using real LLM call."""
    agent = ReActAgent(
        name='test_react_agent',
        model_name=MODEL_NAME,
        tools=[calculator],
        description='Test ReAct agent for arithmetic tasks',
        max_iterations=3
    )

    task_description = 'What is 2 plus 3?'
    agent._run_init(task_description)
    tools = await agent.get_relevant_tools(task_description)

    assert len(tools) > 0
    tool_names = {t.name for t in tools}
    assert len(tool_names) > 0


@pytest.mark.asyncio
async def test_codeact_agent_e2b():
    """Test the CodeActAgent functionality on a remote E2B sandbox."""
    if not os.getenv('E2B_API_KEY'):
        pytest.skip('E2B_API_KEY environment variable is not set.')

    code_agent = CodeActAgent(
        name='Code agent',
        model_name=MODEL_NAME,
        run_env='e2b',
        max_iterations=3,
        allowed_imports=['datetime'],
        description='Agent that can write and execute Python code',
        pip_packages=None,
    )

    task = "What is today's date in words (without time)? Use the `datetime` module."
    responses = []
    async for response in code_agent.run(task):
        responses.append(response['value'])

    # Get today's date for verification
    today = datetime.datetime.now().strftime('%B %d, %Y')
    response = '\n'.join([str(r) for r in responses])
    assert today.lower() in response.lower()


@pytest.mark.asyncio
async def test_react_agent_llm_integration():
    """Integration test for ReActAgent with real LLM calls."""
    agent = ReActAgent(
        name='test_react_agent',
        model_name=MODEL_NAME,
        tools=[],
        description='Test ReAct agent for integration tests',
        max_iterations=3
    )

    responses = []
    async for response in agent.run('What is 2 plus 2?'):
        responses.append(response)

    # Check that we got responses
    assert any(r['type'] == 'final' for r in responses)
    # Verify we got a numerical answer
    final_response = next(r for r in responses if r['type'] == 'final')
    assert '3' in str(final_response['value'])
