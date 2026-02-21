"""Example runners for kodeagent.

This module contains helper functions to run the bundled example agents from
user code. The API exposes a synchronous `run_examples` function plus an
`async def main()` kept for script usage.
"""

import asyncio
import os
import random
from typing import Any

import rich

from . import tools as dtools
from .kodeagent import CodeActAgent, ReActAgent, print_response


async def _run_examples_async(
    agent_type: str, max_steps: int, model_name: str = 'gemini/gemini-2.0-flash-lite'
) -> None:
    """Run the example agent demos asynchronously.

    Args:
        agent_type: Which agent to run; one of 'react', or 'codeact'.
        max_steps: Maximum iterations/steps for the agent.
        model_name: Which model to use for the agent.
    """
    litellm_params: dict[str, Any] = {'temperature': 0, 'timeout': 30}

    def _make_react_agent() -> ReActAgent:
        return ReActAgent(
            name='Simple agent',
            model_name=model_name,
            tools=[
                dtools.calculator,
                dtools.search_web,
                dtools.read_webpage,
                dtools.extract_as_markdown,
            ],
            max_iterations=max_steps,
            litellm_params=litellm_params,
        )

    def _make_code_agent() -> CodeActAgent:
        return CodeActAgent(
            name='Simple agent',
            model_name=model_name,
            tools=[
                dtools.calculator,
                dtools.search_web,
                dtools.read_webpage,
                dtools.extract_as_markdown,
            ],
            max_iterations=max_steps,
            litellm_params=litellm_params,
            run_env='host',
            allowed_imports=[
                'math',
                'datetime',
                'time',
                're',
                'typing',
                'mimetypes',
                'random',
                'ddgs',
                'bs4',
                'urllib.parse',
                'requests',
                'markitdown',
                'pathlib',
            ],
            pip_packages='ddgs~=9.10.0;beautifulsoup4~=4.14.3;',
            work_dir='./agent_workspace',
        )

    the_tasks = [
        ('What is ten plus 15, raised to 2, expressed in words?', None),
        ('What is the date today? Express it in words like <Month> <Day>, <Year>.', None),
        (
            'Which image has a purple background?',
            [
                (
                    'https://www.slideteam.net/media/catalog/product/cache/1280x720'
                    '/p/r/process_of_natural_language_processing_training_ppt_slide01.jpg'
                ),
                (
                    'https://cdn.prod.website-files.com/61a05ff14c09ecacc06eec05'
                    '/66e8522cbe3d357b8434826a_ai-agents.jpg'
                ),
            ],
        ),
        (
            'What is four plus seven? Also, what are the festivals in Paris?'
            ' How they differ from Kolkata?',
            None,
        ),
        ('Write an elegant haiku in Basho style. Save it as poem.txt', None),
    ]

    async def _run_with_agent(agent: Any) -> None:
        """Run the example tasks with the given agent and print results.

        Args:
            agent: An agent instance with an async .run() method.
        """
        print(f'{agent.__class__.__name__} demo\n')

        for task, img_urls in the_tasks:
            rich.print(f'[yellow][bold]User[/bold]: {task}[/yellow]')
            async for response in agent.run(task, files=img_urls):
                print_response(response, only_final=True)

            if getattr(agent, 'artifacts', None):
                print('Artifacts generated:')
                for art in agent.artifacts:
                    print(f'- {art} (size: {os.path.getsize(art)} bytes)')

            if getattr(agent, 'current_plan', None):
                print(f'Plan:\n{agent.current_plan}')

            await asyncio.sleep(random.uniform(0.15, 0.55))
            print('\n\n')

        print('Demonstrating recurrent mode:\n')
        async for response in agent.run('Find the population of France in 2023'):
            print_response(response, only_final=True)

        async for response in agent.run(
            'What would it be with a 0.5% growth?',
            recurrent_mode=True,
        ):
            print_response(response, only_final=True)

    atype = agent_type.lower()

    if atype == 'codeact':
        agent = _make_code_agent()
        await _run_with_agent(agent)
    elif atype == 'react':
        agent = _make_react_agent()
        await _run_with_agent(agent)
    else:
        raise ValueError(f'Unknown agent_type: {agent_type}')


def run_examples(
    agent_type: str = 'react', max_steps: int = 5, model_name: str = 'gemini/gemini-2.0-flash-lite'
) -> None:
    """Run the bundled examples from a synchronous importable API.

    Args:
        agent_type: Which agent to run; one of 'react', or 'codeact'.
        max_steps: Maximum iterations/steps for the agent.
        model_name: Which model to use for the agent.
    """
    asyncio.run(_run_examples_async(agent_type, max_steps, model_name))


if __name__ == '__main__':
    os.environ['PYTHONUTF8'] = '1'
    run_examples()
