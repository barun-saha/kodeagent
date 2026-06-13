"""Example of a local, offline agent using the FunctionCallingAgent.
This example uses LFM-2.5-8B-A1B (via Ollama), which can be run even on CPU (though GPU
is recommended for best performance)."""

import logging
import time
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from kodeagent.fca import FunctionCallingAgent
from kodeagent import tools as dtools


async def main():
    """Example usage of the FunctionCallingAgent."""
    agent = FunctionCallingAgent(
        model_name='ollama_chat/lfm2.5:8b',
        tools=[
            dtools.search_web,
            dtools.read_webpage,
            dtools.transcribe_youtube,
        ],
        litellm_params={'temperature': 0, 'timeout': 120}, # More wait time
    )

    # Tuple: (task desc, optional list of files/URLs)
    tasks = [
        (
            'Find the current stock price of NVIDIA & calculate how many shares'
            ' I can buy with $1000.',
            None
        ),
    ]

    for idx, (task, files) in enumerate(tasks, start=1):
        print(f'\nTask #{idx}: {task}')
        task_start_time = time.perf_counter()
        async for response in agent.run(
                task,
                files,
                max_iterations=10,
                use_planning=False
        ):
            print(response)

        task_duration = time.perf_counter() - task_start_time
        print(
            f'\n>>> Task #{idx} completed in {task_duration / 60:.2f} minutes'
            f' ({agent.task.steps_taken} steps).'
        )

        print('Answer:', agent.task.result)
        print('-' * 80)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
