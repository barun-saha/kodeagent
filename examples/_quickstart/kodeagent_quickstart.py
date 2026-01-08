"""Quick start example for KodeAgent with web search and webpage reading tools."""

from kodeagent import ReActAgent, print_response
from kodeagent.tools import read_webpage, search_web


async def main():
    """Quick start example for KodeAgent with web search and webpage reading tools."""
    litellm_params = {'temperature': 0, 'timeout': 30}
    agent = ReActAgent(
        name='Web agent',
        model_name='gemini/gemini-2.5-flash-lite',
        tools=[search_web, read_webpage],
        max_iterations=5,
        litellm_params=litellm_params,
    )

    for task in [
        'What are the festivals in Paris? How they differ from Kolkata?',
    ]:
        print(f'User: {task}')

        async for response in agent.run(task):
            print_response(response, only_final=True)

        print(agent.current_plan)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
