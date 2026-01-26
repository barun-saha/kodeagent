"""A script to evaluate a CodeActAgent on the GAIA benchmark dataset.
https://huggingface.co/spaces/gaia-benchmark/leaderboard
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time

import tqdm
from huggingface_hub import snapshot_download
from tabulate import tabulate

MODULE_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(MODULE_ROOT, '..'))
sys.path.append(os.path.join(MODULE_ROOT, '../..'))

import kodeagent as ka

REPO_ID = 'gaia-benchmark/GAIA'
LOCAL_DIR = f'{MODULE_ROOT}/gaia_dataset'

logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('langfuse').disabled = True


def get_agent(agent_type: str, model_name: str, max_steps: int = 10) -> ka.Agent:
    """Create a CodeActAgent for solving the GAIA benchmark tasks.

    Args:
        agent_type: Agent type, should be `codeact` or `react`.
        model_name: The LLM to use.
        max_steps: Maximum number of agent steps to run.

    Returns:
        A configured CodeAgent instance.
    """
    all_tools = [
        ka.search_web,
        ka.extract_as_markdown,
        ka.download_file,
        ka.transcribe_youtube,
        ka.transcribe_audio,
        ka.search_wikipedia,
        ka.search_arxiv,
    ]
    litellm_params = {'temperature': 0, 'timeout': 45}

    if agent_type == 'react':
        print('Creating a ReAct agent...')
        agent = ka.ReActAgent(
            name='Multi-task ReAct agent',
            model_name=model_name,
            tools=all_tools,
            max_iterations=max_steps,
            litellm_params=litellm_params,
        )
    else:
        print('Creating a CodeAct agent...')
        agent = ka.CodeActAgent(
            name='Multi-task CodeAct agent',
            model_name=model_name,
            tools=all_tools,
            run_env='host',
            max_iterations=max_steps,
            litellm_params=litellm_params,
            allowed_imports=[
                'os',
                're',
                'time',
                'random',
                'requests',
                'tempfile',
                'ddgs',
                'markitdown',
                'youtube_transcript_api',
                'wikipedia',
                'arxiv',
            ],
            pip_packages=(
                'ddgs~=9.5.2;"markitdown[all]";'
                'youtube-transcript-api~=1.2.2;wikipedia~=1.4.0;arxiv~=2.2.0'
            ),
            env_vars_to_set={'FIREWORKS_API_KEY': os.environ.get('FIREWORKS_API_KEY', '')},
            timeout=35,
            filter_tools_for_task=False,
        )

    return agent


def download_gaia_dataset():
    """Downloads the GAIA dataset from Hugging Face Hub if not already present locally."""
    if os.path.exists(LOCAL_DIR):
        print(f'Using local saved copy of GAIA dataset from {LOCAL_DIR}...')
    else:
        print(f'Downloading GAIA dataset to {LOCAL_DIR}...')
        try:
            # The GAIA dataset is gated. You must be logged in to huggingface-cli
            # and have accepted the terms on the dataset's page.
            snapshot_download(
                repo_id=REPO_ID,
                repo_type='dataset',
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
            )
            print('Download complete.')
        except Exception as e:
            print('Failed to download dataset. You might need to log in to Hugging Face Hub.')
            print(
                'Please run `huggingface-cli login` and accept the terms for the dataset'
                ' on its Hugging Face page.'
            )
            print(f'Error: {e}')
            sys.exit(1)


async def main(
    split: str,
    model_name: str,
    max_tasks: int,
    task_range: str,
    max_steps: int,
    output_file: str,
    agent_type: str = 'react',
):
    """Iterates through the GAIA dataset metadata, printing questions, answers, and associated files.

    Args:
        split (str): The dataset split to use, either 'test' or 'validation'.
        model_name (str): The LLM model to use.
        max_tasks (int): Maximum number of tasks to process from start. Use -1 for all tasks.
        task_range (str): Range of tasks to process in format "start:end" (1-based).
         Takes precedence over max_tasks.
        max_steps (int): Maximum number of agent steps to run.
        output_file (str): The output file name to store the results.
        agent_type (str): Type of agent to use ('react' or 'codeact').
    """
    if split not in {'test', 'validation'}:
        raise ValueError('Split must be either `test` or `validation`')

    # The dataset files for the 2023 challenge are in a '2023' subdirectory.
    metadata_file = os.path.join(LOCAL_DIR, '2023', split, 'metadata.jsonl')

    if not os.path.exists(metadata_file):
        print(f'Error: Metadata file not found at {metadata_file}')
        return

    with open(metadata_file, encoding='utf-8') as _:
        gaia_data = [json.loads(line) for line in _]

    agent = get_agent(agent_type=agent_type, model_name=model_name, max_steps=max_steps)
    evals = []

    # Handle task selection based on range or max_tasks
    if task_range:
        try:
            start, end = map(int, task_range.split(':'))
            if start < 1 or end > len(gaia_data) or start > end:
                raise ValueError(
                    f'Invalid start or end values in task range. Max is {len(gaia_data)}'
                )
            # Convert to 0-based indexing but make end inclusive
            start_idx = start - 1
            end_idx = end  # This will be inclusive due to Python list slicing
            gaia_data = gaia_data[start_idx:end_idx]
            n_questions = len(gaia_data)
            enum_start = start  # Store the starting number for enumeration
            print(f'{start_idx=}, {end_idx=}, {n_questions=}, {enum_start=}')
        except TypeError:
            print(
                f'Invalid task range: {task_range}. Format should be "start:end" (1-based indices)'
            )
            return
        except ValueError as ve:
            print(str(ve))
            return
    else:
        if max_tasks == -1:
            n_questions = len(gaia_data)
        else:
            n_questions = min(max_tasks, len(gaia_data))
            gaia_data = gaia_data[:n_questions]
        enum_start = 1  # Start from 1 when using ntasks

    n_correct = 0
    # GAIA appears to evaluate results using exact match, so even a correct but differently
    # formatted answer will be marked as incorrect
    special_instructions = (
        'The final response should only with the exact answer without any extra text.'
        ' For CSV output, a comma should be followed by a space.'
        ' Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated'
        " list of numbers and/or strings. If you are asked for a number, don't use comma to write"
        ' your number neither use units such as $ or percent sign unless specified otherwise.'
        " If you are asked for a string, don't use articles, neither abbreviations"
        ' (e.g. for cities), and write the digits in plain text unless specified otherwise.'
        ' If you are asked for a comma separated list, apply the above rules depending of whether'
        ' the element to be put in the list is a number or a string.'
    )

    print(f'\n--- Processing `{split}` split with {n_questions} tasks ---')

    sanitized_model_name = model_name.replace('/', '_').replace('.', '_')
    jsonl_output_file = (
        f'gaia_{split}_tasks-{enum_start}-{enum_start + n_questions - 1}_'
        f'steps-{max_steps}_model-{sanitized_model_name}.jsonl'
    )
    if os.path.exists(jsonl_output_file):
        os.remove(jsonl_output_file)

    for idx, item in tqdm.tqdm(
        enumerate(gaia_data, enum_start),
        total=n_questions,
        desc=f'Processing tasks {enum_start} to {enum_start + n_questions - 1}',
    ):
        task_id = item.get('task_id', 'N/A')
        question = item['Question']
        true_answer = item.get('Final answer', 'N/A')
        file_name = item.get('file_name', None)
        file_path = os.path.join(os.path.dirname(metadata_file), file_name)
        print(f'\n#{idx}\nQuestion: {question}\nFile: {file_path if file_name else "N/A"}')
        question += f'\n\n{special_instructions}'

        try:
            final_answer_provided = False
            async for response in agent.run(
                task=question,
                files=[file_path] if file_name else None,
            ):
                if response['type'] == 'final':
                    final_answer_provided = True
                    answer = (
                        response['value'].content
                        if isinstance(response['value'], ka.ChatMessage)
                        else response['value']
                    )
                    print(f'Agent: {answer}\n')
                    is_correct = str(true_answer).strip().lower() == str(answer).strip().lower()
                    if is_correct:
                        n_correct += 1

                    # Somehow the last update to the plan is not captured, so adding a delay
                    await asyncio.sleep(random.uniform(1.25, 2))
                    evals.append(
                        (
                            task_id,
                            question.replace('\n', '<br>'),
                            true_answer.replace('\n', '<br>'),
                            answer.replace('\n', '<br>'),
                            is_correct,
                            agent.current_plan.replace('\n', '<br>'),
                        )
                    )
                    with open(jsonl_output_file, 'a', encoding='utf-8') as f:
                        f.write(
                            json.dumps(
                                {
                                    'task_id': task_id,
                                    'model_answer': answer,
                                    'reasoning_trace': agent.current_plan,
                                }
                            )
                            + '\n'
                        )

            if not final_answer_provided:
                answer = 'Agent failed to provide an answer.'
                plan = agent.current_plan or 'N/A'
                evals.append(
                    (
                        task_id,
                        question.replace('\n', '<br>'),
                        true_answer.replace('\n', '<br>'),
                        answer,
                        False,
                        plan.replace('\n', '<br>'),
                    )
                )
                with open(jsonl_output_file, 'a', encoding='utf-8') as f:
                    f.write(
                        json.dumps(
                            {'task_id': task_id, 'model_answer': answer, 'reasoning_trace': plan}
                        )
                        + '\n'
                    )

            print(f'True Answer: {true_answer}')
            print(f'Plan:\n{agent.current_plan}\n\n')

            await asyncio.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f'Error processing task {task_id}: {e}')
            evals.append((task_id, question, true_answer, 'Execution error', False, 'N/A'))
            with open(jsonl_output_file, 'a', encoding='utf-8') as f:
                f.write(
                    json.dumps(
                        {
                            'task_id': task_id,
                            'model_answer': 'Execution error',
                            'reasoning_trace': 'N/A',
                        }
                    )
                    + '\n'
                )

    table = tabulate(
        evals,
        headers=['Task ID', 'Question', 'True Answer', 'Agent Answer', 'Correct', 'Plan'],
        tablefmt='github',
    )
    evals_md = (
        f'**Date:** {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n\n'
        f'**Model:** {model_name}\n\n'
        f'**Agent steps:** {max_steps}\n\n'
        f'**Dataset split:** {split}\n\n'
        f'**Accuracy:** {n_correct / n_questions * 100 if n_questions > 0 else 0:.2f}%'
        f' ({n_correct} correct out of {n_questions})\n\n{table} '
    )
    print(evals_md)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as _:
        _.write(evals_md)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GAIA dataset.')
    parser.add_argument(
        '--split',
        type=str,
        choices=['test', 'validation'],
        help='The dataset split to process ("test" or "validation").',
        default='validation',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='The LLM model to use.',
        default='gemini/gemini-2.5-flash-lite',
    )

    # Create mutually exclusive group for task selection
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        '--ntasks',
        type=int,
        help='The max no. of tasks to run from start. Use -1 for all tasks.',
        default=3,
    )
    task_group.add_argument(
        '--task-range',
        type=str,
        help='Range of tasks to process "start:end" (1-based, both inclusive). Example: "2:23"',
        default='',
    )

    parser.add_argument(
        '--agent',
        type=str,
        choices=['react', 'codeact'],
        help='The type of agent to use (default: `react`).',
        default='react',
    )

    parser.add_argument('--nsteps', type=int, help='The max no. of agent steps to run.', default=10)
    parser.add_argument(
        '--output_file',
        type=str,
        help='The output file name to store the results.',
        default='gaia_results.md',
    )
    args = parser.parse_args()
    download_gaia_dataset()

    asyncio.run(
        main(
            args.split,
            args.model,
            max_tasks=args.ntasks,
            task_range=args.task_range,
            max_steps=args.nsteps,
            output_file=args.output_file,
            agent_type=args.agent,
        )
    )
