"""
Use KodeAgent to solve the hands-on exercises from Hugging Face's Agents course:
https://huggingface.co/learn/agents-course/en/unit4/hands-on

As of May 25, 2025, the evaluation scores are:
gemini/gemini-2.0-flash-lite: 45%
gemini/gemini-2.5-flash-preview-04-17: 60%
azure/gpt-4.1-mini: 35%

The scores may vary for different runs.
"""
import asyncio
import os
import random
import sys
import time

import requests
import rich
import tqdm

sys.path.append('.')
sys.path.append('..')

import kodeagent as ka


DEFAULT_API_URL = 'https://agents-course-unit4-scoring.hf.space'
IMG_EXTENSIONS = ['png', 'jpg', 'jpeg']
TXT_EXTENSIONS = ['txt', 'py', 'js', 'csv']
OTHER_DOCS = ['xlsx', 'docx', 'pdf']
ALLOWED_EXTENSIONS = IMG_EXTENSIONS + TXT_EXTENSIONS + OTHER_DOCS

os.environ['PYTHONUTF8'] = '1'


def get_questions_list() -> list[dict]:
    """
    Get the questions via HTTP request to https://agents-course-unit4-scoring.hf.space/questions

    Returns:
        A list of available questions.
    """
    url = f'{DEFAULT_API_URL}/questions'
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()  # Raise an error for HTTP request issues
    return response.json()


async def main():
    """
    Evaluate a subset of the GAIA benchmark for HF Agents Course finale.
    """
    questions = get_questions_list()
    # model_name = 'gemini/gemini-2.0-flash-lite'  # 45% score
    # model_name = 'gemini/gemini-2.5-flash-preview-04-17'  # 60% score
    model_name = 'azure/gpt-4.1-mini'  # 35% score
    litellm_params = {'temperature': 0}
    agent = ka.CodeAgent(
        name='Multi-task agent',
        model_name=model_name,
        tools=[
            ka.web_search, ka.extract_as_markdown, ka.file_download, ka.get_youtube_transcript,
            ka.search_wikipedia, ka.get_audio_transcript,
        ],
        run_env='host',
        max_iterations=10,
        litellm_params=litellm_params,
        allowed_imports=[
            'os', 're', 'time', 'random', 'requests', 'tempfile',
            'duckduckgo_search', 'markdownify', 'markitdown', 'youtube_transcript_api',
            'wikipedia',
        ],
        pip_packages=(
            'duckduckgo_search~=8.0.1;markitdown[pdf, docx, xlsx];'
            'youtube-transcript-api~=1.0.3;wikipedia~=1.4.0'
        ),
        timeout=35,
    )

    answers_payload = []
    agent_code = 'https://github.com/barun-saha/kodeagent'
    username = 'barunsaha'
    submit_url = f'{DEFAULT_API_URL}/submit'

    # GAIA appears to evaluate results using exact match, so even a correct but differently
    # formatted answer will be marked as incorrect
    special_instructions = (
        'The final response should only with the exact answer without any extra text.'
        ' For CSV output, a comma should be followed by a space.'
    )

    # Solve the tasks
    for a_question in tqdm.tqdm(questions):
        task_id = a_question.get('task_id')
        question = a_question.get('question')
        file_name = a_question.get('file_name')
        files = None

        # if task_id != '99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3':
        #     continue

        print(f'Task# {task_id}: {question}\nFile: {file_name}')
        if not task_id or question is None:
            print(f'Skipping item with missing task_id or question: {a_question}')
            continue

        if file_name:
            file_url = f'{DEFAULT_API_URL}/files/{task_id}'
            print(f'{file_url=}')
            files = [file_url]

        answer = ''

        try:
            print(f'> Solving task {question} with files {files}...')
            question += f'\n\n{special_instructions}'

            async for response in agent.run(task=question, files=files):
                if response['type'] == 'final':
                    answer = (
                        response['value'].content
                        if isinstance(response['value'], ka.ChatMessage) else response['value']
                    )
                    rich.print(f'[blue][bold]Agent[/bold]: {answer}[/blue]\n')
                elif response['type'] == 'log':
                    rich.print(f'[white]{response}[/white]')
                else:
                    rich.print(f'{response}')
            answers_payload.append({'task_id': task_id, 'submitted_answer': answer})

            time.sleep(random.uniform(2, 5))
        except Exception as ex:
            print(f'!!! An error occurred while solving the task: {ex}')
            continue

    submission_data = {
        'username': username.strip(),
        'agent_code': agent_code,
        'answers': answers_payload
    }
    print(f'Agent finished. Submitting {len(answers_payload)} answers for user \'{username}\'...')

    # Submit
    print(f'Submitting {len(answers_payload)} answers to: {submit_url}')
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f'Submission Successful!\n'
            f'User: {result_data.get("username")}\n'
            f'Overall Score: {result_data.get("score", "N/A")}% '
            f'({result_data.get("correct_count", "?")}/{result_data.get("total_attempted", "?")}'
            f' correct)\n'
            f'Message: {result_data.get("message", "No message received.")}'
        )
        rich.print(final_status)
    except requests.exceptions.HTTPError as e:
        error_detail = f'Server responded with status {e.response.status_code}.'
        try:
            error_json = e.response.json()
            error_detail += f' Detail: {error_json.get("detail", e.response.text)}'
        except requests.exceptions.JSONDecodeError:
            error_detail += f' Response: {e.response.text[:500]}'
        status_message = f'Submission Failed: {error_detail}'
        print(status_message)
    except requests.exceptions.Timeout:
        status_message = 'Submission Failed: The request timed out.'
        print(status_message)
    except requests.exceptions.RequestException as e:
        status_message = f'Submission Failed: Network error - {e}'
        print(status_message)
    except Exception as e:
        status_message = f'An unexpected error occurred during submission: {e}'
        print(status_message)


if __name__ == '__main__':
    asyncio.run(main())
