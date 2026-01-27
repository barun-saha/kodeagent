"""A minimal set of utilities used by KodeAgent.
This module will be copied along with code for CodeAgent, so keep it minimum.
"""

import base64
import logging
import mimetypes
import os
import re
from typing import Any

import litellm
import pydantic as pyd
import requests
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .usage_tracker import UsageMetrics

DEFAULT_MAX_LLM_RETRIES = 3
LOGGERS_TO_SUPPRESS = [
    'asyncio',
    'cookie_store',
    'e2b',
    'e2b_code_interpreter',
    'hpack',
    'httpx',
    'httpcore',
    'langfuse',
    'LiteLLM',
    'litellm',
    'openai',
    'pdfminer',
    'primp',
    'rquest',
    'urllib3',
    'urllib3.connectionpool',
]

for _lg in LOGGERS_TO_SUPPRESS:
    logger_obj = logging.getLogger(_lg)
    logger_obj.setLevel(logging.ERROR)
    # Prevent these logs from propagating to the root logger
    logger_obj.propagate = False

# Capture warnings from the warnings module (optional, helps centralize output)
if hasattr(logging, 'captureWarnings'):
    logging.captureWarnings(True)


def get_logger(name: str | None = 'KodeAgent') -> logging.Logger:
    """Get a logger for KodeAgent.

    Returns:
        A logger instance.
    """
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
    logging.getLogger('langfuse').disabled = True

    return logging.getLogger(name)


# Get a logger for the current module
logger = logging.getLogger('KodeAgent')


def read_prompt(filename: str) -> str:
    """Reads a prompt from the `prompts` directory.

    Args:
        filename: Name of the prompt file to read.

    Returns:
        The content of the prompt file as a string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        RuntimeError: If there is an error reading the file.
    """
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', filename)

    try:
        with open(prompt_path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(
            f'Prompt file `{filename}` not found in the prompts directory: {prompt_path}'
        ) from fnfe
    except Exception as e:
        raise RuntimeError(f'Error reading prompt file `{filename}`: {e}') from e


def is_it_url(path: str) -> bool:
    """Check whether a given path is a URL.

    Args:
        path: The path.

    Returns:
        `True` if it's a URL; `False` otherwise.
    """
    return path.startswith(('http://', 'https://'))


def detect_file_type(url: str) -> str:
    """Identify the content/MIME type of file pointed by a URL.

    Args:
        url: The URL to the file.

    Returns:
        The detected MIME type or `Unknown file type`.
    """
    try:
        # Step 1: Try HEAD request to get Content-Disposition
        response = requests.head(url, allow_redirects=True, timeout=15)
        content_disposition = response.headers.get('Content-Disposition')

        if content_disposition and 'filename=' in content_disposition:
            file_name = content_disposition.split('filename=')[1].strip()
            file_extension = file_name.split('.')[-1]
            return file_extension  # If this works, return immediately

        # Step 2: If HEAD didn't give useful info, send GET request for more details
        response = requests.get(url, stream=True, timeout=20)
        content_type = response.headers.get('Content-Type')

        if content_type and content_type != 'application/json':  # Avoid false positives
            return content_type

        return 'Unknown file type'
    except requests.RequestException as e:
        logger.error('Error detecting file type: %s', str(e))
        return 'Unknown file type'


def is_image_file(file_type) -> bool:
    """Identify whether a given MIME type is an image.

    Args:
        file_type: The file/content type.

    Returns:
        `True` if an image file; `False` otherwise.
    """
    return file_type.startswith('image/')


async def call_llm(
    model_name: str,
    litellm_params: dict,
    messages: list[dict],
    response_format: type[pyd.BaseModel] | None = None,
    trace_id: str | None = None,
    max_retries: int = DEFAULT_MAX_LLM_RETRIES,
    usage_tracker: Any | None = None,
    component_name: str = 'unknown',
) -> str | None:
    """Call the LLM with the given parameters and response format.

    Args:
        model_name: The name of the LLM model to use.
        litellm_params: Dictionary of parameters to pass to litellm.
        messages: List of message dictionaries.
        response_format: Optional pydantic model for structured output.
        trace_id: Optional trace ID for observability.
        max_retries: Maximum number of retries for the LLM call.
        usage_tracker: Optional UsageTracker instance to record usage.
        component_name: Name of the component making the call (for tracking).

    Returns:
        The LLM response as string.

    Raises:
        RetryError: If the LLM call fails after maximum retries.
        ValueError: If the LLM returns an empty or invalid response body.
    """
    params = {'model': model_name, 'messages': messages}

    if response_format:
        params['response_format'] = response_format

    # Add a timeout to prevent indefinite hangs
    if 'timeout' not in litellm_params:
        params['timeout'] = 30  # seconds

    params.update(litellm_params)

    try:
        # Use AsyncRetrying to handle retries in a non-blocking way
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_random_exponential(multiplier=1, max=60),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ):
            with attempt:
                # Use the asynchronous litellm call
                response = await litellm.acompletion(
                    **params,
                    metadata={
                        'trace_id': str(trace_id) if trace_id else None,
                        'trace_name': 'kodeagent',
                        'generation_name': 'kodeagent-generation',
                        'generation_metadata': {
                            'response_format': (
                                response_format.__name__ if response_format else 'None'
                            ),
                            'attempt': attempt.retry_state.attempt_number,
                        },
                        'tags': [model_name],
                    },
                )

                # Check for empty choices list
                if not response.choices or len(response.choices) == 0:
                    raise ValueError('LLM returned an empty choices list.')

                # Check for empty content
                response_content = response.choices[0].message.content
                if not response_content or not response_content.strip():
                    raise ValueError('LLM returned an empty or invalid response body.')

                token_usage = {
                    'cost': response._hidden_params.get('response_cost', 0.0),
                    'prompt_tokens': response.usage.get('prompt_tokens', 0),
                    'completion_tokens': response.usage.get('completion_tokens', 0),
                    'total_tokens': response.usage.get('total_tokens', 0),
                }
                logger.info(token_usage)

                # Record usage if tracker provided
                if usage_tracker:
                    try:
                        metrics = UsageMetrics(
                            prompt_tokens=token_usage['prompt_tokens'],
                            completion_tokens=token_usage['completion_tokens'],
                            total_tokens=token_usage['total_tokens'],
                            cost=token_usage['cost'] or 0.0,
                        )
                        await usage_tracker.record_usage(component_name, metrics)
                    except Exception as e:
                        logger.warning('Failed to record usage: %s', str(e))

                return response_content

    except RetryError:
        raise
    except Exception as e:
        logger.exception('LLM call failed after repeated attempts: %s', str(e), exc_info=True)
        # print('\n\ncall_llm MESSAGES:\n', '\n'.join([str(msg) for msg in messages]), '\n\n')
        raise ValueError('Failed to get a valid response from LLM after multiple retries.') from e


def make_user_message(text_content: str, files: list[str] | None = None) -> list[dict[str, Any]]:
    """Create a single user message to be sent to LiteLLM.

    Args:
        text_content: The text content of the message.
        files: An optional list of file paths or URLs, which can include images
               or other file types.

    Returns:
        A list of dict items representing the messages.
    """
    content: list[dict[str, Any]] = [{'type': 'text', 'text': str(text_content)}]
    message: list[dict[str, Any]] = [{'role': 'user'}]

    if files:
        for item in files:
            is_image = False
            if is_it_url(item):
                if any(
                    ext in item.lower()
                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
                ) or is_image_file(detect_file_type(item)):
                    is_image = True
            elif os.path.isfile(item):
                try:
                    mime_type, _ = mimetypes.guess_type(item)
                    if mime_type and 'image' in mime_type:
                        is_image = True
                except Exception:
                    logger.error(
                        'Error guessing MIME type for local file %s...will ignore it',
                        item,
                        exc_info=True,
                    )
                    # If an error occurs, treat it as not an image to continue processing
                    is_image = False

            if is_image:
                if is_it_url(item):
                    content.append({'type': 'image_url', 'image_url': {'url': item}})
                elif os.path.isfile(item):
                    try:
                        with open(item, 'rb') as img_file:
                            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

                        try:
                            mime_type, _ = mimetypes.guess_type(item)
                        except Exception:
                            logger.warning(
                                'Could not guess MIME type, defaulting to octet-stream',
                                exc_info=True,
                            )
                            mime_type = 'application/octet-stream'

                        mime_type = mime_type if mime_type else 'application/octet-stream'
                        content.append(
                            {
                                'type': 'image_url',
                                'image_url': {'url': f'data:{mime_type};base64,{encoded_image}'},
                            }
                        )
                    except FileNotFoundError:
                        logger.error('Image file not found: %s...will ignore it', item)
                    except Exception as e:
                        logger.error(
                            'Error processing local image %s: %s...will ignore it', item, e
                        )
                else:
                    logger.error('Invalid image file path or URL: %s...will ignore it', item)
            else:  # Handle as a general file or URL (not an image)
                if is_it_url(item):
                    content.append({'type': 'text', 'text': f'File URL: {item}'})
                elif os.path.isfile(item):
                    try:
                        mime_type, _ = mimetypes.guess_type(item)
                        if mime_type and (
                            'text' in mime_type
                            or mime_type in ('application/json', 'application/xml')
                        ):
                            try:
                                with open(item, encoding='utf-8') as f:
                                    file_content = f.read()
                                content.append(
                                    {
                                        'type': 'text',
                                        'text': f'File {item} content:\n{file_content}',
                                    }
                                )
                            except Exception:
                                logger.error(
                                    'Error reading text file `%s`...will fallback to path only',
                                    item,
                                )
                                content.append({'type': 'text', 'text': f'Input file: {item}'})
                        else:
                            # Non-text or unknown types: include only the path reference
                            content.append({'type': 'text', 'text': f'Input file: {item}'})
                    except Exception:
                        logger.error(
                            'Error guessing MIME type for local file %s...will ignore it',
                            item,
                            exc_info=True,
                        )
                        # content.append({'type': 'text', 'text': f'Input file: {item}'})
                else:
                    logger.error('Invalid file path or URL: %s...will ignore it', item)

    message[0]['content'] = content
    return message


def combine_user_messages(messages: list) -> list:
    """Combines consecutive user messages into a single message with a list of content items.

    Returns:
        A new list of messages with combined user messages.
    """
    combined = []
    for msg in messages:
        if msg.get('role') == 'user':
            if combined and combined[-1].get('role') == 'user':
                # Merge content lists
                prev_content = combined[-1]['content']
                curr_content = msg.get('content', [])
                if not isinstance(prev_content, list):
                    prev_content = [prev_content]
                if not isinstance(curr_content, list):
                    curr_content = [curr_content]
                combined[-1]['content'] = prev_content + curr_content
            else:
                # Ensure content is a list
                content = msg.get('content', [])
                if not isinstance(content, list):
                    content = [content]
                combined.append({'role': 'user', 'content': content})
        else:
            combined.append(msg)
    return combined


def clean_json_string(json_str: str) -> str:
    """Clean and repair common JSON formatting issues from LLM responses.

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Cleaned JSON string
    """
    if not json_str or not isinstance(json_str, str):
        return json_str

    # Remove Markdown code blocks
    json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
    json_str = json_str.strip()

    # Try to find the actual JSON object: look for the first { and last }
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = json_str[start_idx : end_idx + 1]

    # Remove trailing quotes/whitespace
    json_str = re.sub(r'[\'"\s]*$', '', json_str)

    # Fix common escaping issues
    # Sometimes LLMs add extra backslashes or quotes
    json_str = json_str.replace("\\'", "'")  # Fix over-escaped single quotes

    return json_str.strip()
