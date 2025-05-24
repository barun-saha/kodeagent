"""
A minimal set of utilities used by KodeAgent.
This module will be copied along with code for CodeAgent, so keep it minimum.
"""
import base64
import logging
import mimetypes
import os
from typing import Optional, Any

import requests


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get a logger for the current module
logger = logging.getLogger('KodeAgent')


def is_it_url(path: str) -> bool:
    """
    Check whether a given path is a URL.

    Args:
        path: The path.

    Returns:
        `True` if it's a URL; `False` otherwise.
    """
    return path.startswith(('http://', 'https://'))


def detect_file_type(url: str) -> str:
    """
    Identify the content/MIME type of file pointed by a URL.

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
            print(f'Detected file type from Content-Disposition: {file_extension}')
            return file_extension  # If this works, return immediately

        # Step 2: If HEAD didn't give useful info, send GET request for more details
        response = requests.get(url, stream=True, timeout=20)
        content_type = response.headers.get('Content-Type')

        if content_type and content_type != 'application/json':  # Avoid false positives
            print(f'Detected Content-Type from GET request: {content_type}')
            return content_type

        return 'Unknown file type'
    except requests.RequestException as e:
        print(f'Error detecting file type: {e}')
        return 'Unknown file type'


def is_image_file(file_type) -> bool:
    """
    Identify whether a given MIME type is an image.

    Args:
        file_type: The file/content type.

    Returns:
        `True` if an image file; `False` otherwise.
    """
    return file_type.startswith('image/')


def make_user_message(
    text_content: str,
    files: Optional[list[str]] = None
) -> list[dict[str, Any]]:
    """
    Create a single user message to be sent to LiteLLM.

    Args:
        text_content: The text content of the message.
        files: An optional list of file paths or URLs, which can include images
               or other file types.

    Returns:
        A list of dict items representing the messages.
    """
    content: list[dict[str, Any]] = [{'type': 'text', 'text': text_content.strip()}]
    message: list[dict[str, Any]] = [{'role': 'user'}]

    if files:
        for item in files:
            is_image = False
            if is_it_url(item):
                if any(
                        ext in item.lower() for ext in [
                            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'
                        ]
                ) or is_image_file(detect_file_type(item)):
                    is_image = True
            elif os.path.isfile(item):
                mime_type, _ = mimetypes.guess_type(item)
                if mime_type and 'image' in mime_type:
                    is_image = True

            if is_image:
                if is_it_url(item):
                    content.append({'type': 'image_url', 'image_url': {'url': item}})
                elif os.path.isfile(item):
                    try:
                        with open(item, 'rb') as img_file:
                            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                        mime_type, _ = mimetypes.guess_type(item)
                        mime_type = mime_type if mime_type else 'application/octet-stream'
                        content.append({
                            'type': 'image_url',
                            'image_url': {'url': f'data:{mime_type};base64,{encoded_image}'}
                        })
                    except FileNotFoundError:
                        logger.error('Image file not found: %s...will ignore it', item)
                    except Exception as e:
                        logger.error(
                            'Error processing local image %s: %s...will ignore it',
                            item, e
                        )
                else:
                    logger.error('Invalid image file path or URL: %s...will ignore it', item)
            else:  # Handle as a general file or URL (not an image)
                if is_it_url(item):
                    content.append({'type': 'text', 'text': f'File URL: {item}'})
                elif os.path.isfile(item):
                    mime_type, _ = mimetypes.guess_type(item)
                    if mime_type:
                        if any(
                                keyword in mime_type for keyword in [
                                    'application', 'audio', 'video'
                                ]
                        ):
                            content.append({'type': 'text', 'text': f'Input file: {item}'})
                        elif 'text' in mime_type or mime_type in [
                            'application/json', 'application/xml'
                        ]:
                            try:
                                with open(item, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                content.append(
                                    {
                                        'type': 'text',
                                        'text': f'File {item} content:\n{file_content}'
                                    }
                                )
                            except Exception as e:
                                logger.error(
                                    'Error reading text file %s: %s...will ignore it',
                                    item, e
                                )
                        else:
                            content.append({'type': 'text', 'text': f'Input file: {item}'})
                    else:
                        content.append({'type': 'text', 'text': f'Input file: {item}'})
                else:
                    logger.error('Invalid file path or URL: %s...will ignore it', item)

    message[0]['content'] = content
    return message
