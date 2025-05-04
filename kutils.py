"""
A minimal set of utilities used by KodeAgent.
This module will be copied along with code for CodeAgent, so keep it minimum.
"""
import base64
import logging
import os
from typing import Optional


logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Get a logger for the current module
logger = logging.getLogger('KodeAgent')


def make_user_message(
        text_content: str,
        image_files: Optional[list[str]] = None
) -> list[dict]:
    """
    Create a single user message to be sent to LiteLLM.

    Args:
        text_content: The text content of the message.
        image_files: An optional list of image file paths or URLs.

    Returns:
        A list of dict items representing the messages.
    """
    content = [{'type': 'text', 'text': text_content.strip()}]
    message = [{'role': 'user'}]
    mime_mapping = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif'
    }

    if image_files:
        for image_file in image_files:
            if image_file.startswith('http://') or image_file.startswith('https://'):
                # It's a URL
                content.append({'type': 'image_url', 'image_url': {'url': image_file}})
            elif os.path.isfile(image_file):
                # It's a local file
                try:
                    with open(image_file, 'rb') as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                    _, ext = os.path.splitext(image_file.lower())
                    mime_type = mime_mapping.get(ext, 'application/octet-stream')
                    content.append({
                        'type': 'image_url',
                        'image_url': {'url': f'data:{mime_type};base64,{encoded_image}'},
                    })
                except FileNotFoundError:
                    logger.error('Image file not found: %s...will ignore it', image_file)
                except Exception as e:
                    logger.error(
                        'Error processing local image %s: %s\n...will ignore it',
                        image_file, e
                    )
            else:
                logger.error('Invalid image file path or URL: %s...will ignore it', image_file)

    message[0]['content'] = content
    return message
