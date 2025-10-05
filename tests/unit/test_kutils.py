"""
Unit tests for the kutils module.
"""
import base64
import logging
from unittest.mock import patch, mock_open, MagicMock

import pytest
import requests

from kodeagent.kutils import (
    is_it_url,
    detect_file_type,
    is_image_file,
    make_user_message,
    combine_user_messages,
    logger,
    call_llm
)


@pytest.mark.parametrize('messages,expected', [
    # Single user message
    ([{'role': 'user', 'content': ['a']}], [{'role': 'user', 'content': ['a']}]),
    # Two consecutive user messages
    (
            [
                {'role': 'user', 'content': ['a']},
                {'role': 'user', 'content': ['b']}
            ],
            [{'role': 'user', 'content': ['a', 'b']}]
    ),
    # User and assistant messages
    (
            [
                {'role': 'user', 'content': ['a']},
                {'role': 'assistant', 'content': ['b']}
            ],
            [{'role': 'user', 'content': ['a']}, {'role': 'assistant', 'content': ['b']}]
    ),
    # Mixed content types
    (
            [
                {'role': 'user', 'content': 'a'},
                {'role': 'user', 'content': ['b']}
            ],
            [{'role': 'user', 'content': ['a', 'b']}]
    ),
    # Non-list content
    ([{'role': 'user', 'content': 'a'}], [{'role': 'user', 'content': ['a']}]),
    # Multiple merges
    (
            [
                {'role': 'user', 'content': ['a']},
                {'role': 'user', 'content': ['b']},
                {'role': 'user', 'content': ['c']}
            ],
            [{'role': 'user', 'content': ['a', 'b', 'c']}]
    ),
])
def test_combine_user_messages(messages, expected):
    """
    Test combining consecutive user messages.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        expected: The expected combined message list.
    """
    assert combine_user_messages(messages) == expected


@pytest.mark.asyncio
async def test_call_llm_success(monkeypatch):
    """
    Test successful LLM call.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    # Mock litellm.acompletion to return a valid response
    class DummyResponse:
        class Choices:
            class Message:
                content = 'response text'
            message = Message()
        choices = [Choices()]
        usage = {
            'prompt_tokens': 1,
            'completion_tokens': 2,
            'total_tokens': 3
        }
        _hidden_params = {'response_cost': 0.01}


    async def dummy_acompletion(**kwargs):
        """
        Mocked acompletion function.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            DummyResponse: A dummy response object.
        """
        return DummyResponse()


    monkeypatch.setattr('src.kodeagent.kutils.litellm.acompletion', dummy_acompletion)
    result = await call_llm(
        'model', {}, [{'role': 'user', 'content': 'hi'}]
    )
    assert result == 'response text'


@pytest.mark.asyncio
async def test_call_llm_empty_response(monkeypatch):
    """
    Test LLM call with empty response content.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    # Mock litellm.acompletion to return empty content
    class DummyResponse:
        class Choices:
            class Message:
                content = ''
            message = Message()
        choices = [Choices()]
        usage = {
            'prompt_tokens': 1,
            'completion_tokens': 2,
            'total_tokens': 3
        }
        _hidden_params = {'response_cost': 0.01}


    async def dummy_acompletion(**kwargs):
        return DummyResponse()


    monkeypatch.setattr('src.kodeagent.kutils.litellm.acompletion', dummy_acompletion)
    with pytest.raises(ValueError):
        await call_llm(
            'model', {}, [{'role': 'user', 'content': 'hi'}]
        )


@pytest.mark.asyncio
async def test_call_llm_exception(monkeypatch):
    """
    Test LLM call that raises an exception.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
    """
    # Mock litellm.acompletion to raise exception
    async def dummy_acompletion(**kwargs):
        raise Exception('fail')


    monkeypatch.setattr('src.kodeagent.kutils.litellm.acompletion', dummy_acompletion)
    with pytest.raises(ValueError):
        await call_llm(
            'model', {}, [{'role': 'user', 'content': 'hi'}]
        )


def test_is_it_url():
    """Test URL detection function."""
    # Test valid URLs
    assert is_it_url('http://example.com')
    assert is_it_url('https://example.com/image.jpg')
    assert is_it_url('https://api.github.com/v1/repos')
    assert is_it_url('http://localhost:8000')

    # Test invalid URLs
    assert not is_it_url('file.txt')
    assert not is_it_url('/path/to/file')
    assert not is_it_url('ftp://example.com')  # Only http(s) supported
    assert not is_it_url('git@github.com:user/repo.git')
    assert not is_it_url('C:\\path\\to\\file')


@pytest.mark.parametrize("url,headers,expected", [
    (
        'https://example.com/image.jpg',
        {'Content-Disposition': 'attachment; filename=photo.jpg'},
        'jpg'
    ),
    (
        'https://example.com/doc',
        {'Content-Type': 'application/pdf'},
        'application/pdf'
    ),
    (
        'https://example.com/unknown',
        {},
        'Unknown file type'
    )
])
def test_detect_file_type(url, headers, expected):
    """Test file type detection with different response headers."""
    with patch('requests.head') as mock_head:
        mock_head.return_value = MagicMock(headers=headers)
        if not headers.get('Content-Disposition'):
            with patch('requests.get') as mock_get:
                mock_get.return_value = MagicMock(headers=headers)
                assert detect_file_type(url) == expected
        else:
            assert detect_file_type(url) == expected


def test_detect_file_type_errors():
    """Test file type detection error handling."""
    with patch('requests.head') as mock_head:
        # Test connection timeout
        mock_head.side_effect = requests.exceptions.Timeout()
        assert detect_file_type('https://example.com/timeout') == 'Unknown file type'

        # Test connection error
        mock_head.side_effect = requests.exceptions.ConnectionError()
        assert detect_file_type('https://example.com/error') == 'Unknown file type'

        # Test invalid URL
        mock_head.side_effect = requests.exceptions.InvalidURL()
        assert detect_file_type('https://invalid-url') == 'Unknown file type'


@pytest.mark.parametrize('file_type,expected', [
    ('image/jpeg', True),
    ('image/png', True),
    ('image/gif', True),
    ('image/webp', True),
    ('image/svg+xml', True),
    ('application/pdf', False),
    ('text/plain', False),
    ('video/mp4', False),
    ('audio/mpeg', False),
    ('application/json', False),
    ('Unknown file type', False)
])
def test_is_image_file(file_type, expected):
    """Test image file type detection."""
    assert is_image_file(file_type) == expected


def test_make_user_message_basic():
    """Test basic message creation without files."""
    # Test simple message
    message = make_user_message('Hello world')
    assert len(message) == 1
    assert message[0]['role'] == 'user'
    assert len(message[0]['content']) == 1
    assert message[0]['content'][0]['type'] == 'text'
    assert message[0]['content'][0]['text'] == 'Hello world'

    # Test message with whitespace
    message = make_user_message('  Trim me  ')
    assert message[0]['content'][0]['text'] == '  Trim me  '

    # Test empty message
    message = make_user_message('')
    assert message[0]['content'][0]['text'] == ''


@patch('requests.head')
@patch('requests.get')
def test_make_user_message_with_urls(mock_head, mock_get):
    """Test message creation with URLs."""

    def mock_request_side_effect(url, *args, **kwargs):
        if 'image.jpg' in url:
            # This is a mock response for the image file
            return MagicMock(headers={'Content-Type': 'image/jpeg'})
        if 'document.pdf' in url:
            # This is a mock response for the PDF file
            return MagicMock(headers={'Content-Type': 'application/pdf'})
        return MagicMock(headers={})

    # Assign the same side effect function to both mocks
    mock_get.side_effect = mock_request_side_effect
    mock_head.side_effect = mock_request_side_effect

    # Test with both URLs
    message = make_user_message(
        'Check these URLs',
        files=[
            'https://example.com/image.jpg',
            'https://example.com/document.pdf'
        ]
    )

    content = message[0]['content']
    assert len(content) == 3
    assert content[0]['text'] == 'Check these URLs'
    assert content[1]['type'] == 'image_url'
    assert content[2]['type'] == 'text'
    assert 'File URL:' in content[2]['text']


@patch('os.path.isfile')
@patch('mimetypes.guess_type')
def test_make_user_message_with_local_files(mock_mime, mock_isfile):
    """Test message creation with local files."""
    mock_isfile.return_value = True

    # Test with text file
    mock_mime.return_value = ('text/plain', None)
    m = mock_open(read_data='Hello from file')
    with patch('builtins.open', m):
        message = make_user_message(
            'Read this file',
            files=['test.txt']
        )

    content = message[0]['content']
    assert len(content) == 2
    assert 'Hello from file' in content[1]['text']

    # Test with binary file
    mock_mime.return_value = ('application/pdf', None)
    message = make_user_message(
        'Check this PDF',
        files=['doc.pdf']
    )

    content = message[0]['content']
    assert len(content) == 2
    assert 'Input file:' in content[1]['text']


@patch('os.path.isfile')
@patch('mimetypes.guess_type')
def test_make_user_message_with_images(mock_mime, mock_isfile):
    """Test message creation with image files."""
    mock_isfile.return_value = True
    mock_mime.return_value = ('image/jpeg', None)

    # Create fake image data
    fake_image = b'fake_image_data'
    expected_b64 = base64.b64encode(fake_image).decode('utf-8')

    m = mock_open(read_data=fake_image)
    with patch('builtins.open', m):
        message = make_user_message(
            'Check this image',
            files=['photo.jpg']
        )

    content = message[0]['content']
    assert len(content) == 2
    assert content[1]['type'] == 'image_url'
    assert expected_b64 in content[1]['image_url']['url']
    assert 'data:image/jpeg;base64,' in content[1]['image_url']['url']


def test_make_user_message_error_handling():
    """Test error handling in message creation."""

    # We mock os.path.isfile and mimetypes.guess_type at the function level
    with patch('os.path.isfile', return_value=True) as mock_isfile:
        with patch('mimetypes.guess_type') as mock_guess_type:
            # Test file read error
            mock_guess_type.side_effect = Exception('Read error')
            message = make_user_message(
                'Test errors',
                files=['error.txt']
            )
            # Should only have the original message
            assert len(message[0]['content']) == 1

            # Test with nonexistent file
            mock_isfile.return_value = False
            mock_guess_type.side_effect = None  # Reset the side effect for the next test
            message = make_user_message(
                'Test missing file',
                files=['missing.txt']
            )
            assert len(message[0]['content']) == 1


@patch('os.path.isfile')
def test_make_user_message_complex_scenario(mock_isfile):
    """Test message creation with mixed content types."""
    def side_effect(path):
        return path in ['local.txt', 'image.jpg']


    mock_isfile.side_effect = side_effect

    with patch('mimetypes.guess_type') as mock_mime:
        def mime_side_effect(path):
            if path == 'local.txt':
                return 'text/plain', None
            if path == 'image.jpg':
                return 'image/jpeg', None
            return None, None
        mock_mime.side_effect = mime_side_effect

        with patch('requests.head') as mock_head:
            mock_head.return_value = MagicMock(headers={})

            with patch('requests.get') as mock_get:
                mock_get.return_value = MagicMock(headers={'Content-Type': 'image/png'})

                with patch('builtins.open', mock_open(read_data='file content')):
                    message = make_user_message(
                        'Complex test',
                        files=[
                            'https://example.com/image.png',
                            'local.txt',
                            'image.jpg',
                            'nonexistent.pdf'
                        ]
                    )

    content = message[0]['content']
    # Check all parts are present
    assert len(content) > 1
    assert content[0]['text'] == 'Complex test'
    assert any(
        part.get('type') == 'image_url'
        and isinstance(part.get('image_url'), dict)
        and 'https://example.com/image.png' in part['image_url'].get('url', '')
        for part in content
    )
    assert any('file content' in str(item) for item in content)


def test_logging_configuration():
    """Test logger configuration."""
    assert logger.name == 'KodeAgent'
    assert logger.getEffectiveLevel() == logging.WARNING

    # Test log message format
    with patch.object(logger, 'warning') as mock_warning:
        logger.warning('Test message')
        mock_warning.assert_called_once_with('Test message')


# Edge case: make_user_message with empty file list
def test_make_user_message_empty_files():
    msg = make_user_message('hello', files=[])
    assert msg[0]['content'][0]['text'] == 'hello'
    assert len(msg[0]['content']) == 1


# Edge case: make_user_message with unknown MIME type
@patch('os.path.isfile', return_value=True)
@patch('mimetypes.guess_type', return_value=(None, None))
def test_make_user_message_unknown_mime(mock_mime, mock_isfile):
    msg = make_user_message('unknown mime', files=['file.unknown'])
    assert any('Input file:' in c['text'] for c in msg[0]['content'] if c['type'] == 'text')


# Edge case: make_user_message with invalid file path
@patch('os.path.isfile', return_value=False)
def test_make_user_message_invalid_path(mock_isfile):
    """
    Test make_user_message with an invalid file path.

    Args:
        mock_isfile: Mock for os.path.isfile.
    """
    msg = make_user_message('invalid path', files=['notfound.txt'])
    assert msg[0]['content'][0]['text'] == 'invalid path'
    assert len(msg[0]['content']) == 1


# Logger error branch for detect_file_type
@patch('requests.head', side_effect=requests.RequestException('fail'))
def test_detect_file_type_logger_error(mock_head):
    """
    Test detect_file_type logging on request exception.

    Args:
        mock_head: A mock for requests.head.
    """
    with patch.object(logger, 'error') as mock_log:
        result = detect_file_type('http://bad.url')
        assert result == 'Unknown file type'
        mock_log.assert_called()


# Logger error branch for make_user_message (invalid image)
def test_make_user_message_logger_error():
    """
    Test make_user_message logging on invalid image file.
    """
    with patch('os.path.isfile', return_value=False):
        with patch.object(logger, 'error') as mock_log:
            msg = make_user_message('bad image', files=['badimage.jpg'])
            mock_log.assert_called()
            assert msg[0]['content'][0]['text'] == 'bad image'
