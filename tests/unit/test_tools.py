"""
Test suite for tools.py module.
Tests all tools with appropriate mocking for web APIs and external dependencies.
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import os

from kodeagent.tools import (
    tool,
    calculator,
    search_web,
    download_file,
    extract_file_contents_as_markdown,
    search_wikipedia,
    search_arxiv,
    get_youtube_transcript,
    get_audio_transcript
)


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_adds_metadata(self):
        """Test that @tool decorator adds required metadata."""

        @tool
        def sample_func(x: int, y: str) -> str:
            """Sample function for testing."""
            return f"{x}-{y}"

        assert hasattr(sample_func, 'name')
        assert hasattr(sample_func, 'description')
        assert hasattr(sample_func, 'args_schema')
        assert sample_func.name == 'sample_func'
        assert sample_func.description == 'Sample function for testing.'

    def test_tool_decorator_preserves_functionality(self):
        """Test that decorated function still works correctly."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5

    def test_tool_decorator_rejects_async_functions(self):
        """Test that @tool raises error for async functions."""
        with pytest.raises(ValueError, match='async functions is not supported'):
            @tool
            async def async_func():
                """Async function."""
                pass

    def test_tool_creates_pydantic_schema(self):
        """Test that tool creates proper Pydantic schema for arguments."""

        @tool
        def func_with_args(name: str, age: int, active: bool = True) -> str:
            """Function with multiple arguments."""
            return f"{name}-{age}-{active}"

        schema = func_with_args.args_schema
        assert schema is not None
        # Test that schema can validate correct inputs
        validated = schema(name="John", age=30, active=False)
        assert validated.name == "John"
        assert validated.age == 30
        assert validated.active is False


class TestCalculator:
    """Tests for the calculator tool."""

    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert calculator("2 + 3") == 5.0
        assert calculator("10 - 4") == 6.0
        assert calculator("3 * 4") == 12.0
        assert calculator("15 / 3") == 5.0

    def test_complex_expressions(self):
        """Test complex expressions with parentheses."""
        assert calculator("(2 + 3) * 4") == 20.0
        assert calculator("10 / (2 + 3)") == 2.0
        assert calculator("((5 + 3) * 2) - 4") == 12.0

    def test_exponentiation(self):
        """Test exponentiation with **."""
        assert calculator("2 ** 3") == 8.0
        assert calculator("5 ** 2") == 25.0

    def test_caret_conversion(self):
        """Test that ^ is converted to **."""
        assert calculator("2 ^ 3") == 8.0

    def test_negative_numbers(self):
        """Test unary minus."""
        assert calculator("-5 + 3") == -2.0
        assert calculator("10 + (-5)") == 5.0

    def test_decimal_numbers(self):
        """Test decimal arithmetic."""
        assert calculator("2.5 + 3.5") == 6.0
        assert calculator("10.5 / 2") == 5.25

    def test_invalid_characters(self):
        """Test that invalid characters return None."""
        assert calculator("2 + a") is None
        assert calculator("import os") is None
        assert calculator("__import__('os')") is None
        assert calculator("2 + 3; print('hello')") is None

    def test_invalid_syntax(self):
        """Test that invalid syntax returns None."""
        assert calculator("2 + + 3") == 5.0  # This is valid syntax; 2 + (+3)
        assert calculator("(2 + 3") is None
        assert calculator("2 +") is None

    def test_division_by_zero(self):
        """Test division by zero returns None."""
        assert calculator("5 / 0") is None

    def test_quote_removal(self):
        """Test that quotes are removed from expression."""
        assert calculator("'2 + 3'") == 5.0


class TestSearchWeb:
    """Tests for the search_web tool."""

    @patch('ddgs.DDGS')
    def test_search_web_basic(self, mock_ddgs):
        """Test basic web search."""
        mock_results = [
            {'title': 'Result 1', 'href': 'https://example.com/1', 'body': 'Description 1'},
            {'title': 'Result 2', 'href': 'https://example.com/2', 'body': 'Description 2'}
        ]
        mock_ddgs.return_value.text.return_value = mock_results

        result = search_web("test query", max_results=2, show_description=False)

        assert '## Search Results' in result
        assert '[Result 1](https://example.com/1)' in result
        assert '[Result 2](https://example.com/2)' in result
        assert 'Description 1' not in result

    @patch('ddgs.DDGS')
    def test_search_web_with_description(self, mock_ddgs):
        """Test web search with descriptions."""
        mock_results = [
            {'title': 'Result 1', 'href': 'https://example.com/1', 'body': 'Description 1'}
        ]
        mock_ddgs.return_value.text.return_value = mock_results

        result = search_web("test query", max_results=1, show_description=True)

        assert 'Description 1' in result

    @patch('ddgs.DDGS')
    def test_search_web_no_results(self, mock_ddgs):
        """Test web search with no results."""
        mock_ddgs.return_value.text.return_value = []

        result = search_web("nonexistent query")

        assert 'No results found' in result

    @patch('ddgs.DDGS')
    def test_search_web_calls_ddgs_correctly(self, mock_ddgs):
        """Test that DDGS is called with correct parameters."""
        mock_ddgs.return_value.text.return_value = []

        search_web("test query", max_results=5)

        mock_ddgs.return_value.text.assert_called_once_with("test query", max_results=5)


class TestDownloadFile:
    """Tests for the download_file tool."""

    @patch('requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert result is not None
        assert os.path.exists(result)

        # Cleanup
        if os.path.exists(result):
            os.remove(result)

    @patch('requests.get')
    def test_download_file_with_headers(self, mock_get):
        """Test that download includes user agent header."""
        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response

        download_file("https://example.com/file.pdf")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert 'headers' in call_kwargs
        assert call_kwargs['headers']['user-agent'] == 'kodeagent/0.0.1'

    @patch('requests.get')
    def test_download_file_http_error(self, mock_get):
        """Test download with HTTP error."""
        mock_get.side_effect = Exception("HTTP Error")

        with pytest.raises(Exception):
            download_file("https://example.com/file.pdf")


class TestExtractFileContentsAsMarkdown:
    """Tests for the extract_file_contents_as_markdown tool."""

    @patch('markitdown.MarkItDown')
    def test_extract_html_content(self, mock_markitdown):
        """Test extracting HTML content."""
        mock_result = Mock()
        mock_result.text_content = "# Heading\n\nSome content"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_file_contents_as_markdown("https://example.com/page.html")

        assert "# Heading" in result
        assert "Some content" in result

    @patch('markitdown.MarkItDown')
    def test_extract_with_link_scrubbing(self, mock_markitdown):
        """Test that links are removed when scrub_links=True."""
        mock_result = Mock()
        mock_result.text_content = "Check [this link](https://example.com) out"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_file_contents_as_markdown("file.html", scrub_links=True)

        assert "this link" in result
        assert "https://example.com" not in result
        assert "[this link]" not in result

    @patch('markitdown.MarkItDown')
    def test_extract_without_link_scrubbing(self, mock_markitdown):
        """Test that links are kept when scrub_links=False."""
        mock_result = Mock()
        mock_result.text_content = "Check [this link](https://example.com) out"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_file_contents_as_markdown("file.html", scrub_links=False)

        assert "[this link](https://example.com)" in result

    @patch('markitdown.MarkItDown')
    def test_extract_with_max_length(self, mock_markitdown):
        """Test truncation with max_length."""
        mock_result = Mock()
        mock_result.text_content = "A" * 1000
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_file_contents_as_markdown("file.html", max_length=100)

        assert len(result) == 100

    @patch('markitdown.MarkItDown')
    @patch('mimetypes.guess_type')
    def test_extract_pdf_with_cid_handling(self, mock_guess_type, mock_markitdown):
        """Test PDF extraction with (cid:NNN) handling."""
        mock_guess_type.return_value = ('application/pdf', None)
        mock_result = Mock()
        mock_result.text_content = "Text with (cid:65) and (cid:66)"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_file_contents_as_markdown("file.pdf")

        assert "(cid:" not in result
        # cid:65 + 29 = 94 (^), cid:66 + 29 = 95 (_)
        assert chr(94) in result or chr(95) in result

    @patch('markitdown.MarkItDown')
    def test_extract_handles_exceptions(self, mock_markitdown):
        """Test that exceptions are caught and returned as strings."""
        mock_markitdown.return_value.convert.side_effect = Exception("Conversion failed")

        result = extract_file_contents_as_markdown("file.html")

        assert "Conversion failed" in result


class TestSearchWikipedia:
    """Tests for the search_wikipedia tool."""

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    def test_search_wikipedia_success(self, mock_page, mock_search):
        """Test successful Wikipedia search."""
        mock_search.return_value = ['Python (programming language)']

        mock_page_obj = Mock()
        mock_page_obj.title = 'Python (programming language)'
        mock_page_obj.url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
        mock_page_obj.summary = 'Python is a high-level programming language.'
        mock_page.return_value = mock_page_obj

        result = search_wikipedia("Python programming")

        assert 'Python (programming language)' in result
        assert 'Python is a high-level programming language.' in result
        assert 'https://en.wikipedia.org/wiki/Python_(programming_language)' in result

    @patch('wikipedia.search')
    def test_search_wikipedia_no_results(self, mock_search):
        """Test Wikipedia search with no results."""
        mock_search.return_value = []

        result = search_wikipedia("xyznonexistent")

        assert 'No results found' in result

    @patch('wikipedia.search')
    def test_search_wikipedia_disambiguation(self, mock_search):
        """Test Wikipedia disambiguation error."""
        from wikipedia.exceptions import DisambiguationError

        mock_search.side_effect = DisambiguationError(
            'Python',
            ['Python (programming language)', 'Python (genus)', 'Monty Python']
        )

        result = search_wikipedia("Python")

        assert 'DisambiguationError' in result
        assert 'Python (programming language)' in result

    @patch('wikipedia.search')
    @patch('wikipedia.page')
    def test_search_wikipedia_max_results(self, mock_page, mock_search):
        """Test that max_results parameter is respected."""
        mock_search.return_value = ['Result1', 'Result2', 'Result3']
        mock_page.return_value = Mock(title='Test', url='http://test.com', summary='Summary')

        search_wikipedia("test", max_results=2)

        mock_search.assert_called_once_with("test", results=2)


class TestSearchArxiv:
    """Tests for the search_arxiv tool."""

    @patch('arxiv.Client')
    def test_search_arxiv_success(self, mock_client):
        """Test successful arXiv search."""
        mock_result = Mock()
        mock_result.title = 'Test Paper'
        mock_result.pdf_url = 'https://arxiv.org/pdf/1234.5678'
        mock_result.summary = 'This is a test paper.'
        mock_result.published = Mock()
        mock_result.published.strftime.return_value = '2024-01-01'

        mock_author = Mock()
        mock_author.name = 'John Doe'
        mock_result.authors = [mock_author]

        mock_client.return_value.results.return_value = [mock_result]

        result = search_arxiv("machine learning")

        assert 'Test Paper' in result
        assert 'John Doe' in result
        assert 'This is a test paper.' in result
        assert '2024-01-01' in result

    @patch('arxiv.Client')
    def test_search_arxiv_no_results(self, mock_client):
        """Test arXiv search with no results."""
        mock_client.return_value.results.return_value = []

        result = search_arxiv("nonexistent topic")

        assert 'No results found' in result

    @patch('arxiv.Client')
    def test_search_arxiv_multiple_authors(self, mock_client):
        """Test arXiv result with multiple authors."""
        mock_result = Mock()
        mock_result.title = 'Multi-author Paper'
        mock_result.pdf_url = 'https://arxiv.org/pdf/1234.5678'
        mock_result.summary = 'Summary'
        mock_result.published = Mock()
        mock_result.published.strftime.return_value = '2024-01-01'

        mock_author1 = Mock()
        mock_author1.name = 'John Doe'
        mock_author2 = Mock()
        mock_author2.name = 'Jane Smith'
        mock_result.authors = [mock_author1, mock_author2]

        mock_client.return_value.results.return_value = [mock_result]

        result = search_arxiv("test")

        assert 'John Doe, Jane Smith' in result

    @patch('arxiv.Client')
    def test_search_arxiv_exception_handling(self, mock_client):
        """Test arXiv search exception handling."""
        mock_client.side_effect = Exception("API Error")

        result = search_arxiv("test")

        assert 'error occurred' in result.lower()
        assert 'API Error' in result


class TestGetYoutubeTranscript:
    """Tests for the get_youtube_transcript tool."""

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_get_youtube_transcript_success(self, mock_api):
        """Test successful YouTube transcript retrieval."""
        mock_snippet1 = Mock()
        mock_snippet1.text = 'Hello world'
        mock_snippet2 = Mock()
        mock_snippet2.text = 'This is a test'

        mock_transcript = Mock()
        mock_transcript.snippets = [mock_snippet1, mock_snippet2]

        mock_api.return_value.fetch.return_value = mock_transcript

        result = get_youtube_transcript('aBc4E')

        assert 'Hello world' in result
        assert 'This is a test' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_get_youtube_transcript_disabled(self, mock_api):
        """Test YouTube transcript when subtitles are disabled."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        mock_api.return_value.fetch.side_effect = TranscriptsDisabled("video_id")

        result = get_youtube_transcript('aBc4E')

        assert 'ERROR' in result
        assert 'disabled' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_get_youtube_transcript_not_found(self, mock_api):
        """Test YouTube transcript when no transcript is found."""
        from youtube_transcript_api._errors import NoTranscriptFound

        mock_api.return_value.fetch.side_effect = NoTranscriptFound(
            'video_id', [], None
        )

        result = get_youtube_transcript('aBc4E')

        assert 'ERROR' in result
        assert 'No transcript found' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_get_youtube_transcript_generic_error(self, mock_api):
        """Test YouTube transcript with generic error."""
        mock_api.return_value.fetch.side_effect = Exception('Network error')

        result = get_youtube_transcript('aBc4E')

        assert 'ERROR' in result
        assert 'Network error' in result


class TestGetAudioTranscript:
    """Tests for the get_audio_transcript tool."""

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_get_audio_transcript_success(self, mock_getenv, mock_file, mock_post):
        """Test successful audio transcription."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Transcribed audio content'}
        mock_post.return_value = mock_response

        result = get_audio_transcript('/path/to/audio.mp3')

        assert result == {'text': 'Transcribed audio content'}
        mock_post.assert_called_once()

        # Verify API call parameters
        call_kwargs = mock_post.call_args[1]
        assert 'Authorization' in call_kwargs['headers']
        assert call_kwargs['headers']['Authorization'] == 'Bearer test_api_key'

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_get_audio_transcript_error(self, mock_getenv, mock_file, mock_post):
        """Test audio transcription with API error."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response

        result = get_audio_transcript('/path/to/audio.mp3')

        assert 'Audio transcription error' in result
        assert '400' in result

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_get_audio_transcript_correct_model(self, mock_getenv, mock_file, mock_post):
        """Test that correct model and parameters are used."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Test'}
        mock_post.return_value = mock_response

        get_audio_transcript('/path/to/audio.mp3')

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['data']['model'] == 'whisper-v3-turbo'
        assert call_kwargs['data']['temperature'] == '0'
        assert call_kwargs['data']['vad_model'] == 'silero'
