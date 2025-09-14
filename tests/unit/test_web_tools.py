"""Tests for web search and related tools."""
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime
import wikipedia
from kodeagent import (
    search_web,
    download_file,
    extract_file_contents_as_markdown,
    search_wikipedia,
    search_arxiv,
    get_youtube_transcript
)

def test_search_web_with_results():
    """Test search_web with mock results."""
    mock_results = [
        {'title': 'Result 1', 'href': 'http://test1.com', 'body': 'Description 1'},
        {'title': 'Result 2', 'href': 'http://test2.com', 'body': 'Description 2'}
    ]

    with patch('ddgs.DDGS') as mock_ddgs:
        mock_instance = mock_ddgs.return_value
        mock_instance.text.return_value = mock_results

        # Test without descriptions
        result = search_web("test query", max_results=2)
        assert '[Result 1](http://test1.com)' in result
        assert '[Result 2](http://test2.com)' in result
        assert 'Description' not in result

        # Test with descriptions
        result = search_web("test query", max_results=2, show_description=True)
        assert '[Result 1](http://test1.com)\nDescription 1' in result
        assert '[Result 2](http://test2.com)\nDescription 2' in result

def test_search_web_no_results():
    """Test search_web with no results."""
    with patch('ddgs.DDGS') as mock_ddgs:
        mock_instance = mock_ddgs.return_value
        mock_instance.text.return_value = []
        result = search_web("empty query")
        assert 'No results found!' in result

def test_download_file_success():
    """Test successful file download."""
    with patch('requests.get') as mock_get:
        mock_response = mock_get.return_value
        mock_response.iter_content.return_value = [b'test content']

        result = download_file('http://test.com/file.txt')
        mock_get.assert_called_with(
            'http://test.com/file.txt',
            timeout=20,
            stream=True,
            headers={'user-agent': 'kodeagent/0.0.1'}
        )
        assert isinstance(result, str)
        assert len(result) > 0

def test_download_file_error():
    """Test file download with error."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.side_effect = Exception("404 Not Found")
        with pytest.raises(Exception):
            download_file('http://test.com/error')

def test_extract_file_contents():
    """Test file content extraction."""
    with patch('markitdown.MarkItDown') as mock_markitdown:
        mock_instance = mock_markitdown.return_value
        mock_instance.convert.return_value.text_content = 'Test content with (cid:123)'

        # Test PDF with CID replacement
        result = extract_file_contents_as_markdown('test.pdf')
        assert 'Test content with' in result
        assert '(cid:123)' not in result

        # Test with links
        mock_instance.convert.return_value.text_content = '[Link](http://test.com) text'
        result = extract_file_contents_as_markdown('test.html', scrub_links=False)
        assert '[Link](http://test.com)' in result

        # Test link removal
        result = extract_file_contents_as_markdown('test.html', scrub_links=True)
        assert '[Link](http://test.com)' not in result
        assert 'Link' in result

        # Test max length
        mock_instance.convert.return_value.text_content = 'A' * 100
        result = extract_file_contents_as_markdown('test.txt', max_length=50)
        assert len(result) <= 50

def test_search_wikipedia_success():
    """Test successful Wikipedia search."""
    with patch('wikipedia.search') as mock_search:
        with patch('wikipedia.page') as mock_page:
            mock_search.return_value = ['Page1', 'Page2']

            class MockPage:
                def __init__(self, title):
                    self.title = title
                    self.url = f'https://en.wikipedia.org/wiki/{title}'
                    self.summary = f'Summary of {title}'

            mock_page.side_effect = lambda title: MockPage(title)
            result = search_wikipedia("test query", max_results=2)

            assert '[Page1](https://en.wikipedia.org/wiki/Page1)' in result
            assert '[Page2](https://en.wikipedia.org/wiki/Page2)' in result
            assert 'Summary of Page1' in result
            assert 'Summary of Page2' in result

def test_search_wikipedia_no_results():
    """Test Wikipedia search with no results."""
    with patch('wikipedia.search') as mock_search:
        mock_search.return_value = []
        result = search_wikipedia("nonexistent")
        assert 'No results found!' in result

def test_search_wikipedia_disambiguation():
    """Test Wikipedia search with disambiguation."""
    with patch('wikipedia.search') as mock_search:
        mock_search.side_effect = wikipedia.exceptions.DisambiguationError("Query", ['Option1', 'Option2'])
        result = search_wikipedia("ambiguous")
        assert 'DisambiguationError' in result
        assert 'Option1' in result
        assert 'Option2' in result

def test_search_arxiv():
    """Test arXiv search functionality."""
    with patch('arxiv.Client') as mock_client:
        with patch('arxiv.Search') as mock_search:
            class MockAuthor:
                def __init__(self, name):
                    self.name = name

            class MockResult:
                def __init__(self, title):
                    self.title = title
                    self.authors = [MockAuthor('Author1'), MockAuthor('Author2')]
                    self.summary = 'Paper summary'
                    self.pdf_url = f'https://arxiv.org/pdf/{title}.pdf'
                    self.published = datetime.now()

            mock_results = [MockResult('paper1'), MockResult('paper2')]
            mock_client.return_value.results.return_value = mock_results

            result = search_arxiv("quantum computing", max_results=2)
            assert 'paper1' in result
            assert 'paper2' in result
            assert 'Author1, Author2' in result
            assert 'Paper summary' in result
            assert 'arxiv.org/pdf/' in result

def test_search_arxiv_no_results():
    """Test arXiv search with no results."""
    with patch('arxiv.Client') as mock_client:
        mock_client.return_value.results.return_value = []
        result = search_arxiv("nonexistent topic")
        assert 'No results found' in result

def test_get_youtube_transcript():
    """Test YouTube transcript retrieval."""
    with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
        class MockTranscript:
            def __init__(self, text):
                self.text = text
            @property
            def snippets(self):
                return [self]

        mock_api.return_value.fetch.return_value = MockTranscript("Test transcript")
        result = get_youtube_transcript("video123")
        assert "Test transcript" in result

def test_get_youtube_transcript_disabled():
    """Test YouTube transcript when disabled."""
    with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
        from youtube_transcript_api import TranscriptsDisabled
        video_id = 'video123'
        mock_api.return_value.fetch.side_effect = TranscriptsDisabled(video_id)
        result = get_youtube_transcript(video_id)
        assert 'subtitles appear to be disabled' in result

def test_get_youtube_transcript_not_found():
    """Test YouTube transcript when not found."""
    with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
        from youtube_transcript_api import NoTranscriptFound
        video_id = 'video123'
        mock_api.return_value.fetch.side_effect = NoTranscriptFound(
            video_id,
            requested_language_codes=['en'],
            transcript_data=None
        )
        result = get_youtube_transcript(video_id)
        assert 'No transcript found' in result
