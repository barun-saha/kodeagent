"""
Test suite for tools.py module.
Tests all tools with appropriate mocking for web APIs and external dependencies.
"""
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
import os

from kodeagent.tools import (
    tool,
    calculator,
    search_web,
    download_file,
    extract_as_markdown,
    read_webpage,
    search_wikipedia,
    search_arxiv,
    transcribe_youtube,
    transcribe_audio
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
        assert calculator('2 + 3') == 5.0
        assert calculator('10 - 4') == 6.0
        assert calculator('3 * 4') == 12.0
        assert calculator('15 / 3') == 5.0

    def test_complex_expressions(self):
        """Test complex expressions with parentheses."""
        assert calculator('(2 + 3) * 4') == 20.0
        assert calculator('10 / (2 + 3)') == 2.0
        assert calculator('((5 + 3) * 2) - 4') == 12.0

    def test_exponentiation(self):
        """Test exponentiation with **."""
        assert calculator('2 ** 3') == 8.0
        assert calculator('5 ** 2') == 25.0

    def test_caret_conversion(self):
        """Test that ^ is converted to **."""
        assert calculator('2 ^ 3') == 8.0

    def test_negative_numbers(self):
        """Test unary minus."""
        assert calculator('-5 + 3') == -2.0
        assert calculator('10 + (-5)') == 5.0

    def test_decimal_numbers(self):
        """Test decimal arithmetic."""
        assert calculator('2.5 + 3.5') == 6.0
        assert calculator('10.5 / 2') == 5.25

    def test_invalid_characters(self):
        """Test that invalid characters return None."""
        assert calculator('2 + a') is None
        assert calculator('import os') is None
        assert calculator("__import__('os')") is None
        assert calculator("2 + 3; print('hello')") is None

    def test_invalid_syntax(self):
        """Test that invalid syntax returns None."""
        assert calculator('2 + + 3') == 5.0  # This is valid syntax; 2 + (+3)
        assert calculator('(2 + 3') is None
        assert calculator('2 +') is None

    def test_division_by_zero(self):
        """Test division by zero returns None."""
        assert calculator('5 / 0') is None

    def test_quote_removal(self):
        """Test that quotes are removed from expression."""
        assert calculator("'2 + 3'") == 5.0

    def test_calculator_handles_basic_expression(self):
        """Covers Constant/Num and BinOp happy-path."""
        result = calculator('2 + 3 * 4 - 5 / 2')
        assert result == pytest.approx(11.5)

    def test_calculator_supports_unary_plus_and_minus(self):
        """Exercises the UnaryOp branch when the operator *is* allowed."""
        result = calculator('-(-4) + +3')
        assert result == pytest.approx(7.0)

    def test_calculator_rejects_disallowed_binary_operator(self):
        """Triggers the BinOp branch that raises for unsupported operators."""
        assert calculator('2 << 1') is None

    def test_calculator_rejects_disallowed_unary_operator(self):
        """Triggers the UnaryOp branch that raises for unsupported operators."""
        assert calculator('~5') is None

    def test_calculator_rejects_function_calls(self):
        """Exercises the final 'else' branch (unsupported node types)."""
        assert calculator('abs(-5)') is None

    @patch('re.compile')
    def test_calculator_rejects_disallowed_binary_operator_after_regex(self, mock_compile):
        """
        Tests that disallowed binary operators are rejected by eval_node,
        by mocking the regex to allow the operator through.
        """
        # Mock the regex to be permissive
        mock_regex = MagicMock()
        mock_regex.match.return_value = True  # Pretend every expression is valid
        mock_compile.return_value = mock_regex

        assert calculator('2 << 1') is None

    @patch('re.compile')
    def test_calculator_rejects_disallowed_unary_operator_after_regex(self, mock_compile):
        """
        Tests that disallowed unary operators are rejected by eval_node,
        by mocking the regex to allow the operator through.
        """
        mock_regex = MagicMock()
        mock_regex.match.return_value = True
        mock_compile.return_value = mock_regex

        assert calculator('~5') is None

    @patch('re.compile')
    def test_calculator_rejects_function_calls_after_regex(self, mock_compile):
        """
        Tests that function calls are rejected by eval_node,
        by mocking the regex to allow the operator through.
        """
        mock_regex = MagicMock()
        mock_regex.match.return_value = True
        mock_compile.return_value = mock_regex

        assert calculator('abs(-5)') is None


class TestSearchWeb:
    """Tests for the search_web tool."""

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_basic(self, mock_ddgs, mock_sleep):
        """Test basic web search."""
        mock_results = [
            {'title': 'Result 1', 'href': 'https://example.com/1', 'body': 'Description 1'},
            {'title': 'Result 2', 'href': 'https://example.com/2', 'body': 'Description 2'}
        ]
        mock_ddgs.return_value.text.return_value = iter(mock_results)

        result = search_web("test query", max_results=2)

        assert '# Search Results for: test query' in result
        assert 'Result 1' in result
        assert 'Result 2' in result
        assert 'https://example.com/1' in result
        assert 'Description 1' in result

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_empty_query(self, mock_ddgs, mock_sleep):
        """Test search with empty query."""
        result = search_web("")
        assert 'ERROR' in result
        assert 'cannot be empty' in result

        result = search_web("   ")
        assert 'ERROR' in result

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_max_results_validation(self, mock_ddgs, mock_sleep):
        """Test max_results validation."""
        mock_results = [
            {'title': 'Result 1', 'href': 'https://example.com/1', 'body': 'Description 1'},
        ]
        mock_ddgs.return_value.text.return_value = iter(mock_results)
        result = search_web('test', max_results=0)
        assert 'Result 1' in result

        # Test capping at 20
        mock_ddgs.return_value.text.return_value = iter([])
        result = search_web("test", max_results=100)
        mock_ddgs.return_value.text.assert_called_with("test", max_results=20)

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_no_results(self, mock_ddgs, mock_sleep):
        """Test web search with no results."""
        mock_ddgs.return_value.text.return_value = iter([])

        result = search_web("nonexistent query")

        assert 'No results found' in result
        assert 'Try:' in result

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_ssl_fallback(self, mock_ddgs, mock_sleep):
        """Test SSL error fallback to verify=False."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs.side_effect = [
            Exception("SSL certificate verify failed"),
            mock_ddgs_instance
        ]
        mock_ddgs_instance.text.return_value = iter([
            {'title': 'Result', 'href': 'https://example.com', 'body': 'Description'}
        ])

        result = search_web("test query")

        assert 'Result' in result
        assert mock_ddgs.call_count == 2

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_non_ssl_error_reraised(self, mock_ddgs, mock_sleep):
        """Test that a non-SSL error is re-raised and caught by the outer try-except."""
        mock_ddgs.side_effect = Exception("A non-SSL error")

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_non_ssl_error_reraised(self, mock_ddgs, mock_sleep):
        """Test that a non-SSL error is re-raised and caught by the outer try-except."""
        mock_ddgs.side_effect = Exception("A non-SSL error")

        result = search_web("test query")

        assert "error: search failed - a non-ssl error" in result.lower()

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_rate_limit_error(self, mock_ddgs, mock_sleep):
        """Test rate limit error handling."""
        mock_ddgs.return_value.text.side_effect = Exception("Ratelimit exceeded")

        result = search_web("test query")

        assert 'ERROR' in result
        assert 'rate limit' in result.lower()

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_timeout_error(self, mock_ddgs, mock_sleep):
        """Test timeout error handling."""
        mock_ddgs.return_value.text.side_effect = Exception("Request timeout")

        result = search_web("test query")

        assert 'ERROR' in result
        assert 'timeout' in result.lower() or 'timed out' in result.lower()

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_generic_error(self, mock_ddgs, mock_sleep):
        """Test generic error handling."""
        mock_ddgs.return_value.text.side_effect = Exception("Unknown error")

        result = search_web("test query")

        assert 'ERROR' in result
        assert 'unknown error' in result.lower()

    @patch('time.sleep')
    @patch('ddgs.DDGS')
    def test_search_web_import_error(self, mock_ddgs, mock_sleep):
        """Test missing duckduckgo-search library."""
        with patch('ddgs.DDGS', side_effect=ImportError("No module named 'duckduckgo_search'")):
            # Need to reload or handle differently
            result = search_web("test")
        assert 'ERROR: Required library `ddgs` not installed' in result


class TestDownloadFile:
    """Tests for the download_file tool."""

    def test_download_file_import_error(self):
        """Test download_file with ImportError."""
        import sys
        with patch.dict('sys.modules', {'requests': None}):
            result = download_file("https://example.com/file.pdf")
            assert 'ERROR: Required lib `requests` not installed' in result

    @patch('requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf', 'Content-Length': '1024'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'SUCCESS' in result
        assert 'file.pdf' in result
        assert 'application/pdf' in result

        # Extract path and cleanup
        if 'Saved to:' in result:
            from pathlib import Path
            path = Path(result.split('Saved to:')[1].split('\n')[0].strip())
            if path.exists():
                path.unlink()

    @patch('requests.get')
    def test_download_file_empty_url(self, mock_get):
        """Test download with empty URL."""
        result = download_file("")
        assert 'ERROR' in result
        assert 'cannot be empty' in result

    @patch('requests.get')
    def test_download_file_invalid_url_scheme(self, mock_get):
        """Test download with invalid URL scheme."""
        result = download_file("ftp://example.com/file.pdf")
        assert 'ERROR' in result
        assert 'must start with http' in result

    @patch('requests.get')
    def test_download_file_invalid_url_format(self, mock_get):
        """Test download with malformed URL."""
        result = download_file("http://")
        assert 'ERROR' in result
        assert 'Invalid URL' in result

    @patch('requests.get')
    def test_download_file_custom_filename(self, mock_get):
        """Test download with custom filename."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf", save_filename="custom.pdf")

        assert 'SUCCESS' in result
        assert 'custom.pdf' in result

        # Cleanup
        if 'Saved to:' in result:
            from pathlib import Path
            path = Path(result.split('Saved to:')[1].split('\n')[0].strip())
            if path.exists():
                path.unlink()

    @patch('requests.get')
    def test_download_file_sanitizes_filename(self, mock_get):
        """Test that invalid characters in filename are sanitized."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf", save_filename="bad<>name.pdf")

        assert 'SUCCESS' in result
        # Should have sanitized the filename

        if 'Saved to:' in result:
            from pathlib import Path
            path = Path(result.split('Saved to:')[1].split('\n')[0].strip())
            if path.exists():
                path.unlink()

    @patch('requests.get')
    def test_download_file_403_error(self, mock_get):
        """Test download with 403 Forbidden error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.reason = 'Forbidden'
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '403' in result
        assert 'forbidden' in result.lower()

    @patch('requests.get')
    def test_download_file_404_error(self, mock_get):
        """Test download with 404 Not Found error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '404' in result

    @patch('requests.get')
    def test_download_file_429_error(self, mock_get):
        """Test download with 429 Rate Limit error."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.reason = 'Too Many Requests'
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '429' in result
        assert 'rate limit' in result.lower()

    @patch('requests.get')
    def test_download_file_file_too_large_header(self, mock_get):
        """Test download with file size exceeding limit in headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': str(101 * 1024 * 1024)}  # 101 MB
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'too large' in result.lower()

    @patch('requests.get')
    def test_download_file_file_too_large_during_download(self, mock_get):
        """Test download aborted when file exceeds size during download."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        # Create chunks that exceed 100MB
        large_chunk = b'x' * (10 * 1024 * 1024)  # 10MB chunks
        mock_response.iter_content.return_value = [large_chunk] * 11  # 110MB total
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '100 MB' in result

    @patch('requests.get')
    def test_download_file_timeout(self, mock_get):
        """Test download timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'timed out' in result.lower()

    @patch('requests.get')
    def test_download_file_connection_error(self, mock_get):
        """Test download connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'Connection failed' in result

    @patch('requests.get')
    def test_download_file_generic_400_error(self, mock_get):
        """Test download with a generic 4xx error."""
        mock_response = Mock()
        mock_response.status_code = 418
        mock_response.reason = "I'm a teapot"
        mock_get.return_value = mock_response

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '418' in result
        assert "I'm a teapot" in result

    @patch('requests.get')
    def test_download_file_request_exception(self, mock_get):
        """Test download with a generic RequestException."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Some request error")

        result = download_file("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'Download failed' in result
        assert 'Some request error' in result

    def test_download_file_urlparse_exception(self):
        """Test that an exception during URL parsing is handled."""
        # urlparse on non-string type can raise an exception
        with patch('urllib.parse.urlparse', side_effect=Exception("mocked exception")):
            result = download_file("http://example.com")
            assert 'ERROR: Invalid URL format' in result
            assert 'mocked exception' in result

    @patch('requests.get')
    def test_download_file_no_filename_in_url(self, mock_get):
        """Test download from a URL with no filename like http://example.com/"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response

        result = download_file("http://example.com/")

        assert 'SUCCESS' in result
        assert 'downloaded_file' in result

        # Cleanup
        if 'Saved to:' in result:
            from pathlib import Path
            path = Path(result.split('Saved to:')[1].split('\n')[0].strip())
            if path.exists():
                path.unlink()

    @patch('requests.get')
    def test_download_file_with_headers(self, mock_get):
        """Test that download includes proper browser-like headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response

        download_file("https://example.com/file.pdf")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert 'headers' in call_kwargs
        assert 'User-Agent' in call_kwargs['headers']
        assert 'Mozilla' in call_kwargs['headers']['User-Agent']
        assert 'Chrome' in call_kwargs['headers']['User-Agent']

        # Cleanup
        from pathlib import Path
        path = Path('/tmp/kodeagent_file.pdf')
        if path.exists():
            path.unlink()


class TestReadWebpage:
    """Tests for the read_webpage tool."""

    @patch('requests.get')
    def test_read_webpage_success(self, mock_get):
        """Test successful webpage reading."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.content = b'''
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <main>
                        <h1>Main Content</h1>
                        <p>This is the main content.</p>
                    </main>
                    <nav>Navigation</nav>
                    <footer>Footer</footer>
                </body>
            </html>
        '''
        mock_get.return_value = mock_response

        result = read_webpage('https://example.com/page')

        assert 'Test Page' in result
        assert 'Main Content' in result
        assert 'This is the main content' in result
        assert 'Navigation' not in result  # Should be removed
        assert 'Footer' not in result  # Should be removed

    @patch('requests.get')
    def test_read_webpage_empty_url(self, mock_get):
        """Test reading with empty URL."""
        result = read_webpage("")
        assert 'ERROR' in result
        assert 'cannot be empty' in result

    @patch('requests.get')
    def test_read_webpage_invalid_url_scheme(self, mock_get):
        """Test reading with invalid URL scheme."""
        result = read_webpage("ftp://example.com")
        assert 'ERROR' in result
        assert 'must start with http' in result

    @patch('requests.get')
    def test_read_webpage_invalid_url_format(self, mock_get):
        """Test reading with malformed URL."""
        result = read_webpage("http://")
        assert 'ERROR' in result
        assert 'Invalid URL' in result

    @patch('requests.get')
    def test_read_webpage_pdf_url(self, mock_get):
        """Test that PDF URLs are rejected with helpful message."""
        result = read_webpage("https://example.com/document.pdf")
        assert 'ERROR' in result
        assert 'document file' in result
        assert 'extract_as_markdown' in result

    @patch('requests.get')
    def test_read_webpage_max_length_validation(self, mock_get):
        """Test max_length validation and clamping."""
        # Properly mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        # Use actual bytes, not MagicMock
        mock_response.content = b'<html><head><title>Test</title></head><body><main>Test content here</main></body></html>'
        mock_get.return_value = mock_response

        # Test that values below 100 are clamped to 100 (no error)
        result = read_webpage('https://example.com', max_length=50)
        assert 'ERROR' not in result
        assert 'Test content' in result

    @patch('requests.get')
    def test_read_webpage_max_length_truncation(self, mock_get):
        """Test content truncation with max_length."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        long_content = 'A' * 10000
        mock_response.content = f'<html><body><main>{long_content}</main></body></html>'.encode()
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com", max_length=500)

        assert 'truncated' in result.lower()
        assert len(result) < 1000  # Should be truncated

    @patch('requests.get')
    def test_read_webpage_403_error(self, mock_get):
        """Test 403 Forbidden error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.reason = 'Forbidden'
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert '403' in result
        assert 'blocking automated access' in result

    @patch('requests.get')
    def test_read_webpage_404_error(self, mock_get):
        """Test 404 Not Found error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = 'Not Found'
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert '404' in result

    @patch('requests.get')
    def test_read_webpage_non_html_content(self, mock_get):
        """Test non-HTML content type."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert 'does not point to a webpage' in result

    @patch('requests.get')
    def test_read_webpage_pdf_content_type(self, mock_get):
        """Test PDF content type detection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com/file")

        assert 'ERROR' in result
        assert 'PDF' in result
        assert 'extract_as_markdown' in result

    @patch('requests.get')
    def test_read_webpage_empty_content(self, mock_get):
        """Test webpage with no extractable content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.content = b'<html><body></body></html>'
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert 'Could not extract meaningful content' in result

    @patch('requests.get')
    def test_read_webpage_timeout(self, mock_get):
        """Test timeout error."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert 'timed out' in result.lower()

    @patch('requests.get')
    def test_read_webpage_connection_error(self, mock_get):
        """Test connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert 'Could not connect' in result

    @patch('requests.get')
    def test_read_webpage_503_error(self, mock_get):
        """Test 503 Service Unavailable error."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.reason = 'Service Unavailable'
        mock_get.return_value = mock_response

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert '503' in result
        assert 'Service unavailable' in result

    @patch('requests.get')
    def test_read_webpage_request_exception(self, mock_get):
        """Test generic RequestException."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Some other request error")

        result = read_webpage("https://example.com")

        assert 'ERROR' in result
        assert 'Request failed' in result
        assert 'Some other request error' in result

    def test_read_webpage_urlparse_exception(self):
        """Test that an exception during URL parsing is handled."""
        with patch('urllib.parse.urlparse', side_effect=Exception("mocked exception")):
            result = read_webpage("http://example.com")
            assert 'ERROR: Invalid URL format' in result
            assert 'mocked exception' in result

    @patch('requests.get')
    def test_read_webpage_no_body(self, mock_get):
        """Test webpage with no body tag."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.content = b'<html><head><title>Test Page</title></head></html>'
        mock_get.return_value = mock_response

        result = read_webpage('https://example.com/page')

        assert 'Test Page' in result
        assert 'ERROR' not in result

    @patch('requests.get')
    def test_read_webpage_with_browser_headers(self, mock_get):
        """Test that proper browser-like headers are sent."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.content = b'<html><body><main>Content</main></body></html>'
        mock_get.return_value = mock_response

        read_webpage("https://example.com")

        call_kwargs = mock_get.call_args[1]
        headers = call_kwargs['headers']
        assert 'Mozilla' in headers['User-Agent']
        assert 'Chrome' in headers['User-Agent']
        assert headers['Accept-Language'] == 'en-US,en;q=0.9'
        assert 'Referer' in headers


class TestExtractAsMarkdown:
    """Tests for the extract_as_markdown tool."""

    @patch('markitdown.MarkItDown')
    def test_extract_pdf_success(self, mock_markitdown):
        """Test successful PDF extraction."""
        mock_result = Mock()
        mock_result.text_content = "# PDF Content\n\nThis is extracted text."
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.pdf")

        assert '# Extracted Content' in result
        assert 'PDF Content' in result
        assert 'This is extracted text' in result
        assert 'PDF' in result

    @patch('markitdown.MarkItDown')
    def test_extract_empty_input(self, mock_markitdown):
        """Test extraction with empty input."""
        result = extract_as_markdown("")
        assert 'ERROR' in result
        assert 'cannot be empty' in result

    @patch('markitdown.MarkItDown')
    def test_extract_invalid_url(self, mock_markitdown):
        """Test extraction with invalid URL."""
        result = extract_as_markdown("http://")
        assert 'ERROR' in result
        assert 'Invalid URL' in result

    @patch('markitdown.MarkItDown')
    @patch('pathlib.Path')
    def test_extract_file_not_found(self, mock_path, mock_markitdown):
        """Test extraction with non-existent file."""
        mock_path.return_value.exists.return_value = False
        result = extract_as_markdown('/path/to/nonexistent.pdf')

        assert 'ERROR' in result
        assert 'not found' in result

    @patch('markitdown.MarkItDown')
    @patch('pathlib.Path')
    def test_extract_path_is_not_file(self, mock_path, mock_markitdown):
        """Test extraction when path is not a file (e.g., a directory)."""
        mock_path_instance = mock_path.return_value
        mock_path_instance.exists.return_value = True
        mock_path_instance.is_file.return_value = False

        result = extract_as_markdown('/path/to/directory.pdf')

        assert 'ERROR: Path is not a file' in result

    def test_extract_urlparse_exception(self):
        """Test that an exception during URL parsing is handled."""
        with patch('urllib.parse.urlparse', side_effect=Exception("mocked exception")):
            result = extract_as_markdown("http://example.com/file.pdf")
            assert 'ERROR: Invalid URL format' in result
            assert 'mocked exception' in result

    def test_extract_import_error(self):
        """Test that ImportError is handled when markitdown is not installed."""
        import sys
        with patch.dict('sys.modules', {'markitdown': None}):
            result = extract_as_markdown("http://example.com/file.pdf")
            assert 'ERROR: Required lib `markitdown` is missing' in result

    @patch('markitdown.MarkItDown')
    def test_extract_memory_error(self, mock_markitdown):
        """Test MemoryError handling."""
        mock_markitdown.return_value.convert.side_effect = MemoryError("Out of memory")

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'out of memory' in result.lower()

    @patch('markitdown.MarkItDown')
    def test_extract_generic_exception(self, mock_markitdown):
        """Test generic exception handling."""
        mock_markitdown.return_value.convert.side_effect = Exception("Some other error")

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'some other error' in result.lower()

    @patch('markitdown.MarkItDown')
    def test_extract_with_excessive_whitespace(self, mock_markitdown):
        """Test that excessive whitespace is cleaned from the output."""
        mock_result = Mock()
        mock_result.text_content = "This   is a    test.\n\n\n\nThis is another line."
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'This  is a  test.' in result
        assert '\n\n\nThis is another line.' in result

    @patch('markitdown.MarkItDown')
    def test_extract_unsupported_format(self, mock_markitdown):
        """Test extraction with unsupported file format."""
        result = extract_as_markdown("https://example.com/file.txt")

        assert 'ERROR' in result
        assert 'Unsupported file format' in result
        assert '.txt' in result

    @patch('markitdown.MarkItDown')
    def test_extract_html_suggests_read_webpage(self, mock_markitdown):
        """Test that HTML files suggest using read_webpage."""
        result = extract_as_markdown("https://example.com/page.html")

        assert 'ERROR' in result
        assert 'read_webpage' in result

    @patch('markitdown.MarkItDown')
    def test_extract_max_length_validation(self, mock_markitdown):
        """Test max_length validation."""
        mock_result = Mock()
        mock_result.text_content = "Test content"
        mock_markitdown.return_value.convert.return_value = mock_result

        # Values below 100 should be clamped to 100 (no error)
        result = extract_as_markdown("https://example.com/file.pdf", max_length=50)
        assert 'ERROR' not in result
        assert 'Test content' in result

        # Values above 1000000 should be clamped to 1000000 (no error)
        result = extract_as_markdown("https://example.com/file.pdf", max_length=2000000)
        assert 'ERROR' not in result

    @patch('markitdown.MarkItDown')
    def test_extract_max_length_truncation(self, mock_markitdown):
        """Test content truncation with max_length."""
        mock_result = Mock()
        mock_result.text_content = "A" * 10000
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.pdf", max_length=500)

        assert 'truncated' in result.lower()
        assert '10,000' in result or '10000' in result

    @patch('markitdown.MarkItDown')
    def test_extract_403_error(self, mock_markitdown):
        """Test 403 error handling."""
        mock_markitdown.return_value.convert.side_effect = Exception("403 Forbidden")

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '403' in result
        assert 'download_file' in result

    @patch('markitdown.MarkItDown')
    def test_extract_404_error(self, mock_markitdown):
        """Test 404 error handling."""
        mock_markitdown.return_value.convert.side_effect = Exception("404 Not Found")

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert '404' in result

    @patch('markitdown.MarkItDown')
    def test_extract_timeout_error(self, mock_markitdown):
        """Test timeout error handling."""
        mock_markitdown.return_value.convert.side_effect = Exception('Request timeout')
        result = extract_as_markdown('https://example.com/file.pdf')

        assert 'ERROR' in result
        assert 'timeout' in result.lower() or 'timed out' in result.lower()

    @patch('markitdown.MarkItDown')
    def test_extract_pdf_specific_error(self, mock_markitdown):
        """Test PDF-specific error handling."""
        mock_markitdown.return_value.convert.side_effect = Exception("PDF extraction failed")

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'PDF' in result

    @patch('markitdown.MarkItDown')
    def test_extract_empty_content(self, mock_markitdown):
        """Test extraction with empty content."""
        mock_result = Mock()
        mock_result.text_content = ''
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.pdf")

        assert 'ERROR' in result
        assert 'No content could be extracted' in result

    @patch('markitdown.MarkItDown')
    @patch('pathlib.Path')
    def test_extract_local_file_with_size(self, mock_path, mock_markitdown):
        """Test extraction from local file shows file size."""
        mock_path.return_value.exists.return_value = True
        MOCKED_FILE_SIZE = 2048  # KB
        mock_path.return_value.stat.return_value.st_size = MOCKED_FILE_SIZE
        mock_path.return_value.suffix = '.pdf'
        mock_result = Mock()
        mock_result.text_content = 'File content'
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown('/path/to/file.pdf')

        assert 'File:' in result
        assert f'{MOCKED_FILE_SIZE / 1024} KB' in result

    @patch('markitdown.MarkItDown')
    def test_extract_docx_format(self, mock_markitdown):
        """Test DOCX file extraction."""
        mock_result = Mock()
        mock_result.text_content = "DOCX content"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.docx")

        assert 'DOCX content' in result
        assert 'DOCX' in result

    @patch('markitdown.MarkItDown')
    def test_extract_xlsx_format(self, mock_markitdown):
        """Test XLSX file extraction."""
        mock_result = Mock()
        mock_result.text_content = "Spreadsheet data"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.xlsx")

        assert 'Spreadsheet data' in result
        assert 'XLSX' in result

    @patch('markitdown.MarkItDown')
    def test_extract_pptx_format(self, mock_markitdown):
        """Test PPTX file extraction."""
        mock_result = Mock()
        mock_result.text_content = "Presentation content"
        mock_markitdown.return_value.convert.return_value = mock_result

        result = extract_as_markdown("https://example.com/file.pptx")

        assert 'Presentation content' in result
        assert 'PPTX' in result


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

    def test_search_arxiv_import_error(self):
        """Test arXiv search with ImportError."""
        import sys
        with patch.dict('sys.modules', {'arxiv': None}):
            result = search_arxiv("test")
            assert 'An error occurred' in result

    def test_search_wikipedia_import_error(self):
        """Test Wikipedia search with ImportError."""
        import sys
        with patch.dict('sys.modules', {'wikipedia': None}):
            result = search_wikipedia("test")
            assert '`wikipedia` was not found' in result

class TestTranscribeYoutube:
    """Tests for the transcribe_youtube tool."""

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_transcribe_youtube_success(self, mock_api):
        """Test successful YouTube transcript retrieval."""
        mock_snippet1 = Mock()
        mock_snippet1.text = 'Hello world'
        mock_snippet2 = Mock()
        mock_snippet2.text = 'This is a test'

        mock_transcript = Mock()
        mock_transcript.snippets = [mock_snippet1, mock_snippet2]

        mock_api.return_value.fetch.return_value = mock_transcript

        result = transcribe_youtube('aBc4E')

        assert 'Hello world' in result
        assert 'This is a test' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_transcribe_youtube_disabled(self, mock_api):
        """Test YouTube transcript when subtitles are disabled."""
        from youtube_transcript_api._errors import TranscriptsDisabled

        mock_api.return_value.fetch.side_effect = TranscriptsDisabled("video_id")

        result = transcribe_youtube('aBc4E')

        assert 'ERROR' in result
        assert 'disabled' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_transcribe_youtube_not_found(self, mock_api):
        """Test YouTube transcript when no transcript is found."""
        from youtube_transcript_api._errors import NoTranscriptFound

        mock_api.return_value.fetch.side_effect = NoTranscriptFound(
            'video_id', [], None
        )

        result = transcribe_youtube('aBc4E')

        assert 'ERROR' in result
        assert 'No transcript found' in result

    @patch('youtube_transcript_api.YouTubeTranscriptApi')
    def test_transcribe_youtube_generic_error(self, mock_api):
        """Test YouTube transcript with generic error."""
        mock_api.return_value.fetch.side_effect = Exception('Network error')

        result = transcribe_youtube('aBc4E')

        assert 'ERROR' in result
        assert 'Network error' in result


class TestTranscribeAudio:
    """Tests for the transcribe_audio tool."""

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_transcribe_audio_success(self, mock_getenv, mock_file, mock_post):
        """Test successful audio transcription."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Transcribed audio content'}
        mock_post.return_value = mock_response

        result = transcribe_audio('/path/to/audio.mp3')

        assert result == {'text': 'Transcribed audio content'}
        mock_post.assert_called_once()

        # Verify API call parameters
        call_kwargs = mock_post.call_args[1]
        assert 'Authorization' in call_kwargs['headers']
        assert call_kwargs['headers']['Authorization'] == 'Bearer test_api_key'

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_transcribe_audio_error(self, mock_getenv, mock_file, mock_post):
        """Test audio transcription with API error."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = 'Bad Request'
        mock_post.return_value = mock_response

        result = transcribe_audio('/path/to/audio.mp3')

        assert 'Audio transcription error' in result
        assert '400' in result

    @patch('requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'audio data')
    @patch('os.getenv')
    def test_transcribe_audio_correct_model(self, mock_getenv, mock_file, mock_post):
        """Test that correct model and parameters are used."""
        mock_getenv.return_value = 'test_api_key'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'text': 'Test'}
        mock_post.return_value = mock_response

        transcribe_audio('/path/to/audio.mp3')

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['data']['model'] == 'whisper-v3-turbo'
        assert call_kwargs['data']['temperature'] == '0'
        assert call_kwargs['data']['vad_model'] == 'silero'

    def test_transcribe_audio_import_error(self):
        """Test audio transcription with ImportError."""
        import sys
        with patch.dict('sys.modules', {'requests': None}):
            result = transcribe_audio('/path/to/audio.mp3')
            assert 'Audio transcription error' in result
