"""
This module defines the `tool` decorator and a set of built-in tools for KodeAgent.
All tools import necessary dependencies within their function bodies to ensure they are
self-contained and can operate in isolated environments. Similarly, all variables are declared
locally within the functions.
"""
import asyncio
import inspect
import textwrap
from functools import wraps
from typing import (
    Callable,
    Any,
    Union,
    Optional
)

import pydantic as pyd


DEFAULT_TOOLS_IMPORTS = [
    'ast',
    'operator',
    're',
    'time',
    'random',
    'ddgs',
    'pathlib',
    'tempfile',
    'requests',
    'markitdown',
    'bs4',
    'wikipedia',
    'arxiv',
    'youtube_transcript_api',
    'urllib.parse',
    'os',
    'base64',
    'litellm',
]
"""List of default modules (stdlib and third-party) to be available in tools."""


def tool(func: Callable) -> Callable:
    """
    A decorator to convert any Python function into a tool with additional metadata.
    Tooling based on async functions is not supported.

    Args:
        func (Callable): The function to be converted into a tool.

    Returns:
        Callable: The decorated function with additional metadata.
    """
    if asyncio.iscoroutinefunction(func):
        raise ValueError(
            'Tooling based on async functions is not supported. Please remove `async` from'
            f' the signature of the `{func.__name__}` function or remove the `@tool` decorator.'
        )

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    signature = inspect.signature(func)
    fields = {}

    for name, param in signature.parameters.items():
        if param.default is inspect.Parameter.empty:
            # Required parameter: (Type, ...)
            fields[name] = (param.annotation, ...)
        else:
            # Optional parameter: (Type, default_value)
            fields[name] = (param.annotation, param.default)

    # Add metadata to the function
    wrapper.name = func.__name__
    wrapper.description = textwrap.dedent(func.__doc__).strip() if func.__doc__ else ''
    wrapper.args_schema = pyd.create_model(func.__name__, **fields)

    return wrapper


@tool
def calculator(expression: str) -> Union[float, None]:
    """
    A simple calculator tool that can evaluate basic arithmetic expressions.

    Examples:
        - "2 + 2" returns 4.0
        - "2 ** 3" returns 8.0 (exponentiation)

    Supported operations: +, -, *, /, ** (exponent), parentheses
    Note: Use ** for exponents, not ^
    In case the expression has any invalid symbol, the function returns `None`.

    Args:
        expression (str): The arithmetic expression as a string.

    Returns:
        The numerical result or `None` if the expression is invalid.

    Raises:
        ValueError: If the expression contains invalid characters.
    """
    import ast
    import operator
    import re

    # Clean the expression
    expression = expression.replace("'", "").replace('^', '**')

    # Define a regex pattern for valid mathematical expressions
    calculator_regex = re.compile(r'^[\d+\-*/().\s]+$')

    if calculator_regex.match(expression) is None:
        return None

    try:
        # Parse the expression into an AST
        node = ast.parse(expression, mode='eval').body

        # Define allowed operations
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def eval_node(node):
            """Recursively evaluate the AST node."""
            if isinstance(node, ast.Constant):  # Python 3.8+
                return node.value
            if isinstance(node, ast.BinOp):  # Binary operation
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f'Operator {op_type} not allowed')
                left = eval_node(node.left)
                right = eval_node(node.right)
                return allowed_operators[op_type](left, right)
            if isinstance(node, ast.UnaryOp):  # Unary operation
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f'Operator {op_type} not allowed')
                operand = eval_node(node.operand)
                return allowed_operators[op_type](operand)

            raise ValueError(f'Unsupported node type: {type(node)}')

        result = eval_node(node)
        return float(result)

    except Exception:
        return None


@tool
def search_web(query: str, max_results: int = 10) -> str:
    """
    Search the web using DuckDuckGo and return top results with titles and links.
    Use this when you need current information, news, or general web search.

    The results include clickable links that can be visited using the `read_webpage` tool
    to get the full content of any page.

    Examples:
        - "latest news about AI" - finds recent AI news articles
        - "Python tutorial" - finds Python learning resources
        - "weather in Tokyo" - finds weather information

    Args:
        query: Search terms (keep it concise, 2-5 words work best).
        max_results: Number of results to return (default 10, min 1, max 20).

    Returns:
        Markdown formatted search results with titles, URLs, and snippets, or an error message.

    Next Step:
        For any question requiring a specific, accurate answer, always use `read_webpage`
        on a result URL.
    """
    import time
    import random
    from datetime import datetime

    today = datetime.now().strftime('%Y-%m-%d')

    try:
        from ddgs import DDGS
    except ImportError:
        return 'ERROR: Required library `ddgs` not installed. Install with: `pip install ddgs`'

    # Validate inputs
    if not query or not query.strip():
        return 'ERROR: Search query cannot be empty.'

    query = query.strip()

    if max_results < 1:
        max_results = 1
    elif max_results > 20:
        max_results = 20  # Cap at reasonable limit

    try:
        # Use verify=True for security, but handle SSL errors gracefully
        try:
            engine = DDGS(timeout=20)
            results = list(engine.text(query, max_results=max_results))
        except Exception as ssl_error:
            # Fallback to verify=False only if SSL fails
            if 'SSL' in str(ssl_error) or 'certificate' in str(ssl_error).lower():
                engine = DDGS(timeout=20, verify=False)
                results = list(engine.text(query, max_results=max_results))
            else:
                raise

        # Small random delay to be respectful
        time.sleep(random.uniform(0.5, 1.5))

        if not results or len(results) == 0:
            return (
                f"No results found for '{query}'. Try:\n"
                "- Using fewer, more common words\n"
                "- Removing special characters\n"
                "- Being less specific"
            )

        # Format results as clean Markdown
        output = f'# Search Discovery (System Date: {today})\n\n'
        output += f'Found {len(results)} result(s)\n\n'

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', '')
            body = result.get('body', '')

            # Clean up title and body
            title = title.replace('\n', ' ').strip()
            body = body.replace('\n', ' ').strip()

            output += f'## {i}. {title}\n'
            output += f'**URL:** {url}\n'
            output += f'**Snippet:** {body}\n\n'

        # output += "\n**Next Step:** Use the 'read_webpage' tool with any URL above to get full page content."
        output += '> **Note:** These are summaries. Use `read_webpage` with a URL to verify facts.'

        return output

    except ImportError:
        return 'ERROR: Required library `ddgs` not installed. Install with: `pip install ddgs`'
    except Exception as e:
        error_msg = str(e).lower()
        if 'ratelimit' in error_msg:
            return 'ERROR: DuckDuckGo rate limit reached. Please wait 30s before searching again.'
        if 'timeout' in error_msg:
            return 'ERROR: Search request timed out. Please try again with a simpler query.'

        return f'ERROR: Search failed - {error_msg}'


@tool
def download_file(url: str, save_filename: str = None) -> str:
    """
    Download a file from the internet and save it locally.
    Use this for downloading images, PDFs, data files, or any binary content.

    For reading webpage content as text, use 'read_webpage' instead.
    For extracting content from PDFs/DOCX/XLSX, use 'extract_as_markdown' instead.

    Examples:
        - Download an image: url="https://example.com/photo.jpg"
        - Download a dataset: url="https://example.com/data.csv"
        - Download a PDF: url="https://example.com/paper.pdf"

    Args:
        url: The complete URL of the file to download (must start with http:// or https://).
        save_filename: Optional custom filename. If not provided, uses the filename from URL.

    Returns:
        Success message with file path, or error message if download fails.
    """
    import re
    import tempfile
    from pathlib import Path
    from urllib.parse import urlparse, unquote

    try:
        import requests
    except ImportError:
        return 'ERROR: Required lib `requests` not installed. Install with: `pip install requests`'

    # Validate URL
    if not url or not url.strip():
        return 'ERROR: URL cannot be empty.'

    url = url.strip()

    if not url.startswith(('http://', 'https://')):
        return 'ERROR: URL must start with http:// or https://'

    # Validate URL format
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return 'ERROR: Invalid URL format - missing domain name.'
    except Exception as e:
        return f'ERROR: Invalid URL format - {str(e)}'

    # Determine filename
    if save_filename:
        # Sanitize custom filename
        save_filename = re.sub(r'[<>:"/\\|?*]', '_', save_filename)
    else:
        # Extract from URL
        path = unquote(parsed.path)
        save_filename = Path(path).name
        if not save_filename or save_filename == '/':
            save_filename = 'downloaded_file'

    # Browser-like headers to avoid 403 errors
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
    }

    try:
        # Make request with streaming for large files
        response = requests.get(
            url,
            headers=headers,
            timeout=30,
            stream=True,
            allow_redirects=True
        )

        # Check for errors
        if response.status_code == 403:
            return (
                f'ERROR: Access forbidden (403) for {url}\n'
                'The website is blocking automated access. Possible reasons:\n'
                '- Website requires login/authentication\n'
                '- Website blocks bots/scrapers\n'
                '- Geographic restrictions\n'
                'Try accessing the URL in a browser first to verify it works.'
            )
        if response.status_code == 404:
            return f'ERROR: File not found (404) at {url}'
        if response.status_code == 429:
            return 'ERROR: Too many requests (429). The server is rate limiting. Wait and retry.'
        if response.status_code >= 400:
            return f'ERROR: HTTP {response.status_code} - {response.reason}\nURL: {url}'

        response.raise_for_status()

        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > 100:
                return (
                    f'ERROR: File too large ({size_mb:.1f} MB). Maximum supported size is 100 MB.'
                )

        # Create temp file with proper extension
        file_ext = Path(save_filename).suffix
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_ext,
                prefix='kodeagent_'
        ) as tmp_file:
            # Download in chunks
            downloaded_size = 0
            chunk_size = 8192

            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded_size += len(chunk)

                    # Safety check during download
                    if downloaded_size > 100 * 1024 * 1024:  # 100 MB
                        tmp_file.close()
                        Path(tmp_file.name).unlink()
                        return 'ERROR: File exceeded 100 MB during download. Aborted.'

            tmp_file_path = tmp_file.name

        # Normalize path for cross-platform compatibility
        tmp_file_path = str(Path(tmp_file_path).as_posix())

        # Get actual file size
        actual_size = Path(tmp_file_path).stat().st_size
        size_str = (
            f'{actual_size / 1024:.1f} KB'
            if actual_size < 1024 * 1024
            else f'{actual_size / (1024 * 1024):.1f} MB'
        )

        return (
            f'SUCCESS: File downloaded successfully!\n'
            f'- Saved to: {tmp_file_path}\n'
            f'- Original filename: {save_filename}\n'
            f'- Size: {size_str}\n'
            f'- Content type: {response.headers.get("Content-Type", "unknown")}\n\n'
            f'You can now use this file path with other tools.'
        )

    except requests.exceptions.Timeout:
        return (
            'ERROR: Download timed out after 30 seconds.\n'
            'The file may be too large or the server is slow. Try again.'
        )
    except requests.exceptions.ConnectionError as e:
        return (
            f'ERROR: Connection failed - {str(e)}\nPossible causes:\n'
            '- No internet connection\n'
            '- Invalid domain name\n'
            '- Server is down'
        )
    except requests.exceptions.RequestException as e:
        return f'ERROR: Download failed - {str(e)}'
    except Exception as e:
        return f'ERROR: Unexpected error - {type(e).__name__}: {str(e)}'


@tool
def extract_as_markdown(url_or_path: str, max_length: Optional[int] = None) -> str:
    """
    Extract content from documents (PDF, DOCX, XLSX, PPTX) as Markdown text.
    Works with both URLs and local file paths.

    Supported formats:
    - PDF files (.pdf)
    - Word documents (.docx)
    - Excel spreadsheets (.xlsx)
    - PowerPoint presentations (.pptx)

    For reading HTML web pages, use 'read_webpage' instead (faster and cleaner).

    Examples:
        - Extract from PDF: "https://example.com/paper.pdf"
        - Extract from local file: "/tmp/document.docx"
        - Extract from Excel: "https://example.com/data.xlsx"

    Args:
        url_or_path: URL or file path to a PDF, DOCX, XLSX, or PPTX file.
        max_length: Optional limit on output length in characters. Use this to
                   truncate very long documents (may lose information).

    Returns:
        Document content as Markdown text, or an error message if extraction fails.
    """
    import re
    from pathlib import Path
    from urllib.parse import urlparse

    try:
        from markitdown import MarkItDown
    except ImportError:
        return (
            'ERROR: Required lib `markitdown` is missing. Install with: `pip install markitdown`'
        )

    # Validate input
    if not url_or_path or not url_or_path.strip():
        return 'ERROR: url_or_path cannot be empty.'

    url_or_path = url_or_path.strip()

    # Check if it's a URL or file path
    is_url = url_or_path.startswith(('http://', 'https://'))

    if is_url:
        # Validate URL format
        try:
            parsed = urlparse(url_or_path)
            if not parsed.netloc:
                return 'ERROR: Invalid URL format - missing domain name.'
        except Exception as e:
            return f'ERROR: Invalid URL format - {str(e)}'
    else:
        # Validate file path
        path_obj = Path(url_or_path)
        if not path_obj.exists():
            return f'ERROR: File not found at path: {url_or_path}'

        if not path_obj.is_file():
            return f'ERROR: Path is not a file: {url_or_path}'

    # Determine file type
    if is_url:
        file_ext = Path(urlparse(url_or_path).path).suffix.lower()
    else:
        file_ext = Path(url_or_path).suffix.lower()

    # Check supported formats
    supported_formats = {'.pdf', '.docx', '.xlsx', '.pptx'}
    if file_ext not in supported_formats:
        return (
            f'ERROR: Unsupported file format \'{file_ext}\'\n'
            f'Supported formats: {", ".join(supported_formats)}\n\n'
            f'For HTML web pages, use \'read_webpage\' instead.\n'
            f'For other files, use \'download_file\' first.'
        )

    # Validate max_length
    if max_length is not None:
        if max_length < 100:
            max_length = 100
        elif max_length > 1000000:
            max_length = 1000000  # Cap at 1M chars

    try:
        # Initialize MarkItDown
        md = MarkItDown()

        # Convert document
        try:
            result = md.convert(url_or_path)
            text = result.text_content
        except Exception as convert_error:
            error_str = str(convert_error).lower()

            # Provide helpful error messages
            if '403' in error_str or 'forbidden' in error_str:
                return (
                    'ERROR: Access forbidden (403) when trying to download from URL.\n'
                    'The server is blocking automated access.\n\n'
                    'Solution: Use `download_file` tool first to save it locally, '
                    'then use this tool with the local file path.'
                )
            if '404' in error_str or 'not found' in error_str:
                return f'ERROR: File not found (404) at URL: {url_or_path}'
            if 'timeout' in error_str:
                return 'ERROR: Request timed out. The file may be too large or server is slow.'
            if 'pdf' in error_str and file_ext == '.pdf':
                return (
                    'ERROR: Failed to extract PDF content.\nThe PDF may be:\n'
                    '- Scanned images without OCR text\n'
                    '- Password protected\n'
                    '- Corrupted or malformed\n\n'
                    f'Original error: {str(convert_error)}'
                )

            raise  # Re-raise for generic handling below

        if not text:
            return (
                f'ERROR: No content could be extracted from the {file_ext} file.\n'
                'The file may be empty, corrupted, or contain only images.'
            )

        # Clean up the text
        text = text.strip()

        # Handle PDF-specific issues (CID characters)
        if file_ext == '.pdf':
            # Remove common PDF artifacts
            text = re.sub(r'\(cid:\d+\)', '', text)  # Remove CID references
            text = re.sub(r'\x00', '', text)  # Remove null bytes

        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        text = re.sub(r' {3,}', '  ', text)  # Max 2 spaces

        # Get file info for output
        if is_url:
            source_info = f'URL: {url_or_path}'
        else:
            path_obj = Path(url_or_path)
            file_size = path_obj.stat().st_size
            size_str = (
                f'{file_size / 1024:.1f} KB'
                if file_size < 1024 * 1024
                else f'{file_size / (1024 * 1024):.1f} MB'
            )
            source_info = f'File: {path_obj.name} ({size_str})'

        # Truncate if needed
        original_length = len(text)
        if max_length and len(text) > max_length:
            text = text[:max_length]
            truncation_msg = (
                f'\n\n---\n**[Content truncated from {original_length:,}'
                f' to {max_length:,} characters]**'
            )
        else:
            truncation_msg = ''

        # Format output
        output = f'# Extracted Content\n\n**Source:** {source_info}\n'
        output += f'**Format:** {file_ext.upper()}\n'
        output += f'**Length:** {original_length:,} characters\n\n---\n\n'
        output += text
        output += truncation_msg

        return output

    except ImportError as e:
        return (
            f'ERROR: Missing required library for {file_ext} files.\n'
            f'Install with: `pip install markitdown[pdf]`\nDetails: {str(e)}'
        )
    except MemoryError:
        return (
            'ERROR: File too large to process (out of memory).\n'
            'Try using max_length parameter to limit output size.'
        )
    except Exception as e:
        error_type = type(e).__name__
        return f'ERROR: {error_type} - {str(e)}'


@tool
def read_webpage(url: str, max_length: int = 50000) -> str:
    """
    Read and extract the main text content from HTML web pages as clean Markdown.
    Use this after 'search_web' to read articles, blogs, documentation, or any HTML content.

    This tool intelligently extracts readable text while removing ads, navigation,
    footers, and other clutter. It's optimized for web pages.

    For documents (PDF, DOCX, XLSX), use 'extract_as_markdown' instead.

    Examples:
        - Read a news article: "https://www.bbc.com/news/article"
        - Read documentation: "https://docs.python.org/3/tutorial/"
        - Read a blog post: "https://example.com/blog/post"

    Args:
        url: The complete URL of the webpage (must start with http:// or https://).
        max_length: Maximum characters to return (default 50000). Lower values
                   process faster with small LLMs.

    Returns:
        Clean webpage content as Markdown text, or an error message.
    """
    import re
    from urllib.parse import urlparse

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        missing_lib = 'requests' if 'requests' in str(e) else 'beautifulsoup4'
        return (
            f'ERROR: Required library `{missing_lib}` not installed.\n'
            f'Install with: `pip install {missing_lib}`'
        )

    # Validate URL
    if not url or not url.strip():
        return 'ERROR: URL cannot be empty.'

    url = url.strip()

    if not url.startswith(('http://', 'https://')):
        return 'ERROR: URL must start with http:// or https://'

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return 'ERROR: Invalid URL format - missing domain name.'
    except Exception as e:
        return f'ERROR: Invalid URL format - {str(e)}'

    # Check if URL points to a document file
    path_lower = parsed.path.lower()
    doc_extensions = ('.pdf', '.docx', '.xlsx', '.pptx', '.doc', '.xls', '.ppt')
    if any(path_lower.endswith(ext) for ext in doc_extensions):
        ext = next(ext for ext in doc_extensions if path_lower.endswith(ext))
        return (
            f'ERROR: URL points to a document file ({ext}), not a webpage.\n'
            'Use `extract_as_markdown` tool instead for document files.'
        )

    # Validate max_length
    if max_length < 100:
        max_length = 100
    elif max_length > 50000:
        max_length = 50000  # Cap at 50K chars

    # Browser-like headers to avoid 403 errors
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://www.google.com/',
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20,
            allow_redirects=True
        )

        # Handle HTTP errors
        if response.status_code == 403:
            return (
                f'ERROR: Access forbidden (403) for {url}\n\n'
                'The website is blocking automated access. This could be because:\n'
                '1. The site requires login/authentication\n'
                '2. The site uses anti-bot protection (Cloudflare, etc.)\n'
                '3. Geographic restrictions apply\n'
                '4. The site blocks all automated tools\n\n'
                'Suggestions:\n'
                '- Verify the URL works in your browser\n'
                '- Check if the site has an API\n'
                '- Try a different source for the same information'
            )
        if response.status_code == 404:
            return f'ERROR: Page not found (404). The URL may be incorrect:\n{url}'
        if response.status_code == 429:
            return 'ERROR: Rate limited (429). Too many requests. Wait before retrying.'
        if response.status_code == 503:
            return 'ERROR: Service unavailable (503). The website may be down. Try again later.'
        if response.status_code >= 400:
            return f'ERROR: HTTP {response.status_code} - {response.reason}\nURL: {url}'

        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            if 'application/pdf' in content_type:
                return (
                    'ERROR: URL points to a PDF file, not a webpage.\n'
                    'Use `extract_as_markdown` instead.'
                )
            return (
                f'ERROR: URL does not point to a webpage (Content-Type: {content_type}).\n'
                'For non-HTML content, use `download_file` or `extract_as_markdown`.'
            )

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup([
            'script', 'style', 'nav', 'footer', 'header',
            'aside', 'iframe', 'noscript', 'svg', 'form',
            'button', '[role="navigation"]', '[role="banner"]',
            '[role="complementary"]', '.advertisement', '.ad',
            '.sidebar', '.menu', '.navigation'
        ]):
            element.decompose()

        # Try to find main content area
        for selector in [
            'main', 'article', '[role="main"]', '.main-content',
            '#main-content', '#content', '.post-content',
            '.entry-content', '.article-content', '.page-content'
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text
        text = main_content.get_text(separator='\n', strip=True)

        # Clean up text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n\n'.join(lines)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        if not text:
            return (
                f'ERROR: Could not extract meaningful content from {url}\nThe page may be:\n'
                '- Dynamically loaded with JavaScript (not supported)\n'
                '- Behind a login wall\n'
                '- Empty or malformed'
            )

        # Get page title
        title = soup.title.string if soup.title else 'No title'
        title = title.strip()

        # Truncate if needed
        original_length = len(text)
        if len(text) > max_length:
            text = text[:max_length]
            truncated_msg = (
                f'\n\n---\n**[Content truncated from {original_length:,}'
                f' to {max_length:,} characters]**'
            )
        else:
            truncated_msg = ''

        # Format output
        output = f'# {title}\n\n**Source:** {url}\n'
        output += f'**Length:** {original_length:,} characters\n\n---\n\n'
        output += text
        output += truncated_msg

        return output

    except requests.exceptions.Timeout:
        return 'ERROR: Request timed out after 20s. The website may be slow or unresponsive.'
    except requests.exceptions.ConnectionError:
        return (
            f'ERROR: Could not connect to {url}\nPossible causes:\n'
            '- No internet connection\n'
            '- Invalid domain name\n'
            '- Server is down'
        )
    except requests.exceptions.RequestException as e:
        return f'ERROR: Request failed - {str(e)}'
    except Exception as e:
        return f'ERROR: Unexpected error - {type(e).__name__}: {str(e)}'


@tool
def search_wikipedia(query: str, max_results: Optional[int] = 3) -> str:
    """
    Search Wikipedia (only) and return the top search results as Markdown text.
    The input should be a search query. The output will contain the title, summary, and link
    to the Wikipedia page.

    Args:
        query: The search query string.
        max_results: The max. no. of search results to consider (default 3).

    Returns:
        The search results in Markdown format.
    """
    try:
        import wikipedia
    except ImportError:
        return '`wikipedia` was not found! Please run `pip install wikipedia`'

    try:
        results = wikipedia.search(query, results=max_results)
        if not results:
            return 'No results found! Try a less restrictive/shorter query.'

        markdown_results = []
        for title in results:
            page = wikipedia.page(title)
            markdown_results.append(f"### [{page.title}]({page.url})\n{page.summary}")

        return '\n\n'.join(markdown_results)
    except wikipedia.exceptions.DisambiguationError as de:
        return f'DisambiguationError: Please select an option from {", ".join(de.options)}'


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search for academic papers on arXiv.org. The input is a search query.
    This tool is highly specialized and should be used exclusively for
    finding scientific and academic papers. It returns the top search results
    with the title, authors, summary, and a link to the PDF.

    Args:
        query: The search query string for the paper.
        max_results: The maximum number of search results to return (default is 5).

    Returns:
        The search results in Markdown format or a message indicating no results were found.
    """
    try:
        import arxiv

        # Construct the default API client
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))

        if not results:
            return f'No results found for the query: {query}'

        output = f'## ArXiv Search Results for: {query}\n\n'
        for result in results:
            authors = ', '.join([author.name for author in result.authors])
            output += f'### [{result.title}]({result.pdf_url})\n'
            output += f'**Authors:** {authors}\n'
            output += f'**Abstract:** {result.summary}\n'
            output += f'**Published:** {result.published.strftime("%Y-%m-%d")}\n\n'

        return output

    except Exception as e:
        return f'An error occurred during the arXiv search: {str(e)}'


@tool
def transcribe_youtube(video_id: str) -> str:
    """
    Retrieve the transcript/subtitles for YouTube videos (only). It also works for automatically
    generated subtitles, supports translating subtitles. The input should be a valid YouTube
    video ID. E.g., the URL https://www.youtube.com/watch?v=aBc4E has the video ID `aBc4E`.

    Args:
        video_id: YouTube video ID from the URL.

    Returns:
        The transcript/subtitle of the video, if available.
    """
    from youtube_transcript_api import YouTubeTranscriptApi, _errors as yt_errors

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        transcript_text = ' '.join([item.text for item in transcript.snippets])
    except yt_errors.TranscriptsDisabled:
        transcript_text = (
            '*** ERROR: Could not retrieve a transcript for the video -- subtitles appear to be'
            ' disabled for this video, so this tool cannot help, unfortunately.'
        )
    except yt_errors.NoTranscriptFound:
        return '*** ERROR: No transcript found for this video.'
    except Exception as e:
        return f'*** ERROR: YouTube transcript retrieval failed: {e}'

    return transcript_text


@tool
def transcribe_audio(file_path: str) -> Any:
    """
    Convert audio files to text using OpenAI's Whisper model via Fireworks API.
    The input should be a path to an audio file (e.g., .mp3, .wav, .flac).
    The audio file should be in a format that Whisper supports.

    Args:
        file_path: Local file system path to the audio file.

    Returns:
        The transcript of the audio file as text.
    """
    import os
    try:
        import requests

        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://audio-turbo.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions',
                headers={'Authorization': f'Bearer {os.getenv("FIREWORKS_API_KEY")}'},
                files={'file': f},
                data={
                    'model': 'whisper-v3-turbo',
                    'temperature': '0',
                    'vad_model': 'silero'
                },
                timeout=15,
            )

        if response.status_code == 200:
            return response.json()

        return f'Audio transcription error: {response.status_code}: {response.text}'
    except ImportError:
        return 'Audio transcription error: `requests` library not found. Please install it with `pip install requests`.'


@tool
def generate_image(prompt: str, model_name: str) -> str:
    """
    Generate an image based on a text prompt using the specified model.
    It returns the image URL or the file path of the generated image.

    Args:
        prompt: Text description of the desired image.
        model_name: The name of the image generation model to use.

    Returns:
        The file path or URL of the generated image or error message.
    """
    import base64
    import os

    import litellm

    try:
        response = litellm.image_generation(prompt=prompt, model=model_name)
        image_data = response.data[0]

        # 1. If a URL is provided, return it
        if image_data.url:
            return image_data.url

        # 2. If URL is None, check for b64_json and save it locally
        if hasattr(image_data, 'b64_json') and image_data.b64_json:
            file_path = 'generated_image.png'
            with open(file_path, 'wb') as f:
                f.write(base64.b64decode(image_data.b64_json))
            return os.path.abspath(file_path)

        return 'Error: No image data (URL or Base64) found in response.'
    except Exception as ex:
        return f'Error: Image generation failed: {ex}'


if __name__ == '__main__':
    img_url = generate_image(
        prompt='A futuristic cityscape at sunset, with flying cars and neon lights',
        model_name='gemini/imagen-4.0-generate-001'
    )
    print(f'Generated image URL: {img_url}')
