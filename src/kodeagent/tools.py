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

    # Create a schema for the function arguments using Pydantic
    signature = inspect.signature(func)
    fields = {name: (param.annotation, ...) for name, param in signature.parameters.items()}

    # Add metadata to the function
    wrapper.name = func.__name__
    wrapper.description = textwrap.dedent(func.__doc__).strip() if func.__doc__ else ''
    wrapper.args_schema = pyd.create_model(func.__name__, **fields)

    return wrapper


@tool
def calculator(expression: str) -> Union[float, None]:
    """
    A simple calculator tool that can evaluate basic arithmetic expressions.
    The expression must contain only the following allowed mathematical symbols:
    digits, +, -, *, /, ., (, )

    The ^ symbol, for example, is not allowed. For exponent, use **.
    In case the expression has any invalid symbol, the function returns `None`.

    Args:
        expression (str): The arithmetic expression as a string.

    Returns:
        The numerical result or `None` in case an incorrect arithmetic expression is provided
         or any other error occurs.

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
            elif isinstance(node, ast.Num):  # Python 3.7 and earlier
                return node.n
            elif isinstance(node, ast.BinOp):  # Binary operation
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f'Operator {op_type} not allowed')
                left = eval_node(node.left)
                right = eval_node(node.right)
                return allowed_operators[op_type](left, right)
            elif isinstance(node, ast.UnaryOp):  # Unary operation
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError(f'Operator {op_type} not allowed')
                operand = eval_node(node.operand)
                return allowed_operators[op_type](operand)
            else:
                raise ValueError(f'Unsupported node type: {type(node)}')

        result = eval_node(node)
        return float(result)

    except Exception:
        return None


@tool
def search_web(query: str, max_results: int = 10, show_description: bool = False) -> str:
    """
    Search the Web using DuckDuckGo. The input should be a search query.
    Use this tool when you need to answer questions about current events and general web search.
    It returns (as Markdown text) the top search results with titles, links, and optional
    descriptions.
    NOTE: The returned URLs can be visited using the `extract_as_markdown` tool to retrieve
    the contents the respective pages.

    Args:
        query: The query string.
        max_results: Maximum no. of search results (links) to return.
        show_description: If `True`, includes the description of each search result.
         Default is `False`.

    Returns:
         The search results.
    """
    import time
    import random

    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError(
            '`ddgs` was not found! Please run `pip install ddgs`.'
        ) from e

    # Note: In general, `verify` should be `True`
    # In some cases, DDGS may fail because of proxy or something else;
    # can set it to `False` but generally not recommended
    results = DDGS(verify=False).text(query, max_results=max_results)
    # DDGS throws a rate limit error
    time.sleep(random.uniform(1.5, 3.5))
    if len(results) == 0:
        return 'No results found! Try a less restrictive/shorter query.'

    if not show_description:
        # If descriptions are not needed, only return titles and links
        results = [f"[{result['title']}]({result['href']})" for result in results]
    else:
        results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
    return '## Search Results\n\n' + '\n\n'.join(results)


@tool
def download_file(url: str) -> str:
    """
    Use this tool only when `extract_as_markdown` cannot be used.
    Download a file from the Web and save it locally on the disk. This tool is to be used to
    when the goal is to download a file (binary) rather than retrieve its contents as text.

    Args:
        url: The URL pointing to the file (must be a correct URL).

    Return:
        Path to the locally saved file or error messages in case of any exception.
    """
    import os
    import requests
    import tempfile

    response = requests.get(url, timeout=20, stream=True, headers={'user-agent': 'kodeagent/0.0.1'})
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file_path = tmp_file.name
        if os.name == 'nt':
            tmp_file_path = tmp_file_path.replace('\\', '/')
    return tmp_file_path


@tool
def extract_file_contents_as_markdown(
        url_or_file_path: str,
        scrub_links: bool = True,
        max_length: int = None
) -> str:
    """
    Always use this tool to extract the contents of HTML files (.html), PDF files (.pdf),
    Word documents (.docx), and Excel spreadsheets (.xlsx) as Markdown text. No other file type
    is supported. The input can point to a URL or a local file path.
    The extracted text can be used for analysis with LLMs.
    This tool can directly work with URLs, so no need to download the files using
    `file_download` separately.
    NOTE: The output returned by this function can be long and may involve lots of quote marks.

    Args:
        url_or_file_path: URL or Path to a .html, .pdf, .docx, or .xlsx file.
        scrub_links: Defaults to `True`, which removes all links from the extracted Markdown text.
         Set it to `False` if you want to retain the links in the text.
        max_length: If set, limit the output to the first `max_length` characters. This can be used
         to truncate the output but doing so can also lead to loss of information.

    Returns:
        The content of the file in Markdown format.
    """
    import re
    import mimetypes

    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise ImportError(
            '`markitdown` was not found! Please run `pip install markitdown[pdf,docx,xlsx]`.'
        ) from e

    md = MarkItDown(enable_plugins=False)
    try:
        result = md.convert(url_or_file_path.strip()).text_content

        if mimetypes.guess_type(url_or_file_path)[0] == 'application/pdf':
            # Handling (cid:NNN) occurrences in PDFs
            cid_pattern = re.compile(r'\(cid:(\d+)\)')
            matches = set(cid_pattern.findall(result))
            for cid_num in matches:
                cid_str = f'(cid:{cid_num})'
                result = result.replace(cid_str, chr(int(cid_num) + 29))

        if scrub_links:
            # Remove Markdown links [text](url)
            result = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', r'\1', result)

        if max_length is not None:
            result = result[:max_length]

        return result
    except Exception as e:
        return str(e)


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
    except ImportError as e:
        raise ImportError('`wikipedia` was not found! Please run `pip install wikipedia`.') from e

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
def get_youtube_transcript(video_id: str) -> str:
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
def get_audio_transcript(file_path: str) -> Any:
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
