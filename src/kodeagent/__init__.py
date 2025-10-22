"""KodeAgent: An intelligent code agent"""

from .kodeagent import (
    llm_vision_support,
    Agent,
    ReActAgent,
    CodeActAgent,
    ChatMessage,
    ReActChatMessage,
    CodeActChatMessage,
    AgentPlan,
    PlanStep,
    Planner,
    Task,
    Observer,
    ObserverResponse,
    CodeRunner,
    AgentResponse,
    print_response
)
from .tools import (
    tool,
    calculator,
    search_web,
    download_file,
    search_arxiv,
    extract_file_contents_as_markdown,
    search_wikipedia,
    get_youtube_transcript,
    get_audio_transcript,
)
from .kutils import (
    is_it_url,
    detect_file_type,
    is_image_file,
    make_user_message
)

# Alphabetical order is recommended
__all__ = [
    'Agent',
    'AgentPlan',
    'AgentResponse',
    'ChatMessage',
    'CodeActAgent',
    'CodeActChatMessage',
    'CodeRunner',
    'Observer',
    'ObserverResponse',
    'PlanStep',
    'Planner',
    'ReActAgent',
    'ReActChatMessage',
    'Task',
    'calculator',
    'detect_file_type',
    'download_file',
    'extract_file_contents_as_markdown',
    'get_audio_transcript',
    'get_youtube_transcript',
    'is_image_file',
    'is_it_url',
    'llm_vision_support',
    'make_user_message',
    'print_response',
    'search_arxiv',
    'search_web',
    'search_wikipedia',
    'tool',
]


# Prefer a single-source file inside the package for the version, with fallbacks.
try:
    # Primary: local single-source file created/updated by maintainers or build tooling
    from ._version import __version__  # type: ignore
except Exception:
    try:
        # Secondary: package metadata (works for installed packages)
        from importlib.metadata import version as _pkg_version  # Python 3.8+
        __version__ = _pkg_version('kodeagent')
    except Exception:
        # Final fallback: best-effort default
        __version__ = '0.1.0'
