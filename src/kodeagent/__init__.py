"""KodeAgent: An intelligent code agent"""

from .kodeagent import (
    Agent,
    AgentResponse,
    ChatMessage,
    CodeActAgent,
    CodeActChatMessage,
    CodeRunner,
    ReActAgent,
    ReActChatMessage,
    Task,
    llm_vision_support,
    print_response,
)
from .kutils import detect_file_type, is_image_file, is_it_url, make_user_message
from .models import AgentPlan, ObserverResponse, PlanStep
from .orchestrator import Observer, Planner
from .tools import (
    calculator,
    download_file,
    extract_as_markdown,
    read_webpage,
    search_arxiv,
    search_web,
    search_wikipedia,
    tool,
    transcribe_audio,
    transcribe_youtube,
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
    'extract_as_markdown',
    'transcribe_audio',
    'transcribe_youtube',
    'is_image_file',
    'is_it_url',
    'llm_vision_support',
    'make_user_message',
    'print_response',
    'read_webpage',
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
