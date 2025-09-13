"""KodeAgent: An intelligent code agent"""

from .kodeagent import (
    tool,
    call_llm,
    llm_vision_support,
    calculator,
    search_web,
    download_file,
    search_arxiv,
    extract_file_contents_as_markdown,
    search_wikipedia,
    get_youtube_transcript,
    get_audio_transcript,
    Agent,
    ReActAgent,
    CodeActAgent,
    ChatMessage,
    ReActChatMessage,
    CodeChatMessage,
    AgentPlan,
    PlanStep,
    Planner,
    Task,
    Observer,
    ObserverResponse,
    CodeRunner,
    AgentResponse
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
    'CodeChatMessage',
    'CodeRunner',
    'Observer',
    'ObserverResponse',
    'PlanStep',
    'Planner',
    'ReActAgent',
    'ReActChatMessage',
    'Task',
    'call_llm',
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
    'search_arxiv',
    'search_web',
    'search_wikipedia',
    'tool',
]


try:
    from importlib.metadata import version as _pkg_version  # Python 3.8+
    __version__ = _pkg_version('kodeagent')
except Exception:
    __version__ = '0.1.0'
