"""KodeAgent: An intelligent code agent"""

from .examples import run_examples
from .fca import FunctionCallingAgent
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
    transcribe_audio,
    transcribe_youtube,
)

# Work directory guidance for CodeActAgent
_CODEACT_WORK_DIR_INSTRUCTION = (
    'Use {work_dir} as the working directory for all file creation and modification.'
    ' If that path does not exist, use the system temporary directory instead.'
)


def _get_codeact_work_dir(agent: 'CodeActAgent') -> str | None:
    """Return the effective working directory for a CodeActAgent, if available."""
    for attr in ('work_dir', 'working_dir', 'workdir', 'code_runner', 'runner', 'env'):
        value = getattr(agent, attr, None)
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, str):
            for inner_attr in ('work_dir', 'working_dir', 'workdir', 'cwd'):
                inner_value = getattr(value, inner_attr, None)
                if isinstance(inner_value, str) and inner_value:
                    return inner_value
    try:
        for attr, value in vars(agent).items():
            if isinstance(value, str) and value and ('dir' in attr.lower() or 'cwd' in attr.lower()):
                return value
    except Exception:
        pass
    return None


def _add_work_dir_to_codeact_prompt(agent: 'CodeActAgent') -> None:
    """Append work-dir guidance to the CodeActAgent system prompt."""
    work_dir = _get_codeact_work_dir(agent)
    if not work_dir:
        return
    instruction = _CODEACT_WORK_DIR_INSTRUCTION.format(work_dir=work_dir)
    for attr in ('system_prompt', 'persona', '_system_prompt', 'system_prompt_template', 'prompt'):
        try:
            current = getattr(agent, attr)
        except Exception:
            continue
        if isinstance(current, str):
            if instruction not in current:
                try:
                    setattr(agent, attr, current.rstrip() + f'\n\n{instruction}')
                except Exception:
                    continue
            return
    try:
        current = getattr(agent, 'persona', None)
        agent.persona = (current or '') + f'\n\n{instruction}'
    except Exception:
        pass


_original_codeact_init = CodeActAgent.__init__


def _codeact_init_with_work_dir(self: 'CodeActAgent', *args, **kwargs):
    """Wrap CodeActAgent.__init__ to include the work dir in the system prompt."""
    _original_codeact_init(self, *args, **kwargs)
    _add_work_dir_to_codeact_prompt(self)


CodeActAgent.__init__ = _codeact_init_with_work_dir

# Alphabetical order is recommended
__all__ = [
    'Agent',
    'AgentPlan',
    'AgentResponse',
    'ChatMessage',
    'CodeActAgent',
    'CodeActChatMessage',
    'CodeRunner',
    'FunctionCallingAgent',
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
    'run_examples',
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
