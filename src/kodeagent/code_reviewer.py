"""Review code for security vulnerabilities."""

import uuid

from . import kutils as ku
from .models import CodeReview
from .usage_tracker import UsageTracker

CODE_SECURITY_SYSTEM_PROMPT = ku.read_prompt('code_guardrail.txt')


class CodeSecurityReviewer:
    """Review code for security vulnerabilities."""

    def __init__(
        self,
        model_name: str,
        litellm_params: dict | None = None,
        usage_tracker: UsageTracker | None = None,
        tool_names: set[str] | None = None,
    ):
        """Initialize the CodeSecurityReviewer.

        Args:
            model_name: The name of the LLM model to use.
            litellm_params: Optional parameters for the LLM.
            usage_tracker: Optional UsageTracker instance.
            tool_names: Optional set of whitelisted tool names provided by the user.
        """
        self.model_name = model_name
        self.litellm_params = litellm_params or {}
        self.usage_tracker = usage_tracker
        self.tool_names = tool_names or set()

    async def review(self, code: str) -> CodeReview:
        """Review the code for security vulnerabilities.

        Args:
            code: The code to review.

        Returns:
            A CodeReview object containing the review results.
        """
        # Format the system prompt with whitelisted tools
        if self.tool_names:
            tools_list = '\n'.join(f'- {tool}' for tool in sorted(self.tool_names))
        else:
            tools_list = '- [None provided]'

        system_prompt = CODE_SECURITY_SYSTEM_PROMPT.format(whitelisted_tools=tools_list)

        messages = [
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': f'Review this code:\n{code}',
            },
        ]
        review_response = await ku.call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=messages,
            response_format=CodeReview,
            trace_id=uuid.uuid4().hex,
            usage_tracker=self.usage_tracker,
            component_name='CodeSecurityReviewer',
        )
        return CodeReview.model_validate_json(review_response)
