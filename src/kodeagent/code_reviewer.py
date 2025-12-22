"""
Review code for security vulnerabilities.
"""
import uuid
from typing import Optional

from . import kutils as ku
from .models import CodeReview


CODE_SECURITY_SYSTEM_PROMPT = ku.read_prompt('code_guardrail.txt')


class CodeSecurityReviewer:
    """
    Review code for security vulnerabilities.
    """
    def __init__(self, model_name: str, litellm_params: Optional[dict] = None):
        """
        Initialize the CodeSecurityReviewer.

        Args:
            model_name: The name of the LLM model to use.
            litellm_params: Optional parameters for the LLM.
        """
        self.model_name = model_name
        self.litellm_params = litellm_params or {}

    async def review(self, code: str) -> CodeReview:
        """
        Review the code for security vulnerabilities.

        Args:
            code: The code to review.

        Returns:
            A CodeReview object containing the review results.
        """
        messages = [
            {
                "role": "system",
                "content": CODE_SECURITY_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f'Review this code:\n{code}',
            },
        ]
        review_response = await ku.call_llm(
            model_name=self.model_name,
            litellm_params=self.litellm_params,
            messages=messages,
            response_format=CodeReview,
            trace_id=uuid.uuid4().hex
        )
        return CodeReview.model_validate_json(review_response)
