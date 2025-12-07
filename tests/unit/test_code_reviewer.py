"""
Unit tests for the CodeSecurityReviewer class in kodeagent.code_reviewer.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from kodeagent.code_reviewer import CodeSecurityReviewer
from kodeagent.models import CodeReview


@pytest.mark.asyncio
async def test_code_reviewer_initialization():
    """Test CodeSecurityReviewer initialization with model parameters."""
    model_name = 'gemini/gemini-2.0-flash-lite'
    litellm_params = {'temperature': 0.3, 'max_tokens': 500}
    
    reviewer = CodeSecurityReviewer(
        model_name=model_name,
        litellm_params=litellm_params
    )
    
    assert reviewer.model_name == model_name
    assert reviewer.litellm_params == litellm_params


@pytest.mark.asyncio
async def test_code_reviewer_review_safe_code():
    """Test that safe code is approved by the reviewer."""
    mock_response = '{"is_secure": true, "reason": "Code performs simple arithmetic operations without security risks"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        safe_code = "result = 2 + 2\nprint(result)"
        review = await reviewer.review(safe_code)
        
        assert isinstance(review, CodeReview)
        assert review.is_secure is True
        assert len(review.reason) > 0  # Verify a reason was provided
        
        # Verify LLM was called with the code
        mock_llm.assert_called_once()
        call_args = mock_llm.call_args
        messages = call_args.kwargs['messages']
        # Check that the user message contains the review request
        assert any('Review this code' in str(msg) for msg in messages)


@pytest.mark.asyncio
async def test_code_reviewer_review_unsafe_code():
    """Test that unsafe code is rejected by the reviewer."""
    mock_response = '{"is_secure": false, "reason": "Code attempts to access environment variables which could expose sensitive information"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        unsafe_code = "import os\nprint(os.environ)"
        review = await reviewer.review(unsafe_code)
        
        assert isinstance(review, CodeReview)
        assert review.is_secure is False
        assert 'environment' in review.reason.lower() or 'sensitive' in review.reason.lower()


@pytest.mark.asyncio
async def test_code_reviewer_with_system_prompt():
    """Test that the reviewer uses the security system prompt."""
    mock_response = '{"is_secure": true, "reason": "Code is safe"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        code = "print('Hello, World!')"
        await reviewer.review(code)
        
        # Verify system prompt was included
        call_args = mock_llm.call_args
        messages = call_args.kwargs['messages']
        
        # Should have system message and user message
        assert len(messages) >= 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert code in messages[1]['content']


@pytest.mark.asyncio
async def test_code_reviewer_with_litellm_params():
    """Test that litellm_params are passed to the LLM call."""
    mock_response = '{"is_secure": true, "reason": "Safe code"}'
    litellm_params = {'temperature': 0.1, 'max_tokens': 200}
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params=litellm_params
        )
        
        code = "x = 1 + 1"
        await reviewer.review(code)
        
        # Verify litellm_params were passed
        call_args = mock_llm.call_args
        assert call_args.kwargs['litellm_params'] == litellm_params


@pytest.mark.asyncio
async def test_code_reviewer_response_format():
    """Test that the reviewer expects CodeReview response format."""
    mock_response = '{"is_secure": true, "reason": "No security issues detected"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        code = "print('test')"
        await reviewer.review(code)
        
        # Verify response_format was set to CodeReview
        call_args = mock_llm.call_args
        assert call_args.kwargs['response_format'] == CodeReview


@pytest.mark.asyncio
async def test_code_reviewer_handles_complex_code():
    """Test reviewer with more complex code patterns."""
    mock_response = '{"is_secure": false, "reason": "Code uses subprocess which can execute arbitrary commands"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        complex_code = """
import subprocess

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout
"""
        review = await reviewer.review(complex_code)
        
        assert isinstance(review, CodeReview)
        assert review.is_secure is False
        assert 'subprocess' in review.reason.lower() or 'command' in review.reason.lower()


@pytest.mark.asyncio
async def test_code_reviewer_model_name_passed():
    """Test that model_name is correctly passed to LLM call."""
    mock_response = '{"is_secure": true, "reason": "Safe"}'
    model_name = 'gpt-4'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name=model_name,
            litellm_params={}
        )
        
        code = "x = 42"
        await reviewer.review(code)
        
        # Verify model_name was passed
        call_args = mock_llm.call_args
        assert call_args.kwargs['model_name'] == model_name


@pytest.mark.asyncio
async def test_code_reviewer_with_empty_code():
    """Test reviewer behavior with empty code."""
    mock_response = '{"is_secure": true, "reason": "No code to review"}'
    
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.return_value = mock_response
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        empty_code = ""
        review = await reviewer.review(empty_code)
        
        assert isinstance(review, CodeReview)
        # The reviewer should still process empty code
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_code_reviewer_llm_error_propagation():
    """Test that LLM errors are properly propagated."""
    with patch('kodeagent.kutils.call_llm') as mock_llm:
        mock_llm.side_effect = Exception("LLM API error")
        
        reviewer = CodeSecurityReviewer(
            model_name='test-model',
            litellm_params={}
        )
        
        code = "print('test')"
        
        with pytest.raises(Exception) as excinfo:
            await reviewer.review(code)
        
        assert 'LLM API error' in str(excinfo.value)
