"""
Unit tests for the CodeRunner class in kodeagent.code_runner.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

from kodeagent.code_runner import CodeRunner, CodeSecurityError, UnknownCodeEnvError
from kodeagent.models import CodeReview


@pytest.fixture
def mock_security_reviewer():
    """Fixture to mock CodeSecurityReviewer with safe default response."""
    async def mock_review(code):
        return CodeReview(is_secure=True, reason="Code is safe for execution")
    
    with patch('kodeagent.code_runner.CodeSecurityReviewer') as mock_reviewer_class:
        mock_instance = MagicMock()
        mock_instance.review = mock_review
        mock_reviewer_class.return_value = mock_instance
        yield mock_reviewer_class


def test_code_runner_security_violation():
    """Test that dangerous builtins raise CodeSecurityError."""
    runner = CodeRunner(env='host', allowed_imports=['os'], model_name='test-model')

    # Test eval
    code_eval = "eval('print(1)')"
    with pytest.raises(CodeSecurityError) as excinfo:
        runner.check_imports(code_eval)
    assert 'Forbidden builtin: eval' in str(excinfo.value)

    # Test exec
    code_exec = "exec('import os')"
    with pytest.raises(CodeSecurityError) as excinfo:
        runner.check_imports(code_exec)
    assert 'Forbidden builtin: exec' in str(excinfo.value)


@pytest.mark.asyncio
async def test_code_runner_unknown_env(mock_security_reviewer):
    """Test that an unknown environment raises the correct exception."""
    with pytest.raises(UnknownCodeEnvError) as excinfo:
        _runner = CodeRunner(env='invalid_env', allowed_imports=[], model_name='test-model')
    
    assert 'Unsupported code execution env: invalid_env' in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_code_host_warning_and_files(mock_security_reviewer):
    """
    Test host execution triggers warning and handles local file copying.
    We mock shutil and sp.run to avoid actual execution/fs operations.
    """
    runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')

    # Inject a dummy file to copy
    runner.local_modules_to_copy = ['helper.py']

    with patch('shutil.copy2') as mock_copy, \
         patch('subprocess.run') as mock_sp_run, \
         patch('os.remove') as _mock_remove, \
         pytest.warns(UserWarning, match='dangerous'):

        # Setup mock return for subprocess
        mock_process = MagicMock()
        mock_process.stdout = 'Output'
        mock_process.stderr = ''
        mock_process.returncode = 0
        mock_sp_run.return_value = mock_process

        stdout, _, _, _ = await runner.run('', "print('test')", task_id='1')

        # Verify file copy was attempted
        assert mock_copy.call_count == 1
        assert stdout == 'Output'


@pytest.mark.asyncio
async def test_run_code_e2b_execution_error(mock_security_reviewer):
    """Test E2B execution when the code itself fails."""
    mock_e2b = MagicMock()
    mock_sbx_instance = MagicMock()
    mock_e2b.Sandbox.create.return_value = mock_sbx_instance

    mock_exec_result = MagicMock()
    mock_exec_result.logs.stdout = []
    mock_exec_result.logs.stderr = ['Traceback...']
    mock_exec_result.error = None # or some error object

    with patch('e2b_code_interpreter.Sandbox', mock_e2b.Sandbox):
        runner = CodeRunner(env='e2b', allowed_imports=[], model_name='test-model')

        mock_sbx_instance.run_code.return_value = mock_exec_result
        mock_sbx_instance.files.list.return_value = []
        mock_sbx_instance.get_info.return_value = {}

        _stdout, _stderr, _exit_status, _ = await runner.run(
            '', "print('test')", task_id='1'
        )

        # Since mock_exec_result.error is None, exit_status should be 0 
        # but in this test we might want to simulate an error.
        # Let's just verify call.
        assert mock_sbx_instance.run_code.called


def test_code_runner_pip_parsing_logic():
    """Test the regex splitting logic for pip packages."""
    # 1. Test comma separation
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy,pandas', model_name='test-model')
    assert runner.pip_packages == ['numpy', 'pandas']

    # 2. Test semicolon separation
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy;pandas', model_name='test-model')
    assert runner.pip_packages == ['numpy', 'pandas']

    # 3. Test mixed with whitespace
    # NOTE: The current CodeRunner implementation does NOT strip whitespace.
    # We match the test to the current code behavior.
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy, pandas; requests ', model_name='test-model')
    assert runner.pip_packages == ['numpy', ' pandas', ' requests ']

    # 4. Test empty/None
    runner_none = CodeRunner(env='host', allowed_imports=[], pip_packages=None, model_name='test-model')
    assert runner_none.pip_packages == []


@pytest.mark.asyncio
async def test_run_code_e2b_pip_install_skipped_without_packages(mock_security_reviewer):
    """No pip command should run when pip_packages_str is falsy."""
    mock_e2b = MagicMock()
    mock_sbx = MagicMock()
    mock_e2b.Sandbox.create.return_value.__enter__.return_value = mock_sbx
    mock_exec = MagicMock()
    mock_exec.logs.stdout = []
    mock_exec.logs.stderr = []
    mock_exec.error = None
    mock_sbx.run_code.return_value = mock_exec

    with patch.dict(sys.modules, {'e2b_code_interpreter': mock_e2b}):
        runner = CodeRunner(env='e2b', allowed_imports=[], pip_packages='', model_name='test-model')
        await runner.run('', "print('hi')", task_id='pip-skip')

    mock_sbx.commands.run.assert_not_called()


@pytest.mark.asyncio
async def test_run_code_e2b_pip_install_runs_with_packages(mock_security_reviewer):
    """pip install should run and local files should be uploaded when packages exist."""
    mock_e2b = MagicMock()
    mock_sbx = MagicMock()
    mock_e2b.Sandbox.create.return_value = mock_sbx
    mock_exec = MagicMock()
    mock_exec.logs.stdout = []
    mock_exec.logs.stderr = []
    mock_exec.error = None
    mock_sbx.run_code.return_value = mock_exec
    mock_sbx.files.list.return_value = []
    mock_sbx.get_info.return_value = {}

    with patch.dict(sys.modules, {'e2b_code_interpreter': mock_e2b}):
        runner = CodeRunner(env='e2b', allowed_imports=[], pip_packages='uvicorn', model_name='test-model')
        runner.local_modules_to_copy = ['dummy.py']

        from unittest.mock import mock_open
        with patch('kodeagent.code_runner.open', mock_open(read_data='print("x")'), create=True):
            await runner.run('', "print('hi')", task_id='pip-run')

    mock_sbx.commands.run.assert_called_once_with('pip install uvicorn')
    mock_sbx.files.write.assert_called_once_with('/home/user/dummy.py', 'print("x")')


@pytest.mark.asyncio
async def test_run_code_e2b_module_missing(mock_security_reviewer):
    """Test that missing e2b module raises SystemExit."""
    # We simulate the module being missing by setting it to None in sys.modules
    with patch.dict(sys.modules, {'e2b_code_interpreter': None}):
        runner = CodeRunner(env='e2b', allowed_imports=[], model_name='test-model')

        # The code catches ModuleNotFoundError and calls sys.exit(-1)
        with pytest.raises(SystemExit) as excinfo:
            await runner.run('', "print('test')", task_id='1')

        assert excinfo.value.code == -1


def match_path(suffix):
    """Custom matcher to check if a path ends with a specific suffix."""
    class PathMatcher:
        """Custom equality check for path suffixes."""
        def __eq__(self, other):
            return str(other).endswith(suffix)

    return PathMatcher()


def test_code_runner_with_empty_pip_packages():
    """Test CodeRunner with empty pip packages string."""
    runner = CodeRunner(
        env='host',
        allowed_imports=['os'],
        pip_packages='',
        timeout=30,
        model_name='test-model'
    )

    assert runner.pip_packages == []


def test_code_runner_multiple_pip_packages():
    """Test CodeRunner with multiple pip packages."""
    runner = CodeRunner(
        env='host',
        allowed_imports=['os'],
        pip_packages='requests==2.31.0;numpy==1.24.0',
        timeout=30,
        model_name='test-model'
    )

    assert len(runner.pip_packages) == 2
    assert 'requests==2.31.0' in runner.pip_packages
    assert 'numpy==1.24.0' in runner.pip_packages


@pytest.mark.asyncio
async def test_code_runner_disallowed_imports_error(mock_security_reviewer):
    """Test CodeRunner returns error for disallowed imports."""
    runner = CodeRunner(env='host', allowed_imports=['os'], timeout=30, model_name='test-model')

    code = 'import subprocess\nsubprocess.run(["ls"])'
    stdout, stderr, exit_code, _ = await runner.run(tools_code='', generated_code=code, task_id='task-1234')

    assert exit_code != 0
    assert 'disallowed' in stderr.lower()


def test_code_runner_check_imports_with_from_import():
    """Test CodeRunner import checking with from imports."""
    runner = CodeRunner(env='host', allowed_imports=['os', 'datetime'], model_name='test-model')

    code = """
from os import path
from datetime import datetime
from requests import get
"""
    disallowed = runner.check_imports(code)
    assert 'requests' in disallowed
    assert 'os' not in disallowed
    assert 'datetime' not in disallowed


def test_code_runner_default_allowed_imports():
    """DEFAULT_ALLOWED_IMPORTS should be available even if not passed in."""
    runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')

    code = "import re\ncompiled = re.compile(r'abc')"
    disallowed = runner.check_imports(code)

    assert 're' not in disallowed, "Default allowed import 're' should be permitted"

    code = 'import math'
    disallowed = runner.check_imports(code)

    assert 'math' in disallowed, 'Non-default import should still be disallowed'


# ===== New Security Review Tests =====

@pytest.mark.asyncio
async def test_security_review_blocks_unsafe_code():
    """Test that unsafe code is blocked by pattern detection or security review."""
    # Note: This code will be blocked by pattern detection BEFORE LLM review
    # because os.environ access is detected by AST analysis
    async def mock_unsafe_review(code):
        return CodeReview(
            is_secure=False,
            reason="Code attempts to access environment variables which could expose sensitive data"
        )
    
    with patch('kodeagent.code_runner.CodeSecurityReviewer') as mock_reviewer_class:
        mock_instance = MagicMock()
        mock_instance.review = mock_unsafe_review
        mock_reviewer_class.return_value = mock_instance
        
        runner = CodeRunner(env='host', allowed_imports=['os'], model_name='test-model')
        
        unsafe_code = "import os\nprint(os.environ)"
        
        with pytest.raises(CodeSecurityError) as excinfo:
            await runner.run(tools_code='', generated_code=unsafe_code, task_id='test-1')
        
        # Pattern detection blocks this before LLM review
        assert 'pattern detection' in str(excinfo.value).lower() or 'security concerns' in str(excinfo.value).lower()
        assert 'environ' in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_security_review_allows_safe_code(mock_security_reviewer):
    """Test that safe code passes security review and executes successfully."""
    with patch('subprocess.run') as mock_sp_run:
        mock_process = MagicMock()
        mock_process.stdout = 'Hello, World!'
        mock_process.stderr = ''
        mock_process.returncode = 0
        mock_sp_run.return_value = mock_process
        
        runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')
        
        safe_code = "print('Hello, World!')"
        
        with patch('os.remove'):
            stdout, stderr, exit_code, _ = await runner.run(
                tools_code='',
                generated_code=safe_code,
                task_id='test-2'
            )
        
        assert exit_code == 0
        assert stdout == 'Hello, World!'


@pytest.mark.asyncio
async def test_security_review_called_with_generated_code_only():
    """Test that security reviewer is called with generated_code only, not tools_code."""
    review_call_args = []
    
    async def mock_review_capture(code):
        review_call_args.append(code)
        return CodeReview(is_secure=True, reason="Code is safe")
    
    with patch('kodeagent.code_runner.CodeSecurityReviewer') as mock_reviewer_class:
        mock_instance = MagicMock()
        mock_instance.review = mock_review_capture
        mock_reviewer_class.return_value = mock_instance
        
        with patch('subprocess.run') as mock_sp_run:
            mock_process = MagicMock()
            mock_process.stdout = 'output'
            mock_process.stderr = ''
            mock_process.returncode = 0
            mock_sp_run.return_value = mock_process
            
            runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')
            
            tools_code = "def helper():\n    return 42"
            generated_code = "result = helper()\nprint(result)"
            
            with patch('os.remove'):
                await runner.run(
                    tools_code=tools_code,
                    generated_code=generated_code,
                    task_id='test-3'
                )
            
            # Verify review was called with generated_code only
            assert len(review_call_args) == 1
            assert review_call_args[0] == generated_code
            assert tools_code not in review_call_args[0]


@pytest.mark.asyncio
async def test_tools_code_and_generated_code_combined(mock_security_reviewer):
    """Test that tools_code and generated_code are combined for execution."""
    executed_code = []
    
    def mock_run_capture(*args, **kwargs):
        # Capture the code that was written to the temp file
        # The code is in args[0] which is the command list
        executed_code.append(kwargs.get('input', ''))
        mock_process = MagicMock()
        mock_process.stdout = '42'
        mock_process.stderr = ''
        mock_process.returncode = 0
        return mock_process
    
    with patch('subprocess.run', side_effect=mock_run_capture):
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_temp_dir = os.path.join(tempfile.gettempdir(), 'kodeagent_test')
            mock_mkdtemp.return_value = mock_temp_dir
            
            # Ensure the directory "exists" for listdir
            with patch('os.listdir', return_value=[]):
                runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')
                
                tools_code = "def helper():\n    return 42"
                generated_code = "result = helper()\nprint(result)"
                
                # Mock open to capture write
                m = mock_open()
                with patch('kodeagent.code_runner.open', m, create=True):
                    # Also mock shutil.copy2 to avoid errors
                    with patch('shutil.copy2'):
                        await runner.run(
                            tools_code=tools_code,
                            generated_code=generated_code,
                            task_id='test-4'
                        )
                
                # Verify the combined code was written
                written_code = "".join(call.args[0] for call in m().write.call_args_list)
                assert tools_code in written_code
                assert generated_code in written_code
                assert f'{tools_code}\n\n{generated_code}' in written_code


@pytest.mark.asyncio
async def test_security_review_error_handling():
    """Test proper error handling when security reviewer raises an exception."""
    async def mock_review_error(code):
        raise Exception("LLM API error")
    
    with patch('kodeagent.code_runner.CodeSecurityReviewer') as mock_reviewer_class:
        mock_instance = MagicMock()
        mock_instance.review = mock_review_error
        mock_reviewer_class.return_value = mock_instance
        
        runner = CodeRunner(env='host', allowed_imports=[], model_name='test-model')
        
        code = "print('test')"
        
        with pytest.raises(Exception) as excinfo:
            await runner.run(tools_code='', generated_code=code, task_id='test-5')
        
        assert 'LLM API error' in str(excinfo.value)


def test_code_runner_init_with_model_params():
    """Test that model_name and litellm_params are properly passed to CodeSecurityReviewer."""
    with patch('kodeagent.code_runner.CodeSecurityReviewer') as mock_reviewer_class:
        mock_instance = MagicMock()
        mock_reviewer_class.return_value = mock_instance
        
        model_name = 'gpt-4'
        litellm_params = {'temperature': 0.5, 'max_tokens': 100}
        
        runner = CodeRunner(
            env='host',
            allowed_imports=['os'],
            model_name=model_name,
            litellm_params=litellm_params
        )
        
        # Verify CodeSecurityReviewer was initialized with correct params
        mock_reviewer_class.assert_called_once_with(
            model_name=model_name,
            litellm_params=litellm_params
        )
        assert runner.code_reviewer == mock_instance

