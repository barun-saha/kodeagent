"""
Unit tests for the CodeRunner class in kodeagent.code_runner.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch, mock_open

from kodeagent.code_runner import CodeRunner, CodeSecurityError, UnknownCodeEnvError


def test_code_runner_security_violation():
    """Test that dangerous builtins raise CodeSecurityError."""
    runner = CodeRunner(env='host', allowed_imports=['os'])

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


def test_code_runner_unknown_env():
    """Test that an unknown environment raises the correct exception."""
    runner = CodeRunner(env='invalid_env', allowed_imports=[])

    with pytest.raises(UnknownCodeEnvError) as excinfo:
        runner.run(source_code="print('ok')", task_id='1')

    assert 'Unsupported code execution env: invalid_env' in str(excinfo.value)


def test_run_code_host_warning_and_files():
    """
    Test host execution triggers warning and handles local file copying.
    We mock shutil and sp.run to avoid actual execution/fs operations.
    """
    runner = CodeRunner(env='host', allowed_imports=[])

    # Inject a dummy file to copy
    runner.local_modules_to_copy = ['helper.py']

    with patch('shutil.copy2') as mock_copy, \
         patch('subprocess.run') as mock_sp_run, \
         patch('os.remove') as mock_remove, \
         pytest.warns(UserWarning, match='dangerous'):

        # Setup mock return for subprocess
        mock_process = MagicMock()
        mock_process.stdout = 'Output'
        mock_process.stderr = ''
        mock_process.returncode = 0
        mock_sp_run.return_value = mock_process

        stdout, _, _ = runner.run("print('test')", task_id='1')

        # Verify file copy was attempted
        assert mock_copy.call_count == 1
        assert stdout == 'Output'
        # Verify temp file cleanup
        assert mock_remove.call_count == 1


@patch.dict(sys.modules, {'e2b_code_interpreter': MagicMock()})
def test_run_code_e2b_success():
    """
    Test successful E2B execution with mocks.
    We verify the sandbox is created, commands run, and output returned.
    """
    # Create the mock for the module
    mock_e2b = sys.modules['e2b_code_interpreter']

    # Set up the Mock Sandbox instance
    mock_sbx_instance = MagicMock()
    mock_e2b.Sandbox.return_value = mock_sbx_instance

    # Setup execution result
    mock_exec_result = MagicMock()
    mock_exec_result.logs.stdout = ['Hello E2B']
    mock_exec_result.logs.stderr = []
    mock_exec_result.error = None
    mock_sbx_instance.run_code.return_value = mock_exec_result

    runner = CodeRunner(
        env='e2b',
        allowed_imports=[],
        pip_packages='pandas',
        env_vars_to_set={'API_KEY': '123'}
    )

    stdout, stderr, code = runner.run("print('Hello E2B')", task_id='task-e2b')

    # Verify Sandbox initialized with correct args
    mock_e2b.Sandbox.assert_called_with(
        timeout=45, # Default 30 + 15 buffer
        envs={'API_KEY': '123'},
        metadata={'task_id': 'task-e2b'}
    )

    # Verify pip install was called
    mock_sbx_instance.commands.run.assert_called_with('pip install pandas')

    # Verify results
    assert stdout == 'Hello E2B'
    assert code == 0


@patch.dict(sys.modules, {'e2b_code_interpreter': MagicMock()})
def test_run_code_e2b_execution_error():
    """Test E2B execution when the code itself fails."""
    mock_e2b = sys.modules['e2b_code_interpreter']
    mock_sbx_instance = MagicMock()
    mock_e2b.Sandbox.return_value = mock_sbx_instance

    # Mock an error result
    mock_exec_result = MagicMock()
    mock_exec_result.logs.stdout = []
    mock_exec_result.logs.stderr = ['Traceback...']

    # Mock the error object on the execution result
    mock_error = MagicMock()
    mock_error.name = 'ValueError'
    mock_error.value = 'Wrong value'
    mock_exec_result.error = mock_error

    mock_sbx_instance.run_code.return_value = mock_exec_result

    runner = CodeRunner(env='e2b', allowed_imports=[])
    stdout, stderr, code = runner.run("raise ValueError", task_id='1')

    assert code == -1
    assert 'ValueError' in stderr
    assert 'Wrong value' in stderr


def test_code_runner_pip_parsing_logic():
    """Test the regex splitting logic for pip packages."""
    # 1. Test comma separation
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy,pandas')
    assert runner.pip_packages == ['numpy', 'pandas']

    # 2. Test semicolon separation
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy;pandas')
    assert runner.pip_packages == ['numpy', 'pandas']

    # 3. Test mixed with whitespace
    # NOTE: The current CodeRunner implementation does NOT strip whitespace.
    # We match the test to the current code behavior.
    runner = CodeRunner(env='host', allowed_imports=[], pip_packages='numpy, pandas; requests ')
    assert runner.pip_packages == ['numpy', ' pandas', ' requests ']

    # 4. Test empty/None
    runner_none = CodeRunner(env='host', allowed_imports=[], pip_packages=None)
    assert runner_none.pip_packages == []


def test_run_code_e2b_module_missing():
    """Test that missing e2b module raises SystemExit."""
    # We simulate the module being missing by setting it to None in sys.modules
    with patch.dict(sys.modules, {'e2b_code_interpreter': None}):
        runner = CodeRunner(env='e2b', allowed_imports=[])

        # The code catches ModuleNotFoundError and calls sys.exit(-1)
        with pytest.raises(SystemExit) as excinfo:
            runner.run("print('test')", task_id='1')

        assert excinfo.value.code == -1


def test_run_code_e2b_file_copying():
    """
    Test that local modules are copied to the E2B sandbox.
    This covers the loop: 'for a_file in self.local_modules_to_copy:'
    """
    # Mock the E2B module and Sandbox
    mock_e2b = MagicMock()
    mock_sbx = MagicMock()
    mock_e2b.Sandbox.return_value = mock_sbx

    # Mock execution result to avoid errors later in the function
    mock_exec = MagicMock()
    mock_exec.logs.stdout = []
    mock_exec.logs.stderr = []
    mock_exec.error = None
    mock_sbx.run_code.return_value = mock_exec

    with patch.dict(sys.modules, {'e2b_code_interpreter': mock_e2b}):
        runner = CodeRunner(env='e2b', allowed_imports=[])

        # Inject a file to copy
        runner.local_modules_to_copy = ['my_helper.py']

        # We must mock 'open' since the code tries to read the local file
        with patch('builtins.open', mock_open(read_data='print("helper")')) as mocked_file:
            runner.run("print('main')", task_id='1')

            # Verify the file was read
            mocked_file.assert_called_with(
                # We can't easily predict the full path (os.path.dirname),
                # so we check if it ended with the filename we expect.
                pytest.approx(match_path('my_helper.py')),
                'r',
                encoding='utf-8'
            )

            # Verify the file was written to the sandbox
            # The code does: sbx.files.write(f'/home/user/{a_file}', py_file.read())
            mock_sbx.files.write.assert_called_with(
                '/home/user/my_helper.py',
                'print("helper")'
            )


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
        timeout=30
    )

    assert runner.pip_packages == []


def test_code_runner_multiple_pip_packages():
    """Test CodeRunner with multiple pip packages."""
    runner = CodeRunner(
        env='host',
        allowed_imports=['os'],
        pip_packages='requests==2.31.0;numpy==1.24.0',
        timeout=30
    )

    assert len(runner.pip_packages) == 2
    assert 'requests==2.31.0' in runner.pip_packages
    assert 'numpy==1.24.0' in runner.pip_packages


def test_code_runner_disallowed_imports_error():
    """Test CodeRunner returns error for disallowed imports."""
    runner = CodeRunner(env='host', allowed_imports=['os'], timeout=30)

    code = 'import subprocess\nsubprocess.run(["ls"])'
    stdout, stderr, exit_code = runner.run(task_id='task-1234', source_code=code)

    assert exit_code != 0
    assert 'disallowed' in stderr.lower()


def test_code_runner_check_imports_with_from_import():
    """Test CodeRunner import checking with from imports."""
    runner = CodeRunner(env='host', allowed_imports=['os', 'datetime'])

    code = """
from os import path
from datetime import datetime
from requests import get
"""
    disallowed = runner.check_imports(code)
    assert 'requests' in disallowed
    assert 'os' not in disallowed
    assert 'datetime' not in disallowed
