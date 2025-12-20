"""
Additional tests for CodeRunner class to increase coverage.
"""
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from kodeagent.code_runner import CodeRunner, CodeRunResult


class TestCodeRunnerExtra(unittest.IsolatedAsyncioTestCase):
    """Additional tests for CodeRunner class to increase coverage."""

    def setUp(self):
        self.model_name = 'test-model'
        self.allowed_imports = ['math']

    @patch('kodeagent.code_runner.HostCodeRunnerEnv')
    @patch('kodeagent.code_runner.CodeSecurityReviewer')
    @patch('kodeagent.code_runner.analyze_code_patterns')
    async def test_code_runner_syntax_error(self, _mock_analyze, _mock_reviewer, _mock_host_env):
        """Test CodeRunner.run with a syntax error."""
        runner = CodeRunner(env='host', allowed_imports=self.allowed_imports, model_name=self.model_name)
        
        # Invalid python code
        bad_code = 'if True:'
        result = await runner.run('', bad_code, 'task-id')
        
        self.assertIsInstance(result, CodeRunResult)
        self.assertEqual(result.return_code, -1)
        self.assertIn('Code parsing failed due to:', result.stderr)
        self.assertIn('Error:', result.stderr)

    @patch('kodeagent.code_runner.HostCodeRunnerEnv')
    async def test_code_runner_local_modules_properties(self, mock_host_env):
        """Test local_modules_to_copy getter and setter."""
        mock_env = MagicMock()
        mock_env.local_modules_to_copy = []
        mock_host_env.return_value = mock_env
        
        runner = CodeRunner(env='host', allowed_imports=self.allowed_imports, model_name=self.model_name)
        
        runner.local_modules_to_copy = ['module1.py']
        self.assertEqual(mock_env.local_modules_to_copy, ['module1.py'])
        self.assertEqual(runner.local_modules_to_copy, ['module1.py'])

    @patch('kodeagent.code_runner.HostCodeRunnerEnv')
    async def test_code_runner_download_remote(self, mock_host_env):
        """Test CodeRunner.download_files_from_remote."""
        mock_env = MagicMock()
        mock_env.download_files_from_remote = AsyncMock(return_value=['/local/path'])
        mock_host_env.return_value = mock_env
        
        runner = CodeRunner(env='host', allowed_imports=self.allowed_imports, model_name=self.model_name)
        result = await runner.download_files_from_remote(['/remote/path'])
        
        self.assertEqual(result, ['/local/path'])
        mock_env.download_files_from_remote.assert_called_once_with(['/remote/path'])

    @patch('kodeagent.code_runner.HostCodeRunnerEnv')
    async def test_code_runner_cleanup(self, mock_host_env):
        """Test CodeRunner.cleanup."""
        mock_env = MagicMock()
        mock_host_env.return_value = mock_env
        
        runner = CodeRunner(env='host', allowed_imports=self.allowed_imports, model_name=self.model_name)
        runner.cleanup()
        
        mock_env.cleanup.assert_called_once()
