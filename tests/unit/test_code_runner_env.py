"""
Unit tests for HostCodeRunnerEnv and E2BCodeRunnerEnv classes.
"""
import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from kodeagent.code_runner import HostCodeRunnerEnv, E2BCodeRunnerEnv


class TestCodeRunnerEnv(unittest.IsolatedAsyncioTestCase):
    """Test cases for HostCodeRunnerEnv and E2BCodeRunnerEnv."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_host_env_effective_work_dir(self):
        """Test that HostCodeRunnerEnv uses the provided work directory."""
        env = HostCodeRunnerEnv(work_dir=self.temp_dir)
        self.assertEqual(env.effective_work_dir, self.temp_dir)

    async def test_host_env_temp_dir(self):
        """Test that HostCodeRunnerEnv creates and cleans up a temp directory."""
        env = HostCodeRunnerEnv()
        eff_dir = env.effective_work_dir
        self.assertTrue(os.path.exists(eff_dir))
        self.assertIn('kodeagent_run_', eff_dir)
        env.cleanup()
        self.assertFalse(os.path.exists(eff_dir))

    @patch('subprocess.run')
    async def test_host_env_run(self, mock_run):
        """Test running code in HostCodeRunnerEnv."""
        mock_run.return_value = MagicMock(stdout='hello', stderr='', returncode=0)
        env = HostCodeRunnerEnv(work_dir=self.temp_dir)
        stdout, stderr, exit_code, generated_files = await env.run(
            'print("hello")', 'task1', 30
        )
        self.assertEqual(stdout, 'hello')
        self.assertEqual(exit_code, 0)
        # task_code.py should be in the directory but not in generated_files
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'task_code.py')))
        self.assertEqual(generated_files, [])

    @patch('e2b_code_interpreter.Sandbox')
    async def test_e2b_env_run(self, mock_cls):
        """Test running code in E2BCodeRunnerEnv."""
        mock_sbx = MagicMock()
        mock_sbx.files.list.side_effect = [[], [MagicMock(path='/home/user/new.txt')]]
        mock_sbx.run_code.return_value = MagicMock(
            logs=MagicMock(stdout=['out'], stderr=['err']), error=None
        )
        mock_cls.create.return_value = mock_sbx

        env = E2BCodeRunnerEnv(work_dir=self.temp_dir)
        stdout, stderr, exit_code, generated_files = await env.run(
            'print("ok")', 'task2', 30
        )
        
        self.assertEqual(stdout, 'out')
        self.assertEqual(generated_files, ['/home/user/new.txt'])
        
        # Test download
        mock_sbx.files.read.return_value = 'file content'
        local_files = await env.download_files_from_remote(['/home/user/new.txt'])
        
        self.assertEqual(len(local_files), 1)
        self.assertEqual(os.path.basename(local_files[0]), 'new.txt')
        with open(local_files[0], 'r') as f:
            self.assertEqual(f.read(), 'file content')

        env.cleanup()
        # mock_sbx.close.assert_called_once()
