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
    async def test_host_env_run_with_generated_files(self, mock_run):
        """Test HostCodeRunnerEnv when it generates files."""
        dummy_file = os.path.join(self.temp_dir, 'output.txt')
        
        def mock_run_side_effect(*_args, **_kwargs):
            with open(dummy_file, 'w') as f:
                f.write('new content')
            return MagicMock(stdout='hello', stderr='', returncode=0)
            
        mock_run.side_effect = mock_run_side_effect
        env = HostCodeRunnerEnv(work_dir=self.temp_dir)
        
        stdout, _stderr, exit_code, generated_files = await env.run(
            'print("hello")', 'task1', 30
        )
        self.assertEqual(stdout, 'hello')
        self.assertIn(dummy_file, generated_files)

    async def test_host_env_download(self):
        """Test HostCodeRunnerEnv.download_files_from_remote."""
        env = HostCodeRunnerEnv(work_dir=self.temp_dir)
        paths = ['/a/b/c.txt']
        result = await env.download_files_from_remote(paths)
        self.assertEqual(result, paths)

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
        stdout, _stderr, _exit_code, generated_files = await env.run(
            'print("ok")', 'task2', 30
        )
        
        self.assertEqual(stdout, 'out')
        self.assertEqual(generated_files, ['/home/user/new.txt'])
        
        # Test download
        mock_sbx.files.read.return_value = 'file content'
        local_files = await env.download_files_from_remote(['/home/user/new.txt'])
        
        self.assertEqual(len(local_files), 1)
        self.assertEqual(os.path.basename(local_files[0]), 'new.txt')
        with open(local_files[0], 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), 'file content')

    @patch('e2b_code_interpreter.Sandbox')
    async def test_e2b_env_get_sandbox_reuse(self, mock_cls):
        """Test that E2BCodeRunnerEnv reuses the sandbox."""
        mock_sbx = MagicMock()
        mock_cls.create.return_value = mock_sbx
        
        env = E2BCodeRunnerEnv(work_dir=self.temp_dir)
        sbx1 = await env._get_sandbox('t1', 30)
        sbx2 = await env._get_sandbox('t1', 30)
        
        self.assertIs(sbx1, sbx2)
        self.assertEqual(mock_cls.create.call_count, 1)

    async def test_e2b_env_download_edge_cases(self):
        """Test E2BCodeRunnerEnv.download_files_from_remote edge cases."""
        env = E2BCodeRunnerEnv(work_dir=self.temp_dir)
        # 1. No paths
        self.assertEqual(await env.download_files_from_remote([]), [])
        # 2. No sandbox
        self.assertEqual(await env.download_files_from_remote(['/p']), [])

    @patch('e2b_code_interpreter.Sandbox')
    async def test_e2b_env_cleanup(self, mock_cls):
        """Test E2BCodeRunnerEnv cleanup calls kill."""
        mock_sbx = MagicMock()
        mock_cls.create.return_value = mock_sbx
        
        env = E2BCodeRunnerEnv(work_dir=self.temp_dir)
        await env._get_sandbox('t1', 30)
        env.cleanup()
        
        mock_sbx.kill.assert_called_once()
