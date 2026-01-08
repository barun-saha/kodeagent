"""Unit tests for CodeRunnerEnv work_dir handling."""

import os
import shutil
import tempfile

from kodeagent.code_runner import CodeRunnerEnv


class ConcreteCodeRunnerEnv(CodeRunnerEnv):
    """Concrete implementation of CodeRunnerEnv for testing."""

    async def run(self, source_code, task_id, timeout):
        """Dummy implementation of run method."""
        pass

    async def download_files_from_remote(self, remote_paths):
        """Dummy implementation of download_files_from_remote method."""
        pass


def test_work_dir_none():
    """Test that a temporary work_dir is created when None is provided."""
    env = ConcreteCodeRunnerEnv(work_dir=None)
    assert env.work_dir is not None
    assert os.path.isdir(env.work_dir)
    assert 'kodeagent_run_' in env.work_dir
    os.rmdir(env.work_dir)


def test_work_dir_abs_exists():
    """Test that an existing absolute work_dir is used as is."""
    temp_dir = tempfile.mkdtemp()
    try:
        abs_path = os.path.abspath(temp_dir)
        env = ConcreteCodeRunnerEnv(work_dir=abs_path)
        assert env.work_dir == abs_path
    finally:
        os.rmdir(temp_dir)


def test_work_dir_rel_exists():
    """Test that an existing relative work_dir is converted to absolute."""
    # Create a temp dir in current folder
    current_dir = os.getcwd()
    rel_name = 'test_rel_dir'
    abs_path = os.path.join(current_dir, rel_name)
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)

    try:
        env = ConcreteCodeRunnerEnv(work_dir=rel_name)
        assert env.work_dir == abs_path
    finally:
        os.rmdir(abs_path)


def test_work_dir_not_exists():
    """Test that a new temporary work_dir is created when the provided one does not exist."""
    non_existent = os.path.abspath('non_existent_dir_12345')
    if os.path.exists(non_existent):
        shutil.rmtree(non_existent)

    env = ConcreteCodeRunnerEnv(work_dir=non_existent)
    assert env.work_dir != non_existent
    assert os.path.isdir(env.work_dir)
    assert 'kodeagent_run_' in env.work_dir
    os.rmdir(env.work_dir)
