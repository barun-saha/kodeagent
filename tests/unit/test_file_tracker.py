"""Unit tests for the file tracker output interceptor."""

import os
import tempfile
import threading

import pytest

from kodeagent.file_tracker import OutputInterceptor, install_interceptor, uninstall_interceptor


@pytest.fixture(scope='module', autouse=True)
def setup_interceptor():
    """Install the interceptor for the duration of the tests."""
    install_interceptor()
    yield
    uninstall_interceptor()


def test_interceptor_basic_write():
    """Test that writes are intercepted and redirected to the sandbox."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox_dir = os.path.join(tmp_dir, 'sandbox')
        test_file = os.path.join(tmp_dir, 'test.txt')

        with OutputInterceptor(sandbox_root=sandbox_dir) as interceptor:
            with open(test_file, 'w') as f:
                f.write('hello')

            manifest = interceptor.get_manifest()
            assert len(manifest) == 1
            # Should be redirected to sandbox
            expected_path = os.path.join(sandbox_dir, 'test.txt')
            assert manifest[0] == expected_path
            assert os.path.exists(expected_path)
            with open(expected_path) as f:
                assert f.read() == 'hello'


def test_interceptor_no_redirect():
    """Test that files outside the sandbox are not redirected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, 'test_noredirect.txt')

        with OutputInterceptor() as interceptor:
            with open(test_file, 'w') as f:
                f.write('no redirect')

            manifest = interceptor.get_manifest()
            assert len(manifest) == 1
            assert manifest[0] == test_file
            assert os.path.exists(test_file)


def test_interceptor_thread_safety():
    """Test that the interceptor works correctly in a multi-threaded context."""
    results = {}

    def run_agent(name, sandbox_path, file_name):
        _interceptor = OutputInterceptor(sandbox_root=sandbox_path)
        with _interceptor:
            with open(file_name, 'w') as f:
                f.write(f'content from {name}')
            results[name] = _interceptor.get_manifest()

    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox1 = os.path.join(tmp_dir, 'sb1')
        sandbox2 = os.path.join(tmp_dir, 'sb2')

        t1 = threading.Thread(target=run_agent, args=('A', sandbox1, 'fileA.txt'))
        t2 = threading.Thread(target=run_agent, args=('B', sandbox2, 'fileB.txt'))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results['A']) == 1
        assert 'fileA.txt' in results['A'][0]
        assert 'sb1' in results['A'][0]

        assert len(results['B']) == 1
        assert 'fileB.txt' in results['B'][0]
        assert 'sb2' in results['B'][0]

        # Ensure no cross-talk
        assert results['A'] != results['B']


def test_interceptor_read_not_intercepted():
    """Test that read operations are not intercepted."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = os.path.join(tmp_dir, 'read_test.txt')
        with open(test_file, 'w') as f:
            f.write('data')

        with OutputInterceptor() as interceptor:
            with open(test_file) as f:
                content = f.read()
            assert content == 'data'
            assert len(interceptor.get_manifest()) == 0
