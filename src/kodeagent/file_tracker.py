"""File tracking and interception utility for KodeAgent.
Provides a thread-safe way to monitor and redirect file creation using builtins.open patching.
"""

import builtins
import contextvars
import os
import threading
from typing import Any, Optional

# Context variable to hold the active interceptor for the current thread/context
_active_interceptor: contextvars.ContextVar[Optional['OutputInterceptor']] = contextvars.ContextVar(
    'active_interceptor', default=None
)

# Store the original open
_original_open = builtins.open


class OutputInterceptor:
    """Intercepts file operations in agent code execution or tool calls.
    Redirects writes to a controlled sandbox directory and logs events.
    """

    def __init__(self, sandbox_root: str | None = None):
        """Initialize the interceptor.

        Args:
            sandbox_root: Target directory for file redirection.
             If None, files are tracked but not redirected.
        """
        self.sandbox_root = sandbox_root
        if self.sandbox_root:
            os.makedirs(self.sandbox_root, exist_ok=True)

        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._token: contextvars.Token | None = None

    def _should_intercept(self, mode: str) -> bool:
        """Only intercept write/append modes.

        Args:
            mode: The mode in which to open the file.

        Returns:
            bool: True if the mode is a write/append mode, False otherwise.
        """
        return any(m in mode for m in ('w', 'a', 'x'))

    def intercept_open(
        self,
        original_open,
        path,
        mode='r',
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        """Replacement for builtins.open.

        Args:
            original_open: The original builtins.open function.
            path: The path to the file to open.
            mode: The mode in which to open the file.
            buffering: The buffering mode.
            encoding: The encoding to use.
            errors: Error handling mode.
            newline: Newline handling mode.
            closefd: Whether to close the file descriptor.
            opener: The opener function.
        """
        is_write = self._should_intercept(mode)

        target_path = path
        if is_write and self.sandbox_root:
            # Redirect to sandbox if root is provided
            # We use basename to keep it flat in the sandbox for now
            target_path = os.path.join(self.sandbox_root, os.path.basename(path))

        if is_write:
            with self._lock:
                # Avoid duplicates
                if not any(e['path'] == target_path for e in self._events):
                    self._events.append({'original_path': path, 'path': target_path, 'mode': mode})

        return original_open(
            target_path, mode, buffering, encoding, errors, newline, closefd, opener
        )

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.stop()

    def start(self):
        """Activate interception for the current context."""
        self._token = _active_interceptor.set(self)

    def stop(self):
        """Deactivate interception for the current context."""
        if self._token:
            _active_interceptor.reset(self._token)
            self._token = None

    def get_manifest(self) -> list[str]:
        """Return a list of unique file paths that were written to."""
        with self._lock:
            return [e['path'] for e in self._events]

    def reset(self):
        """Clear logged events."""
        with self._lock:
            self._events.clear()


def _patched_open(
    file,
    mode='r',
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    closefd=True,
    opener=None,
):
    """Global patch for builtins.open that delegates to the active interceptor if any.

    Args:
        file: The file to open.
        mode: The mode in which to open the file.
        buffering: The buffering mode.
        encoding: The encoding to use.
        errors: Error handling mode.
        newline: Newline handling mode.
        closefd: Whether to close the file descriptor.
        opener: The opener function.
    """
    interceptor = _active_interceptor.get()
    if interceptor:
        return interceptor.intercept_open(
            _original_open, file, mode, buffering, encoding, errors, newline, closefd, opener
        )
    return _original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)


# Apply the global patch once
def install_interceptor():
    """Install the global builtins.open patch."""
    builtins.open = _patched_open


def uninstall_interceptor():
    """Restore the original builtins.open."""
    builtins.open = _original_open
