"""Unit tests for LLMMessage class in history_formatter.py."""

import pytest

from kodeagent.history_formatter import LLMMessage


class TestLLMMessage:
    """Test LLMMessage dictionary subclass."""

    def test_init_with_standard_and_metadata(self):
        """Test initialization with both standard and metadata keys."""
        msg = LLMMessage(role='assistant', content='hello', _thought='thinking', files=['a.txt'])

        # Standard keys in dict
        assert dict(msg) == {'role': 'assistant', 'content': 'hello'}
        assert msg['role'] == 'assistant'
        assert msg['content'] == 'hello'

        # Metadata as attributes
        assert msg._thought == 'thinking'
        assert msg.files == ['a.txt']

        # Metadata accessible via __getitem__
        assert msg['_thought'] == 'thinking'
        assert msg['files'] == ['a.txt']

    def test_getitem_key_error(self):
        """Test that missing keys raise KeyError (Lines 59-60)."""
        msg = LLMMessage(role='user', content='hi')
        with pytest.raises(KeyError) as excinfo:
            _ = msg['non_existent_key']
        assert 'non_existent_key' in str(excinfo.value)

    def test_get_with_default(self):
        """Test get() method with standard and metadata keys."""
        msg = LLMMessage(role='user', content='hi', _meta='data')
        assert msg.get('role') == 'user'
        assert msg.get('_meta') == 'data'
        assert msg.get('missing', 'default') == 'default'

    def test_contains(self):
        """Test __contains__ for standard and metadata keys."""
        msg = LLMMessage(role='user', content='hi', _meta='data')
        assert 'role' in msg
        assert '_meta' in msg
        assert 'missing' not in msg

    def test_update_multiple_args_error(self):
        """Test that update() with multiple positional args raises TypeError (Lines 75-76)."""
        msg = LLMMessage(role='user')
        with pytest.raises(TypeError) as excinfo:
            msg.update({'a': 1}, {'b': 2})
        assert 'expected at most 1 arguments' in str(excinfo.value)

    def test_update_with_dict(self):
        """Test update() with a dictionary (Lines 78-80)."""
        msg = LLMMessage(role='user')
        msg.update({'content': 'new content', '_meta': 'new meta'})
        assert msg['content'] == 'new content'
        assert msg['_meta'] == 'new meta'
        assert '_meta' not in dict(msg)

    def test_update_with_iterable(self):
        """Test update() with an iterable of pairs (Lines 81-83)."""
        msg = LLMMessage(role='user')
        msg.update([('content', 'iter content'), ('_meta', 'iter meta')])
        assert msg['content'] == 'iter content'
        assert msg['_meta'] == 'iter meta'

    def test_update_with_kwargs(self):
        """Test update() with keyword arguments (Lines 84-85)."""
        msg = LLMMessage(role='user')
        msg.update(content='kw content', _meta='kw meta')
        assert msg['content'] == 'kw content'
        assert msg['_meta'] == 'kw meta'

    def test_setitem_metadata(self):
        """Test setting metadata via __setitem__ (Line 92)."""
        msg = LLMMessage(role='user')
        msg['_new_meta'] = 'value'
        assert msg['_new_meta'] == 'value'
        assert '_new_meta' not in dict(msg)
        assert hasattr(msg, '_new_meta')

    def test_repr(self):
        """Test __repr__ includes both standard and metadata."""
        msg = LLMMessage(role='assistant', content='hi', _thought='think')
        representation = repr(msg)
        assert 'LLMMessage' in representation
        assert "'role': 'assistant'" in representation
        assert "'content': 'hi'" in representation
        assert "'_thought': 'think'" in representation
