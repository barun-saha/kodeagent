"""Test observer history optimization."""

import pytest

from kodeagent.kodeagent import ChatMessage, ReActAgent


class TestObserverOptimization:
    """Test observer history optimization."""

    @pytest.fixture
    def agent(self):
        """Fixture to create a ReActAgent instance."""
        # Create a simple agent for testing
        return ReActAgent(
            name='TestAgent',
            model_name='test-model',
            tools=[],
            description='A test agent',
            system_prompt='You are a helpful assistant.',
        )

    def test_exclude_system_prompt(self, agent):
        """Test that system prompt is excluded from observer history."""
        # Initialize history (adds system prompt)
        agent.init_history()

        # Verify system prompt is in main messages
        assert agent.chat_history[0]['role'] == 'system'

        # Get observer history
        history = agent.get_history()

        # Should be empty initially (only system prompt exists)
        assert history == ''

        # Add user message
        agent.add_to_history(ChatMessage(role='user', content='Hello'))
        history = agent.get_history()
        assert '[user]: Hello' in history
        assert 'You are a helpful assistant' not in history

    def test_truncate_long_tool_output(self, agent):
        """Test that detailed tool outputs are truncated."""
        agent.init_history()

        # Create a huge tool output
        huge_text = 'Data ' * 300  # 300 * 5 = 1500 chars
        assert len(huge_text) > 1000

        agent.add_to_history(ChatMessage(role='tool', content=huge_text))
        history = agent.get_history()

        # Verify truncation marker
        assert '... [TRUNCATED]' in history
        # Verify length constraint (roughly 1000 + formatting overhead)
        assert len(history) < 1100

