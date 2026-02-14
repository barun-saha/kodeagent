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
        history = agent._get_observer_history()

        # Should be empty initially (only system prompt exists)
        assert history == ''

        # Add user message
        agent.add_to_history(ChatMessage(role='user', content='Hello'))
        history = agent._get_observer_history()
        assert '[user]: Hello' in history
        assert 'You are a helpful assistant' not in history

    def test_truncate_long_tool_output(self, agent):
        """Test that detailed tool outputs are truncated."""
        agent.init_history()

        # Create a huge tool output
        huge_text = 'Data ' * 300  # 300 * 5 = 1500 chars
        assert len(huge_text) > 1000

        agent.add_to_history(ChatMessage(role='tool', content=huge_text))
        history = agent._get_observer_history()

        # Verify truncation marker
        assert '... [TRUNCATED]' in history
        # Verify length constraint (roughly 1000 + formatting overhead)
        assert len(history) < 1100

    def test_incremental_caching_reuse(self, agent):
        """Test that the method incrementally updates cache."""
        agent.init_history()

        # Add first message
        msg1 = ChatMessage(role='user', content='Step 1')
        agent.add_to_history(msg1)

        history1 = agent._get_observer_history()
        assert agent._observer_history_idx == 1
        assert agent._observer_history_str == '[user]: Step 1'
        assert history1 == '[user]: Step 1'

        # Add second message
        msg2 = ChatMessage(role='assistant', content='I am thinking...')
        agent.add_to_history(msg2)

        agent._get_observer_history()
        assert agent._observer_history_idx == 2

        # Check that history contains both steps
        assert '[user]: Step 1' in agent._observer_history_str
        assert 'I am thinking...' in agent._observer_history_str
        assert agent._observer_history_str == '[user]: Step 1\n[assistant]: I am thinking...'

    def test_observer_history_consistency(self, agent):
        """Verify the content matches expected string format."""
        agent.init_history()
        agent.add_to_history(ChatMessage(role='user', content='User request'))

        # Mocking a ReAct-style message behavior by string manual check
        # Since we use valid ChatMessage objects, str(msg) is reliable

        history = agent._get_observer_history()
        expected = '[user]: User request'
        assert history == expected

        # Verify formatting matches standard string
        assert agent._message_to_string(agent.chat_history[1]) in history
