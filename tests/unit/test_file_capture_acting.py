import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kodeagent.kodeagent import ReActAgent, CodeActAgent
from kodeagent.models import Task, ChatMessage, ReActChatMessage, CodeActChatMessage
from kodeagent.file_tracker import OutputInterceptor
import os
import tempfile

@pytest.fixture
def mock_planner():
    planner = MagicMock()
    planner.plan = MagicMock()
    planner.create_plan = AsyncMock()
    planner.get_formatted_plan.return_value = "- [ ] Step 1"
    return planner

@pytest.fixture
def mock_observer():
    observer = MagicMock()
    observer.observe = AsyncMock(return_value=None)
    return observer

@pytest.mark.asyncio
async def test_react_agent_tool_file_capture():
    # Ensure interceptor is installed for the test
    from kodeagent.file_tracker import install_interceptor
    install_interceptor()
    
    # A tool that creates a file
    def create_file_tool(content: str, filename: str):
        with open(filename, "w") as f:
            f.write(content)
        return f"File {filename} created"

    create_file_tool.name = "create_file_tool" # Mock tool name

    agent = ReActAgent(
        name="TestAgent",
        model_name="test-model",
        tools=[create_file_tool]
    )
    agent.task = Task(description="test task")
    
    # Mock LLM response to trigger the tool
    # ReActAgent parses response in _think and acts in _act.
    # We'll mock _think yield a response that has an action.
    
    mock_react_msg = ReActChatMessage(
        role="assistant",
        thought="I need to create a file",
        action="create_file_tool",
        args='{"content": "hello", "filename": "output.txt"}'
    )
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # We need to use absolute path for the file in the test
        output_path = os.path.join(tmp_dir, "output.txt")
        safe_path = output_path.replace("\\", "\\\\")
        mock_react_msg.args = f'{{"content": "hello", "filename": "{safe_path}"}}'
        
        # We'll manually call _act for testing isolation
        # _act uses self.messages[-1] as the prompt
        agent.messages.append(mock_react_msg)
        
        # Act
        response_data = None
        async for response in agent._act():
            response_data = response
            
        assert response_data is not None
        assert response_data['type'] == 'step'
        assert response_data['metadata']['tool'] == create_file_tool.name
        assert response_data['metadata']['args'] == {"content": "hello", "filename": output_path}
        assert output_path in response_data['metadata']['generated_files']
        
        assert output_path in agent.task_output_files
        assert output_path in agent.task.output_files
        assert os.path.exists(output_path)

@pytest.mark.asyncio
async def test_code_act_agent_file_capture():
    agent = CodeActAgent(
        name="TestCodeAgent",
        model_name="test-model",
        run_env="host"
    )
    agent.task = Task(description="test task")
    
    # Mock CodeRunner to return a fake file
    fake_file = "/fake/path/generated.png"
    agent.code_runner.run = AsyncMock(return_value=("stdout", "stderr", 0, [fake_file]))
    
    # Mock message with code
    mock_code_msg = CodeActChatMessage(
        role="assistant",
        thought="Writing code",
        code="print('creating file')"
    )
    agent.messages.append(mock_code_msg)
    
    # Act
    async for response in agent._act():
        if response['type'] == 'step':
            assert fake_file in response['metadata']['generated_files']
            
    assert fake_file in agent.task_output_files
    assert fake_file in agent.task.output_files
