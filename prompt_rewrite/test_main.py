import pytest 
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
    
@pytest.fixture
def database(mocker):

    mock_connection = mocker.AsyncMock()
    mock_connection.fetchrow.return_value = {"learned_instructions": "Brak", "hard_rules": "Brak"}
    mock_connection.execute.return_value=None
  
    mock_pool = mocker.Mock()
    
    mocker.patch("asyncpg.create_pool", new_callable=mocker.AsyncMock, return_value=mock_pool)
    mock_context_manager = mocker.AsyncMock()
    
    mock_pool.acquire.return_value = mock_context_manager
    mock_pool.close = mocker.AsyncMock() #close chce courtine (await db_pool.close())
    mock_context_manager.__aenter__.return_value=mock_connection
    return mock_connection

def test_main_delete(database, client, mocker):
    payload = {
        "action": "delete",
        "companyName": "test",
        "userId": "test_user",
        "timestamp": "test_timestamp",
        "feedbackId": "test_feeedbackid",
        "instruction": "test_instruction",
        "aiMessage": "test_ai_message",
        "userMessage": "test_user_message",
        "comment": "test_comment",
        "importance": 3,
    }
    op_content = mocker.Mock(content="test content")
    op_message = mocker.Mock(message=op_content)
    op_choices = mocker.Mock(choices=[op_message])
    mock_op_client = mocker.AsyncMock()
    mock_op_client.chat.send_async.return_value=op_choices

    mock_op = mocker.AsyncMock()
    mock_op.__aenter__.return_value=mock_op_client

    mocker.patch("main.OpenRouter", return_value=mock_op)

    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": 200}
    assert database.fetchrow.call_count >= 2
    assert database.execute.call_count >= 3
    assert mock_op_client.chat.send_async.call_count == 1

