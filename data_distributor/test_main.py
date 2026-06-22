import pytest
from fastapi.testclient import TestClient
import httpx
from main import app, worker, Payload

@pytest.fixture
def client(mocker):

    mocker.patch("asyncpg.create_pool", new_callable=mocker.AsyncMock)
    mocker.patch("main.create_async_client")
    mocker.patch("cohere.AsyncClient")

    with TestClient(app) as client:
        yield client

@pytest.fixture
def database(mocker):
    # 1. Fałszywe połączenie, które zwraca wyniki zapytań SQL
    fake_connection = mocker.AsyncMock()
    fake_connection.fetchval.return_value = 1
    fake_connection.execute.return_value = None

    # 2. Pula połączeń musi być ZWYKŁYM Mockiem (bo db_pool.acquire() nie ma 'await')
    fake_pool = mocker.Mock()

    # 3. Tworzymy obiekt asynchronicznego menedżera kontekstu (który obsługuje async with)
    fake_context_manager = mocker.AsyncMock()
    fake_context_manager.__aenter__.return_value = fake_connection

    # 4. Spinamy to w całość: acquire() zwraca menedżera, a menedżer oddaje połączenie
    fake_pool.acquire.return_value = fake_context_manager

    # 5. Patchujemy globalną zmienną w main.py
    mocker.patch("main.db_pool", fake_pool)
    return fake_connection

@pytest.fixture
def mock_supabase(mocker):
    mock_sb = mocker.Mock()
    
    # Udajemy: table() -> zwraca obiekt -> ma insert() -> zwraca obiekt -> ma asynchroniczne execute()
    mock_execute = mocker.AsyncMock()
    mock_sb.table.return_value.insert.return_value.execute = mock_execute
    mock_sb.table.return_value.delete.return_value.eq.return_value.execute = mock_execute
    
    mocker.patch("main.sb", mock_sb)
    return mock_sb

@pytest.fixture
def mock_httpx_router(mocker):

    async def router(url, *args, **kwargs):
        if "api.apify.com" in str(url):
            return mocker.Mock(status_code=200, json=lambda: {"data": {"defaultDatasetId": "dataset-123"}})
        
        elif "supabase.co/functions" in str(url):
            return mocker.Mock(status_code=200)
            
        elif "log-error_container" in str(url):
            return mocker.Mock(status_code=200)
            
        raise ValueError(f"Unknown URL: {url}")

    return mocker.patch("main.client.post", side_effect=router)

def test_successful_send(client, mocker):

    mock_worker = mocker.patch("main.worker", new_callable=mocker.AsyncMock)

    payload = {
        "metadata": {
            "company_name": "test",
            "original_filename": "test",
            "file_hash": "random_number",
            },
        "event": "upload",
        "document_id": "random_id",
        "callback_url": "callback_url"
    }
    response = client.post("/document", json=payload)

    assert response.status_code == 200 
    assert response.json() == {"status": "started"}

    mock_worker.assert_called_once()

@pytest.mark.asyncio
async def test_apify_send_success(mock_httpx_router, database, mocker):
    # Blokujemy pauzy czasowe
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    
    payload_dict = {
        "metadata": {
            "company_name": "test",
            "original_filename": "test.pdf",
            "file_hash": "hash123",
            "mime_type": "application/pdf"
        },
        "event": "url.scan",
        "document_id": "doc-123",
        "website_url": "https://example.com",
        "callback_url": "https://callback.com",
        "single_page_only": True
    }
    test_payload = Payload(**payload_dict)
    
    await worker(test_payload)

    # 1. Sprawdzamy bazę (czy licznik skanów został odpytany i czy wykonano UPDATE/INSERT)
    assert database.fetchval.call_count >= 1
    assert database.execute.call_count == 1
    
    # 2. Sprawdzamy uderzenia do sieci (Router HTTPX)
    called_urls = [str(call.args[0]) for call in mock_httpx_router.call_args_list]
    assert any("api.apify.com" in url for url in called_urls)


@pytest.mark.asyncio
async def test_apify_send_failed_database_crash(mock_httpx_router, mock_supabase, database, mocker):
    mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock)
    
    # Symulujemy twardy błąd bazy danych w momencie próby zapisu
    database.execute.side_effect = Exception("Krytyczny błąd bazy danych")
    
    payload_dict = {
        "metadata": {
            "company_name": "test",
            "original_filename": "test.pdf",
            "file_hash": "hash123",
            "mime_type": "application/pdf"
        },
        "event": "url.scan",
        "document_id": "doc-123",
        "website_url": "https://example.com",
        "callback_url": "https://callback.com",
        "single_page_only": True
    }
    test_payload = Payload(**payload_dict)
    
    await worker(test_payload)
    
    # 1. Sprawdzamy historię uderzeń do sieci
    called_urls = [str(call.args[0]) for call in mock_httpx_router.call_args_list]
    
    # 2. Upewniamy się, że worker uderzył do endpointu z błędem
    assert any("document-callback" in url for url in called_urls)
    
    # 3. Wyciągamy payload błędu wysłany do Supabase
    error_call = [call for call in mock_httpx_router.call_args_list if "document-callback" in str(call.args[0])][0]
    sent_json = error_call.kwargs["json"]
    
    assert sent_json["status"] == "failed"
    assert "Krytyczny błąd bazy danych" in sent_json["error"]



