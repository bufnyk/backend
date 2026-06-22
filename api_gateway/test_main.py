import pytest
from fastapi.testclient import TestClient
import httpx
from main import app  

@pytest.fixture
def client():
    """
    KRYTYCZNE: Użycie bloku 'with' zmusza FastAPI do uruchomienia funkcji 'lifespan'.
    Bez tego 'with', zmienne globalne (redis_client, httpx_client) pozostałyby jako None.
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_redis(mocker):
    """
    Blokujemy wyjście do prawdziwego Redisa upewniając się, że mock jest asynchroniczny.
    """
    mocker.patch("redis.asyncio.Redis.incr", new_callable=mocker.AsyncMock, return_value=1)
    mocker.patch("redis.asyncio.Redis.expire", new_callable=mocker.AsyncMock, return_value=True)

@pytest.fixture
def mock_httpx_success(mocker):
    """
    Udajemy, że wewnętrzny mikroserwis (np. Chatbot) odpowiada prawidłowo (200 OK).
    """
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"message": "Hello from mock service"}'
    mock_response.headers = {"content-type": "application/json"}
    
    return mocker.patch("httpx.AsyncClient.request", return_value=mock_response)

def test_service_not_found(client):
    """
    Test 1: Sprawdzamy, czy Gateway blokuje zapytania do nieznanych usług.
    Nie potrzebujemy tu mockować Redisa, bo kod odrzuca request wcześniej.
    """
    response = client.get("/nieistniejacy_serwis")
    
    assert response.status_code == 404
    assert response.json() == {"detail": "Service not found"}


def test_successful_proxy_request(client, mock_redis, mock_httpx_success):
    """
    Test 2: Sprawdzamy optymalną ścieżkę (Happy Path).
    Wysyłamy POST do /chatbot i oczekujemy, że Gateway przepuści to dalej.
    """
    payload = {"user_id": "user123", "message": "Test"}
    
    response = client.post("/chatbot", json=payload)
    
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from mock service"}
    
    mock_httpx_success.assert_called_once()
    args, kwargs = mock_httpx_success.call_args
    assert kwargs["url"] == "http://chatbot:8000/chatbot"
    assert kwargs["method"] == "POST"


def test_rate_limit_exceeded(client, mocker):
    """
    Test 3: Symulujemy atak DDoS lub przekroczenie limitu przez użytkownika.
    """
    mocker.patch("redis.asyncio.Redis.incr", new_callable=mocker.AsyncMock, return_value=21)
    mocker.patch("redis.asyncio.Redis.expire", new_callable=mocker.AsyncMock, return_value=True)
    
    payload = {"user_id": "spammer123"}
    response = client.post("/chatbot", json=payload)
    
    assert response.status_code == 429
    assert response.json() == {"detail": "Too many Requests"}


def test_microservice_down(client, mock_redis, mocker):
    """
    Test 4: Inżynieria Niezawodności. Sprawdzamy, co zrobi Gateway, 
    gdy mikroserwis wewnątrz Dockera umrze (np. rzuci błędem połączenia).
    """
    fake_request = httpx.Request(method="POST", url="http://chatbot:8000/chatbot")
    mocker.patch(
        "httpx.AsyncClient.request", 
        side_effect=httpx.RequestError("Connection failed", request=fake_request)
    )
    
    payload = {"user_id": "user123"}
    response = client.post("/chatbot", json=payload)
    
    assert response.status_code == 503
    assert response.json() == {"detail": "Service Temporarly Unavailable"}