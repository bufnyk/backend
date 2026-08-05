from unittest.mock import AsyncMock
import json

import httpx
import pytest
import pytest_asyncio

from vectix.backend.api_gateway import main as gateway


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=gateway.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def mock_redis(monkeypatch):
    redis_client = AsyncMock()
    redis_client.incr.return_value = 1
    redis_client.expire.return_value = True
    redis_client.ping.return_value = True
    monkeypatch.setattr(gateway, "redis_client", redis_client)
    return redis_client


@pytest.fixture
def mock_httpx_success(monkeypatch):
    upstream_response = httpx.Response(
        status_code=200,
        json={"message": "Hello from mock service"},
        request=httpx.Request("POST", "http://chatbot:8000/chatbot"),
    )
    http_client = AsyncMock()
    http_client.request.return_value = upstream_response
    monkeypatch.setattr(gateway, "httpx_client", http_client)
    return http_client.request


@pytest.mark.asyncio
async def test_service_not_found(client):
    response = await client.get("/nieistniejacy_serwis")

    assert response.status_code == 404
    assert response.json() == {"detail": "Service not found"}


@pytest.mark.asyncio
async def test_successful_proxy_request(client, mock_redis, mock_httpx_success):
    payload = {
        "user_id": "user123",
        "message": "Test",
        "file_attachments": [
            {
                "name": "sample.pdf",
                "size": 123,
                "type": "application/pdf",
                "url": "https://example.com/sample.pdf",
            }
        ],
    }

    response = await client.post("/chatbot", json=payload)

    assert response.status_code == 200
    assert response.json() == {"message": "Hello from mock service"}

    mock_httpx_success.assert_awaited_once()
    _, kwargs = mock_httpx_success.call_args
    assert kwargs["url"] == "http://chatbot:8000/chatbot"
    assert kwargs["method"] == "POST"
    assert json.loads(kwargs["content"]) == payload


@pytest.mark.asyncio
async def test_rate_limit_exceeded(client, mock_redis, monkeypatch):
    mock_redis.incr.return_value = 21
    monkeypatch.setattr(gateway, "httpx_client", AsyncMock())

    response = await client.post("/chatbot", json={"user_id": "spammer123"})

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many Requests"}


@pytest.mark.asyncio
async def test_microservice_down(client, mock_redis, monkeypatch):
    fake_request = httpx.Request(method="POST", url="http://chatbot:8000/chatbot")
    http_client = AsyncMock()
    http_client.request.side_effect = httpx.RequestError("Connection failed", request=fake_request)
    monkeypatch.setattr(gateway, "httpx_client", http_client)

    response = await client.post("/chatbot", json={"user_id": "user123"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Service temporarily unavailable"}


@pytest.mark.asyncio
async def test_proxy_preserves_subpath_and_query(client, mock_redis, mock_httpx_success):
    response = await client.patch("/chatbot/messages/123?include=files", json={"message": "Test"})

    assert response.status_code == 200
    _, kwargs = mock_httpx_success.call_args
    assert kwargs["url"] == "http://chatbot:8000/chatbot/messages/123"
    assert str(kwargs["params"]) == "include=files"
    assert kwargs["method"] == "PATCH"


@pytest.mark.asyncio
async def test_cors_preflight_is_handled_by_gateway(client):
    response = await client.options(
        "/chatbot",
        headers={
            "Origin": "https://customer.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.mark.asyncio
async def test_health_checks_redis(client, mock_redis):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_redis.ping.assert_awaited_once()
