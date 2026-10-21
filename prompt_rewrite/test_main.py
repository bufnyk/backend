import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import main


class FeedbackEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_feedback_over_http(self):
        connection = AsyncMock()
        connection.fetchrow.return_value = {"learned_instructions": "Brak", "hard_rules": "Brak"}

        async def connection_override():
            yield connection

        main.app.dependency_overrides[main.get_connection] = connection_override
        self.addCleanup(main.app.dependency_overrides.clear)
        send = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="test content"),
        )]))
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(send_async=send))
        payload = {
            "action": "delete", "companyName": "test", "userId": "test_user",
            "timestamp": "test", "feedbackId": "feedback", "instruction": "test",
            "aiMessage": "test", "userMessage": "test", "comment": "test", "importance": 3,
        }
        with patch.object(main, "OpenRouter", return_value=context):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as client:
                response = await client.post("/feedback", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": 200})
        self.assertEqual(connection.execute.await_count, 3)
        send.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()

