import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

import main


class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = AsyncMock()
        self.connection.fetchrow.return_value = {
            "company_context": "Company facts", "base_persona": "Helpful",
            "learned_instructions": "", "hard_rules": [], "live_chat_settings": {},
        }
        self.connection.fetch.return_value = []

        async def connection_override():
            yield self.connection

        main.app.dependency_overrides[main.get_connection] = connection_override
        self.addCleanup(main.app.dependency_overrides.clear)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver")
        self.addAsyncCleanup(self.client.aclose)
        self.logger = AsyncMock()
        self.original_logger = main.logger
        for name, value in (("logger", self.logger), ("generate_reply", AsyncMock(return_value="Answer"))):
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.payload = {"company_name": "company", "message": "Question", "conversationId": "session", "language": None}

    async def test_chat_endpoint_uses_shared_router_and_accepts_null_language(self):
        response = await self.client.post("/chatbot", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Answer", "attachments_processed": 0, "attachments_skipped": 0})
        args = main.generate_reply.await_args.args
        self.assertIn("Question", args[4])
        self.assertEqual(set(args[5]), {"knowledge_base", "human_handoff", "not_resolved"})

    async def test_test_chat_and_copilot_also_use_shared_router(self):
        payloads = {
            "/test_chatbot": {"user_id": "user", "company_name": "company", "message": "Question", "message_history": []},
            "/copilot": {"company_name": "company", "message": "Question", "command": "", "context_session_id": "session", "copilot_message_history": [], "active_conversation_history": []},
        }
        for endpoint, payload in payloads.items():
            with self.subTest(endpoint=endpoint):
                response = await self.client.post(endpoint, json=payload)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"message": "Answer"})
                self.assertEqual(set(main.generate_reply.await_args.args[5]), {"knowledge_base"})

    async def test_failed_current_attachment_is_not_hidden_by_valid_historical_file(self):
        old = main.FileAttachment(name="old.pdf", size=3, type="application/pdf", url="https://example.com/old.pdf")
        current = old.model_copy(update={"name": "current.pdf", "url": "https://example.com/current.pdf"})
        self.connection.fetch.return_value = [{"file_attachments": [old.model_dump()]}]
        with patch.object(main, "download_attachment", new=AsyncMock(side_effect=[
            main.DownloadedAttachment(attachment=old, content=b"old"), RuntimeError("expired"),
        ])):
            response = await self.client.post("/chatbot", json={**self.payload, "file_attachments": [current.model_dump()]})
        self.assertEqual(response.json()["attachments_processed"], 0)
        self.assertIn("załączników", response.json()["message"])
        main.generate_reply.assert_not_awaited()

    async def test_downloaded_attachment_reaches_router_without_gemini_dependency(self):
        file = main.FileAttachment(name="file.pdf", size=3, type="application/pdf", url="https://example.com/file.pdf")
        downloaded = main.DownloadedAttachment(attachment=file, content=b"pdf")
        with patch.object(main, "download_attachment", new=AsyncMock(return_value=downloaded)), patch.object(main, "client", None):
            response = await self.client.post("/chatbot", json={**self.payload, "file_attachments": [file.model_dump()]})
        self.assertEqual(response.json()["attachments_processed"], 1)
        self.assertEqual(main.generate_reply.await_args.args[6], [downloaded])

    async def test_failure_returns_nonempty_message_and_error_flag(self):
        main.generate_reply.side_effect = RuntimeError("both providers failed")
        response = await self.client.post("/chatbot", json=self.payload)
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json()["error"])
        self.assertTrue(response.json()["message"])
        await asyncio.sleep(0)
        self.logger.assert_awaited_once()

    async def test_overall_request_deadline_returns_controlled_error(self):
        async def slow_router(*args):
            await asyncio.sleep(10)

        main.generate_reply.side_effect = slow_router
        with patch.object(main, "CHAT_REQUEST_TIMEOUT_SECONDS", 0.01):
            response = await self.client.post("/chatbot", json=self.payload)
        self.assertEqual(response.status_code, 504)
        self.assertTrue(response.json()["error"])

    async def test_handoff_and_unresolved_updates_are_scoped_to_company(self):
        await main.call_human(self.connection, "summary", "session", "company", "Ada")
        args = self.connection.execute.await_args.args
        self.assertIn("AND company_name = $5", args[0])
        self.assertEqual(args[-2:], ("session", "company"))
        await main.unresolved(self.connection, "session", "company")
        args = self.connection.execute.await_args.args
        self.assertIn("AND company_name = $2", args[0])
        self.assertEqual(args[1:], ("session", "company"))

    async def test_logger_posts_to_existing_error_endpoint(self):
        http_client = AsyncMock()
        with patch.object(main, "httpx_client", http_client):
            await self.original_logger("error", "trace")
        http_client.post.assert_awaited_once_with(
            "http://log-error_container:8000/log-error",
            json={"app_name": "chatbot", "error_message": "error", "traceback": "trace"},
        )

    async def test_database_acquisition_has_a_bounded_wait(self):
        context = AsyncMock()
        context.__aenter__.side_effect = TimeoutError
        pool = Mock()
        pool.acquire.return_value = context
        with patch.object(main, "db_pool", pool):
            with self.assertRaises(main.HTTPException) as error:
                await anext(main.get_connection())
        self.assertEqual(error.exception.status_code, 503)
        pool.acquire.assert_called_once_with(timeout=5)

    async def test_lifespan_closes_all_provider_clients_and_database(self):
        pool = AsyncMock()
        cohere_context = AsyncMock()
        http_context = AsyncMock()
        gemini = Mock()
        gemini.aio.aclose = AsyncMock()
        with (
            patch.object(main.asyncpg, "create_pool", new=AsyncMock(return_value=pool)),
            patch.object(main.cohere, "AsyncClient", return_value=cohere_context),
            patch.object(main.httpx, "AsyncClient", return_value=http_context),
            patch.object(main.genai, "Client", return_value=gemini),
            patch.object(main, "GEMINI_API_KEY", "test-key"),
        ):
            async with main.lifespan(main.app):
                pass
        pool.close.assert_awaited_once()
        gemini.aio.aclose.assert_awaited_once()
        gemini.close.assert_called_once()
        cohere_context.__aexit__.assert_awaited_once()
        http_context.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
