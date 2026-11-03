import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import BackgroundTasks

import main


class ErrorHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sender_uses_mocked_http_without_sending_real_messages(self):
        client = AsyncMock()
        context = AsyncMock()
        context.__aenter__.return_value = client
        client.post.return_value = httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", "https://telegram.test"))
        with patch.object(main.httpx, "AsyncClient", return_value=context):
            self.assertTrue(await main.send_to_telegram("Testowa wiadomość"))
        self.assertEqual(client.post.await_args.kwargs["json"]["text"], "Testowa wiadomość")
        self.assertEqual(client.post.await_args.kwargs["timeout"].read, 15)

    async def test_http_rejection_or_invalid_json_result_cannot_be_reported_as_delivered(self):
        for code, body in ((400, {"ok": False}), (503, {"ok": False}), (200, {"ok": False})):
            with self.subTest(code=code, body=body):
                client = AsyncMock()
                client.post.return_value = httpx.Response(code, json=body, request=httpx.Request("POST", "https://telegram.test"))
                context = AsyncMock()
                context.__aenter__.return_value = client
                with patch.object(main.httpx, "AsyncClient", return_value=context), patch.object(main.log, "error") as log:
                    self.assertFalse(await main.send_to_telegram("alert"))
                    log.assert_called_once()

    async def test_transport_failure_is_logged_without_leaking_bot_token(self):
        client = AsyncMock()
        client.post.side_effect = httpx.ReadTimeout("https://telegram.test/botSECRET/sendMessage")
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch.object(main.httpx, "AsyncClient", return_value=context), patch.object(main.log, "error") as log:
            self.assertFalse(await main.send_to_telegram("alert"))
        self.assertNotIn("SECRET", str(log.call_args))

    async def test_empty_exception_message_uses_traceback_type_and_is_retained_locally(self):
        tasks = BackgroundTasks()
        trace = "Traceback:\nhttpx.ReadTimeout"
        with patch.object(main.log, "error") as log:
            await main.log_error(main.ErrorPayload(app_name="worker", error_message="", traceback=trace), tasks)
        self.assertIn("httpx.ReadTimeout", tasks.tasks[0].args[0])
        self.assertEqual(log.call_args.args[-1], trace)

    async def test_traceback_html_is_escaped_before_it_is_queued(self):
        tasks = BackgroundTasks()
        result = await main.log_error(main.ErrorPayload(
            app_name="chatbot <test>", error_message="invalid <value>", traceback='File "<stdin>", a & b',
        ), tasks)
        self.assertEqual(result, {"status": "Error logged"})
        self.assertEqual(len(tasks.tasks), 1)
        text = tasks.tasks[0].args[0]
        self.assertIn("&lt;stdin&gt;", text)
        self.assertIn("a &amp; b", text)
        self.assertNotIn("<value>", text)
        self.assertIn("<pre>", text)


if __name__ == "__main__":
    unittest.main()
