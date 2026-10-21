import unittest
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks

import main


class ErrorHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sender_uses_mocked_http_without_sending_real_messages(self):
        client = AsyncMock()
        context = AsyncMock()
        context.__aenter__.return_value = client
        with patch.object(main.httpx, "AsyncClient", return_value=context):
            await main.send_to_telegram("Testowa wiadomość")
        self.assertEqual(client.post.await_args.kwargs["json"]["text"], "Testowa wiadomość")

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
