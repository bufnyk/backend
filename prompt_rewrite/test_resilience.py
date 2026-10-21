import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import main


def model_response(text="new instructions"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FeedbackResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = AsyncMock()
        self.connection.fetchrow.side_effect = [
            {"is_locked": True}, {"learned_instructions": "old", "hard_rules": "rules"},
        ]
        self.send = AsyncMock(return_value=model_response())
        context = AsyncMock()
        context.__aenter__.return_value = SimpleNamespace(chat=SimpleNamespace(send_async=self.send))
        patcher = patch.object(main, "OpenRouter", return_value=context)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = main.Payload(action="delete", companyName="company", userId="user", timestamp="test", feedbackId="feedback", instruction="", aiMessage="answer", userMessage="question", comment="feedback", importance=3)

    def assert_unlocked(self):
        args = self.connection.execute.await_args.args
        self.assertIn("SET is_locked = false", args[0])
        self.assertEqual(args[1], "company")

    async def test_success_saves_instructions_and_releases_lock(self):
        self.assertEqual(await main.main(self.payload, self.connection), {"status": 200})
        self.assertEqual(self.connection.execute.await_count, 3)
        self.assert_unlocked()

    async def test_fallback_does_not_send_temperature_to_gemini(self):
        self.send.side_effect = [RuntimeError("primary down"), model_response()]
        await main.main(self.payload, self.connection)
        self.assertEqual(self.send.await_args_list[0].kwargs["temperature"], 0.2)
        self.assertEqual(self.send.await_args_list[1].kwargs["model"], "google/gemini-3.1-pro-preview")
        self.assertNotIn("temperature", self.send.await_args_list[1].kwargs)
        self.assert_unlocked()

    async def test_both_models_failing_still_releases_lock_without_saving(self):
        self.send.side_effect = RuntimeError("provider down")
        with self.assertRaises(RuntimeError):
            await main.main(self.payload, self.connection)
        self.connection.execute.assert_awaited_once()
        self.assert_unlocked()

    async def test_database_read_failure_releases_lock_without_calling_models(self):
        self.connection.fetchrow.side_effect = [{"is_locked": True}, RuntimeError("database down")]
        with self.assertRaises(RuntimeError):
            await main.main(self.payload, self.connection)
        self.send.assert_not_awaited()
        self.assert_unlocked()

    async def test_empty_model_output_cannot_overwrite_saved_instructions(self):
        self.send.return_value = model_response("  ")
        with self.assertRaises(ValueError):
            await main.main(self.payload, self.connection)
        self.connection.execute.assert_awaited_once()
        self.assert_unlocked()

    async def test_database_write_failure_releases_lock(self):
        self.connection.execute.side_effect = [RuntimeError("write failed"), None]
        with self.assertRaises(RuntimeError):
            await main.main(self.payload, self.connection)
        self.assert_unlocked()


if __name__ == "__main__":
    unittest.main()
