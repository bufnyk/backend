import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import main


def model_response(text="new instructions"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FeedbackResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = AsyncMock()
        self.transaction = AsyncMock()
        self.connection.transaction = Mock(return_value=self.transaction)
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

    async def test_feedback_write_failure_rolls_back_the_instruction_update(self):
        self.connection.execute.side_effect = [None, RuntimeError("feedback write failed"), None]
        with self.assertRaisesRegex(RuntimeError, "feedback write failed"):
            await main.main(self.payload, self.connection)
        self.transaction.__aenter__.assert_awaited_once()
        self.assertIs(self.transaction.__aexit__.await_args.args[0], RuntimeError)
        self.assert_unlocked()

    async def test_unlock_failure_rolls_back_both_updates_and_retries_unlock(self):
        self.connection.execute.side_effect = [None, None, RuntimeError("unlock failed"), None]
        with self.assertRaisesRegex(RuntimeError, "unlock failed"):
            await main.main(self.payload, self.connection)
        self.assertIs(self.transaction.__aexit__.await_args.args[0], RuntimeError)
        self.assertEqual(self.connection.execute.await_count, 4)
        self.assert_unlocked()

    async def test_failed_unlock_does_not_mask_original_model_error(self):
        self.send.side_effect = ValueError("invalid model output")
        self.connection.execute.side_effect = RuntimeError("unlock unavailable")
        with patch.object(main.log, "exception") as log:
            with self.assertRaisesRegex(ValueError, "invalid model output"):
                await main.main(self.payload, self.connection)
        log.assert_called_once()


if __name__ == "__main__":
    unittest.main()
