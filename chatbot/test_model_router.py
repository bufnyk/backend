import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from google.genai import types

import model_router as router


def completed(text="answer", output=None):
    return {
        "status": "completed",
        "output": output if output is not None else [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
    }


def function_call(name="knowledge_base", arguments='{"message":"pricing"}', call_id="call-1"):
    return {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id}


class ModelRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.answers = [completed()]
        self.knowledge = AsyncMock(return_value=["Company facts"])
        self.handoff = AsyncMock(return_value="completed")

        async def knowledge_base(message: str):
            return await self.knowledge(message=message)

        async def human_handoff(summary: str = "", client_name: str = ""):
            return await self.handoff(summary=summary, client_name=client_name)

        self.handlers = {"knowledge_base": knowledge_base, "human_handoff": human_handoff}
        self.gemini_chat = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(
            function_calls=None, text="Gemini answer",
        )))
        self.gemini = SimpleNamespace(aio=SimpleNamespace(chats=SimpleNamespace(
            create=Mock(return_value=self.gemini_chat),
        )))
        self.prepare_inputs = AsyncMock(return_value=[])

        async def transport(request):
            self.requests.append(json.loads(request.content))
            result = self.answers.pop(0)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, int):
                return httpx.Response(result, json={"error": {"message": "unavailable"}})
            return httpx.Response(200, json=result)

        self.http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.addAsyncCleanup(self.http_client.aclose)

    async def generate(self, attachments=None, api_key="test-key"):
        return await router.generate_reply(
            self.http_client, api_key, self.gemini, "System rules", "User question",
            self.handlers, attachments or [], self.prepare_inputs,
        )

    async def test_openai_is_primary_and_requests_are_stateless(self):
        self.assertEqual(await self.generate(), "answer")
        body = self.requests[0]
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertFalse(body["store"])
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertNotIn("temperature", body)
        self.assertEqual(body["instructions"], "System rules")
        for tool in body["tools"]:
            self.assertTrue(tool["strict"])
            self.assertFalse(tool["parameters"]["additionalProperties"])
            self.assertEqual(set(tool["parameters"]["required"]), set(tool["parameters"]["properties"]))
        self.gemini.aio.chats.create.assert_not_called()

    async def test_tool_results_and_reasoning_are_passed_back_with_call_id(self):
        reasoning = {"type": "reasoning", "id": "reason-1", "summary": [], "encrypted_content": "encrypted"}
        self.answers = [completed(output=[reasoning, function_call()]), completed("grounded answer")]
        self.assertEqual(await self.generate(), "grounded answer")
        self.knowledge.assert_awaited_once_with(message="pricing")
        inputs = self.requests[1]["input"]
        self.assertEqual(inputs[1], reasoning)
        self.assertEqual(inputs[-1]["call_id"], "call-1")
        self.assertEqual(json.loads(inputs[-1]["output"]), ["Company facts"])

    async def test_provider_errors_fall_back_to_existing_gemini_model(self):
        for error in (429, 500, httpx.ReadTimeout("timed out")):
            with self.subTest(error=type(error).__name__):
                self.answers = [error]
                self.assertEqual(await self.generate(), "Gemini answer")
                kwargs = self.gemini.aio.chats.create.call_args.kwargs
                self.assertEqual(kwargs["model"], "gemini-3.8-flash")
                self.assertTrue(kwargs["config"].automatic_function_calling.disable)
                self.assertIsNone(kwargs["config"].temperature)

    async def test_empty_and_incomplete_openai_responses_trigger_fallback(self):
        for response in (completed(" "), {"status": "incomplete", "output": []}):
            self.answers = [response]
            self.assertEqual(await self.generate(), "Gemini answer")

    async def test_refusal_is_returned_without_fallback(self):
        self.answers = [completed(output=[{
            "type": "message", "content": [{"type": "refusal", "refusal": "Cannot help"}],
        }])]
        self.assertEqual(await self.generate(), "Cannot help")
        self.gemini.aio.chats.create.assert_not_called()

    async def test_missing_openai_key_uses_gemini(self):
        self.assertEqual(await self.generate(api_key=None), "Gemini answer")
        self.assertEqual(self.requests, [])

    async def test_primary_deadline_leaves_time_for_gemini(self):
        async def slow_primary(*args, **kwargs):
            await asyncio.sleep(10)

        with patch.object(router, "PRIMARY_TIMEOUT_SECONDS", 0.01), patch.object(router, "openai_reply", slow_primary):
            self.assertEqual(await self.generate(), "Gemini answer")

    async def test_cancelling_request_does_not_start_fallback(self):
        async def cancelled_primary(*args, **kwargs):
            raise asyncio.CancelledError

        with patch.object(router, "openai_reply", cancelled_primary):
            with self.assertRaises(asyncio.CancelledError):
                await self.generate()
        self.gemini.aio.chats.create.assert_not_called()

    async def test_handoff_is_not_repeated_after_provider_failure(self):
        self.answers = [completed(output=[function_call(
            "human_handoff", '{"summary":"first","client_name":"Ada"}',
        )]), 500]
        self.gemini_chat.send_message.side_effect = [
            SimpleNamespace(function_calls=[types.FunctionCall(
                id="gemini-call-1", name="human_handoff", args={"summary": "different", "client_name": "Ada"},
            )], text=None),
            SimpleNamespace(function_calls=None, text="Handed over"),
        ]
        self.assertEqual(await self.generate(), "Handed over")
        self.handoff.assert_awaited_once_with(summary="first", client_name="Ada")
        first_message = self.gemini_chat.send_message.call_args_list[0].args[0]
        self.assertIn("previous_tool_results", first_message[-1].text)
        function_response = self.gemini_chat.send_message.call_args_list[1].args[0][0].function_response
        self.assertEqual(function_response.id, "gemini-call-1")
        self.assertEqual(function_response.response, {"result": "completed"})

    async def test_invalid_tool_arguments_are_recoverable(self):
        for arguments in ('not-json', '[]', '{"message":123}', '{"unexpected":"value"}'):
            self.answers = [completed(output=[function_call(arguments=arguments)]), completed("Please clarify")]
            self.assertEqual(await self.generate(), "Please clarify")
            self.assertIn("error", json.loads(self.requests[-1]["input"][-1]["output"]))
        self.knowledge.assert_not_awaited()

    async def test_tool_round_limit_stops_loop_before_executing_more_actions(self):
        self.answers = [completed(output=[function_call(arguments=json.dumps({"message": str(i)}))]) for i in range(6)]
        self.assertEqual(await self.generate(), "Gemini answer")
        self.assertEqual(self.knowledge.await_count, 5)

    async def test_final_answer_after_last_allowed_tool_round_is_used(self):
        self.answers = [completed(output=[function_call(arguments=json.dumps({"message": str(i)}))]) for i in range(5)]
        self.answers.append(completed("Final answer"))
        self.assertEqual(await self.generate(), "Final answer")
        self.gemini.aio.chats.create.assert_not_called()

    async def test_pdf_and_image_bytes_are_sent_to_openai(self):
        attachments = [SimpleNamespace(
            attachment=SimpleNamespace(name=name, type=mime), content=content,
        ) for name, mime, content in (
            ("image.png", "image/png", b"image"), ("file.pdf", "application/pdf", b"pdf"),
        )]
        await self.generate(attachments)
        content = self.requests[0]["input"][0]["content"]
        self.assertEqual(content[1], {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="})
        self.assertEqual(content[3], {"type": "input_file", "filename": "file.pdf", "file_data": "data:application/pdf;base64,cGRm"})
        self.prepare_inputs.assert_not_awaited()

    async def test_fallback_receives_same_downloaded_attachments(self):
        attachments = [SimpleNamespace(attachment=SimpleNamespace(name="file.pdf", type="application/pdf"), content=b"pdf")]
        self.answers = [500]
        await self.generate(attachments)
        self.prepare_inputs.assert_awaited_once_with(attachments)

    async def test_both_providers_failing_raises_instead_of_returning_empty_text(self):
        self.answers = [500]
        self.gemini_chat.send_message.return_value = SimpleNamespace(function_calls=None, text="")
        with self.assertRaises(router.ModelResponseError):
            await self.generate()

    async def test_truncated_gemini_response_is_not_returned_as_a_complete_answer(self):
        self.answers = [500]
        self.gemini_chat.send_message.return_value = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.MAX_TOKENS)],
            function_calls=None, text="Partial answer",
        )
        with self.assertRaises(router.ModelResponseError):
            await self.generate()

    async def test_interrupted_mutation_is_not_executed_again(self):
        async def interrupted(summary: str = "", client_name: str = ""):
            raise asyncio.CancelledError

        runner = router.ToolRunner({"human_handoff": interrupted})
        with self.assertRaises(asyncio.CancelledError):
            await runner.run("human_handoff", {})
        result = await runner.run("human_handoff", {"summary": "retry"})
        self.assertIn("interrupted", result["error"])
        self.assertIn("interrupted", runner.history[0]["result"]["error"])


if __name__ == "__main__":
    unittest.main()
