import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

import main
import retry_callback


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.http = AsyncMock()
        self.alert = AsyncMock()
        self.sleep = AsyncMock()
        self.database = Mock()
        self.database.table.return_value.insert.return_value.execute = AsyncMock()
        self.embed = AsyncMock(return_value=SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1, 0.2]]),
        ))
        self.enterContext(patch.object(main, "client", self.http))
        self.enterContext(patch.object(main, "sb", self.database))
        self.enterContext(patch.object(main, "co", SimpleNamespace(embed=self.embed)))
        self.enterContext(patch.object(main, "logger", self.alert))
        self.enterContext(patch.object(main, "log", Mock()))
        self.enterContext(patch.object(main.asyncio, "sleep", self.sleep))
        self.payload = {"document_id": "document", "status": "completed"}
        self.row = SimpleNamespace(
            company_name="company", hash="hash", file_name="website", document_id="document",
        )
        self.item = SimpleNamespace(
            datasetId="dataset", limit=150, offset=0, batchNumber=1, databaseRow=self.row,
        )
        self.http.get.return_value = httpx.Response(
            200, json=[{"markdown": "Document text", "url": "https://example.com"}],
            request=httpx.Request("GET", "https://api.apify.com"),
        )

    def callback_response(self, code=200, body=None):
        return httpx.Response(
            code, json=body if body is not None else {"success": True, "status": "completed"},
            request=httpx.Request("POST", main.CALLBACK_URL),
        )

    async def test_lifespan_initializes_an_awaited_supabase_client(self):
        supabase_client = Mock()
        create = AsyncMock(return_value=supabase_client)
        http_client = AsyncMock()
        with (
            patch.object(main, "create_async_client", create),
            patch.object(main.httpx, "AsyncClient", return_value=http_client),
            patch.object(main.cohere, "AsyncClient", return_value=Mock()),
        ):
            async with main.lifespan(main.app):
                create.assert_awaited_once_with(main.SUPABASE_URL, main.SUPABASE_KEY)
                self.assertIs(main.sb, supabase_client)
                main.sb.table("documents_view")
            http_client.aclose.assert_awaited_once()

    async def test_callback_retries_timeout_without_repeating_import(self):
        self.http.post.side_effect = [httpx.ReadTimeout(""), self.callback_response()]
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item])
        self.embed.assert_awaited_once()
        self.database.table.return_value.insert.return_value.execute.assert_awaited_once()
        cleanup.assert_not_awaited()
        self.assertEqual(self.http.post.await_count, 2)
        for call in self.http.post.await_args_list:
            self.assertEqual(call.kwargs["json"], self.payload)
            self.assertEqual(call.kwargs["timeout"].read, 60)
            self.assertEqual(call.kwargs["timeout"].connect, 10)
        self.alert.assert_not_awaited()

    async def test_exhausted_callback_preserves_vectors_and_never_reports_failed(self):
        self.http.post.side_effect = httpx.ReadTimeout("")
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item])
        cleanup.assert_not_awaited()
        self.assertEqual(self.http.post.await_count, 3)
        self.assertTrue(all(c.kwargs["json"]["status"] == "completed" for c in self.http.post.await_args_list))
        self.assertEqual([c.args[0] for c in self.sleep.await_args_list], [1, 2, 5])
        self.alert.assert_awaited_once()
        self.assertIn("ReadTimeout", self.alert.await_args.args[0])
        self.assertIn("document_id=document", self.alert.await_args.args[0])
        self.embed.assert_awaited_once()

    async def test_callback_retries_transient_http_errors(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.http.post.reset_mock()
                self.http.post.side_effect = [self.callback_response(status), self.callback_response()]
                self.assertTrue(await main.send_callback(self.payload))
                self.assertEqual(self.http.post.await_count, 2)

    async def test_callback_retries_connection_failure(self):
        self.http.post.side_effect = [httpx.ConnectError("offline"), self.callback_response()]
        self.assertTrue(await main.send_callback(self.payload))
        self.assertEqual(self.http.post.await_count, 2)

    async def test_callback_does_not_retry_permanent_http_errors(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.http.post.reset_mock()
                self.http.post.side_effect = None
                self.http.post.return_value = self.callback_response(status)
                self.assertFalse(await main.send_callback(self.payload))
                self.http.post.assert_awaited_once()

    async def test_callback_checks_json_success_and_final_status(self):
        for body in (
            {"success": False}, {"success": True, "status": "failed"},
            {"success": True}, [],
        ):
            with self.subTest(body=body):
                self.http.post.reset_mock()
                self.http.post.return_value = self.callback_response(body=body)
                self.assertFalse(await main.send_callback(self.payload))
                self.http.post.assert_awaited_once()

    async def test_callback_rejects_non_json_success_response(self):
        self.http.post.return_value = httpx.Response(
            200, text="upstream proxy", request=httpx.Request("POST", main.CALLBACK_URL),
        )
        self.assertFalse(await main.send_callback(self.payload))
        self.http.post.assert_awaited_once()

    async def test_failed_verification_does_not_delete_vectors(self):
        self.http.post.return_value = self.callback_response(body={"success": True, "status": "failed"})
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item])
        cleanup.assert_not_awaited()
        self.assertEqual(self.http.post.await_args.kwargs["json"]["status"], "completed")
        self.alert.assert_awaited_once()

    async def test_import_failure_reports_exception_type_and_stage(self):
        self.embed.side_effect = httpx.ReadTimeout("")
        self.http.post.return_value = self.callback_response(body={"success": True, "status": "failed"})
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item])
        cleanup.assert_awaited_once_with(self.row)
        payload = self.http.post.await_args.kwargs["json"]
        self.assertEqual(payload["status"], "failed")
        self.assertIn("ReadTimeout", payload["error"])
        self.assertIn("embedding batch=1", payload["error"])
        self.database.table.return_value.insert.assert_not_called()

    async def test_cleanup_failure_is_included_in_import_error(self):
        self.embed.side_effect = RuntimeError("embedding unavailable")
        self.http.post.return_value = self.callback_response(body={"success": True, "status": "failed"})
        with patch.object(main, "cleanup_vectors", AsyncMock(side_effect=RuntimeError("statement timeout"))):
            await main.worker([self.item])
        payload = self.http.post.await_args.kwargs["json"]
        self.assertIn("embedding unavailable", payload["error"])
        self.assertIn("vector cleanup: RuntimeError: statement timeout", payload["error"])
        self.assertEqual(self.alert.await_count, 2)

    async def test_download_http_error_is_not_treated_as_dataset(self):
        self.http.get.return_value = httpx.Response(
            503, request=httpx.Request("GET", "https://api.apify.com"),
        )
        self.http.post.return_value = self.callback_response(body={"success": True, "status": "failed"})
        with patch.object(main, "cleanup_vectors", AsyncMock()):
            await main.worker([self.item])
        self.embed.assert_not_awaited()
        payload = self.http.post.await_args.kwargs["json"]
        self.assertIn("Apify download", payload["error"])
        self.assertIn("HTTPStatusError", payload["error"])

    async def test_later_batch_download_timeout_does_not_repeat_saved_embeddings(self):
        response = self.http.get.return_value
        self.http.get.side_effect = [response, httpx.ReadTimeout(""), response]
        self.http.post.return_value = self.callback_response()
        second = SimpleNamespace(datasetId="dataset", limit=150, offset=150, batchNumber=2, databaseRow=self.row)
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item, second])
        cleanup.assert_not_awaited()
        self.assertEqual(self.http.get.await_count, 3)
        self.assertEqual(self.embed.await_count, 2)
        self.assertEqual(self.database.table.return_value.insert.return_value.execute.await_count, 2)
        self.http.post.assert_awaited_once()

    async def test_dataset_download_retries_rate_limit_and_has_explicit_timeout(self):
        response = self.http.get.return_value
        self.http.get.side_effect = [httpx.Response(429, request=httpx.Request("GET", "https://api.apify.com")), response]
        self.assertEqual(await main.download_batch(self.item), response.json())
        self.assertEqual(self.http.get.await_count, 2)
        self.assertEqual(self.http.get.await_args.kwargs["timeout"].read, 60)
        self.assertEqual(self.http.get.await_args.kwargs["timeout"].connect, 10)

    async def test_dataset_download_does_not_retry_permanent_rejection(self):
        self.http.get.return_value = httpx.Response(401, request=httpx.Request("GET", "https://api.apify.com"))
        with self.assertRaises(httpx.HTTPStatusError):
            await main.download_batch(self.item)
        self.http.get.assert_awaited_once()

    async def test_exhausted_dataset_download_reports_a_real_import_failure(self):
        self.http.get.side_effect = httpx.ReadTimeout("")
        self.http.post.return_value = self.callback_response(body={"success": True, "status": "failed"})
        with patch.object(main, "cleanup_vectors", AsyncMock()) as cleanup:
            await main.worker([self.item])
        cleanup.assert_awaited_once_with(self.row)
        self.assertEqual(self.http.get.await_count, 3)
        self.embed.assert_not_awaited()
        self.assertEqual(self.http.post.await_args.kwargs["json"]["status"], "failed")
        self.assertIn("ReadTimeout", self.http.post.await_args.kwargs["json"]["error"])

    async def test_cleanup_batches_are_scoped_to_company_and_hash(self):
        query = Mock()
        for method in ("select", "eq", "order", "limit", "delete", "in_"):
            getattr(query, method).return_value = query
        first = [{"id": str(i)} for i in range(100)]
        query.execute = AsyncMock(side_effect=[
            SimpleNamespace(data=first), Mock(),
            SimpleNamespace(data=[{"id": "last"}]), Mock(), SimpleNamespace(data=[]),
        ])
        self.database.table.return_value = query
        await main.cleanup_vectors(self.row)
        self.assertEqual(query.delete.call_count, 2)
        self.assertEqual(query.in_.call_args_list[0].args, ("id", [r["id"] for r in first]))
        self.assertEqual(query.in_.call_args_list[1].args, ("id", ["last"]))
        self.assertEqual(query.eq.call_count, 10)
        self.assertTrue(all(call.args in (("company_name", "company"), ("file_hash", "hash"))
                            for call in query.eq.call_args_list))
        self.assertEqual(query.limit.call_args.args, (100,))

    async def test_manual_reconciliation_only_sends_completed_callback(self):
        with (
            patch.object(retry_callback.httpx, "AsyncClient") as client,
            patch.object(main, "send_callback", AsyncMock(return_value=True)) as callback,
        ):
            client.return_value.__aenter__.return_value = self.http
            self.assertTrue(await retry_callback.retry("document"))
        callback.assert_awaited_once_with(self.payload)
        self.embed.assert_not_awaited()
        self.database.table.assert_not_called()


if __name__ == "__main__":
    unittest.main()
