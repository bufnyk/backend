import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

import main
import retry_callback


class CallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.http = AsyncMock()
        self.http.get.return_value = httpx.Response(
            200, content=b"Document text", request=httpx.Request("GET", "https://example.com/file"),
        )
        self.context = AsyncMock()
        pool = Mock()
        pool.acquire.return_value = self.context
        self.connection = self.context.__aenter__.return_value
        self.connection.fetchval.return_value = 0
        self.connection.transaction = Mock(return_value=AsyncMock())
        self.database = Mock()
        self.insert = AsyncMock()
        self.database.table.return_value.insert.return_value.execute = self.insert
        self.embed = AsyncMock(return_value=SimpleNamespace(embeddings=SimpleNamespace(float=[[0.1, 0.2]])))
        self.original_cleanup = main.cleanup_vectors
        self.cleanup = AsyncMock()
        self.alert = AsyncMock()
        self.sleep = AsyncMock()
        for name, value in (
            ("client", self.http), ("db_pool", pool), ("sb", self.database),
            ("co", SimpleNamespace(embed=self.embed)), ("cleanup_vectors", self.cleanup),
            ("logger", self.alert), ("log", Mock()),
        ):
            self.enterContext(patch.object(main, name, value))
        self.enterContext(patch.object(main.asyncio, "sleep", self.sleep))
        self.data = main.Payload(
            event="upload", document_id="document", file_url="https://example.com/file",
            metadata=main.Metadata(company_name="company", original_filename="file.txt", file_hash="hash"),
            callback_url=main.CALLBACK_URL,
        )
        self.payload = {"document_id": "document", "status": "completed"}

    def response(self, status=200, body=None):
        return httpx.Response(
            status, json=body if body is not None else {"success": True, "status": "completed"},
            request=httpx.Request("POST", main.CALLBACK_URL),
        )

    async def test_callback_timeout_retries_only_reporting_after_connection_is_released(self):
        attempts = 0

        async def post(*args, **kwargs):
            nonlocal attempts
            self.context.__aexit__.assert_awaited_once()
            self.assertEqual(kwargs["json"], self.payload)
            self.assertEqual(kwargs["timeout"].read, 60)
            self.assertEqual(kwargs["timeout"].connect, 10)
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("")
            return self.response()

        self.http.post.side_effect = post
        await main.worker(self.data)
        self.assertEqual(attempts, 2)
        self.embed.assert_awaited_once()
        self.insert.assert_awaited_once()
        self.cleanup.assert_not_awaited()
        self.alert.assert_not_awaited()

    async def test_exhausted_completion_timeout_never_deletes_vectors_or_reports_failed(self):
        self.http.post.side_effect = httpx.ReadTimeout("")
        await main.worker(self.data)
        self.assertEqual(self.http.post.await_count, 3)
        self.assertTrue(all(c.kwargs["json"] == self.payload for c in self.http.post.await_args_list))
        self.cleanup.assert_not_awaited()
        self.insert.assert_awaited_once()
        self.embed.assert_awaited_once()
        self.alert.assert_awaited_once()
        self.assertIn("ReadTimeout", self.alert.await_args.args[0])
        self.assertIn("document_id=document", self.alert.await_args.args[0])
        self.assertEqual([c.args[0] for c in self.sleep.await_args_list], [0.5, 2, 5])

    async def test_transient_statuses_and_connection_errors_are_retried(self):
        for first in (self.response(429), self.response(500), self.response(503), httpx.ConnectError("offline")):
            with self.subTest(first=first):
                self.http.post.reset_mock()
                self.http.post.side_effect = [first, self.response()]
                self.assertTrue(await main.send_callback(self.payload))
                self.assertEqual(self.http.post.await_count, 2)

    async def test_permanent_http_errors_are_not_retried(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.http.post.reset_mock()
                self.http.post.return_value = self.response(status)
                self.assertFalse(await main.send_callback(self.payload))
                self.http.post.assert_awaited_once()

    async def test_application_rejection_never_rolls_back_vectors(self):
        self.http.post.return_value = self.response(body={"success": True, "status": "failed"})
        await main.worker(self.data)
        self.http.post.assert_awaited_once()
        self.cleanup.assert_not_awaited()
        self.assertIn("expected 'completed'", self.alert.await_args.args[0])

    async def test_invalid_json_results_are_reported(self):
        for body in ({"success": False}, {"success": True}, []):
            with self.subTest(body=body):
                self.http.post.reset_mock()
                self.http.post.return_value = self.response(body=body)
                self.assertFalse(await main.send_callback(self.payload))
                self.http.post.assert_awaited_once()
        self.http.post.return_value = httpx.Response(
            200, text="proxy response", request=httpx.Request("POST", main.CALLBACK_URL),
        )
        self.assertFalse(await main.send_callback(self.payload))

    async def test_import_error_cleans_up_and_reports_nonempty_error(self):
        self.embed.side_effect = httpx.ReadTimeout("")
        self.http.post.return_value = self.response(body={"success": True, "status": "failed"})
        await main.worker(self.data)
        self.cleanup.assert_awaited_once_with(self.data.metadata)
        payload = self.http.post.await_args.kwargs["json"]
        self.assertEqual(payload["status"], "failed")
        self.assertIn("ReadTimeout", payload["error"])
        self.assertIn("embedding chunk_offset=0", payload["error"])
        self.insert.assert_not_awaited()

    async def test_document_download_timeout_recovers_without_repeating_embedding(self):
        response = self.http.get.return_value
        self.http.get.side_effect = [httpx.ReadTimeout(""), response]
        self.http.post.return_value = self.response()
        await main.worker(self.data)
        self.assertEqual(self.http.get.await_count, 2)
        self.assertEqual(self.http.get.await_args.kwargs["timeout"].read, 60)
        self.embed.assert_awaited_once()
        self.insert.assert_awaited_once()
        self.cleanup.assert_not_awaited()

    async def test_document_download_does_not_retry_a_missing_file(self):
        self.http.get.return_value = httpx.Response(404, request=httpx.Request("GET", self.data.file_url))
        with self.assertRaises(httpx.HTTPStatusError):
            await main.download_document(self.data.file_url)
        self.http.get.assert_awaited_once()

    async def test_document_download_retries_transient_http_errors(self):
        response = self.http.get.return_value
        for code in (429, 503):
            with self.subTest(code=code):
                self.http.get.reset_mock()
                self.http.get.side_effect = [httpx.Response(code, request=httpx.Request("GET", self.data.file_url)), response]
                self.assertIs(await main.download_document(self.data.file_url), response)
                self.assertEqual(self.http.get.await_count, 2)

    async def test_failed_callback_retries_do_not_repeat_cleanup(self):
        self.embed.side_effect = ValueError("invalid vectors")
        self.http.post.side_effect = [httpx.ReadTimeout(""), self.response(body={"success": True, "status": "failed"})]
        await main.worker(self.data)
        self.cleanup.assert_awaited_once_with(self.data.metadata)
        self.assertEqual(self.http.post.await_count, 2)
        self.assertTrue(all(c.kwargs["json"]["status"] == "failed" for c in self.http.post.await_args_list))

    async def test_cleanup_failure_is_reported_without_hiding_import_error(self):
        self.embed.side_effect = ValueError("invalid vectors")
        self.cleanup.side_effect = RuntimeError("statement timeout")
        self.http.post.return_value = self.response(body={"success": True, "status": "failed"})
        await main.worker(self.data)
        payload = self.http.post.await_args.kwargs["json"]
        self.assertIn("invalid vectors", payload["error"])
        self.assertIn("vector cleanup: RuntimeError: statement timeout", payload["error"])
        self.assertEqual(self.alert.await_count, 2)

    async def test_crawl_start_does_not_send_completion_callback(self):
        self.data.event = "url.scan"
        self.data.website_url = "https://example.com"
        self.data.single_page_only = True
        self.http.post.return_value = self.response(body={"data": {"defaultDatasetId": "dataset"}})
        await main.worker(self.data)
        self.http.post.assert_awaited_once()
        self.assertIn("api.apify.com", self.http.post.await_args.args[0])
        self.assertEqual(self.connection.execute.await_count, 2)
        self.cleanup.assert_not_awaited()

    async def test_failed_crawl_registration_does_not_delete_existing_vectors(self):
        self.data.event = "url.scan"
        self.data.website_url = "https://example.com"
        self.data.single_page_only = True
        self.connection.execute.side_effect = RuntimeError("database unavailable")
        self.http.post.side_effect = [
            self.response(body={"data": {"defaultDatasetId": "dataset"}}),
            self.response(body={"success": True, "status": "failed"}),
        ]
        await main.worker(self.data)
        self.cleanup.assert_not_awaited()
        self.http.post.assert_awaited_once()
        self.assertIn("dataset_id=dataset", self.alert.await_args.args[0])

    async def test_crawl_launch_timeout_is_not_retried_or_reported_failed(self):
        self.data.event = "url.scan"
        self.data.website_url = "https://example.com"
        self.data.single_page_only = True
        self.http.post.side_effect = httpx.ReadTimeout("")
        await main.worker(self.data)
        self.http.post.assert_awaited_once()
        self.assertEqual(self.http.post.await_args.kwargs["timeout"].read, 60)
        self.cleanup.assert_not_awaited()
        self.assertIn("ReadTimeout", self.alert.await_args.args[0])
        self.assertIn("reconciliation", self.connection.execute.await_args.args[1])

    async def test_rejected_crawl_can_report_a_definitive_failure(self):
        self.data.event = "url.scan"
        self.data.website_url = "https://example.com"
        self.data.single_page_only = True
        self.http.post.side_effect = [self.response(401), self.response(body={"success": True, "status": "failed"})]
        await main.worker(self.data)
        self.assertEqual(self.http.post.await_args.kwargs["json"]["status"], "failed")
        self.cleanup.assert_not_awaited()

    async def test_cleanup_deletes_in_batches_and_filters_company_and_hash(self):
        query = Mock()
        for method in ("select", "eq", "order", "limit", "delete", "in_"):
            getattr(query, method).return_value = query
        ids = [{"id": str(i)} for i in range(100)]
        query.execute = AsyncMock(side_effect=[
            SimpleNamespace(data=ids), Mock(),
            SimpleNamespace(data=[{"id": "last"}]), Mock(), SimpleNamespace(data=[]),
        ])
        self.database.table.return_value = query
        # setUp mocks cleanup for worker tests; exercise the original separately.
        await self.original_cleanup(self.data.metadata)
        self.assertEqual(query.delete.call_count, 2)
        self.assertEqual(query.in_.call_args_list[0].args, ("id", [x["id"] for x in ids]))
        self.assertEqual(query.in_.call_args_list[1].args, ("id", ["last"]))
        self.assertEqual(query.eq.call_count, 10)
        self.assertTrue(all(c.args in (("company_name", "company"), ("file_hash", "hash"))
                            for c in query.eq.call_args_list))
        self.assertEqual(query.limit.call_args.args, (100,))

    async def test_manual_reconciliation_only_reports_completion(self):
        with (
            patch.object(retry_callback.httpx, "AsyncClient") as client,
            patch.object(main, "send_callback", AsyncMock(return_value=True)) as callback,
        ):
            client.return_value.__aenter__.return_value = self.http
            self.assertTrue(await retry_callback.retry("document"))
        callback.assert_awaited_once_with(self.payload)
        self.embed.assert_not_awaited()
        self.database.table.assert_not_called()

    def test_http_error_description_omits_url_credentials(self):
        response = httpx.Response(503, request=httpx.Request("POST", "https://api.apify.com/run?token=secret"))
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            message = main.describe_error("starting crawl", error)
        self.assertIn("HTTP 503", message)
        self.assertNotIn("secret", message)


if __name__ == "__main__":
    unittest.main()
