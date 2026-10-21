import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import docx
import httpx

import main


class DocumentResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = AsyncMock()
        self.connection.fetchval.return_value = 0
        self.transaction = AsyncMock()
        self.connection.transaction = Mock(return_value=self.transaction)
        pool = Mock()
        pool.acquire.return_value = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = self.connection
        self.http_client = AsyncMock()
        self.http_client.get.return_value = httpx.Response(200, content=b"Document text", request=httpx.Request("GET", "https://example.com/file"))
        self.http_client.post.return_value = httpx.Response(201, json={"data": {"defaultDatasetId": "dataset"}}, request=httpx.Request("POST", "https://api.apify.com"))
        self.supabase = Mock()
        self.insert = AsyncMock()
        self.supabase.table.return_value.insert.return_value.execute = self.insert
        self.supabase.table.return_value.delete.return_value.eq.return_value.execute = AsyncMock()
        self.cohere = SimpleNamespace(embed=AsyncMock(return_value=SimpleNamespace(
            embeddings=SimpleNamespace(float=[[0.1, 0.2]]),
        )))
        for name, value in (("db_pool", pool), ("client", self.http_client), ("sb", self.supabase), ("co", self.cohere), ("logger", AsyncMock())):
            patcher = patch.object(main, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(main.asyncio, "sleep", new=AsyncMock())
        patcher.start()
        self.addCleanup(patcher.stop)

    def payload(self, filename="file.txt", mime_type="text/plain", event="upload"):
        return main.Payload(
            event=event, document_id="document", file_url="https://example.com/file",
            website_url="https://example.com", single_page_only=True,
            metadata=main.Metadata(company_name="company", original_filename=filename, file_hash="hash", mime_type=mime_type),
            callback_url="https://example.com/callback",
        )

    def assert_callback_status(self, status):
        self.assertEqual(self.http_client.post.await_args.kwargs["json"]["status"], status)

    async def test_plain_text_mime_is_extracted_and_embedded(self):
        await main.worker(self.payload())
        self.cohere.embed.assert_awaited_once()
        self.assertEqual(self.cohere.embed.await_args.kwargs["texts"], ["Document text"])
        self.insert.assert_awaited_once()
        self.assert_callback_status("completed")

    async def test_docx_standard_mime_is_extracted_and_embedded(self):
        document = docx.Document()
        document.add_paragraph("DOCX text")
        content = io.BytesIO()
        document.save(content)
        self.http_client.get.return_value = httpx.Response(200, content=content.getvalue(), request=httpx.Request("GET", "https://example.com/file"))
        await main.worker(self.payload("file.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        self.assertEqual(self.cohere.embed.await_args.kwargs["texts"], ["DOCX text"])
        self.assert_callback_status("completed")

    async def test_missing_mime_falls_back_to_filename(self):
        await main.worker(self.payload(mime_type=None))
        self.cohere.embed.assert_awaited_once()
        self.assert_callback_status("completed")

    async def test_download_error_cannot_be_embedded_as_document_text(self):
        self.http_client.get.return_value = httpx.Response(404, text="Not found", request=httpx.Request("GET", "https://example.com/file"))
        await main.worker(self.payload())
        self.cohere.embed.assert_not_awaited()
        self.insert.assert_not_awaited()
        self.assert_callback_status("failed")

    async def test_empty_document_is_not_marked_completed(self):
        self.http_client.get.return_value = httpx.Response(200, content=b" \n", request=httpx.Request("GET", "https://example.com/file"))
        await main.worker(self.payload())
        self.cohere.embed.assert_not_awaited()
        self.assert_callback_status("failed")

    async def test_apify_201_uses_separate_statements_in_one_transaction(self):
        await main.worker(self.payload(event="url.scan"))
        self.transaction.__aenter__.assert_awaited_once()
        self.assertEqual(self.connection.execute.await_count, 2)
        first, second = self.connection.execute.await_args_list
        self.assertIn('INSERT INTO "apify_url-scraper"', first.args[0])
        self.assertNotIn("UPDATE", first.args[0])
        self.assertIn("UPDATE document_uploads", second.args[0])
        self.assertEqual(second.args[1:], ("document",))


if __name__ == "__main__":
    unittest.main()
