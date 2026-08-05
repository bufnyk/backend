import unittest
from unittest.mock import AsyncMock, patch

from google.genai import types

import main


def attachment(name: str, size: int = 3) -> main.FileAttachment:
    return main.FileAttachment(
        name=name,
        size=size,
        type="image/jpeg",
        url=f"https://ntnoptppvhvezqbdbcyl.supabase.co/storage/v1/object/public/chat-files/{name}",
    )


class AttachmentParsingTests(unittest.TestCase):
    def test_parses_json_and_nested_history_attachments(self):
        raw = '[{"name":"one.jpg","size":3,"type":"image/jpeg","url":"https://example.com/one.jpg"}]'
        parsed = main.parse_file_attachments(raw)
        history = main.get_history_attachments([
            {"file_attachments": raw},
            {"message": {"file_attachments": [attachment("two.jpg").model_dump()]}},
        ])

        self.assertEqual([item.name for item in parsed], ["one.jpg"])
        self.assertEqual([item.name for item in history], ["one.jpg", "two.jpg"])

    def test_merges_multiple_sources_and_deduplicates_by_url(self):
        first = attachment("first.jpg")
        second = attachment("second.jpg")
        third = attachment("third.jpg")

        merged = main.merge_and_limit_attachments(
            [first, second],
            [first],
            [third],
        )

        self.assertEqual([item.name for item in merged], ["first.jpg", "second.jpg", "third.jpg"])

    def test_prioritizes_the_newest_files_when_history_exceeds_the_limit(self):
        files = [attachment("oldest.jpg"), attachment("newer.jpg"), attachment("current.jpg")]

        with patch.object(main, "MAX_CHAT_ATTACHMENTS", 2):
            merged = main.merge_and_limit_attachments(files)

        self.assertEqual([item.name for item in merged], ["newer.jpg", "current.jpg"])


class StoredAttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_current_and_historical_attachments_from_database(self):
        connection = AsyncMock()
        connection.fetch.return_value = [
            {"file_attachments": [attachment("newest.jpg").model_dump()]},
            {"file_attachments": [attachment("oldest.jpg").model_dump()]},
        ]

        stored = await main.get_stored_attachments(connection, "conversation-id", "company")

        self.assertEqual([item.name for item in stored], ["oldest.jpg", "newest.jpg"])
        connection.fetch.assert_awaited_once()

    async def test_builds_multiple_inline_file_parts(self):
        files = [attachment("one.jpg"), attachment("two.jpg")]
        downloaded = [
            main.DownloadedAttachment(attachment=files[0], content=b"one"),
            main.DownloadedAttachment(attachment=files[1], content=b"two"),
        ]

        with patch.object(main, "download_attachment", new=AsyncMock(side_effect=downloaded)):
            inputs, processed, failed = await main.create_attachment_inputs(files)

        self.assertEqual(len(inputs), 4)
        self.assertEqual(inputs[1].inline_data.data, b"one")
        self.assertEqual(inputs[3].inline_data.data, b"two")
        self.assertEqual([item.name for item in processed], ["one.jpg", "two.jpg"])
        self.assertEqual(failed, 0)

    async def test_skips_an_unavailable_historical_file_without_losing_valid_files(self):
        files = [attachment("expired.jpg"), attachment("valid.jpg")]
        downloaded = main.DownloadedAttachment(attachment=files[1], content=b"valid")

        with patch.object(
            main,
            "download_attachment",
            new=AsyncMock(side_effect=[RuntimeError("expired"), downloaded]),
        ):
            inputs, processed, failed = await main.create_attachment_inputs(files)

        self.assertEqual(len(inputs), 2)
        self.assertEqual([item.name for item in processed], ["valid.jpg"])
        self.assertEqual(failed, 1)

    async def test_uses_file_api_after_inline_budget_is_exceeded(self):
        file = attachment("large.jpg")
        downloaded = main.DownloadedAttachment(attachment=file, content=b"large")
        uploaded = types.File(
            name="files/large",
            uri="https://generativelanguage.googleapis.com/v1beta/files/large",
            mime_type="image/jpeg",
        )
        fake_client = AsyncMock()
        fake_client.aio.files.upload.return_value = uploaded

        with (
            patch.object(main, "MAX_INLINE_ATTACHMENTS_SIZE", 0),
            patch.object(main, "client", fake_client),
            patch.object(main, "download_attachment", new=AsyncMock(return_value=downloaded)),
        ):
            inputs, processed, failed = await main.create_attachment_inputs([file])

        self.assertEqual(inputs[1].uri, uploaded.uri)
        self.assertEqual([item.name for item in processed], ["large.jpg"])
        self.assertEqual(failed, 0)
        fake_client.aio.files.upload.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
