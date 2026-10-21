import unittest
from unittest.mock import AsyncMock, Mock, patch

import main


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
