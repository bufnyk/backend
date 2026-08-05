import pytest
from vectix.backend.error_handler.main import send_to_telegram

@pytest.mark.asyncio
async def test_sender():
    result = await send_to_telegram("Testowa wiadomość")
    assert result == None