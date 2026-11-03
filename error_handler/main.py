from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx
import os
import logging
from html import escape
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI()
log = logging.getLogger(__name__)

class ErrorPayload(BaseModel):
    app_name: str
    error_message: str
    traceback: str = "Brak szczegółów"

async def send_to_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML" 
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=httpx.Timeout(15.0, connect=5.0))
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or result.get("ok") is not True:
                raise ValueError("Telegram did not confirm delivery")
        return True
    except Exception as error:
        # Do not log the request URL: it contains the bot token. The complete
        # original alert is already retained in this service's local log.
        log.error("Telegram alert delivery failed (%s)", type(error).__name__)
        return False

@app.post("/log-error")
async def log_error(error: ErrorPayload, background_tasks: BackgroundTasks):
    summary = error.error_message.strip() or next(
        (line.strip() for line in reversed(error.traceback.splitlines()) if line.strip()),
        "Exception without a message",
    )
    log.error("app=%s error=%s\n%s", error.app_name, summary, error.traceback)
    message = (
        f"<b>ALERT: {escape(error.app_name[:100])}</b>\n\n"
        f"<b>Błąd:</b> <code>{escape(summary[:500])}</code>\n\n"
        f"<b>Traceback:</b>\n<pre>{escape(error.traceback[:3000])}</pre>"
    )
    
    background_tasks.add_task(send_to_telegram, message)
    return {"status": "Error logged"}
