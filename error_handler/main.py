from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = FastAPI()

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
    
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload, timeout=5.0)

@app.post("/log-error")
async def log_error(error: ErrorPayload, background_tasks: BackgroundTasks):
    
    message = (
        f"<b>ALERT: {error.app_name}</b>\n\n"
        f"<b>Błąd:</b> <code>{error.error_message}</code>\n\n"
        f"<b>Traceback:</b>\n<pre>{error.traceback[:3000]}</pre>"
    )
    
    background_tasks.add_task(send_to_telegram, message)
    return {"status": "Error logged"}