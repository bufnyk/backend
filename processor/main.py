import os
from dotenv import load_dotenv
from supabase import create_async_client
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import httpx
import cohere 
import traceback
import logging
from contextlib import asynccontextmanager

class Row(BaseModel):
    id: int
    created_at: str
    company_name: str
    dataset_id: str
    document_id: str
    hash: str
    file_name: str

class Payload(BaseModel):
    datasetId: str
    limit: int
    offset: int
    totalItems: int
    batchNumber: int 
    databaseRow: Row

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
load_dotenv()
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
co = None
client = None
sb = None
headers = {
    "Content-Type": "application/json",
    "X-Callback-Secret": CALLBACK_SECRET
}
log = logging.getLogger(__name__)
CALLBACK_URL = "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback"
CALLBACK_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
CALLBACK_RETRY_DELAYS = (2, 5)
CLEANUP_BATCH_SIZE = 100


def describe_error(stage, error):
    detail = str(error) or "exception has no message"
    return f"{stage}: {type(error).__name__}: {detail}"


async def send_callback(payload):
    """Retry reporting only; a reporting failure must never roll back vectors."""
    for attempt in range(len(CALLBACK_RETRY_DELAYS) + 1):
        try:
            response = await client.post(
                CALLBACK_URL, headers=headers, json=payload, timeout=CALLBACK_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or result.get("success") is not True:
                raise ValueError("Callback did not confirm success")
            if result.get("status") != payload["status"]:
                raise ValueError(
                    f"Callback returned status {result.get('status')!r}; "
                    f"expected {payload['status']!r}"
                )
            return True
        except Exception as error:
            retryable = isinstance(error, httpx.TransportError) or (
                isinstance(error, httpx.HTTPStatusError)
                and (error.response.status_code == 429 or 500 <= error.response.status_code < 600)
            )
            message = describe_error(
                f"callback {payload['status']} document_id={payload['document_id']} "
                f"attempt={attempt + 1}", error,
            )
            if retryable and attempt < len(CALLBACK_RETRY_DELAYS):
                log.warning("%s; retrying", message)
                await asyncio.sleep(CALLBACK_RETRY_DELAYS[attempt])
                continue
            log.exception("%s; callback requires reconciliation; vectors retained", message)
            await logger(message, traceback.format_exc())
            return False


async def cleanup_vectors(row):
    """Limit each DELETE and restrict it to the importing company."""
    while True:
        result = await (
            sb.table("documents").select("id")
            .eq("company_name", row.company_name).eq("file_hash", row.hash)
            .order("id").limit(CLEANUP_BATCH_SIZE).execute()
        )
        if not result.data:
            return
        await (
            sb.table("documents").delete()
            .eq("company_name", row.company_name).eq("file_hash", row.hash)
            .in_("id", [item["id"] for item in result.data]).execute()
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global co
    global client
    global sb

    co = cohere.AsyncClient(api_key=COHERE_API_KEY)
    client = httpx.AsyncClient()
    sb = await create_async_client(SUPABASE_URL, SUPABASE_KEY)

    yield 

    await client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/loader")
async def main(data: list[Payload], task: BackgroundTasks):
    
    task.add_task(worker, data)

    return {"status": "processing started"}

async def worker(data):
    if not data:
        log.error("Processor received an empty import")
        return
    stage = "starting import"
    try:
        for item in data:

            documents = []
            stage = f"Apify download batch={item.batchNumber} offset={item.offset}"
            response = await client.get(f"https://api.apify.com/v2/datasets/{item.datasetId}/items?limit={item.limit}&offset={item.offset}")
            response.raise_for_status()
            payload = response.json()
            stage = f"splitting batch={item.batchNumber}"
            
            for element in payload:

                if not element.get("markdown"):
                    continue

                documents.append(
                    Document(
                        page_content=element["markdown"],
                        metadata={
                            "source": element["url"],
                            "file_name": item.databaseRow.file_name,
                            "company_name": item.databaseRow.company_name,
                            "hash": item.databaseRow.hash
                        }
                    )
                )

            splitted_text = text_splitter.split_documents(documents)

            ##list_for_cohere = []
            ##for text in splitted_text:
                ##list_for_cohere.append(text.page_content) TO SAMO ^

            list_for_cohere = [text.page_content for text in splitted_text]
            final_embeddings = []

            for i in range(0, len(list_for_cohere), 90):

                stage = f"embedding batch={item.batchNumber} chunk_offset={i}"
                res = await co.embed(
                    texts=list_for_cohere[i: i + 90],
                    input_type="search_document",
                    embedding_types=["float"],
                    model="embed-multilingual-v3.0",
                    )
                final_embeddings.extend(res.embeddings.float)

            final_list = []

            for vector, metadata in zip(final_embeddings, splitted_text):

                final_list.append({"content": metadata.page_content, "metadata":metadata.metadata, "embedding": vector })
            for i in range(0, len(final_list), 100):

                stage = f"vector insert batch={item.batchNumber} chunk_offset={i}"
                await (
                    sb.table("documents_view")
                    .insert(final_list[i: i + 100])
                    .execute()
                )
                
                await asyncio.sleep(1)
            del payload, documents, splitted_text, list_for_cohere, final_embeddings, final_list
    except Exception as e:
        message = describe_error(stage, e)
        log.exception("%s", message)
        await logger(message, traceback.format_exc())
        try:
            await cleanup_vectors(data[0].databaseRow)
        except Exception as b:
            cleanup_error = describe_error("vector cleanup", b)
            log.exception("%s", cleanup_error)
            await logger(cleanup_error, traceback.format_exc())
            message = f"{message}; {cleanup_error}"
        await send_callback({
            "document_id": data[0].databaseRow.document_id,
            "status": "failed",
            "error": message,
        })
        return

    # Finalization is deliberately outside the import/cleanup exception handler.
    await send_callback({
        "document_id": data[0].databaseRow.document_id,
        "status": "completed",
    })
        
async def logger(error_string, traceback_string):
    try:
        response = await client.post(
            "http://log-error_container:8000/log-error",
            json={
                "app_name": "data_processor",
                "error_message": error_string,
                "traceback": traceback_string,
            }
        )
        response.raise_for_status()
    except Exception as e:
        log.error("%s", describe_error("could not deliver error alert", e))
