import os
import gc
from dotenv import load_dotenv
from supabase import create_async_client
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import httpx
import cohere 
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
CALLBACK_SECRET = os.getenvb("CALLBACK_SECRET")
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global co
    global client
    global sb

    co = cohere.AsyncClient(api_key=COHERE_API_KEY)
    client = httpx.AsyncClient()
    sb = create_async_client(SUPABASE_URL, SUPABASE_KEY)

    yield 

    await client.aclose()

app = FastAPI(lifespan=lifespan)

@app.post("/loader")
async def main(data: list[Payload], task: BackgroundTasks):
    
    task.add_task(worker, data)

    return {"status": "processing started"}

async def worker(data):
    try:
        for item in data:

            documents = []
            payload = (await client.get(f"https://api.apify.com/v2/datasets/{item.datasetId}/items?limit={item.limit}&offset={item.offset}")).json()
            
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

                await (
                    sb.table("documents_view")
                    .insert(final_list[i: i + 100])
                    .execute()
                )
                
                await asyncio.sleep(1)
            del payload, documents, splitted_text, list_for_cohere, final_embeddings, final_list
            gc.collect()
            

        await client.post(
            "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
            headers=headers,
            json={
                "document_id": data[0].databaseRow.document_id,
                "status": "completed"
            }
        )

    except Exception as e:
        try:
            await asyncio.sleep(5)
            await (
                sb.table("documents")
                .delete()
                .eq("file_hash", data[0].databaseRow.hash)
                .execute()
            )

        except Exception as b:
            print(f"error: {b}")
        
        await client.post(
            "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
            headers=headers,
            json={
                "document_id": data[0].databaseRow.document_id,
                "status": "failed",
                "error": str(e)
                }
            )

        

    