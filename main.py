import os
from dotenv import load_dotenv
from supabase import create_async_client
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
import asyncio
import httpx
import cohere 

headers = {
    "Content-Type": "application/json",
    "X-Callback-Secret":"Xy7#b9@Lm2$Qp1z"
}

class N8n_row(BaseModel):
    id: int
    created_at: str
    company_name: str
    dataset_id: str
    document_id: str
    hash: str
    file_name: str

class N8n_payload(BaseModel):
    datasetId: str
    limit: int
    offset: int
    totalItems: int
    batchNumber: int 
    databaseRow: N8n_row



app = FastAPI()
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
co = cohere.AsyncClient(api_key=COHERE_API_KEY)

@app.post("/loader")
async def main(data: list[N8n_payload], task: BackgroundTasks):
    
    task.add_task(worker, data)

    return {"status": "processing started"}

async def worker(data):
    sb = await create_async_client(SUPABASE_URL, SUPABASE_KEY)
    try:
        
        async with httpx.AsyncClient() as client:
        
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
                for i in range(0, len(final_list), 150):

                    response = await (
                        sb.table("documents_view")
                        .insert(final_list[i: i + 150])
                        .execute()
                    )
                    
                    await asyncio.sleep(0.5)
                

            await client.post(
                "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
                headers=headers,
                json={
                    "document_id": data[0].databaseRow.document_id,
                    "status": "completed"
                }
            )

    except Exception as e:
        await asyncio.sleep(5)

        deletion = await (
            sb.table("documents")
            .delete()
            .eq("file_hash", data[0].databaseRow.hash)
            .execute()
        )

        async with httpx.AsyncClient() as client:
            await client.post(
                "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
                headers=headers,
                json={
                    "document_id": data[0].databaseRow.document_id,
                    "status": "failed",
                    "error": str(e)
                    }
                )

        

    