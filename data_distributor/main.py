from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
import asyncio
import asyncpg
import httpx
import math
import os
from dotenv import load_dotenv
if __package__:
    from .functions import globs
    from .document_processing import DocumentExtractor, split_sections
else:
    from functions import globs
    from document_processing import DocumentExtractor, split_sections
import cohere
from supabase import create_async_client
from contextlib import asynccontextmanager
import traceback 

load_dotenv()
PSTG_URL = os.getenv("PSTG_URL")
APIFY = os.getenv("APIFY")
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCUMENT_OCR_MODEL = os.getenv("DOCUMENT_OCR_MODEL", "gpt-6-luna")
DOCUMENT_OCR_MIN_TEXT_CHARS = int(os.getenv("DOCUMENT_OCR_MIN_TEXT_CHARS", "100"))
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "2"))
headers = {
    "Content-Type": "application/json",
    "X-Callback-Secret": CALLBACK_SECRET
}
client = None
co = None
sb = None
db_pool = None


class Metadata(BaseModel):
    company_name: str
    original_filename: str
    file_hash: str
    mime_type: str | None = None
    storage_path: str | None = None

class Payload(BaseModel):
    event: str
    document_id: str
    file_url: str | None = None
    website_url: str | None = None
    excluded_paths: list | None = None
    single_page_only: bool | None = None
    replacement_for: str | None = None
    metadata: Metadata
    callback_url: str
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    global co
    global sb
    global db_pool

    client = httpx.AsyncClient()
    co = cohere.AsyncClient(api_key=COHERE_API_KEY)
    sb = await create_async_client(SUPABASE_URL, SUPABASE_KEY)
    db_pool = await asyncpg.create_pool(
        PSTG_URL,
        min_size=DB_POOL_MIN_SIZE,
        max_size=DB_POOL_MAX_SIZE,
    )

    yield

    await client.aclose()
    await db_pool.close()

app = FastAPI(lifespan=lifespan)
async def get_connection():
    async with db_pool.acquire() as connection:
        yield connection


@app.post("/document")
async def main(data: Payload, task: BackgroundTasks):

    task.add_task(worker, data)

    return {"status": "started"}

async def worker(data: Payload):
    async with db_pool.acquire() as connection:
        try:
            if data.event == "url.scan":
                while True:
                    scan_num = await connection.fetchval('''
                        SELECT COUNT(*)::integer
                        FROM document_uploads
                        WHERE status = 'scanning';
                    ''')
                    if scan_num < 2:
                        break

                    await asyncio.sleep(60)

                if data.single_page_only:
                    r = await client.post(
                        f"https://api.apify.com/v2/acts/apify~website-content-crawler/runs?token={APIFY}&memory=4096&webhooks=W3siZXZlbnRUeXBlcyI6WyJBQ1RPUi5SVU4uU1VDQ0VFREVEIl0sInJlcXVlc3RVcmwiOiJodHRwczovL244bi52ZWN0aXhhaS5jb20vd2ViaG9vay84MjQ0YWQxMy1kMjk3LTQ0MmUtODM3NS1iYTg0ODhjN2ViZDgifV0==",
                        json={
                            "aggressivePrune": False,
                            "blockMedia": True,
                            "clickElementsCssSelector": "[aria-expanded=\"False\"]",
                            "clientSideMinChangePercentage": 15,
                            "crawlerType": "playwright:adaptive",
                            "debugLog": False,
                            "debugMode": False,
                            "expandIframes": True,
                            "ignoreCanonicalUrl": False,
                            "ignoreHttpsErrors": False,
                            "includeUrlGlobs": [
                                {
                                "glob": data.website_url
                                }
                            ],
                            "keepUrlFragments": False,
                            "proxyConfiguration": {
                                "useApifyProxy": True
                            },
                            "readableTextCharThreshold": 100,
                            "htmlTransformer": "readableTextIfPossible",
                            "removeCookieWarnings": True,
                            "removeElementsCssSelector": "nav, footer, script, style, noscript, svg, img[src^='data:'],\n[role=\"alert\"],\n[role=\"banner\"],\n[role=\"dialog\"],\n[role=\"alertdialog\"],\n[role=\"region\"][aria-label*=\"skip\" i],\n[aria-modal=\"True\"]",
                            "renderingTypeDetectionPercentage": 10,
                            "respectRobotsTxtFile": False,
                            "saveFiles": False,
                            "saveHtml": False,
                            "saveHtmlAsFile": False, 
                            "saveMarkdown": True,
                            "saveScreenshots": False,
                            "signHttpRequests": False,
                            "startUrls": [
                                {
                                    "url": data.website_url
                                }
                            ],
                            "storeSkippedUrls": False,
                            "useSitemaps": True
                        }
                    )
                else:
                    exluded_urls = globs(data.website_url, data.excluded_paths)
                    r = await client.post(
                        f"https://api.apify.com/v2/acts/apify~website-content-crawler/runs?token={APIFY}&build=0.3.82&memory=8192&webhooks=W3siZXZlbnRUeXBlcyI6WyJBQ1RPUi5SVU4uU1VDQ0VFREVEIl0sInJlcXVlc3RVcmwiOiJodHRwczovL244bi52ZWN0aXhhaS5jb20vd2ViaG9vay84MjQ0YWQxMy1kMjk3LTQ0MmUtODM3NS1iYTg0ODhjN2ViZDgifV0=",
                        json={
                            "aggressivePrune": False,
                            "blockMedia": True,
                            "clickElementsCssSelector": "[aria-expanded=\"False\"]",
                            "clientSideMinChangePercentage": 15,
                            "crawlerType": "playwright:adaptive",
                            "debugLog": False,
                            "debugMode": False,
                            "excludeUrlGlobs":exluded_urls,
                            "expandIframes": True,
                            "htmlTransformer": "readableTextIfPossible",
                            "ignoreCanonicalUrl": False,
                            "ignoreHttpsErrors": False,
                            "keepUrlFragments": False,
                            "keepUrlFragments": False,
                            "maxConcurrency": 200,
                            "maxCrawlDepth": 20,
                            "maxCrawlPages": 50,
                            "maxRequestRetries": 3,
                            "maxScrollHeightPixels": 20000,
                            "maxSessionRotations": 10,
                            "minFileDownloadSpeedKBps": 128,
                            "proxyConfiguration": {
                                "useApifyProxy": True
                            },
                            "readableTextCharThreshold": 100,
                            "initialConcurrency": 40,
                            "removeCookieWarnings": True,
                            "removeElementsCssSelector": "nav, footer, script, style, noscript, svg, img[src^='data:'],\n[role=\"alert\"],\n[role=\"banner\"],\n[role=\"dialog\"],\n[role=\"alertdialog\"],\n[role=\"region\"][aria-label*=\"skip\" i],\n[aria-modal=\"True\"]",
                            "renderingTypeDetectionPercentage": 10,
                            "respectRobotsTxtFile": False,
                            "saveFiles": False,
                            "saveHtml": False,
                            "saveHtmlAsFile": False,
                            "saveMarkdown": True,
                            "saveScreenshots": False,
                            "signHttpRequests": False,
                            "startUrls": [
                                {
                                    "url": data.website_url
                                }
                            ],
                            "storeSkippedUrls": False,
                            "useSitemaps": True,
                            "maxCrawlPages": 1000
                        }
                    )
                    
                r.raise_for_status()
                response_dict = r.json()
                company_name = data.metadata.company_name
                dataset_id = response_dict["data"]["defaultDatasetId"]
                doc_id = data.document_id
                filename = data.metadata.original_filename
                hash = data.metadata.file_hash

                async with connection.transaction():
                    await connection.execute('''
                        INSERT INTO "apify_url-scraper"(company_name, dataset_id, document_id, file_name, hash)
                        VALUES($1, $2, $3, $4, $5);
                    ''', company_name, dataset_id, doc_id, filename, hash)
                    await connection.execute('''
                        UPDATE document_uploads
                        SET status = 'scanning'
                        WHERE id = $1;
                    ''', doc_id)

            else:
                if not data.file_url:
                    raise ValueError("file_url is required for document uploads")
                r = await client.get(data.file_url, follow_redirects=True, timeout=60)
                r.raise_for_status()
                extractor = DocumentExtractor(
                    client, OPENAI_API_KEY, DOCUMENT_OCR_MODEL, DOCUMENT_OCR_MIN_TEXT_CHARS,
                )
                sections = await extractor.extract(
                    r.content, data.metadata.original_filename,
                    data.metadata.mime_type or r.headers.get("content-type"),
                )
                chunks = split_sections(sections)
                if not chunks:
                    raise ValueError("Document contains no readable text")
                # Store each embedding batch immediately instead of retaining every vector.
                vector_size = None
                for i in range(0, len(chunks), 90):
                    batch = chunks[i: i + 90]
                    cohere_res = await co.embed (
                        texts=[chunk.text for chunk in batch],
                        input_type="search_document",
                        embedding_types=["float"],
                        model="embed-multilingual-v3.0"
                    )
                    vectors = cohere_res.embeddings.float
                    if len(vectors) != len(batch):
                        raise ValueError("Embedding count does not match document chunk count")
                    final_list = []
                    for offset, (vector, chunk) in enumerate(zip(vectors, batch)):
                        if vector_size is None:
                            vector_size = len(vector)
                        if not vector or len(vector) != vector_size or not all(math.isfinite(value) for value in vector):
                            raise ValueError("Embedding response contains an invalid vector")
                        metadata = {
                            "file_name": data.metadata.original_filename,
                            "company_name": data.metadata.company_name,
                            "chatbot": False,
                            "copilot": False,
                            "hash": data.metadata.file_hash,
                            "document_id": data.document_id,
                            "chunk_index": i + offset,
                            "extraction_method": chunk.extraction_method,
                        }
                        if chunk.page is not None:
                            metadata["page"] = chunk.page
                        final_list.append({"content": chunk.text, "metadata": metadata, "embedding": vector})
                    await (
                        sb.table("documents_view")
                        .insert(final_list)
                        .execute()
                    )
                    await asyncio.sleep(0.5)
                await client.post(
                    "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
                    headers=headers,
                    json={
                        "document_id": data.document_id,
                        "status": "completed",
                    }
                )

        except Exception as e:
            await asyncio.sleep(5)

            asyncio.create_task(logger(str(e), traceback.format_exc()))

            await (
                sb.table("documents")
                .delete()
                .eq("file_hash", data.metadata.file_hash)
                .execute()
            )

            await client.post(
                "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
                headers=headers,
                json={
                    "document_id": data.document_id,
                    "status": "failed",
                    "error": str(e)
                }
            )

async def logger(error_string, traceback_string):
    try:
        await client.post(
            "http://log-error_container:8000/log-error",
            json={
                "app_name": "data_distributor",
                "error_message": error_string,
                "traceback": traceback_string,
            }
        )
    except Exception as e:
        print(f"could not log the error {str(e)}")
