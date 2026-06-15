from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
import asyncio
import asyncpg
import httpx
import fitz #PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from dotenv import load_dotenv
from functions import globs
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cohere
from supabase import create_async_client
from contextlib import asynccontextmanager
import traceback 

load_dotenv()
PSTG_URL = os.getenv("PSTG_URL")
APIFY = os.getenv("APIFY")
CALLBACK_SECRET = os.getenv("CALLBACK_SECRET")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
headers = {
    "Content-Type": "application/json",
    "X-Callback-Secret": CALLBACK_SECRET
}
client = None
co = None
sb = None
db_pool = None
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


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
    sb = create_async_client(SUPABASE_URL, SUPABASE_KEY)
    db_pool = await asyncpg.create_pool(PSTG_URL)

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
                    
                response_dict = r.json()
                company_name = data.metadata.company_name
                dataset_id = response_dict["data"]["defaultDatasetId"]
                doc_id = data.document_id
                filename = data.metadata.original_filename
                hash = data.metadata.file_hash

                if r.status_code == 200:
                    await connection.execute('''
                        INSERT INTO "apify_url-scraper"(company_name, dataset_id, document_id, file_name, hash)
                        VALUES($1, $2, $3, $4, $5);

                        UPDATE document_uploads
                        SET status = 'scanning'
                        WHERE id = $3;
                    ''', company_name, dataset_id, doc_id, filename, hash)
                else:
                    await client.post(
                        "https://ntnoptppvhvezqbdbcyl.supabase.co/functions/v1/document-callback",
                        headers=headers,
                        json={
                            "document_id": data.document_id,
                            "status": "failed",
                            "error": str(r)
                            }
                        )

            else:
                extracted_text = []
                r = await client.get(data.file_url)

                if data.metadata.mime_type.endswith('docx'):
                    virtual_file = io.BytesIO(r.content)
                    doc = docx.Document(virtual_file)
                    for paragraph in doc.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            extracted_text.append(text)

                    virtual_file.close()


                else:
                    if data.metadata.mime_type.endswith('pdf'):
                        doc = fitz.open(stream=r.content, filetype="pdf")
                        for page in doc:
                            text = page.get_text().strip()
                            if len(text) < 50:
                                pix = page.get_pixmap(dpi=300)
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                text = pytesseract.image_to_string(img, lang='pol+eng').strip()
                            extracted_text.append(text)
                        doc.close()

                    elif data.metadata.mime_type.endswith('txt'):
                        text = r.content.decode('utf-8')
                        extracted_text.append(text)
                        
                joined_text = " ".join(extracted_text)   
                final_embeddings = []
                list_for_cohere = text_splitter.split_text(joined_text)

                for i in range(0, len(list_for_cohere), 90):
                    cohere_res = await co.embed (
                        texts=list_for_cohere[i: i + 90],
                        input_type="search_document",
                        embedding_types=["float"],
                        model="embed-multilingual-v3.0"
                    )
                    final_embeddings.extend(cohere_res.embeddings.float)
                final_list = []
                for vector, s_text in zip(final_embeddings, list_for_cohere):
                    final_list.append({"content": s_text, "metadata": {
                        "file_name": data.metadata.original_filename,
                        "company_name": data.metadata.company_name,
                        "chatbot": False,
                        "copilot": False,
                        "hash": data.metadata.file_hash
                    }, "embedding": vector})

                for i in range(0, len(final_list), 150):
                    await (
                        sb.table("documents_view")
                        .insert(final_list[i: i + 150])
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
            "http://log-error_container:8000",
            json={
                "app_name": "data_distributor",
                "error_message": error_string,
                "traceback": traceback_string,
            }
        )
    except Exception as e:
        print(f"could not log the error {str(e)}")