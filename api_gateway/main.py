from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
import redis.asyncio as redis

SERVICES = {
    "chatbot": "http://chatbot:8000",
    "copilot": "http://chatbot:8000",
    "test_chatbot": "http://chatbot:8000",
    "loader": "http://processor:8000",
    "document": "http://data_distributor:8000",
    "feedback": "http://prompt_rewrite:8000",
}
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
httpx_client = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    global redis_client

    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    redis_client = redis.Redis(host="redis_db", port=6379, db=0)

    yield

    await httpx_client.aclose()
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def health():
    try:
        if redis_client is None or not await redis_client.ping():
            raise RuntimeError("Redis is unavailable")
    except (redis.RedisError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Gateway is not ready") from exc
    return {"status": "ok"}


@app.api_route("/{service}", methods=PROXY_METHODS)
@app.api_route("/{service}/{path:path}", methods=PROXY_METHODS)
async def proxy_request(service: str, request: Request, path: str = ""):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail="Service not found")

    if not await check_rate_limit(request):
        raise HTTPException(status_code=429, detail="Too many Requests")

    body = await request.body()
    request_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }

    try:
        service_response = await httpx_client.request(
            method=request.method,
            url=f"{SERVICES[service]}{request.url.path}",
            content=body,
            headers=request_headers,
            params=request.query_params,
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc

    response_headers = {
        key: value
        for key, value in service_response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    return Response(
        content=service_response.content,
        status_code=service_response.status_code,
        headers=response_headers,
    )


async def check_rate_limit(request: Request) -> bool:
    body = {}
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except Exception:
            body = {}
    
    user_id = body.get("user_id")
    conversation_id = body.get("conversationId")
    forwarded_for = request.headers.get("x-forwarded-for", "")
    ip = forwarded_for.split(",", 1)[0].strip()
    if not ip and request.client:
        ip = request.client.host


    if user_id:
        key = f"rate_limit:{user_id}"
        get = await redis_client.incr(key)
        if get == 1:
            await redis_client.expire(key, 60)
        elif get > 20:
            return False
    if conversation_id:
        key = f"rate_limit:{conversation_id}"
        get = await redis_client.incr(key)
        if get == 1:
            await redis_client.expire(key, 60)
        elif get > 20:
            return False
    if ip:
        key = f"rate_limit:{ip}"
        get = await redis_client.incr(key)
        if get == 1:
            await redis_client.expire(key, 60)
        elif get > 100:
            return False
    return True
