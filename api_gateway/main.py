from fastapi import FastAPI, Request, HTTPException, Response
from contextlib import asynccontextmanager
import httpx
import redis.asyncio as redis

services = {
    "chatbot": "http://chatbot_container:8000",
    "copilot": "http://chatbot_container:8000",
    "test_chatbot": "http://chatbot_container:8000",
    "loader": "http://processor_container:8000",
    "document": "http://data_distributor_container:8000",
    "feedback": "http://prompt_rewrite_container:8000",
}
httpx_client = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global httpx_client
    global redis_client

    httpx_client = httpx.AsyncClient()
    redis_client = redis.Redis(host="redis_container", port=6379, db=0)

    yield

    await httpx_client.aclose()
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)

@app.route("/{service}", methods=["GET", "POST", "PUT", "DELETE"])
async def main(service: str, request: Request):
    if service not in services:
        raise HTTPException(status_code=404, detail="Service not found")
    
    if await check_rate_limit(request):
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        try:
            service_response = await httpx_client.request(
                method=request.method,
                url=f"{services[service]}/{service}",
                content=body,
                headers=headers,
                params=request.url.query
            )
            return Response(
                content=service_response.content,
                status_code=service_response.status_code,
                headers=service_response.headers,
            )

            
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Service Temporarly Unavailable")
    else:
        raise HTTPException(status_code=429, detail="Too many Requests")

    

async def check_rate_limit(request: Request) -> bool:
    body = {}
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except Exception:
            body = {}
    
    user_id = body.get("user_id")
    converstaion_id = body.get("conversationId")
    ip = request.headers.get("x-forwarded-for")


    if user_id:
        key = f"rate_limit:{user_id}"
        get = await redis_client.incr(key)
        if get == 1:
            await redis_client.expire(key, 60)
        elif get > 20:
            return False
    if converstaion_id:
        key = f"rate_limit:{converstaion_id}"
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
