import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from Mangum import Mangum

app = FastAPI(title="MiniMax-M2.7 API", version="1.0.0")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "MiniMaxAI/MiniMax-M2.7"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.get("/")
async def root():
    return {"status": "ok", "model": "MiniMaxAI/MiniMax-M2.7", "platform": "Vercel Free"}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/chat")
@app.post("/api/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not NVIDIA_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "NVIDIA_API_KEY not configured"}
        )
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": request.model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{NVIDIA_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            return response.json()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

handler = Mangum(app)
