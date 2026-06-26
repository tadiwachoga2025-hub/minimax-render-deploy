import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json

app = FastAPI(title="MiniMax-M2.7 API Proxy", version="1.0.0")

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
    top_p: Optional[float] = 0.9

@app.get("/")
async def root():
    return {"status": "ok", "model": "MiniMaxAI/MiniMax-M2.7"}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "MiniMaxAI/MiniMax-M2.7",
                "object": "model",
                "created": 1234567890,
                "owned_by": "minimax-ai"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": request.model,
        "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream,
        "top_p": request.top_p
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        if request.stream:
            async def stream_generator():
                async with client.stream(
                    "POST",
                    f"{NVIDIA_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = await client.post(
                f"{NVIDIA_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            return response.json()

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{NVIDIA_API_BASE}/completions",
            headers=headers,
            json=body
        )
        return response.json()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
