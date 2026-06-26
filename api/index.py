import os
import json
import httpx

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

def handler(request, response):
    if request["path"] == "/" or request["path"] == "/api":
        response["headers"]["Content-Type"] = "application/json"
        return json.dumps({"status": "ok", "model": "MiniMaxAI/MiniMax-M2.7"})
    
    if request["path"] == "/api/chat" and request["method"] == "POST":
        body = json.loads(request["body"])
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": body.get("model", "MiniMaxAI/MiniMax-M2.7"),
            "messages": body.get("messages", []),
            "max_tokens": body.get("max_tokens", 1024),
            "temperature": body.get("temperature", 0.7)
        }
        
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{NVIDIA_API_BASE}/chat/completions",
                headers=headers,
                json=payload
            )
            response["headers"]["Content-Type"] = "application/json"
            return r.text
    
    response["status"] = 404
    response["headers"]["Content-Type"] = "application/json"
    return json.dumps({"error": "Not found"})
