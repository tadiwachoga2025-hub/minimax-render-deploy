import os
import json
import httpx

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

def handler(request):
    path = request.get("path", "/")
    method = request.get("method", "GET")
    body = request.get("body", "{}")
    
    if path == "/" or path == "/api":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "ok", "model": "MiniMaxAI/MiniMax-M2.7"})
        }
    
    if path == "/api/chat" and method == "POST":
        try:
            data = json.loads(body) if body else {}
            
            headers = {
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": data.get("model", "MiniMaxAI/MiniMax-M2.7"),
                "messages": data.get("messages", []),
                "max_tokens": data.get("max_tokens", 1024),
                "temperature": data.get("temperature", 0.7)
            }
            
            with httpx.Client(timeout=60.0) as client:
                r = client.post(
                    f"{NVIDIA_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload
                )
                return {
                    "statusCode": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": r.text
                }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(e)})
            }
    
    return {
        "statusCode": 404,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": "Not found"})
    }
