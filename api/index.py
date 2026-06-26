import os
import json
import httpx
from flask import Flask, request, jsonify

app = Flask(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

@app.route("/")
def home():
    return jsonify({"status": "ok", "model": "MiniMaxAI/MiniMax-M2.7"})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        
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
            return r.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
