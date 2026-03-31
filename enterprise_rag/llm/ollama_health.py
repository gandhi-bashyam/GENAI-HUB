# enterprise_rag/llm/ollama_health.py

import requests

def is_ollama_healthy(url="http://localhost:11434"):
    try:
        res = requests.get(f"{url}/api/tags", timeout=2)
        return res.status_code == 200
    except Exception:
        return False