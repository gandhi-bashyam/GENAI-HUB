import requests

class LocalOllama:
    def __init__(self, model="llama3"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def generate(self, prompt: str) -> str:
        print("🔥 LOCAL OLLAMA HIT")

        def call():
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60  # ⬅️ production-safe timeout
            )
            response.raise_for_status()
            return response.json()["response"]

        from .retry import retry
        return retry(call)