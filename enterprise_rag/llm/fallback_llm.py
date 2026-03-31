# enterprise_rag/llm/fallback_llm.py

class FallbackLLM:
    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str) -> str:
        print("🛟 Using fallback LLM")
        return self.client.generate(prompt)