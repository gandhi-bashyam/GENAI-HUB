# enterprise_rag/llm/router.py

from .ollama_health import is_ollama_healthy

class LLMRouter:
    def __init__(self, local_llm, fallback_llm=None):
        self.local_llm = local_llm
        self.fallback_llm = fallback_llm

    def generate(self, prompt: str) -> str:
        if is_ollama_healthy():
            try:
                print("🔥 ROUTING → LOCAL OLLAMA")
                response = self.local_llm.generate(prompt)
                print("✅ RESPONSE FROM LOCAL OLLAMA")
                return response
            except Exception as e:
                print(f"❌ Local LLM failed: {e}")

        if self.fallback_llm:
            print("🛟 ROUTING → FALLBACK LLM")
            response = self.fallback_llm.generate(prompt)
            print("✅ RESPONSE FROM FALLBACK")
            return response

        raise Exception("❌ No LLM available")