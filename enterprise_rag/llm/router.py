# enterprise_rag/llm/router.py

from .ollama_health import is_ollama_healthy

def route_query(query: str) -> str:
    words = len(query.split())

    if words <= 5:
        return "mistral"      # fast
    elif words <= 20:
        return "llama3"       # default
    else:
        return "mixtral"      # heavy

class LLMRouter:
    def __init__(self, local_llm, fallback_llm=None):
        self.local_llm = local_llm
        self.fallback_llm = fallback_llm

    # def generate(self, prompt: str) -> str:
    def generate(self, query: str, prompt: str) -> str:
        if is_ollama_healthy():
            try:
                # print("🔥 ROUTING → LOCAL OLLAMA")
                # response = self.local_llm.generate(prompt)
                model_name = route_query(query)

                print(f"🔥 ROUTING → {model_name.upper()}")

                response = self.local_llm.generate(
                    prompt=prompt,
                    model=model_name
                )
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