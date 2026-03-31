# # enterprise_rag/llm_factory.py
# from enterprise_rag.llm.local_ollama import LocalOllama

# class LLMFactory:
#     @staticmethod
#     def create(model_type: str):
#         if model_type == "ollama":
#             return LocalOllama(model="llama3")
#         else:
#             raise ValueError(f"Unsupported LLM type: {model_type}")
        # elif provider == "openai":
        #     from langchain.chat_models import ChatOpenAI
        #     return ChatOpenAI(model="gpt-4")

        # else:
        #     raise ValueError("Unsupported LLM provider")


from enterprise_rag.llm.local_ollama import LocalOllama

class LLMFactory:
    @staticmethod
    def create(model_type: str):
        if model_type == "ollama":
            return LocalOllama(model="llama3")
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")