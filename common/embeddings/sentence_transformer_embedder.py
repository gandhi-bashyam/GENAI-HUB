from common.embeddings.base_embedder import BaseEmbedder
from common.cache.embedding_cache import EmbeddingCache

from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name="sentence-transformers/all-MiniLm-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache()

    def embed(self, text):
        cached = self.cache.get(text)
        if cached:
            print("Embedding Cache Hit")
            return cached

        embedding = self.model.encode(text)
        self.cache.set(text, embedding)

        return embedding

    
    def embed_query(self, query: str):
        return self.model.encode(query, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        # Ensure we return list of lists
        return [vec.tolist() for vec in self.model.encode(texts, normalize_embeddings=True)]

    
    # Make it callable for LangChain
    def __call__(self, texts):
        if isinstance(texts, str):
            return self.embed_query(texts)
        elif isinstance(texts, list):
            return self.embed_documents(texts)
        else:
            raise ValueError("Input must be string or list of strings")