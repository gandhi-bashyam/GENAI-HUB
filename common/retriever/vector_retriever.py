from common.utils.embedding_cache import cached_query
import time


class VectorRetriever:
    def __init__(self, vector_store, k=10):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str, top_k=None) -> list[dict]:

        k = top_k if top_k is not None else self.k

        print(f"Top-{k} results retrieved")

        # ✅ Apply cache BEFORE retrieval
        cached_q = cached_query(query)

        start = time.time()
        results = self.vector_store.similarity_search_with_score(cached_q, k=k)
        print(f"⏱️ Vector retrieval took: {round(time.time() - start, 3)}s")

        print("\nRAW VECTOR RESULTS:")
        for r, score in results:
            print(r.page_content[:50], "->", score)

        seen = set()
        unique_results = []

        for r, score in results:
            text = r.page_content
            if text not in seen:
                seen.add(text)
                unique_results.append({
                    "text": text,
                    "score": float(score),
                    "metadata": getattr(r, "metadata", {})
                })

        return unique_results