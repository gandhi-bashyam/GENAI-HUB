# common/retriever/vector_retriever.py

class VectorRetriever:
    def __init__(self, vector_store, k=10):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self, query: str, top_k=None) -> list[dict]:
            
        k = top_k if top_k is not None else self.k

        
        # print(f"\n🔍 Query: {query}")
        print(f"Top-{self.k} results retrieved")

        # results = self.vector_store.similarity_search(query, k=self.k)

        # return [
        #     {
        #         "text": r.page_content,
        #         "metadata": getattr(r, "metadata", {})
        #     }
        #     for r in results
        # ]

        results = self.vector_store.similarity_search_with_score(query, k=k)
        print("\nRAW VECTOR RESULTS:")
        for r, score in results:
            print(r.page_content[:50], "->", score)
            
        print(results[:3])
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