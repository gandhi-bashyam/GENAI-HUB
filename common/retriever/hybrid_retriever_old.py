class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    # def retrieve(self, query, top_k=3):
    #     vector_results = self.vector_retriever.retrieve(query, top_k)
    #     bm25_results = self.bm25_retriever.retrieve(query, top_k)

    #     # Simple merge + dedupe
    #     combined = list(dict.fromkeys(vector_results + bm25_results))

    #     return combined[:top_k]

    def retrieve(self, query, top_k=3):
        vector_results = self.vector_retriever.retrieve(query, top_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k)

        seen = set()
        combined = []

        for doc in vector_results + bm25_results:
            text = doc["text"]
            if text not in seen:
                seen.add(text)
                combined.append(doc)

        return combined[:top_k]