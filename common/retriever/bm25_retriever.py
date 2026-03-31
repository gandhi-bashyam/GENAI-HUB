from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        # self.tokenized_docs = [doc.split() for doc in documents]
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.tokenized_docs = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query, top_k=10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )

        # return [self.documents[i] for i, _ in ranked[:top_k]]
        # return [{"text": self.documents[i]} for i, _ in ranked[:top_k]]

        return [
            {
                "text": self.documents[i]["text"],
                "score": float(score),
                "metadata": self.documents[i].get("metadata", {})
            }
            for i, score in ranked[:top_k]
        ]