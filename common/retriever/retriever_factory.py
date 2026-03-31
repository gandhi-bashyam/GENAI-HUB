import os
from common.retriever.vector_retriever import VectorRetriever
from common.retriever.bm25_retriever import BM25Retriever
from common.retriever.hybrid_retriever import HybridRetriever

def get_retriever(documents, vector_store):
    retriever_type = os.getenv("RETRIEVER_TYPE", "vector")
    alpha = float(os.getenv("ALPHA", 0.5))  # ✅ ADD THIS
    # print(f"🧪 Loaded ALPHA from ENV: {alpha}")


    vector = VectorRetriever(vector_store)

    if retriever_type == "bm25":
        return BM25Retriever(documents)

    if retriever_type == "hybrid":
        bm25 = BM25Retriever(documents)
        return HybridRetriever(bm25, vector, alpha=alpha)
        # return HybridRetriever(vector, bm25)

    return vector