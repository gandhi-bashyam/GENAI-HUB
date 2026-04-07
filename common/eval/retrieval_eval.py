def precision_at_k(retrieved_docs, relevant_docs, k=5):
    retrieved = retrieved_docs[:k]

    hits = sum(1 for doc in retrieved if doc in relevant_docs)

    return hits / k