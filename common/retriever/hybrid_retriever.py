class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.alpha = alpha  # BM25 weight

    @staticmethod
    def normalize_scores(results, reverse=False):
        scores = [r["score"] for r in results]

        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return results

        normalized = []
        for r in results:
            norm = (r["score"] - min_s) / (max_s - min_s)

            if reverse:
                norm = 1 - norm

            normalized.append({
                **r,
                "score": norm
            })

        return normalized

    def retrieve(self, query, k=5):
        bm25_results = self.bm25.retrieve(query, top_k=k)
        vector_results = self.vector.retrieve(query, top_k=k)

        bm25_results = self.normalize_scores(bm25_results)
        vector_results = self.normalize_scores(vector_results, reverse=True)

        combined = {}

        # BM25 contribution
        for r in bm25_results:
            text = r["text"]
            combined[text] = {
                "text": text,
                "bm25_score": r["score"],
                "vector_score": 0.0,
                "score": self.alpha * r["score"],
                "metadata": r.get("metadata", {})
            }

        # Vector contribution
        for r in vector_results:
            text = r["text"]

            if text in combined:
                combined[text]["vector_score"] = r["score"]
                combined[text]["score"] += (1 - self.alpha) * r["score"]
            else:
                combined[text] = {
                    "text": text,
                    "bm25_score": 0.0,
                    "vector_score": r["score"],
                    "score": (1 - self.alpha) * r["score"],
                    "metadata": r.get("metadata", {})
                }

        final_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        print("\n--- FINAL HYBRID SCORE BREAKDOWN ---")
        for r in final_results:
            print(
                f'{r["text"][:40]} -> '
                f'BM25: {round(r["bm25_score"], 3)} | '
                f'VEC: {round(r["vector_score"], 3)} | '
                f'FINAL: {round(r["score"], 3)}'
            )

        # Fallback
        if all(r["score"] == 0 for r in final_results):
            print("⚠️ Hybrid collapse → fallback to vector")
            return sorted(vector_results, key=lambda x: x["score"], reverse=True)[:k]

        return final_results[:k]