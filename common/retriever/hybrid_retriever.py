class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.alpha = alpha

    @staticmethod
    def normalize_scores(results, reverse=False):
        scores = [r["score"] for r in results]
        
        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return results

        for r in results:
            norm = (r["score"] - min_s) / (max_s - min_s)
            
            if reverse:  # for distance-based scores
                norm = 1 - norm
                
            r["score"] = norm

        return results
    
    def retrieve(self, query, k=10):

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
                "score": self.alpha * r["score"],  # partial
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
        print("\n--- BM25 NORMALIZED ---")
        for r in bm25_results:
            print(f'{r["text"][:40]} -> {round(r["score"], 3)}')

        print("\n--- VECTOR NORMALIZED ---")
        for r in vector_results:
            print(f'{r["text"][:40]} -> {round(r["score"], 3)}')

        # 🔥 SORT FINAL RESULTS
        final_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # 🚨 FALLBACK: Hybrid collapse
        if all(r["score"] == 0 for r in final_results):
            print("⚠️ Hybrid collapse → falling back to vector scores")

            # sort vector results properly (descending similarity)
            vector_sorted = sorted(
                vector_results,
                key=lambda x: x["score"],
                reverse=True
            )

            return vector_sorted[:k]

        print("\n--- FINAL HYBRID SCORE BREAKDOWN ---")
        for r in final_results:
            print(
                f'{r["text"][:40]} -> '
                f'BM25: {round(r["bm25_score"], 3)} | '
                f'VEC: {round(r["vector_score"], 3)} | '
                f'FINAL: {round(r["score"], 3)}'
    )
        return final_results[:k]

    # def retrieve(self, query, k=3):
        bm25_results = self.bm25.retrieve(query, top_k=k)
        vector_results = self.vector.retrieve(query)

        # Normalize
        bm25_results = self.normalize_scores(bm25_results)
        vector_results = self.normalize_scores(vector_results, reverse=True)

        combined = {}

        # Add BM25 results
        for r in bm25_results:
            key = r["text"]
            combined[key] = {
                "text": r["text"],
                "score": self.alpha * r["score"],
                "metadata": r.get("metadata", {})
            }

        # Merge Vector results
        for r in vector_results:
            key = r["text"]
            if key in combined:
                combined[key]["score"] += (1 - self.alpha) * r["score"]
            else:
                combined[key] = {
                    "text": r["text"],
                    "score": (1 - self.alpha) * r["score"],
                    "metadata": r.get("metadata", {})
                }

        # Sort
        ranked = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:k]
    

    