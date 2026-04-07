class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.alpha = alpha  # weight for BM25

    # ----------------------------------------
    # 🔹 SCORE NORMALIZATION
    # ----------------------------------------
    @staticmethod
    def normalize_scores(results, reverse=False):
        if not results:
            return results

        scores = [r.get("score", 0) for r in results]

        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return results

        normalized = []
        for r in results:
            score = r.get("score", 0)
            norm = (score - min_s) / (max_s - min_s)

            if reverse:
                norm = 1 - norm

            normalized.append({
                **r,
                "score": norm
            })

        return normalized

    # ----------------------------------------
    # 🔹 MAIN RETRIEVE (HYBRID)
    # ----------------------------------------
    def retrieve(self, query, top_k=5):
        print("\n🔀 Running Hybrid Retrieval...")

        # Step 1: Get results
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        vector_results = self.vector.retrieve(query, top_k=top_k)

        # Step 2: Normalize
        bm25_results = self.normalize_scores(bm25_results)
        vector_results = self.normalize_scores(vector_results, reverse=True)

        combined = {}

        # ----------------------------------------
        # 🔹 BM25 Contribution
        # ----------------------------------------
        for r in bm25_results:
            text = r.get("text", "")
            if not text:
                continue

            combined[text] = {
                "text": text,
                "bm25_score": r["score"],
                "vector_score": 0.0,
                "score": self.alpha * r["score"],
                "metadata": r.get("metadata", {})
            }

        # ----------------------------------------
        # 🔹 Vector Contribution
        # ----------------------------------------
        for r in vector_results:
            text = r.get("text", "")
            if not text:
                continue

            if text in combined:
                combined[text]["vector_score"] = r["score"]
                # combined[text]["score"] += (1 - self.alpha) * r["score"]
                combined[text]["score"] += 1.2 * (1 - self.alpha) * r["score"]

                # Merge metadata (important!)
                combined[text]["metadata"].update(r.get("metadata", {}))
            else:
                combined[text] = {
                    "text": text,
                    "bm25_score": 0.0,
                    "vector_score": r["score"],
                    "score": 1.2 * (1 - self.alpha) * r["score"],
                    "metadata": r.get("metadata", {})
                }

        # ----------------------------------------
        # 🔹 SORT
        # ----------------------------------------
        # 🔹 SORT
        final_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # 🔥 FILTER LOW-QUALITY RESULTS
        filtered_results = [r for r in final_results if r["score"] > 0.1]

        # ----------------------------------------
        # 🔹 FALLBACK (EMPTY CASE FIRST)
        # ----------------------------------------
        if not filtered_results:
            print("⚠️ No strong results → fallback to vector")
            return sorted(vector_results, key=lambda x: x["score"], reverse=True)[:top_k]

        # ----------------------------------------
        # 🔹 SEMANTIC QUALITY CHECK (CRITICAL)
        # ----------------------------------------
        if not any(r["vector_score"] > 0.3 for r in filtered_results):
            print("⚠️ Weak semantic match → fallback to vector")
            return sorted(vector_results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        # 🔹 DEBUG
        print("\n--- 🔥 HYBRID SCORE BREAKDOWN ---")
        for r in filtered_results[:top_k]:
            print(
                f'{r["text"][:50]}...\n'
                f'   BM25: {round(r["bm25_score"], 3)} | '
                f'VEC: {round(r["vector_score"], 3)} | '
                f'FINAL: {round(r["score"], 3)}'
            )

        # # 🔹 FALLBACK
        # if not filtered_results:
        #     print("⚠️ No strong results → fallback to vector")
        #     return sorted(vector_results, key=lambda x: x["score"], reverse=True)[:top_k]

        return filtered_results[:top_k]