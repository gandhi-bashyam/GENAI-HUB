from common.ranking.fusion import reciprocal_rank_fusion
import asyncio

class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, alpha=0.6):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.alpha = alpha

    # ----------------------------------------
    # 🔹 SAFE NORMALIZATION
    # ----------------------------------------
    @staticmethod
    def normalize_scores(results, reverse=False):
        if not results:
            return []

        scores = [r.get("score", 0) for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return [{**r, "score": 1.0} for r in results]

        normalized = []
        for r in results:
            s = r.get("score", 0)
            norm = (s - min_s) / (max_s - min_s)

            if reverse:
                norm = 1 - norm

            normalized.append({**r, "score": norm})

        return normalized

    # ----------------------------------------
    # 🔹 TEXT CLEANING
    # ----------------------------------------
    @staticmethod
    def clean_text(text):
        return text.strip().replace("\n", " ")

    # ----------------------------------------
    # 🔹 QUALITY FILTER (VERY IMPORTANT)
    # ----------------------------------------
    @staticmethod
    def is_valid_chunk(text):
        if not text:
            return False

        text = text.strip()

        # remove very small / useless chunks
        if len(text) < 50:
            return False

        # remove noisy academic fragments
        noise_patterns = [
            "figure", "table", "section", "et al",
            "copyright", "arxiv"
        ]

        lower = text.lower()
        if any(n in lower for n in noise_patterns):
            return False

        return True

    # ----------------------------------------
    # 🔹 MAIN HYBRID RETRIEVE
    # ----------------------------------------
    async def retrieve_async(self, query, top_k=5):
        print("\n🔀 Running Hybrid Retrieval (ASYNC)...")

        bm25_task = asyncio.to_thread(self.bm25.retrieve, query, top_k * 2)
        vector_task = asyncio.to_thread(self.vector.retrieve, query, top_k * 2)

        bm25_results, vector_results = await asyncio.gather(
            bm25_task, vector_task
        )

        # ⬇️ reuse your existing logic from here
        bm25_rank_list = [
            self.clean_text(r.get("text", ""))
            for r in bm25_results
            if self.is_valid_chunk(r.get("text", ""))
        ]

        vector_rank_list = [
            self.clean_text(r.get("text", ""))
            for r in vector_results
            if self.is_valid_chunk(r.get("text", ""))
        ]

        fused_ranking = reciprocal_rank_fusion(
            [bm25_rank_list, vector_rank_list]
        )

        bm25_results = self.normalize_scores(bm25_results)
        vector_results = self.normalize_scores(vector_results, reverse=True)

        combined = {}


        # ----------------------------------------
        # 🔹 MERGE BM25
        # ----------------------------------------
        for r in bm25_results:
            text = self.clean_text(r.get("text", ""))
            if not self.is_valid_chunk(text):
                continue

            combined[text] = {
                "text": text,
                "bm25_score": r["score"],
                "vector_score": 0.0,
                "score": self.alpha * r["score"],
                "metadata": r.get("metadata", {})
            }

        # ----------------------------------------
        # 🔹 MERGE VECTOR
        # ----------------------------------------
        for r in vector_results:
            text = self.clean_text(r.get("text", ""))
            if not self.is_valid_chunk(text):
                continue

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

        # ----------------------------------------
        # 🔹 SORT
        # ----------------------------------------
        # Create lookup
        combined_lookup = {r["text"]: r for r in combined.values()}

        # Apply fused ranking order
        results = [
            combined_lookup[text]
            for text in fused_ranking
            if text in combined_lookup
        ]
        # ----------------------------------------
        # 🔹 STRONG FILTERING
        # ----------------------------------------
        results = [
            r for r in results
            if r["score"] > 0.15 and r["vector_score"] > 0.2
        ]

        # ----------------------------------------
        # 🔹 FALLBACKS
        # ----------------------------------------
        if not results:
            print("⚠️ No strong hybrid results → fallback to vector")
            return vector_results[:top_k]

        # ----------------------------------------
        # 🔹 DEBUG
        # ----------------------------------------
        print("\n--- 🔥 HYBRID SCORE BREAKDOWN ---")
        for r in results[:top_k]:
            print(
                f'{r["text"][:60]}...\n'
                f'   BM25: {round(r["bm25_score"], 3)} | '
                f'VEC: {round(r["vector_score"], 3)} | '
                f'FINAL: {round(r["score"], 3)}'
            )

        return results[:top_k]
    