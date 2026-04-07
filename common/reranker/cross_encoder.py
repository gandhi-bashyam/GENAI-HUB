from sentence_transformers import CrossEncoder
import time


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=8):
        print("🔧 Loading Cross-Encoder model...")
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return docs

        start_time = time.time()

        pairs = []
        valid_docs = []

        # ----------------------------------------
        # 🔹 PREPARE INPUT
        # ----------------------------------------
        for doc in docs:
            text = doc.get("text", "").strip()
            if not text:
                continue

            pairs.append((query, text))
            valid_docs.append(doc)

        if not pairs:
            return docs[:top_k]

        try:
            # ----------------------------------------
            # 🔥 MANUAL BATCHING (BETTER CONTROL)
            # ----------------------------------------
            scores = []

            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]

                batch_scores = self.model.predict(
                    batch,
                    convert_to_numpy=True
                )

                scores.extend(batch_scores)

            # ----------------------------------------
            # 🔹 NORMALIZATION (IMPORTANT)
            # ----------------------------------------
            if scores:
                min_s, max_s = min(scores), max(scores)
                if max_s != min_s:
                    scores = [(s - min_s) / (max_s - min_s) for s in scores]

            # ----------------------------------------
            # 🔹 ATTACH SCORES
            # ----------------------------------------
            for doc, score in zip(valid_docs, scores):
                doc["rerank_score"] = float(score)

            # ----------------------------------------
            # 🔹 SORT
            # ----------------------------------------
            reranked = sorted(
                valid_docs,
                key=lambda x: x.get("rerank_score", 0),
                reverse=True
            )

            elapsed = time.time() - start_time
            print(f"⏱️ Batch rerank time: {round(elapsed, 3)}s")

            # ----------------------------------------
            # 🔹 DEBUG (OPTIONAL BUT GOLD)
            # ----------------------------------------
            print("\n--- 🔥 RERANK SCORE BREAKDOWN ---")
            for d in reranked[:top_k]:
                print(
                    f'{d["text"][:60]}...\n'
                    f'   RERANK: {round(d.get("rerank_score", 0), 3)}'
                )

            return reranked[:top_k]

        except Exception as e:
            print(f"⚠️ Reranker failed: {e}")
            return docs[:top_k]