import time
from common.cache.query_cache import QueryCache


class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm,
        cache=None,
        embedding_cache=None,
        retrieval_cache=None,

        reranker=None,
        memory=None,
        rerank_top_k=3,
        query_rewriter=None,
        validator=None,

        debug=False,
        prompt_version="v1",
        top_k=5,
        grounding_threshold=0.6  # ✅ NEW: set default

    ):
        self.retriever = retriever
        self.llm = llm
        self.cache = cache or QueryCache()
        self.embedding_cache = embedding_cache
        self.retrieval_cache = retrieval_cache

        self.reranker = reranker
        self.memory = memory
        self.rerank_top_k = rerank_top_k
        self.query_rewriter = query_rewriter
        self.validator = validator

        self.debug = debug
        self.prompt_version = prompt_version
        self.top_k = top_k

        self.metrics = {}
        self.grounding_threshold = grounding_threshold  # ✅ STORE threshold

    # ----------------------------------------
    # 🚀 MAIN ENTRY
    # ----------------------------------------
    def run(self, query: str) -> str:
        start_time = time.time()
        self.metrics = {}

        cached = self._check_cache(query)
        if cached:
            return cached

        print(f"\n🔍 Query: {query}")

        if self.memory:
            self.memory.add_user_message(query)

        queries = self._rewrite_query(query)
        # docs = self._multi_retrieve(queries)

        # if self.reranker and docs:
        #     docs = self._rerank(query, docs)

        docs = self._multi_retrieve(queries)

        # 🔥 CLEAN + PRIORITIZE BEFORE RERANK
        docs = self._clean_docs(docs, query)
        docs = self._prioritize_sources(docs, query)

        if self.reranker and docs:
            docs = self._rerank(query, docs)

        # ✅ UPDATED EVALUATION
        self._evaluate_metrics(docs, query)

        prompt = self._build_prompt(query, docs)
        # response = self._generate(prompt)
        response = self._generate(query, prompt)

        # if self.metrics.get("grounding_score", 0) < 0.6:
        #     print("⚠️ Low grounding → forcing safe response")

        #     response = "Context not sufficient to answer accurately."

        # ✅ Use class-level grounding_threshold
        if self.metrics.get("grounding_score", 0) < self.grounding_threshold:
            print(f"⚠️ Low grounding ({self.metrics.get('grounding_score', 0):.2f}) → forcing safe response")
            response = "Context not sufficient to answer accurately."

        if self.validator:
            valid, response = self.validator.validate(query, response, docs)

            self.metrics["validation_passed"] = valid
            self.metrics["validation_score"] = 1.0 if valid else 0.0

            if not valid:
                print("⚠️ Validation failed")

        if self.memory:
            self.memory.add_ai_message(response)

        self._finalize(query, response, start_time)

        return response

    # ----------------------------------------
    # 🔹 CACHE
    # ----------------------------------------
    def _check_cache(self, query):
        cached = self.cache.get(query)
        if cached:
            print("⚡ Query Cache Hit")
            return cached
        return None

    # ----------------------------------------
    # 🔹 RETRIEVAL
    # ----------------------------------------
    def _retrieve(self, query):
        print("📚 Retrieving relevant documents...")
        start = time.time()

        cache_key = (query, "hybrid", self.top_k)

        cached_docs = None
        if self.retrieval_cache:
            cached_docs = self.retrieval_cache.get(cache_key)

        if cached_docs:
            print("⚡ Retrieval Cache Hit")
            docs = cached_docs
        else:
            import asyncio

            if hasattr(self.retriever, "retrieve_async"):
                docs = asyncio.run(
                    self.retriever.retrieve_async(query, top_k=self.top_k)
                )
            else:
                docs = self.retriever.retrieve(query, top_k=self.top_k)

            if self.retrieval_cache:
                self.retrieval_cache.set(cache_key, docs)

        # Dedup
        seen = set()
        dedup_docs = []
        for doc in docs:
            text = doc.get("text", "")
            if text and text not in seen:
                seen.add(text)
                dedup_docs.append(doc)

        # Score filter
        # filtered_docs = [
        #     d for d in dedup_docs if d.get("score", 0) >= 0.05
        # ]
        filtered_docs = [
            d for d in dedup_docs
            if d.get("score", 0) >= 0.1
            and d.get("metadata", {}).get("source_type")
        ]

        final_docs = filtered_docs if filtered_docs else dedup_docs[:self.top_k]

        if not filtered_docs:
            print("⚠️ Score filtering removed all docs → fallback applied")

        final_docs = sorted(
            final_docs,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:self.top_k]

        elapsed = time.time() - start
        self.metrics["retrieval_time"] = elapsed
        self.metrics["num_docs"] = len(final_docs)

        print(f"⏱️ Retrieval time: {elapsed:.2f}s")
        print(f"📦 Retrieved {len(final_docs)} documents")

        if self.debug:
            print("\n🧪 DEBUG: Retrieval Scores")
            for doc in final_docs:
                print(f"{doc.get('text','')[:60]} -> {doc.get('score')}")

        return final_docs

    # ----------------------------------------
    # 🔥 MULTI QUERY + RRF
    # ----------------------------------------
    def _rewrite_query(self, query):
        if not self.query_rewriter:
            return [query]

        print("🔄 Rewriting query...")
        return self.query_rewriter.rewrite(query)

    def _multi_retrieve(self, queries):
        all_results = []

        for q in queries:
            docs = self._retrieve(q)
            all_results.append(docs)

        return self._rrf_fusion(all_results)

    def _rrf_fusion(self, results, k=60):
        print("🔀 Applying Reciprocal Rank Fusion...")

        scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = doc.get("text")

                if doc_id not in scores:
                    scores[doc_id] = 0

                scores[doc_id] += 1 / (k + rank)

        doc_map = {doc.get("text"): doc for docs in results for doc in docs}
        reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_map[text] for text, _ in reranked][:self.top_k]
    
    # ----------------------------------------
    # 🔥 SOURCE PRIORITY
    # ----------------------------------------
    # def _prioritize_sources(self, docs):
    #     priority = {"wiki": 3, "web": 2, "pdf": 1}

    #     return sorted(
    #         docs,
    #         key=lambda x: (
    #             priority.get(x.get("metadata", {}).get("source_type"), 0),
    #             x.get("score", 0)
    #         ),
    #         reverse=True
    #     )

    def _dynamic_priority(self, query):
        q = query.lower()

        if "who is" in q or "what is" in q:
            return {"wiki": 5, "web": 3, "pdf": 1}

        if "research" in q:
            return {"arxiv": 5, "pdf": 3}

        return {"wiki": 3, "web": 2, "pdf": 1}


    def _prioritize_sources(self, docs, query):
        priority = self._dynamic_priority(query)

        return sorted(
            docs,
            key=lambda x: (
                priority.get(x.get("metadata", {}).get("source_type"), 0),
                x.get("score", 0)
            ),
            reverse=True
        )


    # ----------------------------------------
    # 🔥 CONTEXT CLEANING
    # ----------------------------------------
    def _clean_docs(self, docs, query=None):
        clean = []
        definition_chunks = []

        for d in docs:
            text = d.get("text", "")
            if not text:
                continue

            text_lower = text.lower()

            # remove tiny chunks
            if len(text.strip()) < 50:
                continue

            # remove noise
            if "attention(" in text_lower:
                continue

            if any(noise in text_lower for noise in [
                "figure", "table", "et al", "doi", "arxiv"
            ]):
                continue

            # 🔥 NEW: capture definition-style chunks
            if query and ("who is" in query.lower() or "what is" in query.lower()):
                if any(prefix in text_lower for prefix in [
                    "is a", "is an", "was a", "is the"
                ]):
                    definition_chunks.append(d)

            clean.append(d)

        # 🔥 PRIORITIZE DEFINITION CHUNKS
        if definition_chunks:
            return definition_chunks + clean

        return clean
    # ----------------------------------------
    # 🔥 RERANKING
    # ----------------------------------------
    def _rerank(self, query, docs):
        print("🔄 Reranking documents (Cross-Encoder)...")
        start = time.time()

        reranked = self.reranker.rerank(query, docs, top_k=self.rerank_top_k)

        elapsed = time.time() - start
        self.metrics["rerank_time"] = elapsed

        print(f"⏱️ Rerank time: {elapsed:.2f}s")

        return reranked

    # ----------------------------------------
    # 🔥 FULL EVALUATION
    # ----------------------------------------
    # def _evaluate_metrics(self, docs, query):
    #     if not docs:
    #         self.metrics.update({
    #             "precision@k": 0.0,
    #             "recall@k": 0.0,
    #             "mrr": 0.0,
    #             "grounding_score": 0.0
    #         })
    #         return

    #     # keywords = query.lower().split()
    #     stopwords = {"who", "is", "what", "the", "a", "an", "of", "in"}
    #     keywords = [k for k in query.lower().split() if k not in stopwords]

    #     relevant = 0
    #     first_relevant_rank = None
    #     grounding_scores = []

    #     for i, doc in enumerate(docs):
    #         text = doc.get("text", "").lower()

    #         match_count = sum(1 for k in keywords if k in text)
    #         is_relevant = match_count > 0

    #         if is_relevant:
    #             relevant += 1
    #             if first_relevant_rank is None:
    #                 first_relevant_rank = i + 1

    #         grounding_scores.append(
    #             match_count / len(keywords) if keywords else 0
    #         )

    #     precision = relevant / len(docs)
    #     recall = relevant / max(len(keywords), 1)
    #     mrr = 1 / first_relevant_rank if first_relevant_rank else 0.0
    #     grounding_score = sum(grounding_scores) / len(grounding_scores)

    #     self.metrics.update({
    #         "precision@k": precision,
    #         "recall@k": recall,
    #         "mrr": mrr,
    #         "grounding_score": grounding_score
    #     })

    #     print(f"📊 precision@k: {precision:.2f}")
    #     print(f"📊 recall@k: {recall:.2f}")
    #     print(f"📊 MRR: {mrr:.2f}")
    #     print(f"📊 grounding score: {grounding_score:.2f}")

    def _evaluate_metrics(self, docs, query):
        if not docs:
            self.metrics.update({
                "precision@k": 0.0,
                "recall@k": 0.0,
                "mrr": 0.0,
                "grounding_score": 0.0
            })
            return

        stopwords = {"who", "is", "what", "the", "a", "an", "of", "in"}
        keywords = [k for k in query.lower().split() if k not in stopwords]

        relevant = 0
        first_relevant_rank = None
        grounding_scores = []

        print("\n🔍 Grounding breakdown per doc:")
        for i, doc in enumerate(docs):
            text = doc.get("text", "").lower()
            match_count = sum(1 for k in keywords if k in text)
            is_relevant = match_count > 0

            if is_relevant:
                relevant += 1
                if first_relevant_rank is None:
                    first_relevant_rank = i + 1

            doc_grounding = match_count / len(keywords) if keywords else 0
            grounding_scores.append(doc_grounding)

            # ✅ Debug print
            print(f"Doc {i+1}: [{doc.get('metadata', {}).get('source_type','unknown').upper()}] "
                f"Grounding={doc_grounding:.2f} | Text snippet='{text[:60]}...'")

        precision = relevant / len(docs)
        recall = relevant / max(len(keywords), 1)
        mrr = 1 / first_relevant_rank if first_relevant_rank else 0.0
        grounding_score = sum(grounding_scores) / len(grounding_scores)

        self.metrics.update({
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "grounding_score": grounding_score
        })

        print(f"\n📊 precision@k: {precision:.2f}")
        print(f"📊 recall@k: {recall:.2f}")
        print(f"📊 MRR: {mrr:.2f}")
        print(f"📊 grounding score: {grounding_score:.2f}")
    # ----------------------------------------
    # 🔹 PROMPT
    # ----------------------------------------
    def _build_prompt(self, query, docs):
        # clean_docs = [d.get("text", "") for d in docs[:self.top_k]]
        # clean_docs = [d.get("text", "")[:500] for d in docs[:self.top_k]]
        clean_docs = [
            f"[{d.get('metadata', {}).get('source_type','unknown').upper()}]\n{d.get('text','')[:300]}"
            for d in docs[:self.top_k]
        ]
        context = "\n\n".join(clean_docs)

        history = self.memory.get_context() if self.memory else ""

        return f"""
You are an expert AI assistant.

IMPORTANT RULES:
- Use ONLY the provided context
- Primarily use provided context
- If minor gaps exist, you may infer cautiously
- If answer is not explicitly present → say "Context not sufficient"
- If conflicting info exists → say "Context unclear"
- NEVER guess names, dates, relationships, or numbers

Context:
{context}

Conversation History (for reference only):
{history}

Question:
{query}

Answer:
"""

    # ----------------------------------------
    # 🔹 LLM
    # ----------------------------------------
    # def _generate(self, prompt):
    def _generate(self, query, prompt):
        print("🤖 Sending request to LLM...")
        start = time.time()

        # response = self._call_llm_with_retry(prompt)
        response = self._call_llm_with_retry(query, prompt)

        elapsed = time.time() - start
        self.metrics["llm_time"] = elapsed

        print(f"⏱️ LLM time: {elapsed:.2f}s")
        return response

    # def _call_llm_with_retry(self, prompt):
    def _call_llm_with_retry(self, query, prompt):
        for attempt in range(2):
            try:
                # return self.llm.generate(prompt)
                return self.llm.generate(query, prompt)
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt+1} failed: {e}")
                time.sleep(2)

        raise Exception("❌ LLM failed after retries")

    # ----------------------------------------
    # 🔹 FINALIZE
    # ----------------------------------------
    def _finalize(self, query, response, start_time):
        total_time = time.time() - start_time
        self.metrics["total_time"] = total_time

        self.cache.set(query, response)

        print("💾 Caching response")
        print(f"📦 Cache size: {len(self.cache.cache)}")

        print(f"⏱️ Total latency: {total_time:.2f}s")

        print("\n📊 METRICS BREAKDOWN:")
        for k, v in self.metrics.items():
            print(f"{k}: {v}")

        print("\n📈 FINAL QUALITY METRICS:")
        for k in ["precision@k", "recall@k", "mrr", "grounding_score"]:
            print(f"{k}: {self.metrics.get(k)}")