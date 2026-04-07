import time
import asyncio
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
        grounding_threshold=0.6
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
        self.grounding_threshold = grounding_threshold

        self.metrics = {}

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
        docs = asyncio.run(self._multi_retrieve_async(queries))

        # Clean + prioritize
        docs = self._clean_docs(docs, query)
        docs = self._prioritize_sources(docs, query)

        if self.reranker and docs:
            docs = self._rerank(query, docs)

        self._evaluate_metrics(docs, query)

        prompt = self._build_prompt(query, docs)
        response = self._generate(query, prompt)

        if self.metrics.get("grounding_score", 0) < self.grounding_threshold:
            print(f"⚠️ Low grounding ({self.metrics.get('grounding_score', 0):.2f}) → safe response")
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
    # 🔹 MULTI RETRIEVE ASYNC
    # ----------------------------------------
    async def _multi_retrieve_async(self, queries):
        tasks = [self._retrieve_async(q) for q in queries]
        results = await asyncio.gather(*tasks)
        return self._rrf_fusion(results)

    async def _retrieve_async(self, query):
        # Check embedding cache first
        if self.embedding_cache:
            embedding = self.embedding_cache.get(query)
            if embedding:
                print("⚡ Embedding Cache Hit")
            else:
                embedding = self.retriever.embed(query)
                self.embedding_cache.set(query, embedding)
        else:
            embedding = self.retriever.embed(query) if hasattr(self.retriever, "embed") else None

        # Retrieval cache
        cache_key = (query, "hybrid", self.top_k)
        cached_docs = self.retrieval_cache.get(cache_key) if self.retrieval_cache else None
        if cached_docs:
            print("⚡ Retrieval Cache Hit")
            return cached_docs

        # Async or sync retrieval
        if hasattr(self.retriever, "retrieve_async"):
            docs = await self.retriever.retrieve_async(query, top_k=self.top_k)
        else:
            # fallback to sync in thread
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: self.retriever.retrieve(query, top_k=self.top_k))

        # Save retrieval cache
        if self.retrieval_cache:
            self.retrieval_cache.set(cache_key, docs)

        # Deduplicate
        seen = set()
        dedup_docs = []
        for doc in docs:
            text = doc.get("text", "")
            if text and text not in seen:
                seen.add(text)
                dedup_docs.append(doc)

        # Score filter
        filtered_docs = [
            d for d in dedup_docs
            if d.get("score", 0) >= 0.1 and d.get("metadata", {}).get("source_type")
        ]
        final_docs = filtered_docs if filtered_docs else dedup_docs[:self.top_k]
        final_docs = sorted(final_docs, key=lambda x: x.get("score", 0), reverse=True)[:self.top_k]

        return final_docs

    # ----------------------------------------
    # 🔹 QUERY REWRITE + RRF
    # ----------------------------------------
    def _rewrite_query(self, query):
        if not self.query_rewriter:
            return [query]
        print("🔄 Rewriting query...")
        return self.query_rewriter.rewrite(query)

    def _rrf_fusion(self, results, k=60):
        print("🔀 Applying Reciprocal Rank Fusion...")
        scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = doc.get("text")
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

        doc_map = {doc.get("text"): doc for docs in results for doc in docs}
        reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[text] for text, _ in reranked][:self.top_k]

    # ----------------------------------------
    # 🔥 SOURCE PRIORITY, CLEANING, RERANK, EVALUATION
    # ----------------------------------------
    def _dynamic_priority(self, query):
        q = query.lower()
        if "who is" in q or "what is" in q:
            return {"wiki": 5, "web": 3, "pdf": 1}
        if "research" in q:
            return {"arxiv": 5, "pdf": 3}
        return {"wiki": 3, "web": 2, "pdf": 1}

    def _prioritize_sources(self, docs, query):
        priority = self._dynamic_priority(query)
        return sorted(docs, key=lambda x: (priority.get(x.get("metadata", {}).get("source_type"), 0), x.get("score", 0)), reverse=True)

    def _clean_docs(self, docs, query=None):
        clean = []
        definition_chunks = []
        for d in docs:
            text = d.get("text", "")
            if not text or len(text.strip()) < 50 or "attention(" in text.lower():
                continue
            if any(noise in text.lower() for noise in ["figure", "table", "et al", "doi", "arxiv"]):
                continue
            if query and ("who is" in query.lower() or "what is" in query.lower()):
                if any(prefix in text.lower() for prefix in ["is a", "is an", "was a", "is the"]):
                    definition_chunks.append(d)
            clean.append(d)
        return definition_chunks + clean if definition_chunks else clean

    def _rerank(self, query, docs):
        print("🔄 Reranking documents (Cross-Encoder)...")
        start = time.time()
        reranked = self.reranker.rerank(query, docs, top_k=self.rerank_top_k)
        self.metrics["rerank_time"] = time.time() - start
        print(f"⏱️ Rerank time: {self.metrics['rerank_time']:.2f}s")
        return reranked

    def _evaluate_metrics(self, docs, query):
        if not docs:
            self.metrics.update({"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "grounding_score": 0.0})
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
            if match_count > 0:
                relevant += 1
                if first_relevant_rank is None:
                    first_relevant_rank = i + 1
            grounding_scores.append(match_count / len(keywords) if keywords else 0)
            print(f"Doc {i+1}: [{doc.get('metadata', {}).get('source_type','unknown').upper()}] "
                  f"Grounding={grounding_scores[-1]:.2f} | Text snippet='{text[:60]}...'")

        precision = relevant / len(docs)
        recall = relevant / max(len(keywords), 1)
        mrr = 1 / first_relevant_rank if first_relevant_rank else 0.0
        grounding_score = sum(grounding_scores) / len(grounding_scores)

        self.metrics.update({"precision@k": precision, "recall@k": recall, "mrr": mrr, "grounding_score": grounding_score})

    # ----------------------------------------
    # 🔹 PROMPT + LLM
    # ----------------------------------------
    def _build_prompt(self, query, docs):
        clean_docs = [f"[{d.get('metadata', {}).get('source_type','unknown').upper()}]\n{d.get('text','')[:300]}" for d in docs[:self.top_k]]
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

    def _generate(self, query, prompt):
        print("🤖 Sending request to LLM...")
        start = time.time()
        response = self._call_llm_with_retry(query, prompt)
        self.metrics["llm_time"] = time.time() - start
        return response

    def _call_llm_with_retry(self, query, prompt):
        for attempt in range(2):
            try:
                return self.llm.generate(query, prompt)
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt+1} failed: {e}")
                time.sleep(2)
        raise Exception("❌ LLM failed after retries")

    def _finalize(self, query, response, start_time):
        self.metrics["total_time"] = time.time() - start_time
        self.cache.set(query, response)
        print("💾 Caching response")
        print(f"📦 Cache size: {len(self.cache.cache)}")
        print(f"⏱️ Total latency: {self.metrics['total_time']:.2f}s")