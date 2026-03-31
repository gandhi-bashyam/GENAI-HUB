# enterprise_rag/pipeline.py

import time
from common.cache.query_cache import QueryCache


class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm,
        cache=None,
        debug=False,
        prompt_version="v1"
    ):
        self.retriever = retriever
        self.llm = llm
        self.cache = cache or QueryCache()

        # 🔥 New Features
        self.debug = debug
        self.prompt_version = prompt_version
        self.metrics = {}

    def run(self, query: str) -> str:
        start_time = time.time()

        # Reset metrics
        self.metrics = {}

        # 1. Cache
        cached = self._check_cache(query)
        if cached:
            return cached

        print(f"\n🔍 Query: {query}")

        # 2. Retrieval
        docs = self._retrieve(query)

        # 3. Build Prompt
        prompt = self._build_prompt(query, docs)

        # 4. LLM
        response = self._generate(prompt)

        # 5. Finalize
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
    # 🔹 RETRIEVAL (with DEBUG MODE)
    # ----------------------------------------
    def _retrieve(self, query):
        print("📚 Retrieving relevant documents...")
        start = time.time()

        docs = self.retriever.retrieve(query)

        # Dedup
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc["text"] not in seen:
                seen.add(doc["text"])
                unique_docs.append(doc)

        elapsed = time.time() - start
        self.metrics["retrieval_time"] = elapsed

        print(f"⏱️ Retrieval time: {elapsed:.2f}s")
        print(f"📦 Retrieved {len(unique_docs)} documents")

        # 🔥 DEBUG MODE
        if self.debug:
            print("\n🧪 DEBUG: Retrieval Scores")
            for doc in unique_docs:
                score = doc.get("score", "N/A")
                print(f"{doc['text'][:50]} -> {score}")

        # Print docs
        for i, doc in enumerate(unique_docs):
            print(f"\n📄 Document {i+1}:")
            print(doc["text"][:200])

        return unique_docs

    # ----------------------------------------
    # 🔹 PROMPT VERSIONING
    # ----------------------------------------
    def _build_prompt(self, query, docs):
        context = "\n\n".join([doc["text"] for doc in docs])

        if self.prompt_version == "v1":
            prompt = f"""
                You are an AI assistant.

                Answer ONLY using the context below.
                If the answer is not present, say "I don't know".

                Context:
                {context}

                Question:
                {query}

                Answer:
                """

        elif self.prompt_version == "v2":
            prompt = f"""
                You are a strict retrieval-based QA system.

                Rules:
                - Use ONLY the provided context
                - Do NOT hallucinate
                - If unsure, say "I don't know"

                Context:
                {context}

                User Question:
                {query}

                Final Answer:
                """

        else:
            raise ValueError(f"Unknown prompt version: {self.prompt_version}")

        if self.debug:
            print(f"\n🧪 DEBUG: Using Prompt Version = {self.prompt_version}")

        return prompt

    # ----------------------------------------
    # 🔹 LLM CALL + LATENCY TRACKING
    # ----------------------------------------
    def _generate(self, prompt):
        print("🤖 Sending request to LLM...")
        start = time.time()

        response = self._call_llm_with_retry(prompt)

        elapsed = time.time() - start
        self.metrics["llm_time"] = elapsed

        print(f"⏱️ LLM time: {elapsed:.2f}s")
        return response

    def _call_llm_with_retry(self, prompt):
        retries = 2
        for attempt in range(retries):
            try:
                return self.llm.generate(prompt)
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt+1} failed: {e}")
                time.sleep(2)

        raise Exception("❌ LLM failed after retries")

    # ----------------------------------------
    # 🔹 FINALIZE + METRICS
    # ----------------------------------------
    def _finalize(self, query, response, start_time):
        total_time = time.time() - start_time
        self.metrics["total_time"] = total_time

        self.cache.set(query, response)

        print("💾 Caching response")
        print(f"📦 Cache size: {len(self.cache.cache)}")

        print(f"⏱️ Total latency: {total_time:.2f}s")

        # 🔥 METRICS OUTPUT (PORTFOLIO GOLD)
        print("\n📊 METRICS BREAKDOWN:")
        for k, v in self.metrics.items():
            print(f"{k}: {v:.4f}s")