# enterprise_rag/pipeline.py
from common.cache.query_cache import QueryCache

import time

class RAGPipeline:
    def __init__(self, embedder, vector_store, retriever, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm
        self.cache = QueryCache()


    def _call_llm_with_retry(self, prompt):
        import time

        retries = 2
        for attempt in range(retries):
            try:
                return self.llm.generate(prompt)
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt+1} failed: {e}")
                time.sleep(2)

        raise Exception("❌ LLM failed after retries")

    def run(self, query: str) -> str:
        # Cache check
        start_time = time.time()

        cached = self.cache.get(query)
        if cached:
            print("⚡ Query Cache Hit")
            print("⏱️ Total latency: 0.00s")

            return cached

        print("\n🔍 Query:", query)

        # Step 1: Retrieval
        print("📚 Retrieving relevant documents...")
        retrieval_start = time.time()

        docs = self.retriever.retrieve(query)

        retrieval_time = time.time() - retrieval_start
        print(f"⏱️ Retrieval time: {retrieval_time:.2f}s")

        unique_docs = []
        seen = set()

        for doc in docs:
            text = doc["text"]
            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)

        docs = unique_docs

        context = "\n\n".join([doc["text"] for doc in docs])

        # print(f"📦 Retrieved {len(docs)} documents")
        print(f"📦 Retrieved {len(docs)} documents")

        for i, doc in enumerate(docs):
            print(f"\n📄 Document {i+1}:")
            print(doc["text"][:300])  # limit for readability

        # Step 2: Prompt construction
        
        prompt = f"""
        You are an AI assistant.

        Use the context below to answer the question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # Step 3: LLM call (through router)
        print("🤖 Sending request to LLM...")

        llm_start = time.time()

        try:
            response = self._call_llm_with_retry(prompt)
        except Exception as e:
            print(f"❌ LLM failed: {e}")
            raise

        llm_time = time.time() - llm_start
        print(f"⏱️ LLM time: {llm_time:.2f}s")
                

        # Step 4: Timing
        end_time = time.time()
        latency = end_time - start_time

        print("💾 Caching response")
        self.cache.set(query, response)

        print(f"📦 Cache size: {len(self.cache.cache)}")

        print(f"⏱️ Total latency: {latency:.2f} seconds")

        return response