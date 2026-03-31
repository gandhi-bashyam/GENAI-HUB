# modules/rag/rag_pipeline.py

from common.embeddings.embedder_factory import get_embedder
from common.retriever.retriever_factory import get_retriever
from enterprise_rag.pipeline import RAGPipeline
from enterprise_rag.llm.local_ollama import LocalOllama
from enterprise_rag.llm.router import LLMRouter
from common.vectordb.load_documents import load_documents
from config.config_logger import log_config



def run(vector_store):
    print("🚀 Running Enterprise RAG Module...")

    log_config()   # ✅ ADD THIS HERE

    # 1. Load Documents
    docs = load_documents("data/")
    print(f"✅ Documents loaded: {len(docs)}")

    # 2. Add to Vector DB
    vector_store.add_documents(docs)
    print("✅ Documents added to vector DB")

    # 3. Retriever (Hybrid / Vector / BM25)
    retriever = get_retriever(docs, vector_store)
    print("✅ Retriever ready:", type(retriever))

    # 4. LLM Setup
    local_llm = LocalOllama(model="llama3")

    router = LLMRouter(
        local_llm=local_llm,
        fallback_llm=None
    )

    # 5. Pipeline (🔥 NEW FEATURES ENABLED)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=router,
        debug=True,                # 🔥 enable debug mode
        prompt_version="v2"        # 🔥 test prompt variants
    )

    print("✅ Pipeline ready")

    # 6. Query
    # query = "Who wrote Hamlet?"
    while True:
        query = input("\n💬 Enter your query (or 'exit'): ").strip()

        if query.lower() == "exit":
            print("👋 Exiting RAG module...")
            break

        if not query:
            print("⚠️ Empty query, try again...")
            continue

        response = pipeline.run(query)

        print("\n🧠 FINAL RESPONSE:\n", response)

  