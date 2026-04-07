from common.retriever.retriever_factory import get_retriever
from enterprise_rag.pipeline import RAGPipeline
from enterprise_rag.llm.local_ollama import LocalOllama
from enterprise_rag.llm.router import LLMRouter
from common.vectordb.load_documents import load_documents
from config.config_logger import log_config
from common.reranker.cross_encoder import CrossEncoderReranker
from common.memory.conversation_memory import ConversationMemory
from common.cache.query_cache import QueryCache

def run(vector_store, embedding_cache, retrieval_cache, initial_query=None):
    print("🚀 Running Enterprise RAG Module...\n")

    # ----------------------------------------
    # 🔹 CONFIG
    # ----------------------------------------
    log_config()

    # ----------------------------------------
    # 🔹 1. Load Documents
    # ----------------------------------------
    docs = load_documents("data/")
    print(f"✅ Documents loaded: {len(docs)}")

    # ----------------------------------------
    # 🔹 2. Vector DB Load (SAFE CHECK)
    # ----------------------------------------
    try:
        count = vector_store.count()
    except Exception:
        count = 0

    if count == 0:
        vector_store.add_documents(docs)
        print("✅ Documents added to vector DB")
    else:
        print("ℹ️ Vector DB already populated, skipping ingestion")

    # ----------------------------------------
    # 🔹 3. Retriever
    # ----------------------------------------
    retriever = get_retriever(docs, vector_store)
    print("✅ Retriever ready:", type(retriever))

    # ----------------------------------------
    # 🔹 4. LLM Setup
    # ----------------------------------------
    local_llm = LocalOllama(model="llama3")

    router = LLMRouter(
        local_llm=local_llm,
        fallback_llm=None
    )
    print("✅ LLM Router ready")

    # ----------------------------------------
    # 🔥 5. Reranker (FAST MODEL)
    # ----------------------------------------
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-TinyBERT-L-2",
        batch_size=8
    )
    print("✅ Cross-Encoder Reranker initialized")

    # ----------------------------------------
    # 🔥 6. Memory (FIXED)
    # ----------------------------------------
    memory = ConversationMemory(max_turns=5)
    print("✅ Conversation Memory initialized")

    # ----------------------------------------
    # 🔹 7. Pipeline
    # ----------------------------------------
    cache = QueryCache()  # ✅ create once

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=router,
        cache=cache,
        embedding_cache=embedding_cache,
        retrieval_cache=retrieval_cache,
        reranker=reranker,
        memory=memory,
        rerank_top_k=3,
        debug=True,
        prompt_version="v2"
    )

    print("✅ Pipeline ready\n")

    # ----------------------------------------
    # 🔹 Run initial query first (if provided)
    # ----------------------------------------
    if initial_query:
        print("\n🔍 Running initial query...\n")
        response = pipeline.run(initial_query)
        print("\n🧠 FINAL RESPONSE:\n", response)

    # ----------------------------------------
    # 🔹 8. Query Loop
    # ----------------------------------------
    while True:
        query = input("💬 Enter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting RAG module...")
            break
        if not query:
            print("⚠️ Empty query, try again...")
            continue

        response = pipeline.run(query)
        print("\n🧠 FINAL RESPONSE:\n", response)