import os
from dotenv import load_dotenv

load_dotenv(".env")

from common.embeddings.embedder_factory import get_embedder
from common.vectordb.chroma_store import ChromaStore

from common.cache.embedding_cache import EmbeddingCache
from common.cache.retrieval_cache import RetrievalCache

# Import modules
from modules.rag import rag_pipeline
from modules.guardrail_chatbot import guardrail_pipeline

from enterprise_rag.ingestion import load_enterprise_data

# ----------------------------------------
# 🔹 GLOBAL INFRA
# ----------------------------------------
embedding_cache = EmbeddingCache()
retrieval_cache = RetrievalCache()

# ----------------------------------------
# 🔹 QUERY ROUTER (SOURCE SELECTION)
# ----------------------------------------
# def route_sources(query: str):
#     query_lower = query.lower()
#     sources = []

#     # 🔥 Heuristic routing (upgrade later to LLM router)
#     if "paper" in query_lower or "research" in query_lower:
#         sources.append(("arxiv", query))

#     if "latest" in query_lower or "news" in query_lower:
#         sources.append(("web", query))

#     if "what is" in query_lower or "explain" in query_lower:
#         sources.append(("wiki", query))

#     # ✅ Always include enterprise/internal docs
#     sources.append(("pdf", "enterprise_rag/data/attention.pdf"))

#     return sources

def route_sources(query: str):
    query_lower = query.lower()
    sources = []

    # ✅ General knowledge queries
    if any(word in query_lower for word in ["what", "who", "define", "explain"]):
        sources.append(("wiki", query))

    # ✅ Research queries
    if any(word in query_lower for word in ["paper", "research"]):
        sources.append(("arxiv", query))


    # ✅ Always include enterprise docs
    sources.append(("pdf", "enterprise_rag/data/attention.pdf"))

    print(f"\n🧭 Selected sources: {sources}")  # 🔥 DEBUG LINE

    return sources

# ----------------------------------------
# 🔹 INGESTION LOOP (MULTI-SOURCE)
# ----------------------------------------
# def ingest_data(query, vector_store):
#     sources = route_sources(query)
#     all_chunks = []

#     print("\n🚀 Starting ingestion pipeline...")

#     for source_type, source in sources:
#         print(f"\n📥 Ingesting from {source_type}...")

#         try:
#             chunks = load_enterprise_data(source_type, source)
#             print(f"✅ Loaded {len(chunks)} chunks from {source_type}")

#             all_chunks.extend(chunks)

#         except Exception as e:
#             print(f"❌ Failed {source_type}: {e}")

#     print(f"\n📦 Total chunks collected: {len(all_chunks)}")

#     if all_chunks:
#         vector_store.add_documents(all_chunks)
#         print("✅ Data added to vector DB")
#     else:
#         print("⚠️ No data ingested!")
from common.ingestion.pipeline import multi_source_ingestion

def ingest_data(query, vector_store):
    sources = route_sources(query)

    print("\n🚀 Starting ingestion pipeline...")

    chunks = multi_source_ingestion(sources)

    if chunks:
        vector_store.add_documents(chunks)
        print("✅ Data added to vector DB")
    else:
        print("⚠️ No data ingested!")
# ----------------------------------------
# 🔹 MAIN ENTRY
# ----------------------------------------
def main():

    choice = input("Which module to run? (rag / guardrail): ").strip().lower()

    if choice not in ["rag", "guardrail"]:
        print("❌ Invalid choice! Please select 'rag' or 'guardrail'")
        return

    print(f"\n🚀 Starting {choice.upper()} module...\n")

    # 🔹 Step 1: User Query (first query)
    query = input("💬 Enter your query: ").strip()

    # 🔹 Step 2: Initialize embedder
    embedder = get_embedder()
    print("✅ Embedder initialized")

    # 🔹 Step 3: Vector DB setup
    persist_path = f"./chroma_db/{choice}"

    vector_store = ChromaStore(
        embedder=embedder,
        persist_directory=persist_path
    )

    print(f"📦 Using Vector DB at: {persist_path}")

    # 🔹 Step 4: Clear DB (dev mode)
    vector_store.clear_collection()
    try:
        count = vector_store.db._collection.count()
        print(f"🧪 Collection count after clear: {count}")
    except Exception:
        print("⚠️ Unable to fetch collection count")

    # 🔹 Step 5: Dynamic ingestion
    ingest_data(query, vector_store)

    # 🔹 Step 6: Route to module
    if choice == "rag":
        print("\n📚 Running RAG pipeline...\n")

        # 🔹 Pass first query as initial_query
        rag_pipeline.run(
            vector_store=vector_store,
            embedding_cache=embedding_cache,
            retrieval_cache=retrieval_cache,
            initial_query=query
        )

    elif choice == "guardrail":
        print("\n🛡️ Running Guardrail Chatbot...\n")

        guardrail_pipeline.run(
            vector_store=vector_store,
            embedding_cache=embedding_cache,
            retrieval_cache=retrieval_cache,
            query=query
        )

# ----------------------------------------
if __name__ == "__main__":
    main()