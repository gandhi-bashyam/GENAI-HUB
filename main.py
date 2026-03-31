# genai-hub/main.py

import os
from dotenv import load_dotenv

load_dotenv(".env")

from common.embeddings.embedder_factory import get_embedder
from common.vectordb.chroma_store import ChromaStore

# Import modules
from modules.rag import rag_pipeline
from modules.guardrail_chatbot import guardrail_pipeline


def main():

    choice = input("Which module to run? (rag / guardrail): ").strip().lower()

    if choice not in ["rag", "guardrail"]:
        print("❌ Invalid choice! Please select 'rag' or 'guardrail'")
        return

    print(f"\n🚀 Starting {choice.upper()} module...\n")

    # ✅ Step 1: Initialize embedder (shared infra)
    embedder = get_embedder()
    print("✅ Embedder initialized")

    # ✅ Step 2: Project-specific vector DB (ISOLATION FIX)
    persist_path = f"./chroma_db/{choice}"

    vector_store = ChromaStore(
        embedder=embedder,
        persist_directory=persist_path
    )

    print(f"📦 Using Vector DB at: {persist_path}")

    # ✅ Step 3: Clear DB (controlled lifecycle)
    vector_store.clear_collection()

    # Debug: verify clean state
    try:
        count = vector_store.db._collection.count()
        print(f"🧪 Collection count after clear: {count}")
    except Exception:
        print("⚠️ Unable to fetch collection count")

    # ✅ Step 4: Route to module (dependency injection)
    if choice == "rag":
        print("📚 Running RAG pipeline...\n")
        rag_pipeline.run(vector_store)

    elif choice == "guardrail":
        print("🛡️ Running Guardrail Chatbot...\n")
        guardrail_pipeline.run(vector_store)


if __name__ == "__main__":
    main()