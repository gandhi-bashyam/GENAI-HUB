# main.py

from common.embeddings.embedder_factory import get_embedder
from common.retriever.vector_retriever import VectorRetriever
from common.vectordb.chroma_store import ChromaStore
from enterprise_rag.pipeline import RAGPipeline
from common.retriever.retriever_factory import get_retriever
from config.config_logger import log_config

# LLM imports
from enterprise_rag.llm.local_ollama import LocalOllama
from enterprise_rag.llm.router import LLMRouter

def run():

    print("🚀 Starting Enterprise RAG...")

    log_config()

    # Embedder
    embedder = get_embedder()
    print("✅ Embedder initialized")

    # Load documents
    from common.vectordb.load_documents import load_documents
    docs = load_documents("data/")
    print(f"✅ Documents loaded: {len(docs)}")

    # # Vector store
    # vector_store = ChromaStore(embedder)
    # print("✅ Vector store initialized")
    vector_store.add_documents(docs)
    print("✅ Documents added to vector DB")

    # Retriever
    # retriever = VectorRetriever(vector_store)
    # print("✅ Retriever ready")

    retriever = get_retriever(docs, vector_store)
    print("✅ Retriever ready:", type(retriever))

    # Setup Local Ollama client
    local_llm = LocalOllama(model="llama3")

    # Setup LLM router (production-ready)
    router = LLMRouter(
        local_llm=local_llm,
        fallback_llm=None  # can plug in OpenAI or any other LLM here
    )
    print(f"🧠 LLM TYPE: {type(router)}")

    # Pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        retriever=retriever,
        llm=router
    )
    print("✅ Pipeline ready")

    # Query
    # query = "What is this document about?"
    query = "Who wrote Hamlet?"

    # print(f"🔍 Query: {query}")

    # Run pipeline
    response = pipeline.run(query)
    print("✅ Response generated")

    print("\n🧠 FINAL RESPONSE:\n", response)


if __name__ == "__main__":
    run()