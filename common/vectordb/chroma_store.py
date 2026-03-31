# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
from langchain.docstore.document import Document

from common.vectordb.base_vector_store import BaseVectorStore


class ChromaStore(BaseVectorStore):
    def __init__(self, embedder, persist_directory="./chroma_db"):
        self.embedder = embedder
        self.persist_directory = persist_directory

        # Create or connect to Chroma collection
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedder
        )

    def clear_collection(self):
        """Utility to clear the entire collection (for testing/QA/UAT)."""
        try:
            self.db._client.delete_collection(name=self.db._collection.name)
            print("Vector DB cleared ✅")
            # Recreate collection for fresh additions
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedder
            )
        except Exception as e:
            print(f"Failed to clear DB: {e}")

    def add_documents(self, docs: list):
        """
        Add documents with metadata to the vector store.
        Each doc must be a dict: {"text": str, "metadata": dict (optional)}
        """
        texts = [doc["text"] for doc in docs]
        metadatas = [doc.get("metadata", {}) for doc in docs]  # ensure dict

        print("Collection count:", self.db._collection.count())

        # Add to Chroma
        self.db.add_texts(texts=texts, metadatas=metadatas)

        print(f"{len(docs)} documents added to Vector DB ✅")

    def similarity_search(self, query, k=3):
        """
        Search for top-k similar documents.
        Returns list of langchain Document objects (with page_content and metadata)
        """
        results = self.db.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query, k):
        return self.db.similarity_search_with_score(query, k=k)