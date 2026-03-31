def print_vector_store_contents(vector_store, sample_dim=5):
    """
    Prints all documents, embeddings, and metadata in a Chroma vector store.
    """
    try:
        collection = vector_store.db._collection  # internal Chroma collection
        data = collection.get(include=["documents", "embeddings", "metadatas"])  # only valid fields
    except AttributeError:
        print("Vector store does not have _collection attribute.")
        return

    print(f"Total documents stored: {len(data['documents'])}\n")

    for doc, embedding, meta in zip(
        data['documents'], data['embeddings'], data['metadatas']
    ):
        print("Document:", doc)
        print("Embedding length:", len(embedding))
        print("Embedding sample:", embedding[:sample_dim])
        print("Metadata:", meta)
        print("-" * 50)


docs = [
    {"text": "Employees are entitled to 20 leave days", "metadata": {"category": "HR", "project": "Policy"}},
    {"text": "Leave approval must come from manager", "metadata": {"category": "HR", "project": "Policy"}},
    {"text": "Health insurance benefits are provided", "metadata": {"category": "HR", "project": "Benefits"}},
]

