from .loaders import load_documents
from .metadata import enrich_metadata
from .splitter import split_documents

def ingestion_pipeline(source_type, source):
    print(f"[Ingestion] Starting: {source_type}")

    docs = load_documents(source_type, source)
    print(f"[Ingestion] Loaded docs: {len(docs)}")

    # docs = enrich_metadata(docs, source_type)
    docs = enrich_metadata(docs, source_type, source)

    chunks = split_documents(docs)
    print(f"[Ingestion] Chunks created: {len(chunks)}")

    return chunks

def multi_source_ingestion(sources):
    all_chunks = []

    for source_type, source in sources:
        try:
            chunks = ingestion_pipeline(source_type, source)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Multi-Source Error] {source_type}: {e}")

    print(f"\n📦 Total chunks collected: {len(all_chunks)}")
    return all_chunks