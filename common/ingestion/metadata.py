import uuid
from datetime import datetime

def enrich_metadata(docs, source_type, source):
    for doc in docs:
        doc.metadata["source_type"] = source_type
        doc.metadata["source"] = str(source)
        doc.metadata["ingested_at"] = datetime.utcnow().isoformat()
        doc.metadata["doc_id"] = str(uuid.uuid4())
        doc.metadata["priority"] = 2 if source_type == "pdf" else 1
    return docs