# enterprise_rag/ingestion.py

from common.ingestion.pipeline import ingestion_pipeline

def load_enterprise_data(source_type, source):
    return ingestion_pipeline(source_type, source)