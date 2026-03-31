from dotenv import load_dotenv
from pathlib import Path
import os

# locate project root .env automatically
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

from common.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

# Optional future import
# from common.embeddings.openai_embedder import OpenAIEmbedder

load_dotenv()

def get_embedder():

    model_type = os.getenv("EMBEDDING_MODEL")

    if model_type == "sentence-transformer":
        return SentenceTransformerEmbedder()
    
    # Future support
    # if model_type == "openai":
    #     return OpenAIEmbedder()

    raise ValueError(f"Unsupported embedding model: {model_type}")