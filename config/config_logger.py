import os

def log_config():
    print("\n⚙️ CONFIGURATION")
    print("-" * 30)

    keys = [
        "ENV",
        "EMBEDDING_MODEL",
        "VECTOR_DB",
        "RETRIEVER_TYPE",
        "ALPHA"
    ]

    for key in keys:
        value = os.getenv(key, "NOT_SET")
        print(f"{key}: {value}")

    print("-" * 30)