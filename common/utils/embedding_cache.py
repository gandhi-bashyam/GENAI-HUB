from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_query(text: str):
    return text.strip().lower()