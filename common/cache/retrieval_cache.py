class RetrievalCache:
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            print("Retrieval Cache Hit")
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key, value):
        self.cache[key] = value

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0
        }