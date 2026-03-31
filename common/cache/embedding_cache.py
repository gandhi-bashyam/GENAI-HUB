class EmbeddingCache:
    def __init__(self):
        self.cache = {}

    def get(self, text):
        return self.cache.get(text)

    def set(self, text, embedding):
        self.cache[text] = embedding