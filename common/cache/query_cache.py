class QueryCache:
    def __init__(self):
        self.cache = {}

    def get(self, query):
        return self.cache.get(query)

    def set(self, query, result):
        self.cache[query] = result

    def exists(self, query):
        return query in self.cache