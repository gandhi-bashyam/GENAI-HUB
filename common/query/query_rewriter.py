class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, query):
        prompt = f"""
Generate 3 alternative search queries for:
"{query}"

Make them diverse but relevant.
Return as a list.
"""

        try:
            response = self.llm.generate(prompt)

            # simple parsing
            queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

            return list(set([query] + queries))[:4]

        except Exception:
            return [query]