from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader
def load_documents(source_type, source):
    try:
        if source_type == "pdf":
            return PyPDFLoader(source).load()

        elif source_type == "web":
            return WebBaseLoader(source).load()

        elif source_type == "arxiv":
            return ArxivLoader(query=source).load()

        elif source_type == "wiki":
            try:
                return WikipediaLoader(query=source, load_max_docs=2).load()
            except Exception as e:
                print(f"[Wiki Timeout] {e}")
                return []

        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    except Exception as e:
        print(f"[Loader Error] {source_type}: {e}")
        return []