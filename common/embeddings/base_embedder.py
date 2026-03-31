from abc import ABC, abstractmethod

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_documents(self, texts: list[str]):

        """
            Generate embeddings for list of documents
        """

        pass

    @abstractmethod
    def embed_query(self, query: str):

        """
            Generate embeddings for a single query
        """

        pass


