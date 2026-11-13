"""Qdrant semantic memory for repository context."""

from qdrant_client import QdrantClient, models


class MemoryStore:
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        collection: str,
        url: str | None = None,
    ) -> None:
        # No URL = temporary in-memory Qdrant for Colab.
        self.client = (
            QdrantClient(url=url)
            if url
            else QdrantClient(":memory:")
        )

        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.client.get_embedding_size(self.model_name),
                distance=models.Distance.COSINE,
            ),
        )

    def index(self, documents: list[dict]) -> int:
        """Embed and store repository documents."""

        if not documents:
            return 0

        self.client.upload_collection(
            collection_name=self.collection,
            ids=list(range(1, len(documents) + 1)),
            vectors=[
                models.Document(
                    text=document["text"],
                    model=self.model_name,
                )
                for document in documents
            ],
            payload=documents,
        )

        return len(documents)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Find semantically related repository context."""

        results = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(
                text=query,
                model=self.model_name,
            ),
            limit=limit,
            with_payload=True,
        ).points

        return [
            {
                "score": round(point.score, 4),
                **point.payload,
            }
            for point in results
        ]
