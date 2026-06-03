from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src.embeddings.embedding_client import EmbeddingClient
from src.settings import chroma_db_dir, collection_name


@dataclass
class RetrievedChunk:
    document: str
    metadata: dict
    distance: float | None = None


class ChromaRetriever:
    def __init__(self) -> None:
        self.embedding_client = EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_dir())
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name())

    def _embed_query(self, query: str) -> list[float]:
        return self.embedding_client.embed_query(query)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        if self.collection.count() == 0:
            raise RuntimeError(
                "Chroma collection is empty. Run `python src/ingestion/ingest.py` first."
            )

        query_embedding = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            RetrievedChunk(document=doc, metadata=meta or {}, distance=dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]


def retrieve(query: str, top_k: int = 5) -> dict:
    chunks = ChromaRetriever().retrieve(query=query, top_k=top_k)
    return {
        "documents": [chunk.document for chunk in chunks],
        "metadata": [chunk.metadata for chunk in chunks],
        "sources": [
            {
                "source": chunk.metadata.get("source"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "theme": chunk.metadata.get("theme"),
                "distance": chunk.distance,
            }
            for chunk in chunks
        ],
    }
