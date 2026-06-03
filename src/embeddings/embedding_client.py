from __future__ import annotations

import os
from functools import lru_cache

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from src.settings import embedding_model, embedding_provider, local_embedding_model, require_openai_api_key

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def _local_model() -> SentenceTransformer:
    return SentenceTransformer(local_embedding_model())


class EmbeddingClient:
    def __init__(self) -> None:
        self.provider = embedding_provider()
        self.openai_client = OpenAI(api_key=require_openai_api_key()) if self.provider == "openai" else None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            assert self.openai_client is not None
            response = self.openai_client.embeddings.create(model=embedding_model(), input=texts)
            return [item.embedding for item in response.data]

        embeddings = _local_model().encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
