from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

import chromadb

from src.embeddings.embedding_client import EmbeddingClient
from src.settings import RAW_DOCS_DIR, chroma_db_dir, collection_name


THEME = "AI数据中心液冷"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def iter_source_files(raw_docs_dir: Path = RAW_DOCS_DIR) -> Iterable[Path]:
    yield from sorted(
        path for path in raw_docs_dir.iterdir() if path.suffix.lower() in {".md", ".txt"} and path.is_file()
    )


def build_chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    digest = hashlib.sha1(f"{source}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-{chunk_index}-{digest}"


def ingest(chunk_size: int = 500, overlap: int = 80) -> int:
    embedding_client = EmbeddingClient()
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir())
    collection = chroma_client.get_or_create_collection(name=collection_name())

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for source_path in iter_source_files():
        text = clean_text(source_path.read_text(encoding="utf-8"))
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            source = source_path.name
            ids.append(build_chunk_id(source, chunk_index, chunk))
            documents.append(chunk)
            metadatas.append({"source": source, "chunk_index": chunk_index, "theme": THEME})

    if not documents:
        raise RuntimeError(f"No .md or .txt documents found in {RAW_DOCS_DIR}")

    embeddings = embedding_client.embed_texts(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest liquid cooling industry documents into ChromaDB.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    count = ingest(chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Ingested {count} chunks into Chroma collection '{collection_name()}'.")


if __name__ == "__main__":
    main()
