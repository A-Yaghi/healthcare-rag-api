from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.rag.documents import Chunk


class FaissStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk]) -> None:
        self.index = index
        self.chunks = chunks

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if self.size == 0:
            return []

        k = min(top_k, self.size)
        scores, indices = self.index.search(query_vector.astype(np.float32), k)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        payload = [
            {"doc_id": c.doc_id, "title": c.title, "text": c.text} for c in self.chunks
        ]
        (directory / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> FaissStore:
        index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(index=index, chunks=chunks)

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk]) -> FaissStore:
        if len(vectors) != len(chunks):
            raise ValueError("Vector count must match chunk count")
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        return cls(index=index, chunks=chunks)
