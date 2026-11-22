from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str


def _chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def load_knowledge_chunks(knowledge_dir: Path) -> list[Chunk]:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        title = path.stem.replace("_", " ").title()
        for i, piece in enumerate(_chunk_text(raw)):
            chunks.append(
                Chunk(
                    doc_id=f"{path.stem}-{i}",
                    title=title,
                    text=piece,
                )
            )

    if not chunks:
        raise ValueError(f"No markdown documents found in {knowledge_dir}")
    return chunks
