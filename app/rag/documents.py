from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    source: str = ""
    license: str = ""
    url: str = ""


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(raw.strip())
    if not match:
        return {}, raw.strip()

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, match.group(2).strip()


def _chunk_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def load_knowledge_chunks(knowledge_dir: Path) -> list[Chunk]:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        title = meta.get("title") or path.stem.replace("_", " ").title()
        source = meta.get("source", "")
        license_name = meta.get("license", "")
        url = meta.get("url", "")

        for i, piece in enumerate(_chunk_text(body)):
            chunks.append(
                Chunk(
                    doc_id=f"{path.stem}-{i}",
                    title=title,
                    text=piece,
                    source=source,
                    license=license_name,
                    url=url,
                )
            )

    if not chunks:
        raise ValueError(f"No markdown documents found in {knowledge_dir}")
    return chunks
