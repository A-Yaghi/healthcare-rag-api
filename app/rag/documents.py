from __future__ import annotations

import hashlib
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
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


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


def split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    parts = _SENTENCE_RE.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Chunk on sentence boundaries so answers never start mid-phrase."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        extra = len(sentence) + (1 if current else 0)
        if current and length + extra > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            length = len(sentence)
        else:
            current.append(sentence)
            length += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def knowledge_fingerprint(knowledge_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(knowledge_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(b"chunking:sentence-v2")
    return digest.hexdigest()[:16]


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
