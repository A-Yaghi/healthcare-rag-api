from __future__ import annotations

import re

from app.rag.documents import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Lightweight stopwords so reranking focuses on content terms.
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "about",
    "using",
    "use",
    "used",
}


def tokenize(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 2 and t not in _STOP
    }


def rerank_hits(
    question: str,
    hits: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    """Blend vector score with question-term overlap so answers track the query."""
    if not hits:
        return []

    scored: list[tuple[float, Chunk, float]] = []
    for chunk, vector_score in hits:
        lex = lexical_overlap_score(
            question,
            f"{chunk.title} {chunk.source} {chunk.text}",
        )
        # Prefer chunks that both embed well and share question terms.
        combined = (0.55 * float(vector_score)) + (0.45 * lex)
        scored.append((combined, chunk, float(vector_score)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, vector_score) for _, chunk, vector_score in scored[:top_k]]


_SYNONYMS = {
    "benefits": {"benefit", "advantages", "positive", "improve", "help"},
    "benefit": {"benefits", "advantages", "positive", "improve", "help"},
    "negatives": {"negative", "risks", "risk", "harms", "drawbacks", "challenges"},
    "negative": {"negatives", "risks", "risk", "harms", "drawbacks"},
    "risks": {"risk", "negatives", "harms", "challenges", "threats"},
    "risk": {"risks", "negatives", "harms", "challenges"},
    "regulations": {"regulation", "regulatory", "law", "act", "guidance", "fda", "mdr"},
    "regulation": {"regulations", "regulatory", "law", "act", "guidance"},
    "eu": {"europe", "european", "aia", "mdr", "ivdr"},
    "us": {"usa", "united", "states", "fda", "american"},
    "imaging": {"image", "radiology", "mammography", "mammogram", "medical"},
}


def expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_SYNONYMS.get(token, set()))
    return expanded


def lexical_overlap_score(question: str, text: str) -> float:
    q = expand_tokens(tokenize(question))
    if not q:
        return 0.0
    t = tokenize(text)
    if not t:
        return 0.0
    overlap = q & t
    return len(overlap) / len(q)


def evidence_is_sufficient(question: str, contexts: list[Chunk], min_overlap: float = 0.12) -> bool:
    if not contexts:
        return False
    joined = " ".join(f"{c.title} {c.source} {c.text}" for c in contexts)
    best = max(
        (
            lexical_overlap_score(question, f"{c.title} {c.source} {c.text}")
            for c in contexts
        ),
        default=0.0,
    )
    # Pros/cons questions are grounded if the retrieved text discusses benefits/risks.
    q = question.lower()
    if any(word in q for word in ("benefit", "risk", "negative", "positive", "pros", "cons")):
        if any(word in joined.lower() for word in ("benefit", "risk", "harm", "privacy", "bias")):
            return True
    return best >= min_overlap
