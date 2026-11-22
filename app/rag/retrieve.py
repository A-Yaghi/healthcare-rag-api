from __future__ import annotations

import re

from app.rag.documents import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "the", "to", "what", "when", "where", "which",
    "who", "with", "do", "does", "did", "can", "could", "would", "should", "about",
    "using", "use", "used", "we", "you", "our", "their", "this", "that", "than",
}

# Very common domain words — alone they do not prove the corpus answers the question.
_COMMON_DOMAIN = {
    "ai", "artificial", "intelligence", "health", "healthcare", "medical", "medicine",
    "cancer", "breast", "device", "devices", "system", "systems", "model", "models",
    "data", "research", "patient", "patients", "care", "clinical", "public", "world",
    "organization", "who", "fda", "risk", "risks", "benefit", "benefits",
}

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
    "freely": {"free", "unrestricted", "privacy", "consent", "protection", "gdpr"},
    "profitable": {"profit", "revenue", "commercial", "business", "money"},
    "privacy": {"protection", "confidential", "gdpr", "consent", "personal"},
    "screening": {"mammogram", "mammography", "screen"},
}


def tokenize(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 2 and t not in _STOP
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
    return len(q & t) / len(q)


def distinctive_question_terms(question: str) -> set[str]:
    terms = expand_tokens(tokenize(question))
    return {t for t in terms if t not in _COMMON_DOMAIN and len(t) >= 5}


def rerank_hits(
    question: str,
    hits: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    if not hits:
        return []

    scored: list[tuple[float, Chunk, float]] = []
    for chunk, vector_score in hits:
        lex = lexical_overlap_score(
            question,
            f"{chunk.title} {chunk.source} {chunk.text}",
        )
        combined = (0.50 * float(vector_score)) + (0.50 * lex)
        scored.append((combined, chunk, float(vector_score)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, vector_score) for _, chunk, vector_score in scored[:top_k]]


def evidence_is_sufficient(
    question: str,
    contexts: list[Chunk],
    min_overlap: float = 0.22,
) -> bool:
    """Return True only when retrieved text actually supports the question."""
    if not contexts:
        return False

    joined = " ".join(f"{c.title} {c.source} {c.text}" for c in contexts)
    joined_tokens = tokenize(joined)
    best = max(
        (
            lexical_overlap_score(question, f"{c.title} {c.source} {c.text}")
            for c in contexts
        ),
        default=0.0,
    )

    q = question.lower()

    # Privacy / "freely use medical data" questions need privacy/protection evidence.
    if any(w in q for w in ("freely", "free to", "any data", "medical data", "patient data")):
        privacy_hits = ("privacy", "protection", "consent", "gdpr", "confidential")
        return any(w in joined.lower() for w in privacy_hits)

    # Pros/cons grounded if sources discuss benefits and risks.
    if any(word in q for word in ("benefit", "risk", "negative", "positive", "pros", "cons")):
        if any(word in joined.lower() for word in ("benefit", "risk", "harm", "privacy", "bias")):
            return True

    distinctive = distinctive_question_terms(question)
    if distinctive:
        covered = distinctive & joined_tokens
        # Require most distinctive concepts to appear in evidence.
        if len(covered) / len(distinctive) < 0.34:
            return False

    return best >= min_overlap
