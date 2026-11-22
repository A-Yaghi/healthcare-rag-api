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

_PRIVACY_QUESTION_CUES = (
    "freely",
    "free to",
    "medical data",
    "patient data",
    "health data",
    "personal data",
    "privacy",
    "gdpr",
    "consent",
    "data protection",
    "share data",
    "reuse data",
)

_PRIVACY_DOC_TERMS = (
    "privacy",
    "data protection",
    "consent",
    "gdpr",
    "confidential",
    "freely reusable",
    "medical data",
    "personal health data",
    "lawful",
    "cybersecurity",
)


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


def is_privacy_question(question: str) -> bool:
    q = question.lower()
    return any(cue in q for cue in _PRIVACY_QUESTION_CUES)


def privacy_search_query(question: str) -> str:
    """Rewrite privacy questions so vector search finds governance docs, not cancer pages."""
    return (
        f"{question} privacy data protection consent GDPR medical personal health data "
        "lawful basis safeguards cybersecurity"
    )


def _privacy_doc_score(chunk: Chunk) -> float:
    text = f"{chunk.title} {chunk.source} {chunk.text}".lower()
    hits = sum(1 for term in _PRIVACY_DOC_TERMS if term in text)
    return min(1.0, hits / 4.0)


def rerank_hits(
    question: str,
    hits: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    if not hits:
        return []

    privacy_q = is_privacy_question(question)
    scored: list[tuple[float, Chunk, float]] = []
    for chunk, vector_score in hits:
        blob = f"{chunk.title} {chunk.source} {chunk.text}"
        lex = lexical_overlap_score(question, blob)
        if privacy_q:
            topic = _privacy_doc_score(chunk)
            # Strongly prefer privacy/governance passages over generic medical pages.
            combined = (
                (0.25 * float(vector_score))
                + (0.25 * lex)
                + (0.50 * topic)
            )
        else:
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

    if is_privacy_question(question):
        return any(term in joined.lower() for term in _PRIVACY_DOC_TERMS)

    if any(word in q for word in ("benefit", "risk", "negative", "positive", "pros", "cons")):
        if any(word in joined.lower() for word in ("benefit", "risk", "harm", "privacy", "bias")):
            return True

    distinctive = distinctive_question_terms(question)
    if distinctive:
        covered = distinctive & joined_tokens
        if len(covered) / len(distinctive) < 0.34:
            return False

    return best >= min_overlap
