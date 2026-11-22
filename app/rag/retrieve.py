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
    "safe", "safety",
}

SYNONYMS = {
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
    "imaging": {"image", "radiology", "radiography", "medical"},
    "freely": {"free", "unrestricted", "privacy", "consent", "protection", "gdpr"},
    "profitable": {"profit", "revenue", "commercial", "business", "money"},
    "privacy": {"protection", "confidential", "gdpr", "consent", "personal"},
    # Keep screening generic — do NOT map it to mammography or every screen query
    # collapses onto breast-cancer docs.
    "screening": {"screen", "detection", "early"},
    "mammographies": {"mammography", "mammogram", "mammograms", "breast"},
    "mammography": {"mammogram", "mammograms", "mammographies", "breast"},
    "mammogram": {"mammography", "mammograms", "mammographies", "breast"},
    "mammograms": {"mammogram", "mammography", "mammographies", "breast"},
    "safe": {"safety", "risk", "harm", "harms", "benefit"},
    "safety": {"safe", "risk", "harm", "harms", "benefit"},
    "harmful": {"harm", "harms", "risk", "risks", "danger"},
    "harm": {"harmful", "harms", "risk", "risks"},
    "harms": {"harm", "harmful", "risk", "risks"},
    "xray": {"xrays", "radiography", "radiation", "ionizing", "exposure"},
    "xrays": {"xray", "radiography", "radiation", "ionizing", "exposure"},
    "radiation": {"ionizing", "radiography", "exposure", "xray", "dose"},
    "consent": {"permission", "authorization", "privacy", "gdpr"},
    "ethics": {"ethical", "principles", "governance", "responsible"},
    "ethical": {"ethics", "principles", "governance", "responsible"},
    "advantages": {"benefits", "benefit", "positive", "improve"},
    "drawbacks": {"risks", "negatives", "harms", "challenges"},
    "tomosynthesis": {"mammogram", "mammography", "3d", "imaging"},
    "ultrasound": {"sonography", "imaging", "breast"},
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

_MAMMO_QUESTION_CUES = (
    "mammograph",
    "mammogram",
    "breast screening",
    "breast cancer screening",
    "screening mamm",
    "dense breast",
    "microcalcification",
    "bi-rads",
    "birads",
)

_MAMMO_DOC_TERMS = (
    "mammogram",
    "mammography",
    "breast cancer screening",
    "false-positive",
    "false positive",
    "false-negative",
    "overdiagnosis",
    "bi-rads",
    "microcalcification",
    "dense breast",
    "tomosynthesis",
)

_RADIATION_QUESTION_CUES = (
    "xray",
    "x-ray",
    "x ray",
    "x-rays",
    "xrays",
    "ionizing",
    "radiation",
    "radiography",
    "ct scan",
    "computed tomography",
)

_GENERAL_RADIATION_CUES = (
    "not just",
    "not only",
    "all x",
    "any x",
    "in general",
    "generally",
    "besides breast",
    "other than breast",
    "beyond breast",
    "all screening",
    "every x",
)

_RADIATION_DOC_TERMS = (
    "ionizing radiation",
    "ionizing",
    "radiography",
    "computed tomography",
    "radiation risk",
    "radiation exposure",
    "justified",
    "optimized",
    "paediatric imaging",
    "pediatric imaging",
    "x-ray",
)


def _canonicalize_text(text: str) -> str:
    lowered = text.lower()
    replacements = (
        ("x-rays", "xrays"),
        ("x-ray", "xray"),
        ("x rays", "xrays"),
        ("x ray", "xray"),
        ("ct scans", "ct"),
        ("ct scan", "ct"),
    )
    for old, new in replacements:
        lowered = lowered.replace(old, new)
    return lowered


def tokenize(text: str) -> set[str]:
    prepared = _canonicalize_text(text)
    return {
        t
        for t in _TOKEN_RE.findall(prepared)
        if len(t) > 2 and t not in _STOP
    }


def expand_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))
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


def is_mammography_question(question: str) -> bool:
    q = question.lower()
    if is_general_radiation_question(question):
        return False
    return any(cue in q for cue in _MAMMO_QUESTION_CUES)


def is_radiation_question(question: str) -> bool:
    q = _canonicalize_text(question)
    return any(cue in q for cue in _RADIATION_QUESTION_CUES)


def is_general_radiation_question(question: str) -> bool:
    """True for broad X-ray/radiation questions, not breast-mammography-specific ones."""
    q = _canonicalize_text(question)
    if not is_radiation_question(question):
        return False
    if any(cue in q for cue in _GENERAL_RADIATION_CUES):
        return True
    breast_specific = any(cue in q for cue in _MAMMO_QUESTION_CUES) or "breast" in q
    return not breast_specific


def privacy_search_query(question: str) -> str:
    return (
        f"{question} privacy data protection consent GDPR medical personal health data "
        "lawful basis safeguards cybersecurity"
    )


def mammography_search_query(question: str) -> str:
    return (
        f"{question} mammogram mammography breast cancer screening radiation exposure "
        "benefits harms false positive false negative overdiagnosis"
    )


def radiation_search_query(question: str) -> str:
    return (
        f"{question} ionizing radiation radiography computed tomography X-ray exposure "
        "justified optimized paediatric imaging radiation risk benefit"
    )


def extra_search_queries(question: str) -> list[str]:
    extras: list[str] = []
    if is_privacy_question(question):
        extras.append(privacy_search_query(question))
    if is_general_radiation_question(question):
        extras.append(radiation_search_query(question))
    elif is_mammography_question(question):
        extras.append(mammography_search_query(question))
    return extras


def _topic_doc_score(question: str, chunk: Chunk) -> float:
    text = f"{chunk.title} {chunk.source} {chunk.text}".lower()
    if is_privacy_question(question):
        hits = sum(1 for term in _PRIVACY_DOC_TERMS if term in text)
        return min(1.0, hits / 4.0)
    if is_general_radiation_question(question):
        hits = sum(1 for term in _RADIATION_DOC_TERMS if term in text)
        # Penalize breast-only mammography dumps for general X-ray questions.
        if "mammogram" in text or "mammography" in text:
            if "ionizing" not in text and "computed tomography" not in text:
                hits = max(0, hits - 2)
        return min(1.0, hits / 3.0)
    if is_mammography_question(question):
        hits = sum(1 for term in _MAMMO_DOC_TERMS if term in text)
        return min(1.0, hits / 4.0)
    return 0.0


def _is_general_radiation_chunk(chunk: Chunk) -> bool:
    """True for WHO-style ionizing radiation / radiography guidance, not breast-screening pages."""
    title = chunk.title.lower()
    source = chunk.source.lower()
    text = chunk.text.lower()
    blob = f"{title} {source} {text}"

    if "ionizing radiation" in title:
        return True
    if "radiation" in title and "world health organization" in source:
        return True

    strong = sum(
        1
        for term in (
            "computed tomography",
            "radiography",
            "nuclear medicine",
            "radiotherapy",
            "paediatric imaging",
            "pediatric imaging",
            "appropriately justified",
            "exposure should then be optimized",
        )
        if term in blob
    )
    breast_heavy = sum(
        1
        for term in (
            "mammogram",
            "mammography",
            "breast cancer",
            "dense breast",
            "screening mammogram",
        )
        if term in blob
    )
    return strong >= 2 and breast_heavy <= 1


def filter_hits_for_question(
    question: str,
    hits: list[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """Drop off-topic chunks when a clearer topic match exists."""
    if not hits:
        return hits

    if is_general_radiation_question(question):
        focused = [
            (chunk, score)
            for chunk, score in hits
            if _is_general_radiation_chunk(chunk)
        ]
        if focused:
            return focused
        # Prefer refusing later over answering from breast-only dumps.
        return []

    if is_privacy_question(question):
        focused = [
            (chunk, score)
            for chunk, score in hits
            if any(
                term in f"{chunk.title} {chunk.text}".lower()
                for term in _PRIVACY_DOC_TERMS
            )
        ]
        if focused:
            return focused

    return hits


def rerank_hits(
    question: str,
    hits: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    if not hits:
        return []

    topical = (
        is_privacy_question(question)
        or is_mammography_question(question)
        or is_general_radiation_question(question)
    )
    scored: list[tuple[float, Chunk, float]] = []
    for chunk, vector_score in hits:
        blob = f"{chunk.title} {chunk.source} {chunk.text}"
        lex = lexical_overlap_score(question, blob)
        if topical:
            topic = _topic_doc_score(question, chunk)
            combined = (
                (0.25 * float(vector_score))
                + (0.25 * lex)
                + (0.50 * topic)
            )
        else:
            combined = (0.50 * float(vector_score)) + (0.50 * lex)
        scored.append((combined, chunk, float(vector_score)))

    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [(chunk, vector_score) for _, chunk, vector_score in scored]
    filtered = filter_hits_for_question(question, ranked)
    return filtered[:top_k]


def evidence_is_sufficient(
    question: str,
    contexts: list[Chunk],
    min_overlap: float = 0.22,
) -> bool:
    """Return True only when retrieved text actually supports the question."""
    if not contexts:
        return False

    joined = " ".join(f"{c.title} {c.source} {c.text}" for c in contexts)
    joined_lower = joined.lower()
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
        return any(term in joined_lower for term in _PRIVACY_DOC_TERMS)

    if is_general_radiation_question(question):
        has_general = any(term in joined_lower for term in _RADIATION_DOC_TERMS)
        mammo_only = (
            ("mammogram" in joined_lower or "mammography" in joined_lower)
            and "ionizing" not in joined_lower
            and "computed tomography" not in joined_lower
            and "radiography" not in joined_lower
        )
        return has_general and not mammo_only

    if is_mammography_question(question):
        return any(term in joined_lower for term in _MAMMO_DOC_TERMS)

    if any(word in q for word in ("benefit", "risk", "negative", "positive", "pros", "cons", "harmful", "harm")):
        if any(word in joined_lower for word in ("benefit", "risk", "harm", "privacy", "bias", "radiation")):
            return True

    if "safe" in q or "safety" in q:
        if any(term in joined_lower for term in ("radiation", "risk", "harm", "benefit", "mammogram")):
            return True

    distinctive = distinctive_question_terms(question)
    if distinctive:
        covered = distinctive & joined_tokens
        if len(covered) / len(distinctive) < 0.34:
            return False

    return best >= min_overlap
