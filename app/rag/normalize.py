from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches

from app.rag.documents import Chunk
from app.rag.retrieve import SYNONYMS

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|[0-9]+")

_SEED_TERMS = {
    "mammogram",
    "mammograms",
    "mammography",
    "mammographies",
    "screening",
    "radiation",
    "ionizing",
    "radiography",
    "xray",
    "xrays",
    "privacy",
    "consent",
    "gdpr",
    "protection",
    "regulation",
    "regulations",
    "regulatory",
    "ethics",
    "ultrasound",
    "tomosynthesis",
    "biopsy",
    "calcification",
    "microcalcification",
    "microcalcifications",
    "overdiagnosis",
    "overtreatment",
    "radiology",
    "imaging",
    "healthcare",
    "artificial",
    "intelligence",
    "profitable",
    "profit",
    "harmful",
}


def _cutoff_for(token: str) -> float:
    if len(token) <= 4:
        return 0.86
    if len(token) <= 7:
        return 0.78
    return 0.72


def build_vocabulary(chunks: list[Chunk] | None = None) -> set[str]:
    vocab = set(_SEED_TERMS)
    for key, values in SYNONYMS.items():
        vocab.add(key.lower())
        vocab.update(v.lower() for v in values)

    if chunks:
        for chunk in chunks:
            blob = f"{chunk.title} {chunk.source} {chunk.text}".lower()
            for token in re.findall(r"[a-z]{4,}", blob):
                vocab.add(token)
    return vocab


@dataclass
class NormalizedQuery:
    original: str
    corrected: str
    search_text: str
    corrections: dict[str, str]

    @property
    def changed(self) -> bool:
        return self.corrected.lower() != self.original.lower() or bool(self.corrections)


class QueryNormalizer:
    """Spell-correct user questions against corpus vocabulary + expand synonyms."""

    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = sorted({v.lower() for v in vocabulary})

    def correct_token(self, token: str) -> str:
        lower = token.lower()
        if lower in self.vocabulary:
            return lower
        if len(lower) < 4:
            return lower

        matches = get_close_matches(
            lower,
            self.vocabulary,
            n=3,
            cutoff=_cutoff_for(lower),
        )
        if not matches:
            return lower

        best = matches[0]
        ratio = SequenceMatcher(None, lower, best).ratio()
        if ratio < _cutoff_for(lower):
            return lower
        if abs(len(lower) - len(best)) > 2 and ratio < 0.88:
            return lower
        return best

    def normalize(self, question: str) -> NormalizedQuery:
        corrections: dict[str, str] = {}
        pieces: list[str] = []
        last_end = 0

        for match in _TOKEN_RE.finditer(question):
            raw = match.group(0)
            pieces.append(question[last_end:match.start()])
            last_end = match.end()

            # Keep short acronyms untouched (AI, FDA, WHO, EU, US, MDR).
            if raw.isupper() and 1 < len(raw) <= 5:
                pieces.append(raw)
                continue
            if raw.isdigit():
                pieces.append(raw)
                continue

            corrected = self.correct_token(raw)
            if corrected != raw.lower():
                if raw.istitle():
                    out = corrected.capitalize()
                elif raw.isupper():
                    out = corrected.upper()
                else:
                    out = corrected
                corrections[raw] = out
                pieces.append(out)
            else:
                pieces.append(raw)

        pieces.append(question[last_end:])
        corrected_question = "".join(pieces).strip() or question.strip()

        expansions: list[str] = []
        seen = set(re.findall(r"[a-z]{4,}", corrected_question.lower()))
        for token in list(seen):
            for syn in sorted(SYNONYMS.get(token, set())):
                if syn not in seen and syn not in expansions:
                    expansions.append(syn)
                if len(expansions) >= 6:
                    break
            if len(expansions) >= 6:
                break

        search_text = corrected_question
        if expansions:
            search_text = f"{corrected_question} {' '.join(expansions)}"

        return NormalizedQuery(
            original=question,
            corrected=corrected_question,
            search_text=search_text,
            corrections=corrections,
        )


def build_normalizer_from_chunks(chunks: list[Chunk]) -> QueryNormalizer:
    return QueryNormalizer(build_vocabulary(chunks))
