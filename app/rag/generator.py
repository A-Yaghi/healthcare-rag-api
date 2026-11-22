from __future__ import annotations

import re
from typing import Protocol

from app.rag.documents import Chunk, split_sentences


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> str: ...


def _cite(chunk: Chunk) -> str:
    return chunk.source or chunk.title or chunk.doc_id


def _is_weak_answer(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 80:
        return True
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) < 18:
        return True
    # Tiny one-liners without punctuation are usually FLAN-T5 collapses.
    if cleaned.count(".") == 0 and len(words) < 30:
        return True
    return False


def _wants_pros_cons(question: str) -> bool:
    q = question.lower()
    cues = (
        "benefit",
        "risk",
        "negative",
        "positive",
        "pros",
        "cons",
        "advantage",
        "disadvantage",
        "challenge",
        "drawback",
    )
    return any(cue in q for cue in cues)


def _unique_sentences(contexts: list[Chunk]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for chunk in contexts:
        for sentence in split_sentences(chunk.text):
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append((_cite(chunk), sentence))
    return ordered


def _classify_sentence(sentence: str) -> str:
    lower = sentence.lower()
    benefit_cues = (
        "benefit",
        "improve",
        "assist",
        "support",
        "promise",
        "help",
        "reduce",
        "enable",
        "useful",
        "potential applications",
    )
    risk_cues = (
        "risk",
        "bias",
        "hallucin",
        "privacy",
        "inaccurate",
        "harm",
        "challenge",
        "limitation",
        "false",
        "threat",
        "concern",
    )
    bscore = sum(1 for cue in benefit_cues if cue in lower)
    rscore = sum(1 for cue in risk_cues if cue in lower)
    if bscore and rscore:
        return "risk" if rscore >= bscore else "benefit"
    if bscore:
        return "benefit"
    if rscore:
        return "risk"
    return "other"


def _join_sentences(items: list[str], limit: int = 6) -> str:
    chosen = items[:limit]
    if not chosen:
        return ""
    return " ".join(chosen)


def _lowercase_lead(text: str) -> str:
    if not text:
        return text
    # Keep acronyms like WHO at the start.
    if text[:3].isupper() and len(text) > 3 and text[3] in {" ", "'", ","}:
        return text
    return text[0].lower() + text[1:]


class NarrativeGenerator:
    """Natural-language fallback that uses full sentences (no ellipsis clipping)."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        pairs = _unique_sentences(contexts)

        if _wants_pros_cons(question):
            benefits = [s for _, s in pairs if _classify_sentence(s) == "benefit"]
            risks = [s for _, s in pairs if _classify_sentence(s) == "risk"]
            other = [s for _, s in pairs if _classify_sentence(s) == "other"]

            benefit_text = _join_sentences(benefits) or _join_sentences(other, 3)
            risk_text = (
                _join_sentences(risks)
                or _join_sentences(other[3:], 3)
                or _join_sentences(other, 2)
            )

            answer = (
                f"Drawing only on the indexed sources, here is a concise take on "
                f"{question.rstrip('?')}.\n\n"
                f"On the positive side, {_lowercase_lead(benefit_text)}\n\n"
                f"On the risk side, {_lowercase_lead(risk_text)}"
            )
            return f"{answer}\n\nSources: {citations}"

        overview = _join_sentences([s for _, s in pairs], limit=8)
        orgs = sorted({_cite(c) for c in contexts})
        org_phrase = ", ".join(orgs)
        return (
            f"Based on {org_phrase}, {_lowercase_lead(overview)}\n\n"
            f"Sources: {citations}"
        )


class ExtractiveGenerator:
    """Offline/CI generator — full chunk text, no ellipsis truncation."""

    def __init__(self) -> None:
        self._narrative = NarrativeGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        return self._narrative.generate(question, contexts)


class FlanT5Generator:
    """Small grounded LLM layer: rewrite retrieved evidence into natural prose."""

    def __init__(self, model_name: str, max_new_tokens: int = 320) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
        )
        self.model_name = model_name
        self._tokenizer = self._pipe.tokenizer

    def _fit_context(self, question: str, contexts: list[Chunk], instruction: str) -> str:
        """Pack as many full sentences as fit in the model context window."""
        reserve = instruction + f"\n\nContext:\n\n\nQuestion: {question}\nAnswer:"
        reserve_ids = self._tokenizer.encode(reserve, add_special_tokens=True)
        # FLAN-T5 practical limit is 512 tokens; leave room for generation prefix.
        budget = max(128, 480 - len(reserve_ids))

        pieces: list[str] = []
        used = 0
        for cite, sentence in _unique_sentences(contexts):
            candidate = f"[{cite}] {sentence}"
            token_len = len(
                self._tokenizer.encode(candidate + "\n", add_special_tokens=False)
            )
            if pieces and used + token_len > budget:
                break
            if not pieces and token_len > budget:
                # Extremely long sentence: keep a token-safe prefix ending on a space.
                ids = self._tokenizer.encode(candidate, add_special_tokens=False)[:budget]
                pieces.append(self._tokenizer.decode(ids, skip_special_tokens=True))
                break
            pieces.append(candidate)
            used += token_len
        return "\n".join(pieces)

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        if _wants_pros_cons(question):
            instruction = (
                "You are a helpful assistant. Using ONLY the context, write a natural "
                "conversational answer with two short paragraphs: (1) benefits/positive "
                "uses and (2) risks/negatives. Do not copy the context verbatim. "
                "Do not use ellipses. If evidence is missing, say so briefly."
            )
        else:
            instruction = (
                "You are a helpful assistant. Using ONLY the context, write a natural "
                "conversational answer in 2-5 complete sentences. Synthesize the key "
                "points in plain language. Do not copy the context verbatim. "
                "Do not use ellipses. If the answer is not in the context, say you do not know."
            )

        context_block = self._fit_context(question, contexts, instruction)
        prompt = (
            f"{instruction}\n\nContext:\n{context_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = self._pipe(prompt)[0]["generated_text"].strip()
        return result


class HybridGenerator:
    """Grounded LLM synthesis with natural-language fallback (never ellipsis clips)."""

    def __init__(self, llm: FlanT5Generator) -> None:
        self.llm = llm
        self.narrative = NarrativeGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return self.narrative.generate(question, contexts)

        llm_answer = self.llm.generate(question, contexts)
        citations = "; ".join(sorted({_cite(c) for c in contexts}))

        # Prefer LLM prose; if it under-answers, use narrative synthesis instead of
        # dumping webpage-like truncated snippets.
        if _is_weak_answer(llm_answer) or "…" in llm_answer or "..." in llm_answer:
            return self.narrative.generate(question, contexts)

        if "Sources:" not in llm_answer and citations:
            return f"{llm_answer}\n\nSources: {citations}"
        return llm_answer
