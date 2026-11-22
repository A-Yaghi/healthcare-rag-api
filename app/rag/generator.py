from __future__ import annotations

import re
from typing import Protocol

from app.rag.documents import Chunk, split_sentences
from app.rag.retrieve import (
    distinctive_question_terms,
    evidence_is_sufficient,
    lexical_overlap_score,
    tokenize,
)


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> str: ...


def _cite(chunk: Chunk) -> str:
    return chunk.source or chunk.title or chunk.doc_id


def _is_weak_answer(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 60:
        return True
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) < 12:
        return True
    low = cleaned.lower()
    if low.startswith(("here is what", "based on ", "drawing only")):
        # Narrative dumps are a fallback, not preferred LLM output.
        return False
    if cleaned.count(".") == 0 and len(words) < 28:
        return True
    return False


def _wants_pros_cons(question: str) -> bool:
    q = question.lower()
    cues = (
        "benefit", "risk", "negative", "positive", "pros", "cons",
        "advantage", "disadvantage", "challenge", "drawback",
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
        "benefit", "improve", "assist", "support", "promise", "help",
        "reduce", "enable", "useful", "potential applications",
    )
    risk_cues = (
        "risk", "bias", "hallucin", "privacy", "inaccurate", "harm",
        "challenge", "limitation", "false", "threat", "concern",
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
    if text[:3].isupper() and len(text) > 3 and text[3] in {" ", "'", ","}:
        return text
    return text[0].lower() + text[1:]


def _relevant_sentences(question: str, contexts: list[Chunk], limit: int = 8) -> list[str]:
    scored: list[tuple[float, str]] = []
    for _, sentence in _unique_sentences(contexts):
        score = lexical_overlap_score(question, sentence)
        if score <= 0:
            continue
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def _insufficient_answer(question: str, contexts: list[Chunk]) -> str:
    available = "; ".join(sorted({_cite(c) for c in contexts})) or "none"
    return (
        f"I don't know from the indexed corpus. The sources do not contain a direct "
        f"answer to: '{question}'. "
        f"Closest retrieved sources were: {available}. "
        f"Try questions about FDA/EU AI medical-device rules, WHO AI ethics, "
        f"privacy/data-protection considerations in those docs, mammography, "
        f"breast cancer screening, or medical imaging radiation safety."
    )


def _answer_addresses_question(question: str, answer: str) -> bool:
    low = answer.lower()
    if "don't know" in low or "do not know" in low:
        return True
    if _wants_pros_cons(question):
        return ("benefit" in low or "positive" in low) and (
            "risk" in low or "harm" in low or "negative" in low
        )
    distinctive = distinctive_question_terms(question)
    if not distinctive:
        return True
    answer_terms = tokenize(answer)
    coverage = len(distinctive & answer_terms) / len(distinctive)
    return coverage >= 0.25


class NarrativeGenerator:
    """Question-aware fallback; refuses when evidence is weak."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts or not evidence_is_sufficient(question, contexts):
            return _insufficient_answer(question, contexts)

        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        relevant = _relevant_sentences(question, contexts, limit=8)
        if len(relevant) < 2:
            return _insufficient_answer(question, contexts)

        if _wants_pros_cons(question):
            benefits = [s for s in relevant if _classify_sentence(s) == "benefit"]
            risks = [s for s in relevant if _classify_sentence(s) == "risk"]
            other = [s for s in relevant if _classify_sentence(s) == "other"]
            benefit_text = _join_sentences(benefits) or _join_sentences(other, 3)
            risk_text = (
                _join_sentences(risks)
                or _join_sentences(other[3:], 3)
                or _join_sentences(other, 2)
            )
            answer = (
                f"From the indexed sources:\n\n"
                f"On the positive side, {_lowercase_lead(benefit_text)}\n\n"
                f"On the risk side, {_lowercase_lead(risk_text)}\n\n"
                f"Sources: {citations}"
            )
            if not _answer_addresses_question(question, answer):
                return _insufficient_answer(question, contexts)
            return answer

        overview = _join_sentences(relevant, limit=6)
        answer = (
            f"From the indexed sources most relevant to your question:\n\n"
            f"{overview}\n\n"
            f"Sources: {citations}"
        )
        if not _answer_addresses_question(question, answer):
            return _insufficient_answer(question, contexts)
        return answer


class ExtractiveGenerator:
    def __init__(self) -> None:
        self._narrative = NarrativeGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        return self._narrative.generate(question, contexts)


class FlanT5Generator:
    """Grounded LLM layer — stronger FLAN-T5 variant via settings."""

    def __init__(self, model_name: str, max_new_tokens: int = 384) -> None:
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
        reserve = instruction + f"\n\nContext:\n\n\nQuestion: {question}\nAnswer:"
        reserve_ids = self._tokenizer.encode(reserve, add_special_tokens=True)
        budget = max(128, 480 - len(reserve_ids))

        ranked = []
        for cite, sentence in _unique_sentences(contexts):
            score = lexical_overlap_score(question, sentence)
            ranked.append((score, cite, sentence))
        ranked.sort(key=lambda item: item[0], reverse=True)

        pieces: list[str] = []
        used = 0
        for score, cite, sentence in ranked:
            if score <= 0 and pieces:
                continue
            candidate = f"[{cite}] {sentence}"
            token_len = len(
                self._tokenizer.encode(candidate + "\n", add_special_tokens=False)
            )
            if pieces and used + token_len > budget:
                break
            if not pieces and token_len > budget:
                ids = self._tokenizer.encode(candidate, add_special_tokens=False)[:budget]
                pieces.append(self._tokenizer.decode(ids, skip_special_tokens=True))
                break
            pieces.append(candidate)
            used += token_len
        return "\n".join(pieces)

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I don't know from the indexed corpus — no relevant passages were retrieved."
            )

        instruction = (
            "You are a careful assistant for a retrieval system. "
            "Answer ONLY using the context. "
            "If the context does not directly answer the question, reply exactly: "
            "I don't know from the indexed corpus. "
            "Do not invent facts. Do not dump unrelated regulations. "
            "Write clear natural sentences."
        )
        if _wants_pros_cons(question):
            instruction += (
                " If answering, use two short paragraphs: benefits, then risks."
            )

        context_block = self._fit_context(question, contexts, instruction)
        prompt = (
            f"{instruction}\n\nContext:\n{context_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return self._pipe(prompt)[0]["generated_text"].strip()


class HybridGenerator:
    """Stronger grounded LLM with strict refusal when evidence is weak."""

    def __init__(self, llm: FlanT5Generator) -> None:
        self.llm = llm
        self.narrative = NarrativeGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts or not evidence_is_sufficient(question, contexts):
            return _insufficient_answer(question, contexts)

        llm_answer = self.llm.generate(question, contexts)
        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        low = llm_answer.lower()

        if (
            "don't know" in low
            or "do not know" in low
            or "i don't know from the indexed corpus" in low
        ):
            return _insufficient_answer(question, contexts)

        if (
            _is_weak_answer(llm_answer)
            or "…" in llm_answer
            or "..." in llm_answer
            or not _answer_addresses_question(question, llm_answer)
        ):
            return self.narrative.generate(question, contexts)

        if "Sources:" not in llm_answer and citations:
            return f"{llm_answer}\n\nSources: {citations}"
        return llm_answer
