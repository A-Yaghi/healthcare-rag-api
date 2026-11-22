from __future__ import annotations

import re
from typing import Protocol

from app.rag.documents import Chunk, split_sentences
from app.rag.retrieve import evidence_is_sufficient, lexical_overlap_score, tokenize


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
        f"I do not have enough grounded evidence in the indexed corpus to answer "
        f"this specifically: '{question}'. "
        f"The closest retrieved sources were: {available}. "
        f"Try asking about topics covered by the corpus (FDA device AI guidance, "
        f"EU AI Act / MDR interplay, WHO AI ethics, mammography, or GMLP)."
    )


class NarrativeGenerator:
    """Question-aware natural-language fallback using relevant full sentences."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        if not evidence_is_sufficient(question, contexts):
            return _insufficient_answer(question, contexts)

        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        relevant = _relevant_sentences(question, contexts, limit=8)
        if len(relevant) < 2 and lexical_overlap_score(question, " ".join(relevant)) < 0.2:
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
                f"Drawing only on the indexed sources relevant to your question:\n\n"
                f"On the positive side, {_lowercase_lead(benefit_text)}\n\n"
                f"On the risk side, {_lowercase_lead(risk_text)}"
            )
            return f"{answer}\n\nSources: {citations}"

        overview = _join_sentences(relevant, limit=7)
        return (
            f"Here is what the indexed sources say that is most relevant to "
            f"your question:\n\n{_lowercase_lead(overview)}\n\n"
            f"Sources: {citations}"
        )


class ExtractiveGenerator:
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
        reserve = instruction + f"\n\nContext:\n\n\nQuestion: {question}\nAnswer:"
        reserve_ids = self._tokenizer.encode(reserve, add_special_tokens=True)
        budget = max(128, 480 - len(reserve_ids))

        # Prefer sentences that overlap the question.
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
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        if _wants_pros_cons(question):
            instruction = (
                "You are a helpful assistant. Using ONLY the context, answer THIS "
                "question with two short paragraphs: benefits and risks. "
                "Ignore unrelated context. Do not copy the context verbatim."
            )
        else:
            instruction = (
                "You are a helpful assistant. Using ONLY the context, answer THIS "
                "exact question in 2-5 complete sentences. Focus on the question's "
                "specific entities (for example US vs EU, imaging, FDA, AI Act). "
                "If the context does not answer the question, say you do not know. "
                "Do not reuse a generic WHO summary unless it directly answers the question."
            )

        context_block = self._fit_context(question, contexts, instruction)
        prompt = (
            f"{instruction}\n\nContext:\n{context_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = self._pipe(prompt)[0]["generated_text"].strip()
        return result


class HybridGenerator:
    """Grounded LLM synthesis with question-aware narrative fallback."""

    def __init__(self, llm: FlanT5Generator) -> None:
        self.llm = llm
        self.narrative = NarrativeGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return self.narrative.generate(question, contexts)

        if not evidence_is_sufficient(question, contexts):
            return self.narrative.generate(question, contexts)

        llm_answer = self.llm.generate(question, contexts)
        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        q_terms = tokenize(question)

        # Reject LLM answers that ignore the question's key terms when those terms
        # exist in the retrieved evidence (common failure mode: generic WHO dump).
        answer_terms = tokenize(llm_answer)
        important = {t for t in q_terms if len(t) > 3}
        if important and llm_answer:
            coverage = len(important & answer_terms) / len(important)
        else:
            coverage = 1.0

        if (
            _is_weak_answer(llm_answer)
            or "…" in llm_answer
            or "..." in llm_answer
            or coverage < 0.2
        ):
            return self.narrative.generate(question, contexts)

        if "Sources:" not in llm_answer and citations:
            return f"{llm_answer}\n\nSources: {citations}"
        return llm_answer
