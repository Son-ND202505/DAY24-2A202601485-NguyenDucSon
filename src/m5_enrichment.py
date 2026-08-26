from __future__ import annotations

"""Module 5: Enrichment Pipeline."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_MODEL  # noqa: E402


@dataclass
class EnrichedChunk:
    """A chunk enriched before embedding."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _client():
    if not OPENAI_API_KEY or os.getenv("USE_LLM_ENRICHMENT", "0") != "1":
        return None
    from openai import OpenAI

    return OpenAI()


def _parse_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _fallback_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    questions = []
    for sentence in _fallback_sentences(text):
        if len(sentence) > 10:
            questions.append(f"{sentence.rstrip('.!?')}?")
        if len(questions) >= n_questions:
            break
    return questions


def _fallback_metadata(text: str) -> dict:
    text_l = text.lower()
    if any(term in text_l for term in ["nghỉ", "lương", "nhân viên", "bao hiểm", "đào tạo"]):
        category = "hr"
    elif any(term in text_l for term in ["mật khẩu", "vpn", "thiết bị", "email"]):
        category = "it"
    elif any(term in text_l for term in ["chi phí", "hóa đơn", "ngân sách", "tài chính"]):
        category = "finance"
    else:
        category = "policy"
    return {"topic": "general", "entities": [], "category": category, "language": "vi"}


def summarize_chunk(text: str) -> str:
    """Create a short summary for a chunk."""
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Tom tat doan van sau trong 2-3 cau ngan gon bang tieng Viet."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  Warning: OpenAI summarize failed: {exc}")

    sentences = _fallback_sentences(text)
    return " ".join(sentences[:2]) if sentences else text


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate questions that the chunk can answer."""
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"Dua tren doan van, tao {n_questions} cau hoi ma doan van co the tra loi. Moi cau hoi tren 1 dong.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            lines = resp.choices[0].message.content.strip().splitlines()
            return [line.strip().lstrip("0123456789.-) ") for line in lines if line.strip()][:n_questions]
        except Exception as exc:
            print(f"  Warning: OpenAI HyQA failed: {exc}")

    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend one sentence of document context while preserving the original text."""
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Viet 1 cau ngan mo ta doan van nam o dau trong tai lieu va noi ve chu de gi. Chi tra ve 1 cau.",
                    },
                    {"role": "user", "content": f"Tai lieu: {document_title}\n\nDoan van:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as exc:
            print(f"  Warning: OpenAI contextual failed: {exc}")

    prefix = f"Trich tu {document_title}. " if document_title else "Ngu canh tai lieu: "
    return f"{prefix}{text}"


def extract_metadata(text: str) -> dict:
    """Extract topic, entities, category, and language metadata."""
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": 'Trich xuat metadata tu doan van. Chi tra ve JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return _parse_json(resp.choices[0].message.content)
        except Exception as exc:
            print(f"  Warning: OpenAI metadata failed: {exc}")

    return _fallback_metadata(text)


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata."""
    client = _client()
    if client:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """Phan tich doan van va chi tra ve JSON hop le:
{
  "summary": "tom tat 2-3 cau",
  "questions": ["cau hoi 1", "cau hoi 2", "cau hoi 3"],
  "context": "1 cau mo ta doan van nam o dau trong tai lieu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
                    },
                    {"role": "user", "content": f"Tai lieu: {source}\n\nDoan van:\n{text}"},
                ],
                max_tokens=400,
            )
            result = _parse_json(resp.choices[0].message.content)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            print(f"  Warning: Enrichment API failed: {exc}")

    return {
        "summary": summarize_chunk(text),
        "questions": _fallback_questions(text),
        "context": f"Trich tu {source}." if source else "Ngu canh tai lieu.",
        "metadata": _fallback_metadata(text),
    }


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Run enrichment pipeline on chunks."""
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    enriched = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**chunk.get("metadata", {}), **auto_meta},
                method="+".join(methods),
            )
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = "Nhan vien chinh thuc duoc nghi phep nam 12 ngay lam viec moi nam."
    print("=== Enrichment Pipeline Demo ===")
    print(summarize_chunk(sample))
    print(generate_hypothesis_questions(sample))
    print(contextual_prepend(sample, "So tay nhan vien"))
    print(extract_metadata(sample))
