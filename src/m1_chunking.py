from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies.

Implements semantic, hierarchical, and structure-aware chunking while keeping
lightweight fallbacks so the lab can run in constrained environments.
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer from PDF. Return empty string for scanned PDFs."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown files and PDFs that have a text layer from data/."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Warning: skipping {os.path.basename(fp)} because it has no text layer.")

    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Basic paragraph chunking baseline."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]


def _token_similarity(a: str, b: str) -> float:
    toks_a = set(re.findall(r"\w+", a.lower(), flags=re.UNICODE))
    toks_b = set(re.findall(r"\w+", b.lower(), flags=re.UNICODE))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group neighboring sentences by semantic similarity."""
    metadata = metadata or {}
    sentences = _sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    try:
        if os.getenv("USE_REAL_MODELS", "0") != "1":
            raise RuntimeError("USE_REAL_MODELS is not enabled")
        from numpy import dot
        from numpy.linalg import norm
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, show_progress_bar=False)

        def similarity(i: int) -> float:
            a, b = embeddings[i - 1], embeddings[i]
            return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    except Exception as exc:
        print(f"  Warning: semantic model unavailable, using token fallback: {exc}")

        def similarity(i: int) -> float:
            return _token_similarity(sentences[i - 1], sentences[i])

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        effective_threshold = threshold if os.getenv("USE_REAL_MODELS", "0") == "1" else min(threshold, 0.2)
        if similarity(i) < effective_threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks = []
    for i, group in enumerate(groups):
        chunk_text = " ".join(group).strip()
        if chunk_text:
            chunks.append(Chunk(chunk_text, {**metadata, "strategy": "semantic", "chunk_index": i}))
    return chunks


def _split_long_text(text: str, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    parts = re.findall(r".{1,%d}(?:\s+|$)" % max_size, text, flags=re.DOTALL)
    return [part.strip() for part in parts if part.strip()]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and child chunks for precise retrieval."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()] or ([text.strip()] if text.strip() else [])

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > parent_size:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid}))
            current = para
        else:
            current = candidate
    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid}))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        child_current = ""
        for para in [p.strip() for p in parent.text.split("\n\n") if p.strip()]:
            for piece in _split_long_text(para, child_size):
                candidate = f"{child_current}\n\n{piece}".strip() if child_current else piece
                if child_current and len(candidate) > child_size:
                    children.append(
                        Chunk(
                            child_current.strip(),
                            {**metadata, "chunk_type": "child", "parent_id": pid, "chunk_index": len(children)},
                            parent_id=pid,
                        )
                    )
                    child_current = piece
                else:
                    child_current = candidate
        if child_current.strip():
            children.append(
                Chunk(
                    child_current.strip(),
                    {**metadata, "chunk_type": "child", "parent_id": pid, "chunk_index": len(children)},
                    parent_id=pid,
                )
            )

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk markdown by level 1-3 headers while preserving section text."""
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if not current_header and not "".join(current_lines).strip():
            current_lines = []
            return
        lines = ([current_header] if current_header else []) + current_lines
        chunk_text = "\n".join(lines).strip()
        if chunk_text:
            section = current_header.lstrip("#").strip() if current_header else "root"
            chunks.append(
                Chunk(
                    chunk_text,
                    {**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
                )
            )
        current_lines = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+$", line):
            flush()
            current_header = line.strip()
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        return [Chunk(text.strip(), {**metadata, "section": "root", "strategy": "structure", "chunk_index": 0})]
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies on documents and compare simple length stats."""
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, stats in results.items():
        print(f"{name:<15} {stats['count']:>7} {stats['avg_len']:>5} {stats['min_len']:>5} {stats['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    compare_strategies(docs)
