from __future__ import annotations

"""Module 3: Reranking - CrossEncoder top-N to top-k."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K  # noqa: E402


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if os.getenv("USE_REAL_MODELS", "0") != "1":
                    raise RuntimeError("USE_REAL_MODELS is not enabled")
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  Warning: cross-encoder unavailable, using lexical reranker: {exc}")
                self._model = False
        return self._model

    @staticmethod
    def _lexical_score(query: str, text: str, original_score: float = 0.0) -> float:
        query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
        doc_tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        overlap = sum(1 for token in doc_tokens if token in query_tokens)
        return float(overlap) + float(original_score) * 0.01

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents and return top-k results sorted by rerank_score."""
        if not documents:
            return []

        model = self._load_model()
        if model:
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = model.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
        else:
            scores = [
                self._lexical_score(query, doc.get("text", ""), float(doc.get("score", 0.0)))
                for doc in documents
            ]

        scored = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight optional reranker."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        scored = [
            (
                CrossEncoderReranker._lexical_score(query, doc.get("text", ""), float(doc.get("score", 0.0))),
                doc,
            )
            for doc in documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
