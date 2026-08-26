from __future__ import annotations

"""Module 2: Hybrid Search - BM25 Vietnamese + dense vectors + RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


class HashingEncoder:
    """Small deterministic encoder fallback with the same interface as SentenceTransformer."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _encode_one(self, text: str):
        import numpy as np

        vector = np.zeros(self.dim, dtype=float)
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
            vector[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np

        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.array([self._encode_one(text) for text in texts])


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text and expand underthesea underscores for BM25."""
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in chunks
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i].get("text", ""),
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if float(scores[i]) > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_REAL_MODELS", "0") == "1":
                try:
                    from sentence_transformers import SentenceTransformer

                    self._encoder = SentenceTransformer(EMBEDDING_MODEL)
                except Exception as exc:
                    print(f"  Warning: embedding model unavailable, using hashing encoder: {exc}")
                    self._encoder = HashingEncoder()
            else:
                self._encoder = HashingEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunks[i].get("metadata", {}), "text": chunks[i].get("text", "")},
            )
            for i, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        return [
            SearchResult(
                text=point.payload.get("text", "") if point.payload else "",
                score=float(point.score),
                metadata=point.payload or {},
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = sum(1 / (k + rank + 1))."""
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + dense search + RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhan vien duoc nghi phep nam"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
