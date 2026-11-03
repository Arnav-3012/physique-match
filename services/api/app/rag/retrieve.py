"""BM25 + dense + RRF with metadata filters (ADR-048 section 3).

Protocol, fixed before the first scoring run: BM25 top-20 and dense top-20 are fused by
reciprocal rank fusion with k = 60 (Cormack 2009, not tuned), then cut to top-k. A BM25
candidate needs a score > 0 (at least one query term in the chunk).

`muscle` matches an exercise's primary muscles; `equipment` and `level` match exactly. Any
non-empty filter keeps only exercise chunks from the planner's default pool (strength and
powerlifting, `plannable`), unless `all_categories` is set. A missing or empty filter key means
no filter; unfiltered queries search the whole index.

The pool rule was added after the first scoring run showed cardio, olympic and plyometric chunks
leaking into filtered results (see dev-log).
"""

import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.corpus import Corpus
from app.rag.embed import Embedder

RRF_K = 60
LIST_SIZE = 20


@dataclass(frozen=True)
class Filters:
    muscle: tuple[str, ...] = ()
    equipment: tuple[str, ...] = ()
    level: tuple[str, ...] = ()
    all_categories: bool = False  # default pool is strength + powerlifting (`plannable`), ADR-048

    @classmethod
    def from_dict(cls, d: dict) -> "Filters":
        return cls(*(tuple(d.get(k, ())) for k in ("muscle", "equipment", "level")))


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Retriever:
    corpus: Corpus
    embedder: Embedder
    bm25: BM25Okapi = field(init=False)

    def __post_init__(self):
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.corpus.chunks])

    def mask(self, filters: Filters) -> np.ndarray:
        if filters == Filters():
            return np.ones(len(self.corpus.chunks), dtype=bool)

        def ok(c: dict) -> bool:
            return (
                c["kind"] == "exercise"
                and (filters.all_categories or c["plannable"])
                and (not filters.muscle or bool(set(c["muscle"]) & set(filters.muscle)))
                and (not filters.equipment or c["equipment"] in filters.equipment)
                and (not filters.level or c["level"] in filters.level)
            )

        return np.array([ok(c) for c in self.corpus.chunks])

    def _top(self, scores: np.ndarray, allowed: np.ndarray, k: int, floor: float | None) -> list[int]:
        scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        return [int(i) for i in order if np.isfinite(scores[i]) and (floor is None or scores[i] > floor)]

    def bm25_top(self, query: str, filters: Filters, k: int = LIST_SIZE) -> list[int]:
        return self._top(self.bm25.get_scores(tokens(query)), self.mask(filters), k, floor=0.0)

    def dense_top(self, query: str, filters: Filters, k: int = LIST_SIZE) -> list[int]:
        scores = self.corpus.embeddings @ self.embedder.embed_queries([query])[0]
        return self._top(scores, self.mask(filters), k, floor=None)

    def search(self, query: str, mode: str = "hybrid", filters: Filters | None = None, k: int = 5) -> list[str]:
        """Chunk ids, best first. Modes: bm25, dense, hybrid (the first two are for the eval)."""
        filters = filters or Filters()
        if mode == "bm25":
            ranked = self.bm25_top(query, filters)[:k]
        elif mode == "dense":
            ranked = self.dense_top(query, filters)[:k]
        elif mode == "hybrid":
            fused: dict[int, float] = {}
            for ranking in (self.bm25_top(query, filters), self.dense_top(query, filters)):
                for rank, i in enumerate(ranking, start=1):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(fused, key=lambda i: (-fused[i], i))[:k]
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [self.corpus.chunks[i]["id"] for i in ranked]
