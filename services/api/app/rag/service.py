"""Retrieval for /plan, loaded once at startup (ADR-057).

Hybrid (BM25 + bge-small + RRF) when the model is present. Without it the API falls back to BM25 with a
logged warning, unless REQUIRE_EMBEDDER is true, which fails startup (like REQUIRE_MODEL for the analyzer).
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from app.planner.params import TOP_K
from app.rag.corpus import KB_DIR, Corpus, load
from app.rag.embed import MODEL_FILE, TOKENIZER_FILE, Embedder
from app.rag.retrieve import Filters, Retriever

log = logging.getLogger("retrieval")
MODEL_DIR = Path(os.environ.get("EMBEDDER_DIR", KB_DIR.parent / "ml" / "data" / "models" / "bge-small"))


@dataclass
class Retrieval:
    corpus: Corpus
    retriever: Retriever
    mode: str  # "hybrid" | "bm25"

    def search(self, query: str, filters: Filters) -> list[str]:
        return self.retriever.search(query, self.mode, filters, k=TOP_K)


def load_retrieval(model_dir: Path = MODEL_DIR, require: bool | None = None) -> Retrieval:
    if require is None:
        require = os.environ.get("REQUIRE_EMBEDDER", "").strip().lower() in {"1", "true", "yes"}
    corpus = load()
    if (model_dir / MODEL_FILE).exists() and (model_dir / TOKENIZER_FILE).exists():
        return Retrieval(corpus, Retriever(corpus, Embedder(model_dir)), "hybrid")
    if require:
        raise RuntimeError(f"REQUIRE_EMBEDDER is set and the embedder files are missing in {model_dir}")
    log.warning("embedder files missing in %s: /plan retrieval falls back to BM25", model_dir)
    return Retrieval(corpus, Retriever(corpus, None), "bm25")
