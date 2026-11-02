"""Regenerate the committed planner rank fixtures (ADR-055). From the repo root:

    PYTHONPATH=.:services/api uv run python tests/fixtures/make_planner_ranks.py

`planner_ranks_bm25.json` needs only the corpus. `planner_ranks_hybrid.json` is a snapshot of the
production retriever (BM25 + bge-small + RRF) and needs ml/data/models/bge-small (gitignored); both are
stamped with the corpus hash, and a test fails when the corpus changes without a regeneration.
"""

import json
from pathlib import Path

from app.planner.candidates import ranked_candidates
from app.planner.params import TOP_K
from app.rag import corpus as kb_corpus
from app.rag.embed import Embedder
from app.rag.retrieve import Retriever
from physique_schemas.models import UserProfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
MODEL_DIR = ROOT / "ml" / "data" / "models" / "bge-small"
BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip


def rank_key(profile: UserProfile) -> str:
    return f"{profile.level}|{','.join(sorted(profile.equipment)) or 'body only'}"


def main() -> None:
    cases = [json.loads(line) for line in (ROOT / "evals" / "plan.jsonl").read_text().splitlines()]
    profiles = {
        rank_key(p): p for p in (UserProfile(**{**BASE_PROFILE, **c["input"].get("profile", {})}) for c in cases)
    }
    corpus = kb_corpus.load()
    for mode in ("hybrid", "bm25"):
        retriever = Retriever(corpus, Embedder(MODEL_DIR) if mode == "hybrid" else None)

        def search(query, filters, mode=mode, retriever=retriever):
            return retriever.search(query, mode, filters, k=TOP_K)

        ranks = {key: ranked_candidates(p, search) for key, p in sorted(profiles.items())}
        out = {"corpus_hash": corpus.manifest["corpus_hash"], "mode": mode, "ranks": ranks}
        (OUT / f"planner_ranks_{mode}.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(mode, len(ranks), "profiles")


if __name__ == "__main__":
    main()
