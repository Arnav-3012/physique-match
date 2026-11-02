"""Eval runner: `python -m evals.run --suite rag` (cache, plan, safety suites come in later phases).

RAG suite protocol: evals/README.md and ADR-048 section 3. Scores BM25-only, dense-only and hybrid
on evals/rag.jsonl, prints the gate, the per-type table and every query with recall@5 < 1, and
writes evals/results/ (with the rag.jsonl and corpus hashes).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "services" / "api"))
from app.rag import corpus as corpus_mod  # noqa: E402
from app.rag.embed import Embedder  # noqa: E402
from app.rag.retrieve import Filters, Retriever  # noqa: E402

MODEL_DIR = ROOT / "ml" / "data" / "models" / "bge-small"
RAG_SET = Path(__file__).parent / "rag.jsonl"
MODES = ("bm25", "dense", "hybrid")
K, GATE, BOOTSTRAP, SEED = 5, 0.85, 10_000, 0


def recall_at_k(gold: list[str], ranked: list[str]) -> float:
    return len(set(gold) & set(ranked)) / min(len(gold), K)


def paired_bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = rng.choice(diff, size=(BOOTSTRAP, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_rag() -> dict:
    queries = [json.loads(line) for line in RAG_SET.read_text().splitlines()]
    retriever = Retriever(corpus_mod.load(), Embedder(MODEL_DIR))
    ranked = {
        mode: {q["id"]: retriever.search(q["query"], mode, Filters.from_dict(q["filters"]), K) for q in queries}
        for mode in MODES
    }
    recall = {m: np.array([recall_at_k(q["gold"], ranked[m][q["id"]]) for q in queries]) for m in MODES}
    hit = {m: np.array([bool(set(q["gold"]) & set(ranked[m][q["id"]])) for q in queries]) for m in MODES}
    types = sorted({q["type"] for q in queries})
    by_type = {m: {t: float(recall[m][[q["type"] == t for q in queries]].mean()) for t in types} for m in MODES}
    comparisons = {
        other: {
            "mean_diff": float((recall["hybrid"] - recall[other]).mean()),
            "ci95": paired_bootstrap_ci(recall["hybrid"] - recall[other]),
        }
        for other in ("bm25", "dense")
    }
    result = {
        "n_queries": len(queries),
        "rag_jsonl_sha256": hashlib.sha256(RAG_SET.read_bytes()).hexdigest(),
        "corpus_hash": retriever.corpus.manifest["corpus_hash"],
        "recall_at_5": {m: float(recall[m].mean()) for m in MODES},
        "hit_at_5": {m: float(hit[m].mean()) for m in MODES},
        "recall_at_5_by_type": by_type,
        "hybrid_minus": comparisons,
        "gate": {
            "recall_at_5_ge_0.85": bool(recall["hybrid"].mean() >= GATE),
            "hybrid_beats_bm25": bool(recall["hybrid"].mean() > recall["bm25"].mean()),
            "hybrid_beats_dense": bool(recall["hybrid"].mean() > recall["dense"].mean()),
        },
        "failures": {
            m: [
                {
                    "id": q["id"],
                    "query": q["query"],
                    "recall": float(r),
                    "top5": ranked[m][q["id"]],
                    "gold": q["gold"],
                }
                for q, r in zip(queries, recall[m], strict=False)
                if r < 1
            ]
            for m in MODES
        },
    }
    return result


def run_pipeline_suite(args) -> dict:
    """Plan and safety suites (ADR-058). Writes the full per-case result; returns the summary only."""
    live = args.llm == "live"
    if args.suite == "plan" and args.readable:
        # Scores merge into the SAVED result through the key file; no model call, no new plans.
        from evals.plan_suite import apply_readable

        out = Path(__file__).parent / "results" / (args.out or f"plan{'_live' if live else ''}.json")
        result = apply_readable(json.loads(out.read_text()), args.readable)
        out.write_text(json.dumps(result, indent=2) + "\n")
        return {k: v for k, v in result.items() if k != "rows"}
    if args.suite == "plan":
        from evals.plan_suite import run_plan

        only = args.cases.split(",") if args.cases else None
        result = run_plan(live, args.readable, only)
        if only:
            result["subset"] = f"only {only}; not the gate run, which needs all 15 cases"
    else:
        from evals.safety_suite import run_safety

        result = run_safety(live)
    out = Path(__file__).parent / "results" / (args.out or f"{args.suite}{'_live' if live else ''}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    bad = [r for r in result["rows"] if r.get("status") == "fail" or r.get("schema_valid") is False]
    summary = {k: v for k, v in result.items() if k != "rows"}
    return {**summary, "needs_attention": bad[:20], "full_result": str(out)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["rag", "plan", "safety"], required=True)
    parser.add_argument("--out", help="file name under evals/results/ (default: <suite>[_live].json)")
    parser.add_argument("--llm", choices=["off", "live"], default="off", help="plan / safety only")
    parser.add_argument("--cases", help="plan only: comma-separated case ids for a quick subset, e.g. p03,p11")
    parser.add_argument("--readable", type=Path, help="plan only: the owner's blind scores, 'A 1' per line")
    args = parser.parse_args()
    if args.suite in ("plan", "safety"):
        print(json.dumps(run_pipeline_suite(args), indent=2))
        return
    result = run_rag()
    args.out = args.out or "rag.json"
    out = Path(__file__).parent / "results" / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "failures"}
    print(json.dumps(summary, indent=2))
    print("\nqueries with recall@5 < 1 (hybrid):")
    for f in result["failures"]["hybrid"]:
        print(f"  {f['id']} {f['recall']:.2f}  {f['query']!r}\n    top5: {f['top5']}")
    print(f"\nfull failures for bm25 / dense: {out}")


if __name__ == "__main__":
    main()
