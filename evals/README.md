# Evals

## RAG suite (`rag.jsonl`, ADR-048 section 3)
One JSON object per line: `id`, `type` (`planner`, `synonym`, `exact`, `guideline`), `query`, `filters`, `gold`.
- `filters` keys `muscle` (matches the exercise's primary muscles), `equipment` and `level` take lists; a missing key means no filter. Guideline queries have no filters.
- Chunk ids: exercises `ex:<free-exercise-db id>`; guideline docs `doc:<doc id>#<heading slug>`, one chunk per `#` / `##` heading with a non-empty body, slug = lowercased heading with every run of non-alphanumerics replaced by one `-`, ends trimmed. `kb/build_index.py` must produce exactly these ids.
- Gold was labelled by hand from the full strength/powerlifting id list of `kb/raw/exercises.json` (2026-11-02) and the headings of `kb/docs/*.md`, before any retrieval code existed. The aim was completeness within each query's filter (every matching exercise that fits the query is listed); a miss is possible, and a missed gold item can only lower recall, never raise it. Judgement calls (for example straight-arm pulldowns counted as pulldowns) are in the file as written and are not changed after the first scoring run.

Metrics (fixed before the first run):
- `recall@5 = |gold ∩ top5| / min(|gold|, 5)`, mean over queries; `hit@5` = share of queries with at least one gold item in the top 5.
- Hybrid (BM25 top-20 + dense top-20, RRF k = 60) must have a strictly higher point estimate of recall@5 than BM25-only and dense-only. Paired-bootstrap CI over queries is reported and labelled underpowered (33 queries).
- Gate: recall@5 >= 0.85 and hybrid beats each single retriever, on this demo set. If it fails, look at the failures first; allowed fixes are corpus text, chunking and metadata, each logged. No tuning of RRF k or list sizes after the first run.
- Report per type; `exact` and `planner` queries are expected to favour BM25 and `guideline` and `synonym` the dense retriever.

The cache, plan and safety suites are later phases.
