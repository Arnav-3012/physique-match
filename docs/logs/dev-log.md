# Dev log
One entry per session. ≤ 10 lines. Numbers over adjectives.

```
## YYYY-MM-DD · Phase NN
Done:
Metrics:      (e.g. MAE per region, RAG recall@5, cache hit rate, p50 latency)
Blockers:
Next:
```

## 2026-11-01 · Phase 00 (session 1)
Done: scaffold written by Claude (files only, no commands run): pyproject (uv workspace), schemas + tests, FastAPI `/health`, Dockerfile, Vite/React/TS/Tailwind hello-world, CI + HF-sync workflows, .env.example, CLAUDE.md working-agreement rules.
Metrics: none yet, nothing executed.
Blockers: user must run install/test commands (see chat); accounts/keys requested phase by phase.
Decision: HF Spaces/Hub dropped (not free) -> ADR-022; API + models local, web on Vercel. Removed HF token, deploy-space workflow, HF Space README frontmatter.
Also: ruff isort `known-first-party` fix; prettier pre-commit hook removed (bad tag). 3 entries added to mistakes.md.
Gate (P0): reported passed by user on 2026-11-01 (pytest, Vercel URL, `/health` with both keys); outputs not shared with Claude, so no numbers recorded. Repo: Arnav-3012/physique-match.
Process: `/sensei` learn docs skipped for P0 (setup phase); only P3-P6 get them. Datasets downloaded only in their own phase.
Next: P1 (private 4-view capture). Awaiting user go-ahead.
