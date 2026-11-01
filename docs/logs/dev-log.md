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
Gate (P0): pytest 14 passed, 1 warning in 0.13s (PASS) · web live at https://physique-match.vercel.app/ (PASS) · `/health` both providers ok: user-reported, output not seen. First `/health` run showed `not_configured`; fixed with `--env-file ../../.env`. Repo: Arnav-3012/physique-match.
Closed 2026-11-01 on the user's confirmation.
Process: `/sensei` learn docs skipped for P0 (setup phase); only P3-P6 get them. Datasets downloaded only in their own phase.
Next: P1 (private 4-view capture). Awaiting user go-ahead.

## 2026-11-01 · Phase 01
Done: browser capture in `apps/web/src/capture/` (Pose Landmarker mask, user-paced 4-view turn, upload path, consent + 18+, review "what we send"), `lib/scan.ts`, vitest suites (yaw, sharpness, quality, silhouette, scan). ADR-023; 4 entries in mistakes.md.
Metrics: `npm run build` green (63 modules, 407 kB JS / 129 kB gzip). Laptop 4/4 valid, turn 7.6 s on the passing run (earlier runs 19.6 s, 5.9 s with failures, 52.9 s with the reference bug). Masks 2-4 KB each (384x512).
Gate (P1): captures ~8 of 10 (laptop + phone, user-reported, 4 phone captures valid), **short of 10; accepted by the user 2026-11-01**. Network-tab check **PASS** (user-reported 2026-11-01): every request was localhost, nothing to another domain. Per-capture times not recorded for all runs.
Blockers: none. Untested: busy backgrounds, iOS Safari GPU path, loose-clothing heuristic thresholds.
Process: `/sensei` skipped for P1 per protocol; user asked for explain docs, written to `learn/phase-01/` (model-architecture.md, phase-overview.md).
Next: P2 (BodyM pipeline + geometric baseline + body comp). BodyM download happens in P2 only.
