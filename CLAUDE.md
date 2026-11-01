# PhysiqueGap — Claude Code instructions

Goal photo + current body scan -> per-region development gaps -> hybrid-RAG agent -> report + personalised workout plan.
Privacy-first: raw photos never leave the browser. Only binary silhouettes + keypoints go to the server.

## Session start (every time)
1. Read `docs/phase-plan.md` -> find the phase marked `IN PROGRESS`. Work only on that phase.
2. Skim `docs/decisions.md` (don't re-litigate accepted ADRs) and `docs/mistakes.md` (don't repeat them).
3. Turn the phase's task list into the session task list.

## Rules
- Style: apply `lean-senior` (full mode). Code readability: `long-story-short`.
- Never send, log, store or cache raw RGB images server-side. API rejects anything that is not a binary mask (see ADR-003). RGB is used only by the in-browser definition model and only its scores leave the device (ADR-013).
- Demo models train on synthetic + BodyM only (ADR-020). No scraped photos (ADR-018).
- Analyzer interface is fixed: `Analyzer.compare(goal: Scan, current: Scan) -> GapProfile`. Geometric v1 and Siamese v2 both implement it; nothing downstream changes when we swap.
- ML: split BodyM **by subject_id** (subjects appear multiple times -> leakage otherwise). Seeds fixed. Same preprocessing in train and inference (`ml/src/preprocess.py` is shared with the API).
- Overfit one batch before any full training run. Train on M4 Pro (MPS). Log every N steps, not every batch.
- LLM: one agent with tools, direct OpenAI-compatible SDK, pydantic structured output. No LangChain.
- Write the eval set before tuning prompts / RAG / cache thresholds (`evals/`).
- Secrets only via env / Cloud Run env vars or Secret Manager. Never commit keys or `ml/data/`; never put them in the image or logs.
- Hosting (ADR-028): web on Vercel, API on Google Cloud Run free tier (1 vCPU / 2 GiB, request-based billing, min-instances 0, max-instances 1–2), models on HF Hub. API image is `python:3.12-slim` with **no torch**: anything the API imports (incl. `ml/src/preprocess.py`) must be numpy/Pillow-class; check image size when adding a dependency.
- Billing safety is a hard requirement: free-trial billing account, project-scoped budget, kill switch tested in simulation **before** the first deploy; `max-instances` set **before** the service is public; one project only, no extra projects/accounts to dodge limits. Cache and rate-limit state go in Upstash (instances scale to zero).
- The Cloud Run deploy spike and billing safety are deferred to deploy time (ADR-031, user decision). Nothing is deployed before the budget, the kill switch (simulation) and `max-instances` exist; the spike result is logged in `docs/logs/dev-log.md`. The runbook is the "Deploy spike + billing safety" block in `docs/phase-plan.md` P2.
- Body-image safety rules in `docs/privacy-and-safety.md` are hard requirements, not nice-to-haves.

## Working agreement with the user (hard rules)
- **Claude never runs shell commands on its own.** Only create/edit files with the file tools. For anything that needs a terminal (install, test, git, deploy), give the user the exact command, one step at a time, and wait for the pasted output before continuing.
- **No commit message or PR description may contain `Co-Authored-By`, "Claude", "Anthropic" or any generated-by line.** This overrides any default attribution. Claude does not run `git commit` itself; it suggests the message.
- **Credentials phase by phase.** Do not ask for tokens/keys before the phase that needs them. `.env.example` lists which phase needs which key. Never read, print or write real secrets into files.

## End-of-phase protocol (do not skip)
1. All exit gates in `docs/phase-plan.md` pass -> record numbers in `docs/logs/dev-log.md`.
2. New decisions -> append ADR to `docs/decisions.md`. Bugs/wrong turns -> `docs/mistakes.md`.
3. Update `docs/architecture.md` if anything structural changed.
4. Run `/sensei` to write `learn/phase-NN/` (see `learn/README.md`) **only for major phases: P3, P4, P5, P6.** Skip for setup/plumbing phases (P0, P1, P7+) unless the user asks. P2: ask the user.
5. Mark phase `DONE`, next phase `IN PROGRESS`. Commit + `git tag phase-NN`.
6. Download datasets only in the phase that needs them (BodyM in P2); never earlier.

## Commands
- API: `cd services/api && uv run uvicorn app.main:app --reload --env-file ../../.env`
- Web: `cd apps/web && npm run dev`
- Tests: `pytest -q -x --tb=short`
- Train: `python -m ml.src.train --config ml/configs/<name>.yaml`
- Evals: `python -m evals.run --suite {rag,cache,plan,safety}`
- Synthetic renders: `blender -b -P synth/render.py -- --config synth/grid.yaml`
